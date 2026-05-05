"""
tesbi_e2e_7param.py 

7-Parameter End-to-End TeSBI Pipeline.
- Freed 3 additional parameters: cost_time, cost_go_error, cost_go_missing (Total 7).
- Upgraded Architecture: d_model=128, n_layers=4, Zuko hidden=[256, 256].
- Default N_SIMS = 100,000.
"""
import sys
import os
import math
import random
import argparse
import warnings
import multiprocessing
import gc
from pathlib import Path
from typing import Dict, Optional
import zipfile
import tempfile

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from joblib import Parallel, delayed
import zuko

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

# Local modules
from utils.preprocessing import preprocessing
from core.models import POMDP
from core import simulation

# ==============================================================================
# CPU CORES SETUP (No global GPU init to prevent multi-processing deadlock)
# ==============================================================================
try:
    N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK"))
except (ValueError, TypeError):
    N_JOBS = multiprocessing.cpu_count()

# ==============================================================================
# CONFIGURATION
# ==============================================================================
USE_PREPROCESSING = True # when use  real abcd data
# USE_PREPROCESSING = False # when use example data

MODEL_TAG = "6p_v9"

PARAM_RANGES = {
    "q_d_n": (0.0, 1.0),
    "q_d":   (0.5, 1.0),
    "q_s":   (0.5, 1.0),
    "cost_stop_error": (1, 50.0),
    "cost_time":       (0.01, 0.5),  
    "tau":   (4, 16),
}

LINEAR_PARAMS = ["q_d_n", "q_d", "q_s", "tau"]
LOG_PARAMS = ["cost_stop_error", "cost_time"]
PARAM_ORDER = LINEAR_PARAMS + LOG_PARAMS

FIXED_PARAMS = {
    "rate_stop_trial": 1.0 / 6.0,
    "cost_go_error": 3.0,
    "cost_go_missing": 1.0,
    "q_s_n": 0.01,
    "inv_temp": 20
}

RESULT_LEVELS = ["GS", "GE", "GM", "SS", "SE"]
ART_DIR = Path("outputs/")
DATASET_PATH = ART_DIR / f"simulated_dataset_{MODEL_TAG}.pt"
MODEL_PATH = ART_DIR / f"amortized_inference_net_{MODEL_TAG}.pth"
RECOVERY_CSV = ART_DIR / f"params_recovery_{MODEL_TAG}.csv"
POST_SUMMARY_CSV = ART_DIR / f"params_posteriors_{MODEL_TAG}.csv"

# ==============================================================================
# PARAMETER SCALING
# ==============================================================================
_param_min, _param_max = [], []
for k in PARAM_ORDER:
    lo, hi = PARAM_RANGES[k]
    if k in LOG_PARAMS:
        lo, hi = math.log(lo), math.log(hi)
    _param_min.append(lo)
    _param_max.append(hi)

PARAM_TRANSFORMED_MIN = np.array(_param_min, dtype=np.float32)
PARAM_TRANSFORMED_MAX = np.array(_param_max, dtype=np.float32)

def sample_prior(n_samples: int) -> np.ndarray:
    samples = []
    for k in PARAM_ORDER:
        lo, hi = PARAM_RANGES[k]
        if k in LOG_PARAMS:
            samples.append(np.random.uniform(math.log(lo), math.log(hi), n_samples))
        else:
            samples.append(np.random.uniform(lo, hi, n_samples))
    return np.stack(samples, axis=1)

def scale_params(raw_array: np.ndarray) -> np.ndarray:
    return (raw_array - PARAM_TRANSFORMED_MIN) / (PARAM_TRANSFORMED_MAX - PARAM_TRANSFORMED_MIN + 1e-8)

def unscale_params(scaled_array: np.ndarray) -> np.ndarray:
    if isinstance(scaled_array, torch.Tensor):
        scaled_array = scaled_array.detach().cpu().numpy()
    scaled_array = np.clip(scaled_array, 0.0, 1.0)
    return scaled_array * (PARAM_TRANSFORMED_MAX - PARAM_TRANSFORMED_MIN) + PARAM_TRANSFORMED_MIN


def scaled_to_dict(scaled_vec: np.ndarray) -> Dict[str, float]:
    unscaled = unscale_params(scaled_vec)
    out = {k: (float(np.exp(unscaled[i])) if k in LOG_PARAMS else float(unscaled[i])) for i, k in enumerate(PARAM_ORDER)}
    
    if "tau" in out:
        out["tau"] = int(round(out["tau"]))
        
    out.update(FIXED_PARAMS)
    return out

# ==============================================================================
# SIMULATOR & FEATURES
# ==============================================================================
def simulate_data(params: Dict[str, float], seed: Optional[int] = None) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    pomdp = POMDP(**params)
    pomdp.value_iteration_tensor()
    arr = simulation.simu_task(pomdp)
    return pd.DataFrame(arr, columns=["result", "rt", "ssd"])

def build_per_trial_features(df: pd.DataFrame, use_index=True) -> np.ndarray:
    N = len(df)
    idx_map = {k: i for i, k in enumerate(RESULT_LEVELS)}
    oh = np.zeros((N, len(RESULT_LEVELS)), dtype=np.float32)
    for t, r in enumerate(df["result"].astype(str).values):
        if r in idx_map: oh[t, idx_map[r]] = 1.0

    rt = np.nan_to_num(pd.to_numeric(df["rt"], errors="coerce").values.astype(np.float32), nan=0.0)
    ssd = np.nan_to_num(pd.to_numeric(df["ssd"], errors="coerce").values.astype(np.float32), nan=0.0)
    rt_nan, ssd_nan = (rt == 0.0).astype(np.float32), (ssd == 0.0).astype(np.float32)

    prev_oh = np.roll(oh, 1, axis=0); prev_oh[0, :] = 0.0
    prev_rt = np.roll(rt, 1); prev_rt[0] = 0.0
    prev_ssd = np.roll(ssd, 1); prev_ssd[0] = 0.0
    delta_ssd = ssd - prev_ssd

    cols = [oh, rt[:, None], ssd[:, None], rt_nan[:, None], ssd_nan[:, None], prev_oh, prev_rt[:, None], prev_ssd[:, None], delta_ssd[:, None]]
    if use_index: cols.append(np.linspace(-1.0, 1.0, N, dtype=np.float32)[:, None])
    return np.concatenate(cols, axis=1).astype(np.float32)

def worker_simulate(i, omega, seed_offset=0):
    omega_scaled = scale_params(omega)
    param_dict = scaled_to_dict(omega_scaled)
    df = simulate_data(param_dict, seed=seed_offset + i)
    return build_per_trial_features(df), omega_scaled

# ==============================================================================
# END-TO-END MODEL: TRANSFORMER + ZUKO FLOW
# ==============================================================================
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x): return x + self.pe[:x.size(1)].unsqueeze(0)

class EndToEndTeSBI(nn.Module):
    def __init__(self, in_dim=18, d_model=128, n_heads=8, n_layers=4, param_dim=10):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pe = SinusoidalPE(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, batch_first=True),
            num_layers=n_layers
        )
        self.context_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.flow = zuko.flows.NSF(features=param_dim, context=d_model, hidden_features=[256, 256])

    def forward(self, x_seq, true_params_scaled=None):
        h = self.encoder(self.pe(self.proj(x_seq)))
        context = self.context_head(h.mean(dim=1))
        dist = self.flow(context)
        return dist.log_prob(true_params_scaled) if true_params_scaled is not None else dist

# ==============================================================================
# PIPELINE STAGES
# ==============================================================================
def stage_simulate(n_sims: int):
    print(f"\n [Simulate] Generating {n_sims} datasets in parallel using {N_JOBS} CPUs...")
    ART_DIR.mkdir(parents=True, exist_ok=True)
    
    omegas = sample_prior(n_sims)
    results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(worker_simulate)(i, omegas[i]) for i in range(n_sims)
    )
    
    X_data = [r[0] for r in results]
    Y_data = [r[1] for r in results]
    
    X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_data), dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    torch.save(dataset, DATASET_PATH)
    print(f"[Simulate] Dataset successfully saved to {DATASET_PATH}")

def stage_train(epochs: int, batch_size=128, lr=1e-3, patience=15):
    # Safe to initialize CUDA here since multiprocessing is done
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n [Train] Starting E2E Joint Training on {str(device).upper()}...")
    if device.type == 'cuda': print(f" [Train] GPU Name: {torch.cuda.get_device_name(0)}")
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset missing at {DATASET_PATH}. Run '--stage simulate' first.")
    
    dataset = torch.load(DATASET_PATH, weights_only=False)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = EndToEndTeSBI(in_dim=18, param_dim=len(PARAM_ORDER)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    wait = 0
    
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            log_probs = model(Xb, true_params_scaled=Yb)
            loss = -log_probs.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
            
        train_loss /= len(train_ds)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                log_probs = model(Xb, true_params_scaled=Yb)
                loss = -log_probs.mean()
                val_loss += loss.item() * Xb.size(0)
        val_loss /= len(val_ds)
        
        print(f"Epoch {ep+1:03d}/{epochs} | Train NLL: {train_loss:.4f} | Val NLL: {val_loss:.4f}", end="")
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(" * [Saved Best]")
        else:
            wait += 1
            print(f" (Wait {wait})")
            if wait >= patience:
                print("Early stopping triggered.")
                break

def stage_recover(K: int, num_post: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n [Recovery] Checking {K} ground-truth cases on {str(device).upper()}...")
    
    model = EndToEndTeSBI(in_dim=18, param_dim=len(PARAM_ORDER)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    omegas_raw = sample_prior(K)
    details = []
    
    for i in range(K):
        gt_params = scaled_to_dict(scale_params(omegas_raw[i]))
        df_obs = simulate_data(gt_params, seed=4242 + i)
        
        X = build_per_trial_features(df_obs)
        tensor_x = torch.tensor(X[None, ...], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            posterior_dist = model(tensor_x)
            samples_scaled = posterior_dist.sample((num_post,))[:, 0, :].cpu().numpy()
            
        row = {"case": i}
        for k_idx, k in enumerate(PARAM_ORDER):
            s_unscaled = unscale_params(samples_scaled)[:, k_idx]
            vals = np.exp(s_unscaled) if k in LOG_PARAMS else s_unscaled
            
            mu, lo, hi = np.mean(vals), np.percentile(vals, 5), np.percentile(vals, 95)
            gt_val = gt_params[k]
            row[f"gt_{k}"] = gt_val
            row[f"mu_{k}"] = mu
            row[f"hit90_{k}"] = 1.0 if lo <= gt_val <= hi else 0.0
        details.append(row)

    pd.DataFrame(details).to_csv(RECOVERY_CSV, index=False)
    print(f"\n [Recovery] Results saved to {RECOVERY_CSV}")

def stage_inference(sst_folder: str, glob_pat: str, num_samples: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = ART_DIR / "subjects"
    out_root.mkdir(parents=True, exist_ok=True)
    
    folder_path = Path(sst_folder)
    zip_files = list(folder_path.glob("*.zip"))
    csv_files = list(folder_path.glob(glob_pat))
    
    model = EndToEndTeSBI(in_dim=18, param_dim=len(PARAM_ORDER)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    summaries = []
    
    if zip_files:
        print(f"\n [Inference] Detected ZIP files. Processing {len(zip_files)} real subjects on {str(device).upper()}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            for fp in zip_files:
                try:
                    subject_id = fp.name.split('_baseline_')[0]
                    with zipfile.ZipFile(fp, 'r') as zr:
                        zr.extractall(tmpdir)
                    
                    extracted_csvs = list(Path(tmpdir).rglob(f"*{subject_id}*.csv"))
                    if not extracted_csvs:
                        continue
                        
                    target_csv = extracted_csvs[0]
                    df_obs = preprocessing(str(target_csv)) if USE_PREPROCESSING else pd.read_csv(str(target_csv))
                    
                    X = build_per_trial_features(df_obs)
                    tensor_x = torch.tensor(X[None, ...], dtype=torch.float32).to(device)

                    with torch.no_grad():
                        posterior_dist = model(tensor_x)
                        samples_scaled = posterior_dist.sample((num_samples,))[:, 0, :].cpu().numpy()
                    
                    rows = [scaled_to_dict(s) for s in samples_scaled]
                    post_df = pd.DataFrame(rows)
                    summ = post_df.describe(percentiles=[0.05, 0.5, 0.95]).T[["mean", "std", "5%", "50%", "95%"]]
                    
                    subj_dir = out_root / f"{subject_id}_baseline"
                    subj_dir.mkdir(exist_ok=True)
                    post_df.to_csv(subj_dir / "posterior_samples.csv", index=False)
                    summ.to_csv(subj_dir / "posterior_summary.csv")
                    
                    s_row = summ.copy(); s_row["subject_id"] = subject_id; s_row["subject_year"] = "baseline"
                    summaries.append(s_row.reset_index())
                    
                    target_csv.unlink()
                except Exception as e:
                    print(f" [Error] {fp.name}: {e}")
    else:
        print(f"\n [Inference] Processing {len(csv_files)} real subjects from CSV files on {str(device).upper()}...")
        for fp in csv_files:
            try:
                parts = fp.name.split('_')
                sid = parts[1] if len(parts) > 1 else fp.stem
                year = parts[2] if len(parts) > 2 else "unknown"

                df_obs = preprocessing(str(fp)) if USE_PREPROCESSING else pd.read_csv(str(fp))
                X = build_per_trial_features(df_obs)
                tensor_x = torch.tensor(X[None, ...], dtype=torch.float32).to(device)

                with torch.no_grad():
                    posterior_dist = model(tensor_x)
                    samples_scaled = posterior_dist.sample((num_samples,))[:, 0, :].cpu().numpy()
                
                rows = [scaled_to_dict(s) for s in samples_scaled]
                post_df = pd.DataFrame(rows)
                summ = post_df.describe(percentiles=[0.05, 0.5, 0.95]).T[["mean", "std", "5%", "50%", "95%"]]
                
                subj_dir = out_root / f"{sid}_{year}"
                subj_dir.mkdir(exist_ok=True)
                post_df.to_csv(subj_dir / "posterior_samples.csv", index=False)
                summ.to_csv(subj_dir / "posterior_summary.csv")
                
                s_row = summ.copy(); s_row["subject_id"] = sid; s_row["subject_year"] = year
                summaries.append(s_row.reset_index())
            except Exception as e:
                print(f" [Error] {fp.name}: {e}")

    if summaries:
        pd.concat(summaries).to_csv(POST_SUMMARY_CSV, index=False)
        print(f"\n [Inference] All summaries saved to {POST_SUMMARY_CSV}")
    else:
        print("\n [Inference] No summaries generated. Check if files were properly read.")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    
    SEED = 137
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    parser = argparse.ArgumentParser(description="End-to-End TeSBI Pipeline (Transformer + Zuko)")
    parser.add_argument("--stage", choices=["all", "simulate", "train", "recover", "posterior"], required=True)
    parser.add_argument("--n_sims", type=int, default=100000) # Updated Default
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--K", type=int, default=20, help="Recovery test cases")
    parser.add_argument("--num_post", type=int, default=1000, help="Samples per recovery/inference")
    parser.add_argument("--sst_folder", type=str, default=None, help="Folder of real CSVs")
    parser.add_argument("--glob_pat", type=str, default="*.csv", help="File pattern for inference")

    args = parser.parse_args()
    
    # 1. GENERATE DATA (PURE CPU)
    if args.stage in ["all", "simulate"]:
        stage_simulate(args.n_sims)
        
    # 2. CLEAR MEMORY BEFORE GPU TASKS
    if args.stage == "all":
        print("\n [System] Simulating complete. Running Garbage Collection before loading CUDA...")
        gc.collect()
        
    # 3. TRAIN (GPU INITIALIZED HERE)
    if args.stage in ["all", "train"]:
        stage_train(args.epochs, args.batch_size)
        
    # 4. RECOVER (GPU)
    if args.stage in ["all", "recover"]:
        stage_recover(args.K, args.num_post)
        
    # 5. INFERENCE (GPU)
    if args.stage in ["all", "posterior"]:
        if args.sst_folder:
            stage_inference(args.sst_folder, args.glob_pat, args.num_post)
        else:
            print(" [WARNING] Skipping posterior inference: Please provide --sst_folder.")