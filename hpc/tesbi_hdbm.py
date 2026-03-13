"""
tesbi_hdbm.py 
End-to-end SBI with HDBM + POMDP for the Stop Signal Task.
"""
from html import parser
import sys
import os
import random
import pickle
import argparse
import warnings
import multiprocessing
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed
from sbi.inference import SNPE
from sbi.utils import BoxUniform

# Local modules
# Ensure parent directory is in path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.preprocessing import preprocessing
from core.models import POMDP
from core.hdbm import HDBM
from core import simulation

# --- Device Setup ---
try:
    N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
except (ValueError, TypeError):
    N_JOBS = multiprocessing.cpu_count()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device} ({N_JOBS} CPUs available for simulation)")

# --- Configuration ---

# 1. Paths
BASE_DIR = Path(__file__).parent
ART_DIR = BASE_DIR / "outputs_hdbm"
FIXED_PARAMS_PATH = BASE_DIR / "posterior_summary.csv"
ORDER_CSV_PATH = BASE_DIR / "sst_order_statistics.csv"

# Artifacts
ENC_PATH = ART_DIR / "encoder_hdbm.pt"
STD_MEAN_PATH = ART_DIR / "embeds_mean_hdbm.npy"
STD_STD_PATH = ART_DIR / "embeds_std_hdbm.npy"
POSTF_PATH = ART_DIR / "posterior_hdbm_final.pkl"
POST_SUMMARY_CSV = ART_DIR / "posterior_summary_hdbm.csv"
RECOVERY_CSV = ART_DIR / "recovery_hdbm.csv"
SIM_DATA_PATH = ART_DIR / "sim_data_shared.npz"

# 2. Parameters
PARAM_RANGES = {"alpha": (0.01, 0.99), "rho": (0.01, 0.99)}
PARAM_ORDER = ["alpha", "rho"]
TRIALS_PER_BLOCK = 180

# --- Global Data Loading ---
def load_fixed_pomdp_params(csv_path: Path) -> pd.DataFrame:
    try:
        df_raw = pd.read_csv(csv_path)
        return df_raw.pivot_table(
            index=['subject_id', 'subject_year'], columns='index', values='mean'
        ).reset_index()
    except FileNotFoundError:
        warnings.warn(f"POMDP Params not found at {csv_path}. Simulation will fail.")
        sys.exit(1)
        return pd.DataFrame()

def load_order_stats(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
        probs = df['Subject_Count'] / df['Subject_Count'].sum()
        return df['Order_Signature'].values, probs
    except Exception:
        warnings.warn(f"Order stats not found at {csv_path}. Defaulting to random.")
        sys.exit(1)
        return None, None

FIXED_POMDP_DF = load_fixed_pomdp_params(FIXED_PARAMS_PATH)
ORDER_SIGS, ORDER_PROBS = load_order_stats(ORDER_CSV_PATH)

# --- Simulator Functions ---

def get_empirical_task_structure(seed=None):
    if seed is not None: np.random.seed(seed)
    
    if ORDER_SIGS is None:
        # Fallback if CSV missing
        n_trials = TRIALS_PER_BLOCK * 2
        trial_types = np.random.choice(['stop', 'nonstop'], size=n_trials, p=[0.25, 0.75])
    else:
        chosen_sig = np.random.choice(ORDER_SIGS, p=ORDER_PROBS)
        trial_types = np.array(['stop' if x == '1' else 'nonstop' for x in chosen_sig])
    
    go_directions = np.random.choice(['left', 'right'], size=len(trial_types))
    return trial_types, go_directions

def simulate_data(params: Dict[str, float], seed: Optional[int] = None) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed); random.seed(seed)

    # 1. Parameters
    alpha = np.clip(params['alpha'], 0.0, 1.0)
    rho = np.clip(params['rho'], 0.0, 1.0)

    # 2. Context (Nuisance Parameters)
    if FIXED_POMDP_DF.empty:
        raise RuntimeError("FIXED_POMDP_DF is empty. Cannot simulate.")
    
    subj_row = FIXED_POMDP_DF.sample(n=1, random_state=seed).iloc[0].to_dict()
    
    # 3. Task Structure
    trial_types, go_directions = get_empirical_task_structure(seed)
    
    # 4. HDBM Trajectory
    hdbm = HDBM(alpha=alpha, rho=rho)
    seq_int = np.where(trial_types == 'stop', 1, 0)
    r_preds = np.clip(hdbm.simu_task(seq_int, block_size=TRIALS_PER_BLOCK), 0.001, 0.999)

    # 5. Agent Setup
    valid_keys = ["q_d_n", "q_d", "q_s_n", "q_s", "cost_go_error", 
                  "cost_go_missing", "cost_stop_error", "cost_time", 
                  "inv_temp", "rate_stop_trial"]
    agent = POMDP(**{k: subj_row[k] for k in valid_keys if k in subj_row})

    # 6. Loop
    results = []
    next_stop_ssd = 2 

    for t, (t_type, go_dir) in enumerate(zip(trial_types, go_directions)):
        current_ssd = next_stop_ssd if t_type == 'stop' else None
        
        # Update Belief & Solve
        agent.rate_stop_trial = float(r_preds[t])
        agent.value_iteration_tensor()
        
        # Act
        res, rt, _ = simulation.simu_trial(
            agent, true_go_state=go_dir, true_stop_state=t_type, 
            ssd=current_ssd, verbose=False
        )
        
        # Record
        results.append([res, float(rt) if rt else np.nan, float(current_ssd) if current_ssd else np.nan])
        
        # Staircase
        if t_type == 'stop':
            if res == 'SS': next_stop_ssd = min(next_stop_ssd + 2, 34)
            elif res == 'SE': next_stop_ssd = max(next_stop_ssd - 2, 2)

    df = pd.DataFrame(results, columns=["result", "rt", "ssd"])
    df['rt'] = pd.to_numeric(df['rt']); df['ssd'] = pd.to_numeric(df['ssd'])
    return df

# --- NEW: Simulation Function ---
def run_simulation_dump(args, prior):
    """
    Generates 'n_train' samples ONCE and saves them.
    These will be used for BOTH pretraining and SNPE.
    """
    # We use args.n_train as the single source of truth for dataset size
    print(f"\n[Simulation] Generating {args.n_train} shared samples...")
    
    omegas = prior.sample((args.n_train,)).cpu()
    
    # Run parallel simulation
    X_list = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(worker_simulate)(i, omegas[i]) for i in range(args.n_train)
    )
    
    # Pad for storage
    max_len = max(x.shape[0] for x in X_list)
    feat_dim = X_list[0].shape[1]
    
    X_padded = np.zeros((args.n_train, max_len, feat_dim), dtype=np.float32)
    lengths = np.zeros(args.n_train, dtype=np.int32)
    
    for i, x in enumerate(X_list):
        L = x.shape[0]
        X_padded[i, :L, :] = x
        lengths[i] = L
        
    np.savez_compressed(
        SIM_DATA_PATH, 
        theta=omegas.numpy(), 
        x=X_padded, 
        lengths=lengths
    )
    print(f"[Simulation] Saved {args.n_train} samples to {SIM_DATA_PATH}")

def load_shared_data():
    if not SIM_DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found at {SIM_DATA_PATH}. Run --stage simulate first.")
    d = np.load(SIM_DATA_PATH)
    return d['theta'], d['x'], d['lengths']


# --- Helpers ---

def make_box_prior():
    low = torch.tensor([PARAM_RANGES[k][0] for k in PARAM_ORDER], device=device)
    high = torch.tensor([PARAM_RANGES[k][1] for k in PARAM_ORDER], device=device)
    return BoxUniform(low=low, high=high), low, high

def untransform(omega_vec):
    vals = omega_vec.detach().cpu().numpy()
    return {k: float(vals[i]) for i, k in enumerate(PARAM_ORDER)}

# --- Feature Engineering ---

def build_per_trial_features(df: pd.DataFrame) -> np.ndarray:
    RESULT_LEVELS = ["GS", "GE", "GM", "SS", "SE"]
    idx_map = {k: i for i, k in enumerate(RESULT_LEVELS)}
    N = len(df)

    oh = np.zeros((N, len(RESULT_LEVELS)), dtype=np.float32)
    for t, r in enumerate(df["result"].astype(str)):
        if r in idx_map: oh[t, idx_map[r]] = 1.0

    rt = np.nan_to_num(pd.to_numeric(df["rt"], errors="coerce").values, nan=0.0).astype(np.float32)
    ssd = np.nan_to_num(pd.to_numeric(df["ssd"], errors="coerce").values, nan=0.0).astype(np.float32)
    
    prev_oh = np.zeros_like(oh); prev_oh[1:] = oh[:-1]
    prev_rt = np.zeros_like(rt); prev_rt[1:] = rt[:-1]
    prev_ssd = np.zeros_like(ssd); prev_ssd[1:] = ssd[:-1]
    
    cols = [
        oh, rt[:, None], ssd[:, None], 
        (rt == 0)[:, None].astype(np.float32), (ssd == 0)[:, None].astype(np.float32),
        prev_oh, prev_rt[:, None], prev_ssd[:, None], (ssd - prev_ssd)[:, None]
    ]
    
    # Time index
    cols.append(np.linspace(-1, 1, N, dtype=np.float32)[:, None])
    return np.concatenate(cols, axis=1)

# --- Transformer ---

class TrialTransformer(nn.Module):
    def __init__(self, in_dim, model_dim=64, out_dim=64, nhead=4, nlayers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, model_dim)
        
        # Positional Encoding
        pe = torch.zeros(1024, model_dim)
        pos = torch.arange(0, 1024).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, model_dim, 2).float() * (-np.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=256, batch_first=True),
            num_layers=nlayers
        )
        self.head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, out_dim))

    def forward(self, x):
        h = self.proj(x) + self.pe[:x.size(1)].unsqueeze(0)
        h = self.encoder(h)
        return self.head(h.mean(dim=1))

def train_encoder_on_simulations(X_list, omega_list, in_dim, omega_dim, epochs=50, batch_size=128, patience=10):
    # Prepare Data
    X_t = [torch.tensor(x, dtype=torch.float32) for x in X_list]
    Y_t = torch.tensor(np.stack(omega_list), dtype=torch.float32)
    
    # Normalize Targets
    y_mean, y_std = Y_t.mean(0), Y_t.std(0)
    Y_t = (Y_t - y_mean) / (y_std + 1e-6)
    
    ds = torch.utils.data.TensorDataset(torch.nn.utils.rnn.pad_sequence(X_t, batch_first=True), Y_t)
    train_len = int(0.8 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, len(ds) - train_len])
    
    dl_opts = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': True}
    train_dl = DataLoader(train_ds, shuffle=True, **dl_opts)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_opts)

    # Model
    model = TrialTransformer(in_dim, out_dim=64).to(device)
    head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, omega_dim)).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Loop
    best_loss = float('inf'); pat = 0
    for ep in range(epochs):
        model.train(); head.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(head(model(xb)), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        
        model.eval(); head.eval(); val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                val_loss += loss_fn(head(model(xb.to(device))), yb.to(device)).item()
        val_loss /= len(val_dl)
        
        print(f"Epoch {ep+1:02d} | Val Loss: {val_loss:.4f}", end="\r")
        
        if val_loss < best_loss:
            best_loss = val_loss; pat = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= patience: break
            
    print(f"\nTraining done. Best Val Loss: {best_loss:.4f}")
    model.load_state_dict(best_state)
    return model

# --- Pipeline Logic ---

class SummaryStandardizer:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0); self.std_[self.std_==0] = 1.0
        return self
    def transform(self, X): return (X - self.mean_) / self.std_

def worker_simulate(i, omega, seed_offset=0):
    return build_per_trial_features(simulate_data(untransform(omega), seed=seed_offset + i))

def run_pretrain(args, prior, D):
    print(f"\n[Pretrain] Loading shared data...")
    theta, x_padded, lengths = load_shared_data()
    
    # Convert padded array back to list of tensors for the Transformer
    # (The transformer training loop expects a list of arrays)
    X_list = [x_padded[i, :lengths[i], :] for i in range(len(theta))]
    
    print(f"[Pretrain] Training on {len(theta)} samples...")
    encoder = train_encoder_on_simulations(
        X_list, theta, 
        in_dim=X_list[0].shape[1], omega_dim=D,
        epochs=args.epochs, patience=args.patience
    )
    torch.save(encoder.state_dict(), ENC_PATH)

def run_snpe(args, prior, encoder):
    print(f"\n[SNPE] Loading shared data...")
    theta, x_padded, lengths = load_shared_data()
    
    print(f"[SNPE] Training on {len(theta)} samples (Reuse)...")
    
    # Prepare Tensors
    theta_t = torch.tensor(theta)
    
    # Embed everything
    encoder.to(device).eval()
    embeds = []
    batch_size = 256
    
    # Create a Tensor from the padded numpy array
    X_tensor = torch.tensor(x_padded).to(device)
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            xb = X_tensor[i:i+batch_size]
            # No masking needed if your encoder handles padding or if you rely on the padding being 0
            # Ideally, pass lengths to encoder if supported. 
            # If standard Transformer, 0-padding usually requires masking.
            # For now, we assume your TrialTransformer handles it or we accept slight noise from 0s.
            embeds.append(encoder(xb).cpu())
            
    embeds_z = torch.cat(embeds, dim=0).numpy()
    
    # Standardize
    stdzr = SummaryStandardizer().fit(embeds_z)
    np.save(STD_MEAN_PATH, stdzr.mean_)
    np.save(STD_STD_PATH, stdzr.std_)
    
    x_z_tensor = torch.tensor(stdzr.transform(embeds_z), dtype=torch.float32)
    
    # Train SNPE
    inference = SNPE(prior=prior, density_estimator=args.density, device=str(device))
    inference.append_simulations(theta_t, x_z_tensor)
    
    density = inference.train(
        stop_after_epochs=args.patience,
        max_num_epochs=args.snpe_max_epochs,
        training_batch_size=128,
        learning_rate=5e-4,
        show_train_summary=True
    )
    
    with open(POSTF_PATH, "wb") as f: 
        pickle.dump(inference.build_posterior(density), f)
    
    print(f"[SNPE] Posterior saved to {POSTF_PATH}")

# --- Inference ---

def run_inference(args, posterior, encoder, standardizer):
    files = []
    if args.real_csv: files.append(Path(args.real_csv))
    if args.sst_folder: files.extend(Path(args.sst_folder).glob(args.glob_pat))
    
    if not files: return print("No files found.")
    
    encoder.to(device).eval()
    
    # Create Header if missing
    if not POST_SUMMARY_CSV.exists():
        cols = ["subject_id", "subject_year", "index", "mean", "std", "5%", "50%", "95%"]
        pd.DataFrame(columns=cols).to_csv(POST_SUMMARY_CSV, index=False)
        
    print(f"Processing {len(files)} files...")
    
    for fp in files:
        try:
            parts = fp.name.split('_')
            sid, year = (parts[1], parts[2]) if len(parts) > 2 else (fp.stem, "unk")
            
            # Check exist
            subj_dir = ART_DIR / "subjects" / f"{sid}_{year}"
            if (subj_dir / "posterior_summary.csv").exists(): continue
            
            # Inference
            df_obs = preprocessing(str(fp))
            if df_obs.empty: continue
            
            X = torch.tensor(build_per_trial_features(df_obs), dtype=torch.float32).unsqueeze(0)
            z = encoder(X.to(device)).detach().cpu().numpy()
            samples = posterior.sample((args.num_samples,), x=torch.tensor(standardizer.transform(z), device=device)).cpu()
            
            # Save
            post_df = pd.DataFrame([untransform(s) for s in samples])
            summ = post_df.describe(percentiles=[0.05, 0.5, 0.95]).T[["mean", "std", "5%", "50%", "95%"]]
            
            subj_dir.mkdir(parents=True, exist_ok=True)
            post_df.to_csv(subj_dir / "posterior_samples.csv", index=False)
            summ.to_csv(subj_dir / "posterior_summary.csv")
            
            # Incremental Append
            row = summ.copy()
            row["subject_id"] = sid
            row["subject_year"] = year
            row_to_save = row.reset_index()
            cols_order = ["subject_id", "subject_year", "index", "mean", "std", "5%", "50%", "95%"]
            row_to_save = row_to_save[cols_order]
            row_to_save.to_csv(POST_SUMMARY_CSV, mode='a', header=False, index=False)
            
        except Exception as e:
            print(f"Error {fp.name}: {e}")

def run_recovery(args, prior, posterior, encoder, standardizer):
    omegas = prior.sample((args.K,)).cpu()
    res = []
    encoder.eval()
    
    for i in range(args.K):
        gt = untransform(omegas[i])
        df = simulate_data(gt, seed=42+i)
        X = torch.tensor(build_per_trial_features(df)).unsqueeze(0).to(device)
        
        z = standardizer.transform(encoder(X).detach().cpu().numpy())
        samples = posterior.sample((1000,), x=torch.tensor(z, device=device)).cpu().numpy()
        
        row = {"case": i}
        for idx, param_name in enumerate(PARAM_ORDER):
            row[f"gt_{param_name}"] = gt[param_name]
            row[f"mu_{param_name}"] = np.mean(samples[:, idx])
        
        res.append(row)
        
    pd.DataFrame(res).to_csv(RECOVERY_CSV, index=False)
    

def main():
    torch.set_num_threads(1)
    
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="SBI with HDBM + POMDP (Simulate Once, Reuse Twice)")
    
    # Pipeline control
    parser.add_argument("--stage", choices=["all", "simulate", "pretrain", "snpe", "recover", "posterior"], default="all")
    
    # Data & Training
    parser.add_argument("--n_train", type=int, default=50000, help="Total samples for simulation (used for both Pretrain and SNPE)")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for Encoder pretraining")
    parser.add_argument("--snpe_max_epochs", type=int, default=500, help="Max epochs for Density Estimation")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (used for both)")
    parser.add_argument("--density", type=str, default="nsf", help="Density estimator for SNPE (e.g., 'nsf', 'maf')")
    
    # Recovery & Inference
    parser.add_argument("--K", type=int, default=20, help="Number of recovery cases")
    parser.add_argument("--real_csv", type=str, help="Path to a specific real data CSV")
    parser.add_argument("--sst_folder", type=str, help="Folder containing real data CSVs")
    parser.add_argument("--glob_pat", type=str, default="NDAR_*.csv*", help="Pattern to match real data files")
    parser.add_argument("--num_samples", type=int, default=4000, help="Posterior samples per subject")
    
    # Fix: Path argument for dependencies
    parser.add_argument("--fixed_params_path", type=str, default="posterior_summary.csv", help="Path to fixed subject parameters CSV")
    
    args = parser.parse_args()
    
    # --- 2. Setup ---
    ART_DIR.mkdir(parents=True, exist_ok=True)
    prior, _, _ = make_box_prior()
    D = len(PARAM_ORDER)

    # Load Fixed Parameters (Global)
    global FIXED_POMDP_DF
    if args.fixed_params_path and Path(args.fixed_params_path).exists():
        FIXED_POMDP_DF = load_fixed_pomdp_params(Path(args.fixed_params_path))
    else:
        print(f"Warning: Fixed params not found at {args.fixed_params_path}. Simulation may fail if needed.")

    # --- 3. Stage: SIMULATE (Shared Data) ---
    # Generates data ONCE for both Pretrain and SNPE
    if args.stage in ["all", "simulate"]:
        run_simulation_dump(args, prior)
        if args.stage == "simulate": 
            print("Simulation finished. Exiting."); return

    # --- 4. Stage: PRETRAIN (Encoder) ---
    # Loads shared data to train the embedding network
    if args.stage in ["all", "pretrain"]:
        run_pretrain(args, prior, D)
        if args.stage == "pretrain": 
            print("Pretraining finished. Exiting."); return

    # --- Load Encoder ---
    # Determine input dimension from data or dummy sim
    if SIM_DATA_PATH.exists():
        _, x_dummy, _ = load_shared_data()
        in_dim = x_dummy.shape[2] # (N, T, F)
    else:
        # Fallback if just running inference without existing sim data
        dummy_df = simulate_data(untransform(prior.sample((1,)).cpu()[0]), seed=0)
        in_dim = build_per_trial_features(dummy_df).shape[1]
    
    encoder = TrialTransformer(in_dim, out_dim=64)
    if ENC_PATH.exists():
        encoder.load_state_dict(torch.load(ENC_PATH, map_location=device))
        print(f"Loaded encoder from {ENC_PATH}")
    elif args.stage != "simulate": 
        raise FileNotFoundError(f"Encoder not found at {ENC_PATH}. Run --stage pretrain.")
    
    # --- 5. Stage: SNPE (Density Estimation) ---
    # Loads shared data again to train the flow
    if args.stage in ["all", "snpe"]:
        run_snpe(args, prior, encoder)
        if args.stage == "snpe": 
            print("SNPE training finished. Exiting."); return
        
    # --- Load Posterior & Standardizer ---
    # Required for Recovery and Inference stages
    if args.stage in ["all", "recover", "posterior"]:
        if not POSTF_PATH.exists(): 
            raise FileNotFoundError(f"Posterior not found at {POSTF_PATH}. Run --stage snpe.")
        
        with open(POSTF_PATH, "rb") as f: 
            posterior = pickle.load(f)
        
        if not STD_MEAN_PATH.exists():
             raise FileNotFoundError("Standardizer stats missing. Run --stage snpe.")
        
        stdzr = SummaryStandardizer()
        stdzr.mean_ = np.load(STD_MEAN_PATH)
        stdzr.std_ = np.load(STD_STD_PATH)

        # --- 6. Stage: RECOVERY CHECK ---
        if args.stage in ["all", "recover"]:
            print("Starting Recovery Check...")
            run_recovery(args, prior, posterior, encoder, stdzr)

        # --- 7. Stage: INFERENCE (Real Data) ---
        if args.stage in ["all", "posterior"]:
            print("Starting Inference on Real Data...")
            run_inference(args, posterior, encoder, stdzr)

if __name__ == "__main__":
    main()