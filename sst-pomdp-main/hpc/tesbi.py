"""
tesbi.py 

Transformer-encoded Simulation-Based Inference (TeSBI) Pipeline.
End-to-end for the POMDP-SST parameter inference.

================================================================================
PROJECT STRUCTURE & FILE LOCATIONS
================================================================================
- Main Python Script: 
  ./hpc/tesbi.py

- HPC SLURM Script: 
  ./slurm/run_tesbi.sh

- Example Data: 
  We provide 100 example mock files for public testing located at:
  ./data/example_processed_data/
  (e.g., EXAMPLE_SUB_001.csv)

================================================================================
WORKFLOW
================================================================================
1) Round-1: Simulate pairs of (omega, trials).
2) Pretrain: Train a small Transformer encoder to map trials -> embedding.
3) Freeze Encoder: Generate embeddings and fit a z-score standardizer.
4) Train SNPE: Train Neural Spline Flow (NSF) posterior on (omega, embedding_z).
5) Round-2 (Active Learning): Sample from proposal, simulate, and retrain.
6) Inference: Condition on observed data to sample the posterior.

================================================================================
USAGE EXAMPLES
================================================================================
--- A. HPC EXECUTION (RECOMMENDED FOR FULL RUNS) ---
This pipeline is computationally heavy. It is highly recommended to execute 
the full pipeline on an HPC cluster using SLURM.
    
    sbatch run_tesbi.sh

--- B. LOCAL TESTING (NOT RECOMMENDED FOR PRODUCTION) ---
Use these commands locally strictly to verify the pipeline works (Smoke Test) 
without waiting hours for heavy computation.

    # 1. Pretrain Encoder (Quick: 5 sims, 2 epochs)
    python tesbi.py --stage pretrain --n1_pre 5 --epochs 2

    # 2. Train SNPE (Light: MAF density, small N)
    python tesbi.py --stage snpe --n1_pre 5 --n2 2 --density maf
    
    # 3. Recovery Check (5 test cases)
    python tesbi.py --stage recover --K 5 --num_post 10

    # 4. Inference on Example Data (Public use)
    python tesbi.py --stage posterior \
      --sst_folder "./data/example_processed_data" \
      --glob_pat "EXAMPLE_SUB_*.csv" \
      --num_samples 10

--- C. FULL LOCAL RUN (WARNING: HEAVY COMPUTATION) ---
If you must run the full analysis locally, use the following:

    # 1. Pretrain Encoder (Deep: 30k sims, 8 epochs)
    python tesbi.py --stage pretrain --n1_pre 30000 --epochs 8

    # 2. Train SNPE (Heavy: NSF density, 50k total sims)
    python tesbi.py --stage snpe --n1_pre 30000 --n2 20000 --density nsf

    # 3. Full Recovery Sweep (200 test cases)
    python tesbi.py --stage recover --K 200 --num_post 4000
================================================================================
"""
import sys
import os
import math
import random
import pickle
import argparse
import warnings
import glob
import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed

from sbi.inference import SNPE
from sbi.utils import BoxUniform

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
# Fix path to allow importing 'model'
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

# Local modules
from utils.preprocessing import preprocessing
from core.models import POMDP
from core import simulation

# ==============================================================================
# DEVICE SETUP
# ==============================================================================
# --- CPU Device Setup ---
try:
    # SLURM Allocated CPUs
    N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK"))
except (ValueError, TypeError):
    # Default to all available CPUs locally
    N_JOBS = multiprocessing.cpu_count()

print(f"  [Auto-Config] Detected {N_JOBS} CPU cores available for joblib.")

# --- GPU Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("-" * 60)
print(f"Running on device: {str(device).upper()}")
if device.type == 'cuda':
    print(f"GPU Name:          {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory:        {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("WARNING: Running on CPU. This will be slow!")
print("-" * 60)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Set to False because the example data is already processed. 
# Switch to True when running inference on raw ABCD SST data.
USE_PREPROCESSING = False

# Parameter Ranges
PARAM_RANGES = {
    "q_d_n": (0.0, 1.0),
    "q_d":   (0.5, 1.0),
    "q_s_n": (0.0, 1.0),
    "q_s":   (0.5, 1.0),
    "cost_stop_error": (0.01, 2.0),
    "inv_temp":     (1, 100)
    }

LINEAR_PARAMS = ["q_d_n", "q_d", "q_s_n", "q_s", "inv_temp"]
LOG_PARAMS = ["cost_stop_error"]
PARAM_ORDER = LINEAR_PARAMS + LOG_PARAMS

FIXED_PARAMS = {
    "cost_time": 0.001,
    "cost_go_error":   1.0,
    "cost_go_missing": 1.0,
    "rate_stop_trial": 1.0 / 6.0
    }

RESULT_LEVELS = ["GS", "GE", "GM", "SS", "SE"]

# Artifact Paths
ART_DIR = Path("outputs/tesbi/")
ENC_PATH = ART_DIR / "encoder.pt"
STD_MEAN_PATH = ART_DIR / "embeds_mean.npy"
STD_STD_PATH = ART_DIR / "embeds_std.npy"
POST1_PATH = ART_DIR / "posterior_round1.pkl"
POSTF_PATH = ART_DIR / "posterior_final.pkl"
RECOVERY_CSV = ART_DIR / "params_recovery.csv"
POST_SUMMARY_CSV = ART_DIR / "params_posteriors.csv"


# ==============================================================================
# 1. SIMULATOR WRAPPER
# ==============================================================================
def simulate_data(params: Dict[str, float], seed: Optional[int] = None) -> pd.DataFrame:
    """Runs the POMDP simulator for a single parameter set."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    
    pomdp = POMDP(**params)
    pomdp.value_iteration_tensor()
    arr = simulation.simu_task(pomdp)
    return pd.DataFrame(arr, columns=["result", "rt", "ssd"])


# ==============================================================================
# 2. PARAMETER SETUP
# ==============================================================================
def make_box_prior() -> Tuple[BoxUniform, torch.Tensor, torch.Tensor]:
    """Constructs the SBI BoxUniform prior based on configured ranges."""
    low, high = [], []
    for k in PARAM_ORDER:
        lo, hi = PARAM_RANGES[k]
        if k in LOG_PARAMS:
            lo, hi = math.log(lo), math.log(hi)
        low.append(lo)
        high.append(hi)
    
    low = torch.tensor(low, dtype=torch.float32).to(device)
    high = torch.tensor(high, dtype=torch.float32).to(device)
    prior = BoxUniform(low=low, high=high)
    return prior, low, high


def untransform(omega_vec: torch.Tensor) -> Dict[str, float]:
    """Converts transformed (log) parameters back to original space."""
    vals = omega_vec.detach().cpu().numpy().astype(float)
    out = {}
    for i, k in enumerate(PARAM_ORDER):
        v = vals[i]
        if k in LOG_PARAMS:
            v = float(np.exp(v))
        out[k] = float(v)
    out.update(FIXED_PARAMS)
    return out


# ==============================================================================
# 3. FEATURE ENGINEERING
# ==============================================================================
def build_per_trial_features(df: pd.DataFrame, use_index=True) -> np.ndarray:
    """
    Constructs per-trial feature vectors (N_trials x Features).
    Features: One-hot result, RT, SSD, missing flags, lag-1 history, and trial index.
    """
    N = len(df)
    idx_map = {k: i for i, k in enumerate(RESULT_LEVELS)}

    # Current trial features (One-Hot)
    oh = np.zeros((N, len(RESULT_LEVELS)), dtype=np.float32)
    res = df["result"].astype(str).values
    for t, r in enumerate(res):
        if r in idx_map:
            oh[t, idx_map[r]] = 1.0

    rt = pd.to_numeric(df["rt"], errors="coerce").values.astype(np.float32)
    ssd = pd.to_numeric(df["ssd"], errors="coerce").values.astype(np.float32)
    
    # Missing value indicators & fill
    rt_nan = np.isnan(rt).astype(np.float32)
    ssd_nan = np.isnan(ssd).astype(np.float32)
    rt = np.nan_to_num(rt, nan=0.0)
    ssd = np.nan_to_num(ssd, nan=0.0)

    # Lag-1 features
    prev_oh = np.roll(oh, 1, axis=0); prev_oh[0, :] = 0.0
    prev_rt = np.roll(rt, 1); prev_rt[0] = 0.0
    prev_ssd = np.roll(ssd, 1); prev_ssd[0] = 0.0
    delta_ssd = ssd - prev_ssd

    cols = [
        oh, rt[:, None], ssd[:, None], rt_nan[:, None], ssd_nan[:, None],
        prev_oh, prev_rt[:, None], prev_ssd[:, None], delta_ssd[:, None]
    ]
    
    if use_index:
        t_idx = np.linspace(-1.0, 1.0, N, dtype=np.float32)[:, None]
        cols.append(t_idx)

    return np.concatenate(cols, axis=1).astype(np.float32)


# ==============================================================================
# 4. TRANSFORMER ENCODER
# ==============================================================================
class SinusoidalPE(nn.Module):
    """Sinusoidal Positional Encoding."""
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        N = x.size(1)
        return x + self.pe[:N].unsqueeze(0)


class TrialTransformer(nn.Module):
    """Lightweight Transformer encoder + mean pooling."""
    def __init__(self, in_dim: int, model_dim: int = 64, nhead: int = 4, nlayers: int = 2,
                 dropout: float = 0.1, out_dim: int = 64, use_pos_enc: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, model_dim)
        self.use_pos = use_pos_enc
        if use_pos_enc:
            self.pe = SinusoidalPE(model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=model_dim*4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, out_dim)
        )

    def forward(self, x):
        h = self.proj(x)
        if self.use_pos:
            h = self.pe(h)
        h = self.encoder(h)
        h = h.mean(dim=1)  # Mean pool
        return self.head(h)


# ==============================================================================
# 5. TRAINING UTILS
# ==============================================================================
class OmegaDataset(Dataset):
    def __init__(self, X_list: List[np.ndarray], omega_list: List[np.ndarray]):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X_list]
        self.Y = [torch.tensor(y, dtype=torch.float32) for y in omega_list]

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


def train_encoder_on_simulations(X_list, omega_list, in_dim, omega_dim, 
                                 epochs=50, batch_size=128, lr=1e-3, seed=137,
                                 patience=10):
    # --- Batch Size ---
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        print(f"  [Auto-Scale] Detected {n_gpu} GPUs!")
        print(f"   Batch size scaled from {batch_size} -> {batch_size * n_gpu}")
        batch_size = batch_size * n_gpu
    
    # --- 1. Split & Scale Data ---
    torch.manual_seed(seed); np.random.seed(seed)
    total_len = len(X_list)
    val_len = int(0.2 * total_len)
    train_len = total_len - val_len
    
    perm = np.random.permutation(total_len)
    train_idx, val_idx = perm[:train_len], perm[train_len:]
    
    X_train = [X_list[i] for i in train_idx]
    X_val =   [X_list[i] for i in val_idx]
    
    # Normalize Targets
    Y_all = np.stack(omega_list, axis=0).astype(np.float32)
    y_mean = Y_all[train_idx].mean(axis=0, keepdims=True)
    y_std = Y_all[train_idx].std(axis=0, keepdims=True)
    y_std[y_std < 1e-6] = 1.0 
    
    def scale_y(indices):
        return [(Y_all[i] - y_mean[0]) / y_std[0] for i in indices]

    # Dataloaders (pin_memory=True for GPU)
    train_ds = OmegaDataset(X_train, scale_y(train_idx))
    val_ds   = OmegaDataset(X_val,   scale_y(val_idx))
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # --- 2. Model Setup ---
    encoder = TrialTransformer(in_dim, out_dim=64, use_pos_enc=True).to(device)
    if n_gpu > 1:
        encoder = nn.DataParallel(encoder) # All visible GPUs
    reg_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, omega_dim)).to(device)
    
    params = list(encoder.parameters()) + list(reg_head.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    loss_fn = nn.MSELoss()

    # --- 3. Training Loop ---
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"\n [Encoder] Training on {device}...")

    for ep in range(epochs):
        # -- Train --
        encoder.train(); reg_head.train()
        train_loss = 0.0
        for Xb, Yb in train_dl:
            Xb, Yb = Xb.to(device), Yb.to(device) # <--- Move to GPU
            pred = reg_head(encoder(Xb))
            loss = loss_fn(pred, Yb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_ds)

        # -- Validate --
        encoder.eval(); reg_head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, Yb in val_dl:
                Xb, Yb = Xb.to(device), Yb.to(device) # <--- Move to GPU
                pred = reg_head(encoder(Xb))
                loss = loss_fn(pred, Yb)
                val_loss += loss.item() * Xb.size(0)
        val_loss /= len(val_ds)
        
        print(f"Epoch {ep+1:03d} | Val MSE: {val_loss:.4f}", end="")

        # -- Early Stopping --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            raw_encoder = encoder.module if isinstance(encoder, nn.DataParallel) else encoder
            best_model_state = {
                'enc': {k: v.cpu().clone() for k, v in raw_encoder.state_dict().items()},
                'head': {k: v.cpu().clone() for k, v in reg_head.state_dict().items()}
            }            
            print(" *")
        else:
            patience_counter += 1
            print(f" (Wait {patience_counter})")
            if patience_counter >= patience:
                print("Early stopping.")
                break
    
    # Restore best weights
    if isinstance(encoder, nn.DataParallel):
        encoder = encoder.module
        
    if best_model_state:
        encoder.load_state_dict(best_model_state['enc'])
        reg_head.load_state_dict(best_model_state['head'])
    
    encoder.eval()
    return encoder, reg_head


class SummaryStandardizer:
    """Standardizes transformer embeddings to z-scores."""
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self
    def transform(self, X): return (X - self.mean_) / self.std_
    def fit_transform(self, X): return self.fit(X).transform(X)

def worker_simulate(i, omega, seed_offset=0):
    params = untransform(omega)
    df = simulate_data(params, seed=seed_offset + i)
    return build_per_trial_features(df)


# ==============================================================================
# 6. PIPELINE STAGES
# ==============================================================================

# Stage 1: Pretrain Transformer Encoder on Simulated Data

def run_pretrain(args, prior, D):
    """Simulates initial dataset and pretrains the Transformer."""
    print(f"\n [Pretrain] Simulating {args.n1_pre} pairs...")
    omegas = prior.sample((args.n1_pre,)).cpu()
    
    print(f"Launching Stage 1 parallel simulation...")
    X_list = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(worker_simulate)(i, omegas[i]) for i in range(args.n1_pre)
    )
    Omega_list = [omegas[i].numpy() for i in range(args.n1_pre)]
    encoder, _ = train_encoder_on_simulations(
        X_list, Omega_list, in_dim=X_list[0].shape[1], omega_dim=D,
        epochs=args.epochs, patience=args.patience
    )
    torch.save(encoder.state_dict(), ENC_PATH)
    print(f"[Pretrain] Encoder saved to {ENC_PATH}")

# Stage 2: SNPE Rounds with pretrained Transformer Embeddings

@dataclass
class SimulationBlock:
    """
    Used for SNPE rounds to store parameters and their neural embeddings.
    """
    omegas: torch.Tensor
    embeds: torch.Tensor

def simulate_round(sampler_fn, n_samples, encoder, seed_offset=0):
    """
    Simulates data on CPU, generates embeddings on GPU, 
    and returns a SimulationBlock.
    """
    omegas = sampler_fn((n_samples,)).cpu()
    
    print("Parking encoder to CPU to avoid CUDA/Multiprocessing conflict...")
    encoder.cpu()
    import gc; gc.collect(); torch.cuda.empty_cache()
        
    print(f"Launching Stage 2 parallel simulation...")
    
    # 1. Simulate (CPU)
    X_list = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(worker_simulate)(i, omegas[i], seed_offset=seed_offset) 
        for i in range(n_samples)
    )

    print("Moving encoder back to GPU...")
    encoder.to(device)
    
    # 2. Embed (GPU)
    encoder.eval()
    embeds = []
    with torch.no_grad():
        for x_trial in X_list:
            # Move input to GPU -> Encode -> Move result back to CPU
            # Processing trial-by-trial avoids GPU memory overflow
            tensor_x = torch.tensor(x_trial, dtype=torch.float32).unsqueeze(0)
            emb = encoder(tensor_x.to(device)).cpu()
            embeds.append(emb)
            
    embeds_tensor = torch.cat(embeds, dim=0)
    
    return SimulationBlock(omegas=omegas, embeds=embeds_tensor)

def run_snpe(args, prior, encoder):
    """Trains the Neural Spline Flow (SNPE) posterior."""
    encoder = encoder.to(device)
    
    # Round 1
    print(f"\n [Round1] Simulating {args.n1_pre} pairs...")
    
    # We pass the sampler function directly
    block1 = simulate_round(prior.sample, args.n1_pre, encoder, seed_offset=0)
    
    stdzr = SummaryStandardizer()
    # Note: block1.embeds is already a tensor from our helper
    embeds_1_z = stdzr.fit_transform(block1.embeds.numpy())
    
    np.save(STD_MEAN_PATH, stdzr.mean_)
    np.save(STD_STD_PATH, stdzr.std_)

    print(f"[Round1] Training SNPE ({args.density}) on {device}...")
    
    inference = SNPE(prior=prior, density_estimator=args.density, device=str(device))
    inference.append_simulations(block1.omegas, torch.tensor(embeds_1_z, dtype=torch.float32))
    
    density = inference.train(
        stop_after_epochs=20, 
        max_num_epochs=500,
        show_train_summary=False
    )
    posterior_1 = inference.build_posterior(density)
    
    with open(POST1_PATH, "wb") as f: pickle.dump(posterior_1, f)

    # Round 2 (Optional)
    if args.n2 > 0:
        print(f"\n [Round2] Fine Tuning with {args.n2} simulations...")

        ref_x = torch.tensor(embeds_1_z.mean(axis=0, keepdims=True), dtype=torch.float32).to(device)
        
        # Mixture Proposal
        def proposal_sampler(shape):
            n = shape[0]
            n_prior = int(args.mix_prior_frac * n)
            n_post = n - n_prior
            
            # Sample from posterior (CPU/GPU handled by sbi, but ensure ref_x is compatible)
            post_samples = posterior_1.sample((n_post,), x=ref_x).reshape(n_post, -1)
            # return torch.cat([prior.sample((n_prior,)), post_samples], dim=0)
            return torch.cat([
                    prior.sample((n_prior,)).to(device), 
                    post_samples.to(device)
                ], dim=0)

        block2 = simulate_round(proposal_sampler, args.n2, encoder, seed_offset=1_000_000)
        
        # Transform using existing standardizer
        embeds_2_z = stdzr.transform(block2.embeds.numpy())

        omegas_all = torch.cat([block1.omegas, block2.omegas], 0)
        embeds_all = torch.cat([torch.tensor(embeds_1_z), torch.tensor(embeds_2_z)], 0)

        print(f"[Merged] Training Final Posterior on {device}...")
        
        # Pass device again for new inference object
        inference = SNPE(prior=prior, density_estimator=args.density, device=str(device))
        
        inference.append_simulations(omegas_all, embeds_all.float())
        
        # Early Stopping
        density = inference.train(
            stop_after_epochs=20,
            max_num_epochs=500,
            show_train_summary=False
        )
        posterior_final = inference.build_posterior(density)
    else:
        posterior_final = posterior_1

    with open(POSTF_PATH, "wb") as f: pickle.dump(posterior_final, f)

# Stage 3: Validates the posterior against known ground-truth cases.

def run_recovery(args, prior, posterior, encoder, standardizer):
    print(f"\n [Recovery] Checking {args.K} ground-truth cases...")
    omegas_true = prior.sample((args.K,)).cpu()
    details = []
    
    for i in range(args.K):
        gt_params = untransform(omegas_true[i])
        df_obs = simulate_data(gt_params, seed=4242 + i)
        
        # Inference
        X = build_per_trial_features(df_obs)
        with torch.no_grad():
            tensor_x = torch.tensor(X[None, ...], dtype=torch.float32).to(device)
            z = encoder(tensor_x).cpu().numpy()
            
        z_z = standardizer.transform(z)
        samples = posterior.sample((args.num_post,),
                                   x=torch.tensor(z_z, dtype=torch.float32).to(device)).cpu()
        
        # Stats
        row = {"case": i}
        samples_np = samples.numpy()
        for k_idx, k in enumerate(PARAM_ORDER):
            # Transform back to original space for stats
            vals = [untransform(torch.tensor(s))[k] for s in samples_np]
            mu, lo, hi = np.mean(vals), np.percentile(vals, 5), np.percentile(vals, 95)
            gt_val = gt_params[k]
            row[f"gt_{k}"] = gt_val
            row[f"mu_{k}"] = mu
            row[f"hit90_{k}"] = 1.0 if lo <= gt_val <= hi else 0.0
        details.append(row)

    pd.DataFrame(details).to_csv(RECOVERY_CSV, index=False)
    print(f"\n [Recovery] Saved to {RECOVERY_CSV}")

# Stage 4: Runs inference on real/example observed data.

def run_inference(args, posterior, encoder, standardizer):
    out_root = ART_DIR / "subjects"
    out_root.mkdir(parents=True, exist_ok=True)
    
    files = []
    if args.real_csv: files.append(Path(args.real_csv))
    if args.sst_folder: files.extend(Path(args.sst_folder).glob(args.glob_pat))
    files = sorted(files)
    
    print(f"\n [Inference] Processing {len(files)} files...")
    summaries = []
    
    pd.options.mode.chained_assignment = None
    total_files = len(files)
    for i, fp in enumerate(files):
        if i % 500 == 0:
            print(f"   ... Progress: {i}/{total_files} files processed", flush=True)
        try:
            # Parse filename (Assumes NDAR_ID_YEAR_...)
            parts = fp.name.split('_')
            sid = parts[1] if len(parts) > 1 else fp.stem
            year = parts[2] if len(parts) > 2 else "unknown"

            # Apply conditional preprocessing
            if USE_PREPROCESSING:
                df_obs = preprocessing(str(fp))
            else:
                df_obs = pd.read_csv(str(fp))

            X = build_per_trial_features(df_obs)
            with torch.no_grad():
                tensor_x = torch.tensor(X[None, ...], dtype=torch.float32).to(device)
                z = encoder(tensor_x).cpu().numpy()
            z_z = standardizer.transform(z)
            
            samples = posterior.sample((args.num_samples,), 
                    x=torch.tensor(z_z, dtype=torch.float32).to(device)).cpu()
            
            # Save Results
            rows = [untransform(s) for s in samples]
            post_df = pd.DataFrame(rows)
            summ = post_df.describe(percentiles=[0.05, 0.5, 0.95]).T[["mean", "std", "5%", "50%", "95%"]]
            
            subj_dir = out_root / f"{sid}_{year}"
            subj_dir.mkdir(exist_ok=True)
            post_df.to_csv(subj_dir / "posterior_samples.csv", index=False)
            summ.to_csv(subj_dir / "posterior_summary.csv")
            
            s_row = summ.copy(); s_row["subject_id"] = sid; s_row["subject_year"] = year
            summaries.append(s_row.reset_index())

        except Exception as e:
            print(f"\n [Error] {fp.name}: {e}")

    if summaries:
        pd.concat(summaries).to_csv(POST_SUMMARY_CSV, index=False)
        print(f"\n [Inference] All summaries saved to {POST_SUMMARY_CSV}")


# ==============================================================================
# 7. MAIN
# ==============================================================================
def main():
    warnings.filterwarnings("ignore")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    
    # Global Seed for reproducibility
    SEED = 137
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    # Transformer/SNPE args
    parser = argparse.ArgumentParser(description="Transformer-based SNPE Pipeline")
    parser.add_argument("--stage", choices=["all", "pretrain", "snpe", "recover", "posterior"], default="all")
    parser.add_argument("--n1_pre", type=int, default=20000, help="Simulations for pretraining/Round 1")
    parser.add_argument("--epochs", type=int, default=5, help="Pretraining epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--n2", type=int, default=15000, help="Round 2 simulations")
    parser.add_argument("--mix_prior_frac", type=float, default=0.2, help="Prior fraction in proposal")
    parser.add_argument("--density", choices=["nsf", "maf", "mdn"], default="nsf", help="Density estimator")
    
    # Recovery/Inference args
    parser.add_argument("--K", type=int, default=20, help="Recovery test cases")
    parser.add_argument("--num_post", type=int, default=1000, help="Samples per recovery posterior")
    parser.add_argument("--real_csv", type=str, default=None, help="Single real CSV")
    parser.add_argument("--sst_folder", type=str, default=None, help="Folder of real CSVs")
    parser.add_argument("--glob_pat", type=str, default="NDAR_*.csv*", help="File pattern")
    parser.add_argument("--num_samples", type=int, default=4000, help="Posterior samples per subject")

    args = parser.parse_args()
    ART_DIR.mkdir(exist_ok=True)

    prior, _, _ = make_box_prior()
    D = len(PARAM_ORDER)

    # --- Pipeline Execution ---
    
    # 1. Pretrain
    if args.stage in ["all", "pretrain"]:
        run_pretrain(args, prior, D)
        if args.stage == "pretrain": return

    # Load Encoder
    dummy_df = simulate_data(untransform(prior.sample((1,))[0]), seed=0)
    in_dim = build_per_trial_features(dummy_df).shape[1]
    
    encoder = TrialTransformer(in_dim, out_dim=64, use_pos_enc=True)
    encoder.load_state_dict(torch.load(ENC_PATH))
    encoder.eval()
    encoder.to(device)
    for p in encoder.parameters(): p.requires_grad = False

    # 2. SNPE
    if args.stage in ["all", "snpe"]:
        run_snpe(args, prior, encoder)
    
    # Load Posterior & Standardizer
    if args.stage in ["recover", "posterior"] or args.stage == "all":
        with open(POSTF_PATH, "rb") as f: posterior = pickle.load(f)
        stdzr = SummaryStandardizer()
        stdzr.mean_ = np.load(STD_MEAN_PATH)
        stdzr.std_ = np.load(STD_STD_PATH)

    # 3. Recovery Validation
    if args.stage in ["all", "recover"]:
        run_recovery(args, prior, posterior, encoder, stdzr)

    # 4. Final Inference
    if args.stage in ["all", "posterior"]:
        run_inference(args, posterior, encoder, stdzr)

if __name__ == "__main__":
    main()