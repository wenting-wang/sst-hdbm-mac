"""
embedding.py 

Generates embeddings for individual subjects using a trained TrialTransformer encoder.

This script processes trial data through the frozen encoder to calculate:
1. The Embedding Vector (mean-pooled representation of the trials).
2. The Standard Deviation of trial-level latents (a measure of trial-to-trial variability).

================================================================================
USAGE & CONFIGURATION
================================================================================
--- OPTION A: HPC EXECUTION (RECOMMENDED FOR FULL RUNS) ---
For processing the full dataset, it is highly recommended to run this on an 
HPC cluster. You will need to update the default paths in the configuration 
section below to point to your cluster directories. 
Ensure `USE_PREPROCESSING = True` if using ABCD SST raw data.

--- OPTION B: LOCAL TESTING (SMOKE TEST) ---
The active configuration below is set up for local testing using the public 
example dataset. This is strictly for verifying that the pipeline works without 
needing access to the private dataset. `USE_PREPROCESSING` is set to `False` 
because the example data is already processed.
================================================================================
"""

import os
import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# --- Custom Modules ---
from utils.preprocessing import preprocessing

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
RESULT_LEVELS = ["GS", "GE", "GM", "SS", "SE"]

# --- OPTION A: HPC Production Configuration (Recommended) ---
# Uncomment and update these paths when running on your cluster.
# BASE_DIR = Path('/kyb/agpd/wwang/data/')
# OUT_DIR = Path('/kyb/agpd/wwang/outputs/')
# DEFAULT_DATA_DIR = BASE_DIR / "sst_valid_base"
# DEFAULT_META_CSV = BASE_DIR / "clinical_behavior.csv"
# DEFAULT_MODEL_PATH = OUT_DIR / "encoder.pt"
# DEFAULT_OUT_CSV = OUT_DIR / "embeddings.csv"
# USE_PREPROCESSING = True  # Set to True if HPC data is raw

# --- OPTION B: Local Testing Configuration ---
# Uses the provided example data for testing.
# Dynamically resolve project root for default paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "example_processed_data"
DEFAULT_META_CSV = PROJECT_ROOT / "data" / "example_clinical_behavior.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "encoder.pt"
DEFAULT_OUT_CSV = PROJECT_ROOT / "outputs" / "example_embeddings.csv"
USE_PREPROCESSING = False  # Set to False because the example data is already processed


# ==============================================================================
# MODEL DEFINITIONS (Copied from Pipeline)
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
    """
    A lightweight Transformer encoder + mean pooling to generate simulation embeddings.
    """
    def __init__(self, in_dim: int, model_dim: int = 64, nhead: int = 4, nlayers: int = 2,
                 dropout: float = 0.1, out_dim: int = 32, use_pos_enc: bool = True):
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

    def forward_with_stats(self, x):
        """
        Custom forward pass to return both Mean (Embedding) and Std of trial latents.
        """
        # x shape: (Batch, Trials, Features)
        h = self.proj(x)
        if self.use_pos:
            h = self.pe(h)
        h = self.encoder(h) # (Batch, Trials, ModelDim)
        
        # 1. Mean Pooling (Standard Embedding)
        h_mean = h.mean(dim=1)  
        z_mean = self.head(h_mean) # (Batch, OutDim)
        
        # 2. Std Pooling (Extra Statistic)
        # Calculate std of the latents across trials to measure variability
        h_std_latents = h.std(dim=1) # (Batch, ModelDim)
        
        return z_mean, h_std_latents

    def forward(self, x):
        z, _ = self.forward_with_stats(x)
        return z


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

def build_per_trial_features(df: pd.DataFrame, use_index=True) -> np.ndarray:
    """
    Constructs per-trial feature vectors (N_trials x Features).
    """
    N = len(df)
    idx_map = {k: i for i, k in enumerate(RESULT_LEVELS)}

    # Current trial features
    oh = np.zeros((N, len(RESULT_LEVELS)), dtype=np.float32)
    res = df["result"].astype(str).values
    for t, r in enumerate(res):
        if r in idx_map:
            oh[t, idx_map[r]] = 1.0

    rt = pd.to_numeric(df["rt"], errors="coerce").values.astype(np.float32)
    ssd = pd.to_numeric(df["ssd"], errors="coerce").values.astype(np.float32)
    
    # Missing value indicators
    rt_nan = np.isnan(rt).astype(np.float32)
    ssd_nan = np.isnan(ssd).astype(np.float32)
    
    # Fill NaNs
    rt = np.nan_to_num(rt, nan=0.0)
    ssd = np.nan_to_num(ssd, nan=0.0)

    # Lag-1 features
    prev_oh = np.roll(oh, 1, axis=0)
    prev_oh[0, :] = 0.0
    prev_rt = np.roll(rt, 1)
    prev_rt[0] = 0.0
    prev_ssd = np.roll(ssd, 1)
    prev_ssd[0] = 0.0
    delta_ssd = ssd - prev_ssd

    cols = [
        oh, rt[:, None], ssd[:, None], rt_nan[:, None], ssd_nan[:, None],
        prev_oh, prev_rt[:, None], prev_ssd[:, None], delta_ssd[:, None]
    ]
    
    if use_index:
        t_idx = np.linspace(-1.0, 1.0, N, dtype=np.float32)[:, None]
        cols.append(t_idx)

    X = np.concatenate(cols, axis=1).astype(np.float32)  # Shape: (N, F)
    return X


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser(description="Generate Embeddings for Valid Subjects")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_META_CSV), 
                        help="Input CSV with filenames and metadata")
    parser.add_argument("--folder", type=str, default=str(DEFAULT_DATA_DIR), 
                        help="Folder containing subject data files")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), 
                        help="Path to trained encoder.pt")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_CSV), 
                        help="Output CSV filename for the generated embeddings")
    args = parser.parse_args()

    # --- 1. Load Subject List ---
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV file not found: {csv_path}")
    
    df_meta = pd.read_csv(csv_path)
    if "filename" not in df_meta.columns:
        raise ValueError(f"Column 'filename' missing from {csv_path}")
    
    target_files = set(df_meta["filename"].dropna().unique())
    print(f"Found {len(target_files)} unique target filenames in CSV.")
    print(f"Preprocessing Enabled: {USE_PREPROCESSING}")

    # --- 2. Initialize Model & Device ---
    device = torch.device("cpu") # CPU is sufficient for sequential inference on a list
    encoder = None 
    results = []
    
    data_folder = Path(args.folder)
    print(f"Processing files from {data_folder}...")
    
    count_ok = 0
    count_miss = 0
    count_err = 0

    for fname in target_files:
        fpath = data_folder / fname
        
        # Check existence
        if not fpath.exists():
            count_miss += 1
            continue

        try:
            # Load Data and Toggle Preprocessing based on Configuration
            if USE_PREPROCESSING:
                df_subj = preprocessing(str(fpath))
            else:
                df_subj = pd.read_csv(fpath)

            X = build_per_trial_features(df_subj)
            
            # Init model dynamically once we know the input dimension from the first file
            if encoder is None:
                in_dim = X.shape[1]
                # Default params matching the pipeline
                encoder = TrialTransformer(in_dim, model_dim=64, nhead=4, nlayers=2, out_dim=64) 
                
                model_file = Path(args.model_path)
                if not model_file.exists():
                     raise FileNotFoundError(f"Model not found at {model_file}")
                
                state_dict = torch.load(model_file, map_location=device, weights_only=True)
                encoder.load_state_dict(state_dict)
                encoder.to(device)
                encoder.eval()
                print(f"Model loaded from {model_file} (Input Dim: {in_dim})")

            # Inference
            X_t = torch.tensor(X[None, ...], dtype=torch.float32).to(device) # (1, T, F)
            
            with torch.no_grad():
                # Get Embedding (Mean) and Latent Std
                z_mean, h_std = encoder.forward_with_stats(X_t)
                
                z_vec = z_mean.squeeze(0).cpu().numpy()      # Primary embedding vector
                h_std_vec = h_std.squeeze(0).cpu().numpy()   # Std of trials vector
            
            # Prepare Row Data
            meta_row = df_meta[df_meta["filename"] == fname].iloc[0]
            sid = meta_row.get("subject_id", "unknown")
            year = meta_row.get("year", "unknown")

            row = {
                "subject_id": sid,
                "year": year,
                "filename": fname,
                # Scalar summaries of the embedding vector
                "embedding_vec_mean": float(np.mean(z_vec)),
                "embedding_vec_std": float(np.std(z_vec)),
                # Scalar summary of trial variability
                "trial_latents_mean_std": float(np.mean(h_std_vec)) 
            }
            
            # Save full embedding vector elements
            for i, val in enumerate(z_vec):
                row[f"emb_{i}"] = val
            
            results.append(row)
            count_ok += 1
            
            if count_ok % 100 == 0:
                print(f"Processed {count_ok} files...")

        except Exception as e:
            print(f"[Error] {fname}: {e}")
            count_err += 1

    # --- 3. Save Results ---
    if results:
        df_out = pd.DataFrame(results)
        
        # Ensure output directory exists before saving
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_out.to_csv(out_path, index=False)
        print(f"Done. Saved embeddings for {len(df_out)} subjects to {out_path}")
        print(f"Stats: OK={count_ok}, Missing={count_miss}, Error={count_err}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()