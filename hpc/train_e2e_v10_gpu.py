# USE REAL POMDP - GPU TRAINING SCRIPT
# FULL 8-PARAMETER E2E MODEL (2 HDBM + 6 POMDP) + FAST EVALUATION & REAL DATA INFERENCE

import sys
import os
import ast
import warnings
import math
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List
import concurrent.futures

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm
import zuko
import matplotlib.pyplot as plt
import copy
from torch.utils.data import random_split
warnings.filterwarnings("ignore", message="Can't initialize NVML")

# =============================================================================
# GLOBAL CONFIGURATION FOR CLUSTER RUNS
# =============================================================================
# Options: 'additive_1', 'additive_2', 'multiplicative'
FUSION_MODE = 'multiplicative'  

# Execution Options: 
# 'ALL' : Train network -> Parameter Recovery -> Real Data Inference
# 'RECOVERY_ONLY' : Load trained model -> Parameter Recovery
# 'RECOVERY_AND_INFERENCE' : Load trained model -> Parameter Recovery -> Real Data Inference
# 'INFERENCE_ONLY' : Load trained model -> Real Data Inference
EXECUTION_MODE = 'ALL'
# EXECUTION_MODE = 'INFERENCE_ONLY'

# Define free parameters for HDBM
if FUSION_MODE in ['additive_1', 'additive_2']:
    HDBM_PARAM_RANGE = {
        "eta": (0.0, 10.0),   
        "rho": (0.0, 1.0)
    }
elif FUSION_MODE == 'multiplicative':
    HDBM_PARAM_RANGE = {
        "eta": (0.0, 10.0),   
        "gamma": (0.0, 5.0)  
    }
else:
    raise ValueError(f"Unknown FUSION_MODE: {FUSION_MODE}")

# =============================================================================
# --- PATH RESOLUTION ---
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
sys.path.append(str(BASE_DIR))

ORDERS_CSV_PATH = BASE_DIR / "orders.csv"
PRIOR_CSV_PATH = BASE_DIR / "pomdp_posterior.csv"
REAL_DATA_DIR = Path("/u/wenwang/data/sst_valid_base")

from core.hdbm_v4 import HDBM  
from core.pomdp import POMDP
from core import simulation
from core.preprocessing import preprocessing

# --- 1. CONFIGURATION & PRIORS (8 FREE PARAMS) ---
POMDP_PARAM_RANGE = {
    "q_d_n": (1e-4, 0.9999), "q_d": (0.5, 0.9999),
    "q_s_n": (1e-4, 0.9999), "q_s": (0.5, 0.9999),
    "cost_stop_error": (0.3, 2.0), "inv_temp": (10, 100)
}

FIXED_PARAMS = {"cost_time": 0.001, "cost_go_error": 1.0, "cost_go_missing": 1.0}

FREE_PARAM = list(HDBM_PARAM_RANGE.keys()) + list(POMDP_PARAM_RANGE.keys())
PARAM_RANGE = {**HDBM_PARAM_RANGE, **POMDP_PARAM_RANGE}
OUTCOME_MAP = {0: 'GS', 1: 'GE', 2: 'GM', 3: 'SS', 4: 'SE'}
RES_TO_IDX = {'GS': 0, 'GE': 1, 'GM': 2, 'SS': 3, 'SE': 4} 

# --- 1.5. MAF TARGET SCALING (LOG + MIN-MAX) ---
_param_transformed_min = []
_param_transformed_max = []

for k in FREE_PARAM:
    low, high = PARAM_RANGE[k]
    if k in ['cost_stop_error', 'inv_temp']:
        _param_transformed_min.append(np.log(low))
        _param_transformed_max.append(np.log(high))
    else:
        _param_transformed_min.append(low)
        _param_transformed_max.append(high)

PARAM_TRANSFORMED_MIN = np.array(_param_transformed_min, dtype=np.float32)
PARAM_TRANSFORMED_MAX = np.array(_param_transformed_max, dtype=np.float32)

def transform_params_for_maf(params_dict: dict) -> np.ndarray:
    raw_array = []
    for k in FREE_PARAM:
        val = params_dict[k]
        if k in ['cost_stop_error', 'inv_temp']:
            raw_array.append(np.log(val))
        else:
            raw_array.append(val)
    transformed_array = np.array(raw_array, dtype=np.float32)
    scaled_array = (transformed_array - PARAM_TRANSFORMED_MIN) / (PARAM_TRANSFORMED_MAX - PARAM_TRANSFORMED_MIN + 1e-8)
    return scaled_array

def inverse_transform_from_maf(scaled_tensor: torch.Tensor) -> np.ndarray:
    if isinstance(scaled_tensor, torch.Tensor):
        scaled_array = scaled_tensor.detach().cpu().numpy()
    else:
        scaled_array = scaled_tensor
    transformed_array = scaled_array * (PARAM_TRANSFORMED_MAX - PARAM_TRANSFORMED_MIN) + PARAM_TRANSFORMED_MIN
    for i, k in enumerate(FREE_PARAM):
        if k in ['cost_stop_error', 'inv_temp']:
            transformed_array[..., i] = np.exp(transformed_array[..., i])
    return transformed_array

# --- 2. DATA LOADING & SAMPLING (For evaluation) ---
def sample_prior(n_samples: int, prior_df: pd.DataFrame) -> pd.DataFrame:
    samples = {}
    for k in HDBM_PARAM_RANGE.keys():
        low, high = HDBM_PARAM_RANGE[k]
        samples[k] = np.random.uniform(low, high, n_samples)
    pomdp_keys = list(POMDP_PARAM_RANGE.keys())
    df_filtered = prior_df[prior_df['index'].isin(pomdp_keys)]
    df_mean = df_filtered.pivot(index='subject_id', columns='index', values='mean').dropna()
    df_std = df_filtered.pivot(index='subject_id', columns='index', values='std').dropna()
    subjects = df_mean.index.values
    selected_subject_indices = np.random.choice(len(subjects), size=n_samples, replace=True)
    for k in pomdp_keys:
        low, high = POMDP_PARAM_RANGE[k]
        means = df_mean[k].values[selected_subject_indices]
        stds = np.maximum(df_std[k].values[selected_subject_indices], 1e-6)
        noisy_vals = np.random.normal(loc=means, scale=stds)
        samples[k] = np.clip(noisy_vals, low, high)
    return pd.DataFrame(samples)

def load_order_stats(csv_path: Path = ORDERS_CSV_PATH) -> Tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv(csv_path, dtype={'order_seq': str})
        probs = df['subj_cnt'] / df['subj_cnt'].sum()
        sequences = []
        for seq_str in df['order_seq']:
            seq_str = str(seq_str).strip()
            if seq_str.startswith('['):
                seq = ast.literal_eval(seq_str)
            else:
                seq = [int(char) for char in seq_str if char.isdigit()]
            sequences.append(seq)
        return np.array(sequences, dtype=object), probs.values
    except Exception as e:
        warnings.warn(f"Failed to load '{csv_path}'. Using random sequences. Error: {e}")
        return None, None

def sample_task_sequence(sequences: np.ndarray, probs: np.ndarray) -> List[int]:
    sampled_idx = np.random.choice(len(sequences), p=probs)
    return sequences[sampled_idx]

def simulate_single_dataset(args) -> Tuple[np.ndarray, np.ndarray]:
    params, sequences, probs = args 
    seq_int = sample_task_sequence(sequences, probs) 
    total_trials = len(seq_int)
    go_directions = np.random.choice([0, 1], size=total_trials)
    
    hdbm_kwargs = {
        'alpha_go': 0.85, 'alpha_stop': 0.85, 'eta': params['eta'], 'fusion_type': FUSION_MODE
    }
    if FUSION_MODE in ['additive_1', 'additive_2']:
        hdbm_kwargs['rho'] = params['rho']
    elif FUSION_MODE == 'multiplicative':
        hdbm_kwargs['gamma'] = params['gamma']

    hdbm = HDBM(**hdbm_kwargs)
    r_preds = hdbm.simu_task(seq_int, block_size=180)

    next_stop_ssd = 2
    features_seq = []
    
    for t in range(total_trials):
        is_stop = seq_int[t]
        go_dir = 'right' if go_directions[t] == 1 else 'left'
        r_pred = float(r_preds[t])
        
        true_stop_state = 'stop' if is_stop == 1 else 'nonstop'
        ssd = next_stop_ssd if is_stop == 1 else -1
        
        pomdp = POMDP(
            rate_stop_trial=r_pred,  
            q_d_n=params['q_d_n'], q_d=params['q_d'],
            q_s_n=params['q_s_n'], q_s=params['q_s'],
            cost_stop_error=params['cost_stop_error'], inv_temp=params['inv_temp'],
            **FIXED_PARAMS
        )
        pomdp.value_iteration_tensor()
        
        res_str, rt = simulation.simu_trial(
            pomdp, true_go_state=go_dir, true_stop_state=true_stop_state, ssd=ssd, verbose=False
        )
        
        choice_idx = RES_TO_IDX[res_str]
        actual_rt = rt if choice_idx in [0, 1, 4] else 0.0
        
        features_seq.append([
            float(is_stop), ssd / 40.0, float(go_directions[t]), choice_idx / 4.0, actual_rt / 40.0
        ])
        
        if is_stop == 1:
            if res_str == 'SS': next_stop_ssd = min(next_stop_ssd + 2, 34)
            elif res_str == 'SE': next_stop_ssd = max(next_stop_ssd - 2, 2)
            
    param_scaled = transform_params_for_maf(params)
    return np.array(features_seq, dtype=np.float32), param_scaled

# --- 4. AMORTIZED INFERENCE NETWORK ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x

class AmortizedInferenceNet(nn.Module):
    def __init__(self, trial_feature_dim=5, d_model=64, n_heads=4, n_layers=2, param_dim=8):
        super().__init__()
        self.embedding = nn.Linear(trial_feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=400)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.flow = zuko.flows.MAF(features=param_dim, context=d_model, hidden_features=[128, 128])
        
    def forward(self, x_seq, true_params_scaled=None):
        emb = self.embedding(x_seq)
        emb = self.pos_encoder(emb)
        out_seq = self.transformer(emb)
        context = out_seq.mean(dim=1) 
        
        dist = self.flow(context)
        if true_params_scaled is not None:
            return dist.log_prob(true_params_scaled)
        return dist

# --- 5. EVALUATION: PARAMETER RECOVERY ---
def evaluate_parameter_recovery(model, device, num_test=32): 
    print("\n" + "="*50)
    print("--- Evaluating Parameter Recovery (FAST MULTI-CORE) ---")
    print("="*50)
    model.eval()
    
    if not PRIOR_CSV_PATH.exists():
        raise FileNotFoundError(f"Prior CSV not found at {PRIOR_CSV_PATH}")
    prior_df = pd.read_csv(PRIOR_CSV_PATH)
    
    sequences, probs = load_order_stats(ORDERS_CSV_PATH)
    prior_sampled_df = sample_prior(num_test, prior_df)
    tasks = [(prior_sampled_df.iloc[i].to_dict(), sequences, probs) for i in range(num_test)]
    
    print(f"Generating {num_test} new test datasets on CPU for evaluation...")
    X_test_data, Y_test_scaled_data = [], []
    
    num_workers = min(16, os.cpu_count() or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(simulate_single_dataset, tasks), total=num_test))
        
    for x_seq, y_scaled in results:
        X_test_data.append(x_seq)
        Y_test_scaled_data.append(y_scaled)
        
    X_test_tensor = torch.tensor(np.array(X_test_data), dtype=torch.float32).to(device)
    Y_test_scaled_tensor = torch.tensor(np.array(Y_test_scaled_data), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        posterior_dists = model(X_test_tensor) 
        samples = posterior_dists.sample((1000,)) 
        estimated_scaled_means = samples.mean(dim=0).cpu() 
        
    true_original = inverse_transform_from_maf(Y_test_scaled_tensor)
    estimated_original = inverse_transform_from_maf(estimated_scaled_means)
    
    # --- 5.1 SAVE TO CSV ---
    recovery_df = pd.DataFrame()
    for j, param_name in enumerate(FREE_PARAM):
        recovery_df[f"{param_name}_true"] = true_original[:, j]
        recovery_df[f"{param_name}_est"] = estimated_original[:, j]
        
    csv_out_path = BASE_DIR / f"parameter_recovery_{FUSION_MODE}_finetune.csv"
    recovery_df.to_csv(csv_out_path, index=False)
    print(f"Saved numerical recovery results to '{csv_out_path}'.")

    # --- 5.2 GENERATE SCATTER PLOT ---
    num_params = len(FREE_PARAM)
    cols = 4  
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    
    for j, param_name in enumerate(FREE_PARAM):
        ax = axes[j]
        t_vals = true_original[:, j]
        e_vals = estimated_original[:, j]
        ax.scatter(t_vals, e_vals, alpha=0.7, color='blue', edgecolor='k')
        
        min_val = min(np.min(t_vals), np.min(e_vals))
        max_val = max(np.max(t_vals), np.max(e_vals))
        padding = (max_val - min_val) * 0.05 if max_val != min_val else 0.1
        ax.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 'r--', lw=1.5)
        
        mae = np.mean(np.abs(t_vals - e_vals))
        corr = np.corrcoef(t_vals, e_vals)[0, 1] if np.std(t_vals) > 0 and np.std(e_vals) > 0 else 0
        
        ax.set_title(f"{param_name}\nMAE: {mae:.4f} | r: {corr:.2f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Estimated")
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    plot_out_path = BASE_DIR / f"parameter_recovery_{FUSION_MODE}_finetune.png"
    plt.savefig(plot_out_path, dpi=300)
    plt.close()

# --- 6. REAL DATA PROCESSING & INFERENCE ---
def df_to_tensor(df: pd.DataFrame) -> torch.Tensor:
    """preprocessed data converted to tensor format [seq_len, 5] for model input"""
    features = []
    for _, row in df.iterrows():
        is_stop = float(row['sequence'])
        
        if is_stop == 1.0 and pd.notna(row['ssd']):
            ssd_val = float(row['ssd']) / 40.0
        else:
            ssd_val = -1.0 / 40.0
            
        go_dir = 1.0 if row['true_go_state'] == 'right' else 0.0
        
        res = row['result']
        choice_idx = RES_TO_IDX.get(res, 2)  
        
        actual_rt = 0.0
        if choice_idx in [0, 1, 4] and pd.notna(row['rt']):
            actual_rt = float(row['rt']) / 40.0
            
        features.append([is_stop, ssd_val, go_dir, choice_idx / 4.0, actual_rt])
        
    features = features[:400] 
    return torch.tensor(features, dtype=torch.float32)

def run_real_data_inference(model, device):
    print("\n" + "="*50)
    print("--- Running Inference on REAL Human Data ---")
    print("="*50)
    
    if not REAL_DATA_DIR.exists():
        print(f"[WARNING] Real data directory '{REAL_DATA_DIR}' not found. Skipping inference.")
        return

    zip_files = list(REAL_DATA_DIR.glob("*.zip"))
    if not zip_files:
        print(f"[WARNING] No .zip files found in '{REAL_DATA_DIR}'.")
        return

    model.eval()
    subject_ids = []
    estimated_means_list = []
    
    print(f"Processing {len(zip_files)} real subjects from {REAL_DATA_DIR}...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, zpath in enumerate(zip_files):
            if (i + 1) % 1000 == 0:
                print(f">>> Progress: [{i + 1} / {len(zip_files)}] real subjects processed...", flush=True)
                
            subject_id = zpath.name.split('_baseline_')[0]
         
            try:
                with zipfile.ZipFile(zpath, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                csv_files = list(Path(tmpdir).rglob(f"*{subject_id}*.csv"))
                if not csv_files:
                    continue
                target_csv = csv_files[0]
                
                df = preprocessing(target_csv)
                tensor_seq = df_to_tensor(df).to(device)  
                
                tensor_batch = tensor_seq.unsqueeze(0)
                
                with torch.no_grad():
                    posterior_dist = model(tensor_batch)
                    samples = posterior_dist.sample((2000,)) 
                    est_scaled = samples.mean(dim=0).cpu()   
                    
                est_original = inverse_transform_from_maf(est_scaled)[0] 
                
                subject_ids.append(subject_id)
                estimated_means_list.append(est_original)
                
                target_csv.unlink()
                
            except Exception as e:
                print(f"Error processing {zpath.name}: {e}")
                continue

    if not subject_ids:
        print("No subjects were successfully processed.")
        return
        
    output_df = pd.DataFrame(estimated_means_list, columns=FREE_PARAM)
    output_df.insert(0, 'subject_id', subject_ids)
    
    out_csv_path = BASE_DIR / f"est_param_{FUSION_MODE}.csv"
    output_df.to_csv(out_csv_path, index=False)
    
    print(f"\nSuccessfully inferred parameters for {len(subject_ids)} subjects!")
    print(f"Results saved to: {out_csv_path}")


# --- 7. MAIN TRAINING PIPELINE ---
def train_pipeline(epochs=300, batch_size=128, patience=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Initialization ---")
    print(f"Current Fusion Mode: {FUSION_MODE.upper()}")
    print(f"Using device: {device.type.upper()}")
    
    dataset_files = list(BASE_DIR.glob(f"e2e_dataset_*_{FUSION_MODE}_finetune_part_*.pt"))
    
    if not dataset_files:
        single_file = BASE_DIR / f"e2e_dataset_5000_{FUSION_MODE}_finetune.pt"
        if single_file.exists():
            dataset_files = [single_file]
        else:
            print(f"\n[ERROR] No datasets found matching {FUSION_MODE}! Please run CPU generation scripts.")
            sys.exit(1)

    print(f"\nFound {len(dataset_files)} dataset parts. Loading and concatenating...")
    datasets = [torch.load(f, weights_only=False) for f in dataset_files]
    full_dataset = ConcatDataset(datasets)
    print(f"Total samples loaded: {len(full_dataset)}")

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Initializing Amortized Inference Network...")
    model = AmortizedInferenceNet(trial_feature_dim=5, param_dim=len(FREE_PARAM)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    print(f"\n--- Starting Training Loop ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            log_probs = model(batch_x, true_params_scaled=batch_y)
            loss = -log_probs.mean() 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                log_probs = model(batch_x, true_params_scaled=batch_y)
                loss = -log_probs.mean()
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1:03d}/{epochs}] - LR: {current_lr:.6f} | Train NLL: {avg_train_loss:.4f} | Val NLL: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}! Best Val Loss: {best_val_loss:.4f}")
                break

    model.load_state_dict(best_model_weights)
    model_out_path = BASE_DIR / f"amortized_inference_net_{FUSION_MODE}_finetune.pth"
    torch.save(model.state_dict(), model_out_path)
    
    return model, device

# =============================================================================
# --- MAIN EXECUTION BLOCK ---
# =============================================================================
if __name__ == "__main__":
    
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if EXECUTION_MODE == 'ALL':
        # Train from scratch, then run recovery and inference
        trained_model, comp_device = train_pipeline(epochs=300, batch_size=128, patience=40)
        evaluate_parameter_recovery(trained_model, comp_device, num_test=32)
        run_real_data_inference(trained_model, comp_device)
        
    else:
        # For RECOVERY_ONLY, INFERENCE_ONLY, or RECOVERY_AND_INFERENCE
        # We must load the pre-trained weights first
        pretrained_model_path = BASE_DIR / f"amortized_inference_net_{FUSION_MODE}_finetune.pth"
        
        if not pretrained_model_path.exists():
            print(f"[ERROR] Expected trained model file not found at: {pretrained_model_path}")
            print(f"Please ensure you have run the 'ALL' mode at least once to train the model.")
            sys.exit(1)
            
        print(f"\nLoading pre-trained model from: {pretrained_model_path}")
        trained_model = AmortizedInferenceNet(trial_feature_dim=5, param_dim=len(FREE_PARAM)).to(comp_device)
        trained_model.load_state_dict(torch.load(pretrained_model_path, map_location=comp_device, weights_only=True))
        
        # Execute the requested steps
        if EXECUTION_MODE == 'RECOVERY_ONLY':
            evaluate_parameter_recovery(trained_model, comp_device, num_test=32)
            
        elif EXECUTION_MODE == 'RECOVERY_AND_INFERENCE':
            evaluate_parameter_recovery(trained_model, comp_device, num_test=32)
            run_real_data_inference(trained_model, comp_device)
            
        elif EXECUTION_MODE == 'INFERENCE_ONLY':
            run_real_data_inference(trained_model, comp_device)
            
        else:
            print(f"[ERROR] Unknown EXECUTION_MODE: {EXECUTION_MODE}")