import sys
import os
import ast
import warnings
from pathlib import Path
from typing import Dict, Tuple, List
import concurrent.futures

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import zuko
import matplotlib.pyplot as plt
import copy
from torch.utils.data import random_split
warnings.filterwarnings("ignore", message="Can't initialize NVML")

# --- PATH RESOLUTION ---
# Get the absolute path to the directory containing this script (/kyb/agpd/wwang/sst-hdbm)
BASE_DIR = Path(__file__).resolve().parent

# Ensure the core module and current directory are in the Python path
sys.path.append(str(BASE_DIR.parent))
sys.path.append(str(BASE_DIR))

# Define robust absolute paths for dependencies
ORDERS_CSV_PATH = BASE_DIR / "orders.csv"
SURROGATE_MODEL_PATH = BASE_DIR / "pomdp_surrogate.pth"

# Import custom modules
from train_surrogate import load_surrogate
from core.hdbm import HDBM  # HDBM integration enabled

# --- 1. CONFIGURATION & PRIORS ---
HDBM_PARAM_RANGE = {"alpha": (0.0, 1.0), "rho": (0.0, 1.0)}
POMDP_PARAM_RANGE = {
    "q_d_n": (0.0, 1.0), "q_d": (0.5, 1.0),
    "q_s_n": (0.0, 1.0), "q_s": (0.5, 1.0),
    "cost_stop_error": (0.3, 2.0), "inv_temp": (10, 100)
}
FIXED_PARAMS = {"cost_time": 0.001, "cost_go_error": 1.0, "cost_go_missing": 1.0}

FREE_PARAM = list(HDBM_PARAM_RANGE.keys()) + list(POMDP_PARAM_RANGE.keys())
PARAM_RANGE = {**HDBM_PARAM_RANGE, **POMDP_PARAM_RANGE}
OUTCOME_MAP = {0: 'GS', 1: 'GE', 2: 'GM', 3: 'SS', 4: 'SE'}

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
    """Transforms raw prior parameters to [0, 1] range for stable MAF training."""
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
    """Reverts the [0, 1] scaled MAF outputs back to their original physical scales."""
    if isinstance(scaled_tensor, torch.Tensor):
        scaled_array = scaled_tensor.detach().cpu().numpy()
    else:
        scaled_array = scaled_tensor
        
    transformed_array = scaled_array * (PARAM_TRANSFORMED_MAX - PARAM_TRANSFORMED_MIN) + PARAM_TRANSFORMED_MIN
    
    for i, k in enumerate(FREE_PARAM):
        if k in ['cost_stop_error', 'inv_temp']:
            transformed_array[..., i] = np.exp(transformed_array[..., i])
            
    return transformed_array


# --- 2. DATA LOADING & SAMPLING ---
def sample_prior(n_samples: int) -> pd.DataFrame:
    samples = {k: np.random.uniform(low, high, n_samples) for k, (low, high) in PARAM_RANGE.items()}
    return pd.DataFrame(samples)

def load_order_stats(csv_path: Path = ORDERS_CSV_PATH) -> Tuple[np.ndarray, np.ndarray]:
    try:
        # 1. Force pandas to read 'order_seq' as a string to prevent float overflow
        df = pd.read_csv(csv_path, dtype={'order_seq': str})
        probs = df['subj_cnt'] / df['subj_cnt'].sum()
        sequences = []
        
        for seq_str in df['order_seq']:
            seq_str = str(seq_str).strip()
            
            # 2. Handle if it's formatted like a list: "[0, 1, 0, 1]"
            if seq_str.startswith('['):
                seq = ast.literal_eval(seq_str)
            # 3. Handle if it's formatted as a continuous string: "0100110"
            else:
                seq = [int(char) for char in seq_str if char.isdigit()]
                
            sequences.append(seq)
            
        return np.array(sequences, dtype=object), probs.values
    except Exception as e:
        warnings.warn(f"Failed to load '{csv_path}'. Using random sequences. Error: {e}")
        return None, None

def sample_task_sequence(sequences: np.ndarray, probs: np.ndarray, n_blocks: int = 2) -> List[int]:
    if sequences is None:
        return [1 if np.random.rand() < 1/6 else 0 for _ in range(180 * n_blocks)]
    sampled_idx = np.random.choice(len(sequences), size=n_blocks, p=probs)
    full_seq = []
    for idx in sampled_idx:
        full_seq.extend(sequences[idx])
    return full_seq

# --- 3. SURROGATE & TASK SIMULATION ---
def simulate_trial_surrogate(r_pred, pomdp_params, ssd, true_go_state, true_stop_state, surr_model, X_min, X_max):
    cost_stop_error_log = np.log(pomdp_params['cost_stop_error'])
    inv_temp_log = np.log(pomdp_params['inv_temp'])
    go_val = 1 if true_go_state == 'right' else 0
    stop_val = 1 if true_stop_state == 'stop' else 0
    
    X_raw = np.array([
        r_pred, pomdp_params['q_d_n'], pomdp_params['q_d'],
        pomdp_params['q_s_n'], pomdp_params['q_s'],
        cost_stop_error_log, inv_temp_log, ssd, go_val, stop_val
    ])
    
    X_scaled = (X_raw - X_min) / (X_max - X_min + 1e-8)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        choice_logits, rt_pred_scaled = surr_model(X_tensor)
        choice_probs = torch.softmax(choice_logits, dim=-1)
        choice_idx = torch.distributions.Categorical(choice_probs).sample().item()
        rt = (rt_pred_scaled.squeeze() * 40.0).item()
        
    return choice_idx, rt

def simulate_single_dataset(args) -> Tuple[np.ndarray, np.ndarray]:
    params, sequences, probs, surr_model, X_min, X_max = args
    seq_int = sample_task_sequence(sequences, probs, n_blocks=2) 
    total_trials = len(seq_int)
    go_directions = np.random.choice([0, 1], size=total_trials)
    
    hdbm = HDBM(alpha=params['alpha'], rho=params['rho'])
    r_preds = hdbm.simu_task(seq_int, block_size=180)

    next_stop_ssd = 2
    features_seq = []
    
    for t in range(total_trials):
        is_stop = seq_int[t]
        go_dir = 'right' if go_directions[t] == 1 else 'left'
        r_pred = float(r_preds[t])
        
        true_stop_state = 'stop' if is_stop == 1 else 'nonstop'
        ssd = next_stop_ssd if is_stop == 1 else -1
        
        choice_idx, rt = simulate_trial_surrogate(
            r_pred, params, ssd, go_dir, true_stop_state, surr_model, X_min, X_max
        )
        
        actual_rt = rt if choice_idx in [0, 1, 4] else 0.0
        
        features_seq.append([
            float(is_stop), 
            ssd / 40.0, 
            float(go_directions[t]), 
            choice_idx / 4.0, 
            actual_rt / 40.0
        ])
        
        if is_stop == 1:
            res_str = OUTCOME_MAP[choice_idx]
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
        """
        x: shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x
    
class AmortizedInferenceNet(nn.Module):
    def __init__(self, trial_feature_dim=5, d_model=64, n_heads=4, n_layers=2, param_dim=10):
        super().__init__()
        self.embedding = nn.Linear(trial_feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=400)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=128
        )
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
def evaluate_parameter_recovery(model, device, num_test=16):
    print("\n" + "="*50)
    print("--- Evaluating Parameter Recovery ---")
    print("="*50)
    model.eval()
    
    surr_model, X_min, X_max = load_surrogate(filepath=str(SURROGATE_MODEL_PATH))
    surr_model.eval()
    sequences, probs = load_order_stats(ORDERS_CSV_PATH)
    
    prior_df = sample_prior(num_test)
    tasks = [(prior_df.iloc[i].to_dict(), sequences, probs, surr_model, X_min, X_max) for i in range(num_test)]
    
    print(f"Generating {num_test} new test datasets on CPU for evaluation...")
    X_test_data, Y_test_scaled_data = [], []
    for task in tasks:
        x_seq, y_scaled = simulate_single_dataset(task)
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
    
    # 1. Save to CSV
    recovery_data = []
    for i in range(num_test):
        row_dict = {"subject_idx": i + 1}
        for j, param_name in enumerate(FREE_PARAM):
            row_dict[f"true_{param_name}"] = true_original[i, j]
            row_dict[f"est_{param_name}"] = estimated_original[i, j]
        recovery_data.append(row_dict)
        
    df_recovery = pd.DataFrame(recovery_data)
    csv_out_path = BASE_DIR / "parameter_recovery_results.csv"
    df_recovery.to_csv(csv_out_path, index=False)
    print(f"Saved numerical recovery results to '{csv_out_path}'.")

    # 2. Generate Scatter Plots
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    
    for j, param_name in enumerate(FREE_PARAM):
        ax = axes[j]
        t_vals = true_original[:, j]
        e_vals = estimated_original[:, j]
        
        ax.scatter(t_vals, e_vals, alpha=0.7, color='blue', edgecolor='k')
        
        min_val = min(np.min(t_vals), np.min(e_vals))
        max_val = max(np.max(t_vals), np.max(e_vals))
        padding = (max_val - min_val) * 0.05
        ax.plot([min_val - padding, max_val + padding], 
                [min_val - padding, max_val + padding], 
                'r--', lw=1.5, label='Perfect Recovery')
        
        mae = np.mean(np.abs(t_vals - e_vals))
        corr = np.corrcoef(t_vals, e_vals)[0, 1] if np.std(t_vals) > 0 and np.std(e_vals) > 0 else 0
        
        ax.set_title(f"{param_name}\nMAE: {mae:.4f} | r: {corr:.2f}")
        ax.set_xlabel("True Values")
        ax.set_ylabel("Estimated Values")
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    plot_out_path = BASE_DIR / "parameter_recovery_scatter.png"
    plt.savefig(plot_out_path, dpi=300)
    plt.close()
    print(f"Saved scatter plot matrix to '{plot_out_path}'.")
    print("="*50 + "\n")

# --- 6. MAIN TRAINING PIPELINE ---
def train_pipeline(n_samples=50000, epochs=200, batch_size=256, patience=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Initialization ---")
    print(f"Using device: {device.type.upper()}")
    print(f"Base Directory: {BASE_DIR}")
    
    surr_model, X_min, X_max = load_surrogate(filepath=str(SURROGATE_MODEL_PATH))
    surr_model.eval()
    sequences, probs = load_order_stats(ORDERS_CSV_PATH)
    
    # --- Data Generation & Offline Caching ---
    dataset_file = BASE_DIR / f"e2e_dataset_{n_samples}.pt"
    
    if dataset_file.exists():
        print(f"\n[Fast Track] Found saved offline dataset '{dataset_file}'. Loading directly...")
        dataset = torch.load(dataset_file)
        print("Dataset loaded successfully! Skipping simulation phase.")
    else:
        print(f"\nSimulating {n_samples} training datasets (CPU Multi-threading)...")
        print("This might take a while, but the result will be saved for future runs.")
        
        prior_df = sample_prior(n_samples)
        tasks = [(prior_df.iloc[i].to_dict(), sequences, probs, surr_model, X_min, X_max) for i in range(n_samples)]
        
        X_data, Y_data = [], []
        num_workers = max(1, os.cpu_count() - 2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(simulate_single_dataset, tasks), total=n_samples))
            
        for x_seq, y_param in results:
            X_data.append(x_seq)
            Y_data.append(y_param)
            
        X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32)
        Y_tensor = torch.tensor(np.array(Y_data), dtype=torch.float32)
        
        dataset = TensorDataset(X_tensor, Y_tensor)
        
        # Save the generated dataset to disk to save time in future runs
        torch.save(dataset, dataset_file)
        print(f"Dataset successfully saved to '{dataset_file}'. It will be loaded directly next time.")

    # Validation Split (80/20) matching your surrogate training
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Initializing Amortized Inference Network...")
    model = AmortizedInferenceNet(trial_feature_dim=5, param_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Learning Rate Scheduler and Early Stopping Logic
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    print(f"\n--- Starting Training Loop ---")
    for epoch in range(epochs):
        # Training Phase
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
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                log_probs = model(batch_x, true_params_scaled=batch_y)
                loss = -log_probs.mean()
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:03d}/{epochs}] - LR: {current_lr:.6f} | Train NLL: {avg_train_loss:.4f} | Val NLL: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}! Best Val Loss: {best_val_loss:.4f}")
                break

    # Restore best weights and save
    model.load_state_dict(best_model_weights)
    model_out_path = BASE_DIR / "amortized_inference_net.pth"
    torch.save(model.state_dict(), model_out_path)
    print(f"\nTraining Complete! Best model saved to '{model_out_path}'.")
    
    evaluate_parameter_recovery(model, device, num_test=16)

if __name__ == "__main__":
    # train_pipeline(n_samples=16, epochs=5, batch_size=8)
    # train_pipeline(n_samples=1000, epochs=50, batch_size=64)

    # HPC / Cloud scale settings
    # train_pipeline(n_samples=50000, epochs=200, batch_size=256, patience=30)


