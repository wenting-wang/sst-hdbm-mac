import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
torch.set_float32_matmul_precision('high')
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import concurrent.futures
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict, Tuple, Optional
from scipy.stats import qmc

# Custom imports from the core module
from core.pomdp import POMDP
from core import simulation

# --- 1. SETUP AND SIMULATION ---

HDBM_PARAM_RANGE = {"rate_stop_trial": (0.01, 0.99)}
POMDP_PARAM_RANGE = {
    "q_d_n": (0.0, 1.0),
    "q_d":   (0.5, 1.0),
    "q_s_n": (0.0, 1.0),
    "q_s":   (0.5, 1.0),
    "cost_stop_error": (0.3, 2.0),
    "inv_temp":     (10, 100)
}
FIXED_PARAMS = {
    "cost_time": 0.001,
    "cost_go_error":   1.0,
    "cost_go_missing": 1.0,
}
FREE_PARAM = list(HDBM_PARAM_RANGE.keys()) + list(POMDP_PARAM_RANGE.keys())
PARAM_RANGE = {**HDBM_PARAM_RANGE, **POMDP_PARAM_RANGE}

# def sample_prior(n_samples: int) -> pd.DataFrame:
#     samples = {}
#     for k in FREE_PARAM:
#         low, high = PARAM_RANGE[k]
#         samples[k] = np.random.uniform(low, high, n_samples)
#     return pd.DataFrame(samples)

def sample_prior(n_samples: int) -> pd.DataFrame:
    """
    Latin Hypercube Sampling
    """
    sampler = qmc.LatinHypercube(d=len(FREE_PARAM))
    lhs_samples = sampler.random(n=n_samples)
    
    samples = {}
    for i, k in enumerate(FREE_PARAM):
        low, high = PARAM_RANGE[k]
        samples[k] = lhs_samples[:, i] * (high - low) + low
        
    return pd.DataFrame(samples)


def _worker_simulate_batch(args):
    """
    Solves the POMDP exactly ONCE, then simulates multiple trials
    across all combinations of Go/Stop states and SSDs.
    """
    param_dict, seed = args
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
    # 1. EXPENSIVE STEP: Solve POMDP Once
    pomdp = POMDP(**param_dict, **FIXED_PARAMS)
    pomdp.value_iteration_tensor()
    
    results = []
    
    # 2. FAST STEP: Simulate multiple GO trials (e.g., 10 left, 10 right to capture RT variance)
    for go_state in ['left', 'right']:
        for _ in range(10): # 10 repeats to capture internal stochasticity (if any)
            res, rt = simulation.simu_trial(
                pomdp, true_go_state=go_state, true_stop_state='nonstop', ssd=-1, verbose=False
            )
            row = param_dict.copy()
            row.update({
                'ssd': -1,
                'true_go_state': 1 if go_state == 'right' else 0,
                'true_stop_state': 0,
                'res': {'GS': 0, 'GE': 1, 'GM': 2, 'SS': 3, 'SE': 4}[res], 
                'rt': rt
            })
            results.append(row)
            
    # 3. FAST STEP: Simulate STOP trials covering EVERY SSD from 2 to 34
    for go_state in ['left', 'right']:
        for ssd in range(2, 35):
            res, rt = simulation.simu_trial(
                pomdp, true_go_state=go_state, true_stop_state='stop', ssd=ssd, verbose=False
            )
            row = param_dict.copy()
            row.update({
                'ssd': ssd,
                'true_go_state': 1 if go_state == 'right' else 0,
                'true_stop_state': 1,
                'res': {'GS': 0, 'GE': 1, 'GM': 2, 'SS': 3, 'SE': 4}[res], 
                'rt': rt
            })
            results.append(row)
            
    # Total trials per POMDP solve = 20 Go + 66 Stop = 86 trials
    return results

def get_or_generate_dataset(n_pomdp_solves=2500, filename="pomdp_dataset_200k_dense.csv"):
    """
    n_pomdp_solves: Number of unique POMDPs to solve. 
    Total dataset size will be n_pomdp_solves * 86.
    (e.g., 2500 solves * 86 = 215,000 samples)
    """
    if os.path.exists(filename):
        print(f"Found existing dataset '{filename}'. Loading data from disk...")
        return pd.read_parquet(filename)
        
    # num_workers = max(1, os.cpu_count()) 
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    num_workers = max(1, num_workers)
    
    expected_samples = n_pomdp_solves * 86
    print(f"Solving {n_pomdp_solves} POMDPs to generate ~{expected_samples} trials using {num_workers} CPU cores...")
    
    params_df = sample_prior(n_pomdp_solves)
    base_seed = np.random.randint(0, 1000000)
    tasks = [(params_df.iloc[i].to_dict(), base_seed + i) for i in range(n_pomdp_solves)]
    
    all_results = []
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # returns a list of lists
        # for batch_results in tqdm(executor.map(_worker_simulate_batch, tasks), total=n_pomdp_solves):
        for batch_results in tqdm(executor.map(_worker_simulate_batch, tasks), 
                          total=n_pomdp_solves, 
                          mininterval=60.0, 
                          miniters=500,
                          ascii=True):
            all_results.extend(batch_results)
        
    df = pd.DataFrame(all_results)
    # df.to_csv(filename, index=False)
    df.to_parquet(filename, index=False)
    
    print(f"Data saved successfully to {filename}. Total samples: {len(df)}")
    return df


# --- 2. ENHANCED RESNET SURROGATE MODEL ---

class ResidualBlock(nn.Module):
    """A standard fully-connected residual block with Mish activation."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.Mish() # Mish performs better than ReLU for smooth probabilistic surfaces
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual  # Skip connection
        out = self.act(out)
        out = self.drop(out)
        return out

class POMDPSurrogate(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=512, num_blocks=3):
        super().__init__()
        # Input Layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish()
        )
        
        # Deep Residual Blocks to capture complex parameter interactions
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=0.1) for _ in range(num_blocks)
        ])
        
        # Output compression
        self.shared_out = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.Mish()
        )
        
        # Dual Heads
        self.choice_head = nn.Linear(256, 5) 
        self.rt_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU() # Prevent negative RT
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.shared_out(x)
        
        choice_logits = self.choice_head(x)
        rt_pred = self.rt_head(x)
        return choice_logits, rt_pred

def preprocess_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_features = df.copy()
    if 'cost_stop_error' in df_features.columns:
        df_features['cost_stop_error'] = np.log(df_features['cost_stop_error'])
    if 'inv_temp' in df_features.columns:
        df_features['inv_temp'] = np.log(df_features['inv_temp'])
        
    feature_cols = FREE_PARAM + ['ssd', 'true_go_state', 'true_stop_state']
    X_raw = df_features[feature_cols].values
    
    # Very strict Min-Max scaling to ensure equal gradient priority
    X_min = X_raw.min(axis=0)
    X_max = X_raw.max(axis=0)
    X_scaled = (X_raw - X_min) / (X_max - X_min + 1e-8)
    return X_scaled, X_min, X_max

def calculate_class_weights(y: np.ndarray) -> torch.Tensor:
    """Calculates inverse frequency weights to handle imbalanced classes."""
    class_counts = np.bincount(y, minlength=5)
    class_counts[class_counts == 0] = 1 # Prevent division by zero
    weights = 1.0 / np.sqrt(class_counts)
    weights = weights / weights.sum() * 5 # Normalize
    return torch.tensor(weights, dtype=torch.float32)

def fit_surrogate_model(df: pd.DataFrame, max_epochs=500, batch_size=1024, patience=30) -> Tuple[POMDPSurrogate, np.ndarray, np.ndarray]:
    print("Training Enhanced ResNet Surrogate Model...")
    
    X_scaled, X_min, X_max = preprocess_features(df)
    X = torch.tensor(X_scaled, dtype=torch.float32)
    
    Y_choice = torch.tensor(df['res'].values, dtype=torch.long)
    Y_rt = torch.tensor(df['rt'].values / 40.0, dtype=torch.float32).unsqueeze(1) 
    
    dataset = TensorDataset(X, Y_choice, Y_rt)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    
    # model = POMDPSurrogate(input_dim=X.shape[1]).cuda() if torch.cuda.is_available() else POMDPSurrogate(input_dim=X.shape[1])
    # device = next(model.parameters()).device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")
    model = POMDPSurrogate(input_dim=X.shape[1], hidden_dim=1024, num_blocks=6).to(device)
    
    if torch.__version__ >= "2.0.0" and device.type == "cuda":
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
    
    # Using AdamW for better weight decay generalization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    
    class_weights = calculate_class_weights(df['res'].values).to(device)
    criterion_choice = nn.CrossEntropyLoss(weight=class_weights)
    criterion_rt = nn.HuberLoss() 
    
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_Y_choice, batch_Y_rt in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_Y_choice = batch_Y_choice.to(device, non_blocking=True)
            batch_Y_rt = batch_Y_rt.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            choice_logits, rt_pred = model(batch_X)
            loss_choice = criterion_choice(choice_logits, batch_Y_choice)
            
            mask = (batch_Y_choice == 0) | (batch_Y_choice == 1) | (batch_Y_choice == 4)
            loss_rt = criterion_rt(rt_pred[mask], batch_Y_rt[mask]) if mask.sum() > 0 else 0.0
            
            loss = loss_choice + 5.0 * loss_rt 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_Y_choice, batch_Y_rt in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_Y_choice = batch_Y_choice.to(device, non_blocking=True)
                batch_Y_rt = batch_Y_rt.to(device, non_blocking=True)

                choice_logits, rt_pred = model(batch_X)
                
                loss_choice = criterion_choice(choice_logits, batch_Y_choice)
                mask = (batch_Y_choice == 0) | (batch_Y_choice == 1) | (batch_Y_choice == 4)
                loss_rt = criterion_rt(rt_pred[mask], batch_Y_rt[mask]) if mask.sum() > 0 else 0.0
                
                loss = loss_choice + 10.0 * loss_rt
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d}/{max_epochs} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}! Best Val Loss: {best_val_loss:.4f}")
                break
                
    model.load_state_dict(best_model_weights)
    print("Training complete! Restored best model weights.")
    return model.cpu(), X_min, X_max

def save_surrogate(model: nn.Module, X_min: np.ndarray, X_max: np.ndarray, filepath: str = "pomdp_surrogate.pth"):
    save_dict = {
        'model_state_dict': model.state_dict(),
        'X_min': X_min,
        'X_max': X_max,
        'input_dim': len(X_min)
    }
    torch.save(save_dict, filepath)
    print(f"Surrogate model and scaling parameters saved to {filepath}")

def load_surrogate(filepath: str = "pomdp_surrogate.pth") -> Tuple[POMDPSurrogate, np.ndarray, np.ndarray]:
    checkpoint = torch.load(filepath, weights_only=False)
    model = POMDPSurrogate(input_dim=checkpoint['input_dim'],
                           hidden_dim=1024, 
                           num_blocks=6)
    
    state_dict = checkpoint['model_state_dict']
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            clean_state_dict[k[len('_orig_mod.'):]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)
    model.eval()
    return model, checkpoint['X_min'], checkpoint['X_max']

if __name__ == "__main__":
    # # TEST MODE
    # df_simulated = get_or_generate_dataset(n_pomdp_solves=10, filename="pomdp_dataset_test.csv")
    # surrogate_model, X_min, X_max = fit_surrogate_model(
    #     df_simulated, max_epochs=3, patience=2, batch_size=128
    # )
    # save_surrogate(surrogate_model, X_min, X_max, filepath="pomdp_surrogate_test.pth")


    # FULL MODE    
    df_simulated = get_or_generate_dataset(n_pomdp_solves=200000, filename="pomdp_dataset_17M_lhs.parquet")
    # Train deep ResNet surrogate
    surrogate_model, X_min, X_max = fit_surrogate_model(df_simulated, max_epochs=500, patience=40, batch_size=4096)
    save_surrogate(surrogate_model, X_min, X_max, filepath="pomdp_surrogate.pth")
