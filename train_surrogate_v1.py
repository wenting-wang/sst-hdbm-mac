import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import concurrent.futures
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict, Tuple, Optional

from core.pomdp import POMDP
from core import simulation

# --- 1. SETUP AND SIMULATION ---

HDBM_PARAM_RANGE = {"rate_stop_trial": (0.05, 0.55)}
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

def sample_prior(n_samples: int) -> pd.DataFrame:
    samples = {}
    for k in FREE_PARAM:
        low, high = PARAM_RANGE[k]
        samples[k] = np.random.uniform(low, high, n_samples)
    return pd.DataFrame(samples)

def simulate_trial(params: Dict[str, float], seed: Optional[int] = None) -> Tuple[str, float, int, str, str]:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    
    true_go_state = np.random.choice(['left', 'right'])
    true_stop_state = np.random.choice(['stop', 'nonstop'])
    
    if true_stop_state == 'nonstop':
        ssd = -1 
    else:
        ssd = np.random.choice(list(range(2, 34))) 

    pomdp = POMDP(**params, **FIXED_PARAMS)
    pomdp.value_iteration_tensor()
    res, rt = simulation.simu_trial(
                pomdp, true_go_state=true_go_state, 
                true_stop_state=true_stop_state, 
                ssd=ssd, verbose=False)

    return res, rt, ssd, true_go_state, true_stop_state

def _worker_simulate(args):
    param_dict, seed = args
    res, rt, ssd, go_state, stop_state = simulate_trial(param_dict, seed=seed)
    
    row = param_dict.copy()
    row.update({
        'ssd': ssd,
        'true_go_state': 1 if go_state == 'right' else 0,
        'true_stop_state': 1 if stop_state == 'stop' else 0,
        'res': {'GS': 0, 'GE': 1, 'GM': 2, 'SS': 3, 'SE': 4}[res], 
        'rt': rt
    })
    return row

def get_or_generate_dataset(n_samples=100000, filename="pomdp_dataset_100k.csv"):
    """Loads existing dataset if found, otherwise generates and saves it."""
    if os.path.exists(filename):
        print(f"Found existing dataset '{filename}'. Loading data from disk...")
        return pd.read_csv(filename)
        
    num_workers = max(1, os.cpu_count() - 2) 
    print(f"Generating {n_samples} simulated trials using {num_workers} CPU cores...")
    params_df = sample_prior(n_samples)
    
    base_seed = np.random.randint(0, 1000000)
    tasks = [(params_df.iloc[i].to_dict(), base_seed + i) for i in range(n_samples)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(_worker_simulate, tasks), total=n_samples))
        
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Data saved successfully to {filename}")
    return df


# --- 2. ENHANCED SURROGATE MODEL ---

class POMDPSurrogate(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=256):
        super().__init__()
        # Deeper network with Batch Normalization to learn complex surfaces
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.choice_head = nn.Linear(128, 5) 
        self.rt_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.ReLU() # Prevent negative RT
        )

    def forward(self, x):
        shared_features = self.shared_net(x)
        choice_logits = self.choice_head(shared_features)
        rt_pred = self.rt_head(shared_features)
        return choice_logits, rt_pred

def preprocess_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_features = df.copy()
    if 'cost_stop_error' in df_features.columns:
        df_features['cost_stop_error'] = np.log(df_features['cost_stop_error'])
    if 'inv_temp' in df_features.columns:
        df_features['inv_temp'] = np.log(df_features['inv_temp'])
        
    feature_cols = FREE_PARAM + ['ssd', 'true_go_state', 'true_stop_state']
    X_raw = df_features[feature_cols].values
    
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

def fit_surrogate_model(df: pd.DataFrame, max_epochs=500, batch_size=1024, patience=50) -> Tuple[POMDPSurrogate, np.ndarray, np.ndarray]:
    print("Training Enhanced Surrogate Model...")
    
    X_scaled, X_min, X_max = preprocess_features(df)
    X = torch.tensor(X_scaled, dtype=torch.float32)
    
    Y_choice = torch.tensor(df['res'].values, dtype=torch.long)
    Y_rt = torch.tensor(df['rt'].values / 40.0, dtype=torch.float32).unsqueeze(1) 
    
    dataset = TensorDataset(X, Y_choice, Y_rt)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = POMDPSurrogate(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Class weights for imbalanced Choice outcomes
    class_weights = calculate_class_weights(df['res'].values)
    criterion_choice = nn.CrossEntropyLoss(weight=class_weights)
    
    # HuberLoss is much more robust to outliers in RT than MSE
    criterion_rt = nn.HuberLoss() 
    
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_Y_choice, batch_Y_rt in train_loader:
            optimizer.zero_grad()
            choice_logits, rt_pred = model(batch_X)
            
            loss_choice = criterion_choice(choice_logits, batch_Y_choice)
            mask = (batch_Y_choice == 0) | (batch_Y_choice == 1) | (batch_Y_choice == 4)
            loss_rt = criterion_rt(rt_pred[mask], batch_Y_rt[mask]) if mask.sum() > 0 else 0.0
            
            # Increase weight for RT to force the model to learn it better
            loss = loss_choice + 5.0 * loss_rt 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_Y_choice, batch_Y_rt in val_loader:
                choice_logits, rt_pred = model(batch_X)
                
                loss_choice = criterion_choice(choice_logits, batch_Y_choice)
                mask = (batch_Y_choice == 0) | (batch_Y_choice == 1) | (batch_Y_choice == 4)
                loss_rt = criterion_rt(rt_pred[mask], batch_Y_rt[mask]) if mask.sum() > 0 else 0.0
                
                loss = loss_choice + 5.0 * loss_rt
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 5 == 0:
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
    return model, X_min, X_max

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
    model = POMDPSurrogate(input_dim=checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['X_min'], checkpoint['X_max']

if __name__ == "__main__":
    # Will generate and SAVE 100k samples. If file exists, it skips generation!
    df_simulated = get_or_generate_dataset(n_samples=100000, filename="pomdp_dataset_100k.csv")
    
    surrogate_model, X_min, X_max = fit_surrogate_model(df_simulated, max_epochs=500, patience=50, batch_size=1024)
    save_surrogate(surrogate_model, X_min, X_max, filepath="pomdp_surrogate.pth")