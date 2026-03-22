# USE REAL POMDP - CPU DATA GENERATION SCRIPT
# ADDITIVE HDBM (POMDP FROZEN)

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
torch.set_num_threads(1)
from torch.utils.data import TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Can't initialize NVML")

# --- PATH RESOLUTION ---
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
sys.path.append(str(BASE_DIR))

ORDERS_CSV_PATH = BASE_DIR / "orders.csv"

from core.hdbm_v2 import HDBM  
from core.pomdp import POMDP
from core import simulation

# --- 1. CONFIGURATION & PRIORS (POMDP FROZEN) ---
# HDBM_PARAM_RANGE = {
#     "alpha_go": (0.0, 1.0),
#     "alpha_stop": (0.0, 1.0),
#     "k_go": (0.0, 10.0),  
#     "rho": (0.0, 1.0)
# }

# HDBM_PARAM_RANGE = {
#     "alpha": (0.0, 1.0),
#     "rho": (0.0, 1.0)
# }

HDBM_PARAM_RANGE = {
    "k_go": (0.0, 10.0),
    "rho": (0.0, 1.0)
}


FIXED_PARAMS = {"cost_time": 0.001, "cost_go_error": 1.0, "cost_go_missing": 1.0}

FREE_PARAM = list(HDBM_PARAM_RANGE.keys())
PARAM_RANGE = HDBM_PARAM_RANGE
OUTCOME_MAP = {0: 'GS', 1: 'GE', 2: 'GM', 3: 'SS', 4: 'SE'}
RES_TO_IDX = {'GS': 0, 'GE': 1, 'GM': 2, 'SS': 3, 'SE': 4} 

# --- 1.5. MAF TARGET SCALING (LOG + MIN-MAX) ---
_param_transformed_min = []
_param_transformed_max = []

for k in FREE_PARAM:
    low, high = PARAM_RANGE[k]
    _param_transformed_min.append(low)
    _param_transformed_max.append(high)

PARAM_TRANSFORMED_MIN = np.array(_param_transformed_min, dtype=np.float32)
PARAM_TRANSFORMED_MAX = np.array(_param_transformed_max, dtype=np.float32)

def transform_params_for_maf(params_dict: dict) -> np.ndarray:
    raw_array = [params_dict[k] for k in FREE_PARAM]
    transformed_array = np.array(raw_array, dtype=np.float32)
    scaled_array = (transformed_array - PARAM_TRANSFORMED_MIN) / (PARAM_TRANSFORMED_MAX - PARAM_TRANSFORMED_MIN + 1e-8)
    return scaled_array

# --- 2. DATA LOADING & SAMPLING ---
def sample_prior(n_samples: int) -> pd.DataFrame:
    samples = {k: np.random.uniform(low, high, n_samples) for k, (low, high) in PARAM_RANGE.items()}
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
    
    # hdbm = HDBM(
    #     alpha_go=params['alpha_go'], 
    #     alpha_stop=params['alpha_stop'], 
    #     k_go=params['k_go'], 
    #     rho=params['rho'],
    #     fusion_type='additive'
    # )
    
    # hdbm = HDBM(
    #     alpha_go=params['alpha'],       
    #     alpha_stop=params['alpha'],     
    #     k_go=1.0,                      
    #     rho=params['rho'],
    #     fusion_type='additive'
    # )
    
    hdbm = HDBM(
        alpha_go=0.85,     
        alpha_stop=0.85,   
        k_go=params['k_go'], 
        rho=params['rho'],
        fusion_type='additive'
    )
    
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
            q_d_n=0.518,
            q_d=0.725,
            q_s_n=0.018,
            q_s=0.839,
            cost_stop_error=1.649,
            inv_temp=36.174,
            **FIXED_PARAMS
        )
        pomdp.value_iteration_tensor()
        
        res_str, rt = simulation.simu_trial(
            pomdp, true_go_state=go_dir, true_stop_state=true_stop_state, ssd=ssd, verbose=False
        )
        
        choice_idx = RES_TO_IDX[res_str]
        actual_rt = rt if choice_idx in [0, 1, 4] else 0.0
        
        features_seq.append([
            float(is_stop), 
            ssd / 40.0, 
            float(go_directions[t]), 
            choice_idx / 4.0, 
            actual_rt / 40.0
        ])
        
        if is_stop == 1:
            if res_str == 'SS': next_stop_ssd = min(next_stop_ssd + 2, 34)
            elif res_str == 'SE': next_stop_ssd = max(next_stop_ssd - 2, 2)
            
    param_scaled = transform_params_for_maf(params)
    return np.array(features_seq, dtype=np.float32), param_scaled

def generate_data(n_samples=2000):
    print(f"\nSimulating {n_samples} training datasets with REAL POMDP (FROZEN PARAMETERS)...")
    sequences, probs = load_order_stats(ORDERS_CSV_PATH)
    dataset_file = BASE_DIR / f"e2e_dataset_{n_samples}_REAL_POMDP.pt"
    
    prior_df = sample_prior(n_samples)
    tasks = [(prior_df.iloc[i].to_dict(), sequences, probs) for i in range(n_samples)]
    
    X_data, Y_data = [], []
    num_workers = max(1, os.cpu_count() - 2)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(simulate_single_dataset, tasks), total=n_samples, mininterval=10.0, ascii=True))
        
    for x_seq, y_param in results:
        X_data.append(x_seq)
        Y_data.append(y_param)
        
    X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_data), dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    torch.save(dataset, dataset_file)
    print(f"Dataset successfully saved to '{dataset_file}'.")
    print("Data generation complete on CPU! Exiting script before training starts.")
    sys.exit(0)

if __name__ == "__main__":
    generate_data(n_samples=2000)