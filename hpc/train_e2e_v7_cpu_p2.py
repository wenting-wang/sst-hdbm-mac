# USE REAL POMDP - CPU DATA GENERATION SCRIPT
# FULL 8-PARAMETER E2E MODEL WITH INFORMATIVE PRIORS

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

warnings.filterwarnings("ignore", message="Can't initialize NVML")

# --- PATH RESOLUTION ---
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
sys.path.append(str(BASE_DIR))

ORDERS_CSV_PATH = BASE_DIR / "orders.csv"
POSTERIORS_CSV_PATH = BASE_DIR / "pomdp_params_prior.csv"

from core.hdbm_v2 import HDBM  
from core.pomdp import POMDP
from core import simulation

# --- 1. CONFIGURATION (8 FREE PARAMS) ---
HDBM_PARAM_RANGE = {
    "k_go": (0.0, 10.0),  
    "rho": (0.0, 1.0)
}

POMDP_PARAM_RANGE = {
    "q_d_n": (0.0, 1.0), "q_d": (0.5, 1.0),
    "q_s_n": (0.0, 1.0), "q_s": (0.5, 1.0),
    "cost_stop_error": (0.3, 2.0), "inv_temp": (10, 100)
}
FIXED_PARAMS = {"cost_time": 0.001, "cost_go_error": 1.0, "cost_go_missing": 1.0}
FREE_PARAM = list(HDBM_PARAM_RANGE.keys()) + list(POMDP_PARAM_RANGE.keys())
PARAM_RANGE = {**HDBM_PARAM_RANGE, **POMDP_PARAM_RANGE}

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

# --- 2. DATA LOADING & SAMPLING (INFORMATIVE PRIORS) ---
def load_informative_priors() -> pd.DataFrame:
    df_post = pd.read_csv(POSTERIORS_CSV_PATH)
    pomdp_params = list(POMDP_PARAM_RANGE.keys())
    # 转置成以 subject_id 为行的矩阵
    df_means = df_post.pivot(index='subject_id', columns='index', values='mean')
    df_stds = df_post.groupby('index')['std'].mean() # 获取群体平均不确定度
    return df_means, df_stds

try:
    DF_MEANS, DF_STDS = load_informative_priors()
    print(f"Loaded Informative Priors for {len(DF_MEANS)} baseline subjects.")
except Exception as e:
    print(f"Warning: Failed to load posteriors CSV: {e}")
    sys.exit(1)

def sample_prior(n_samples: int) -> pd.DataFrame:
    # 1. 对 HDBM 认知参数进行 Uniform 采样（全空间探索）
    samples = {k: np.random.uniform(low, high, n_samples) for k, (low, high) in HDBM_PARAM_RANGE.items()}
    
    # 2. 对 POMDP 底层参数进行 Empirical Informative Prior 采样（软固定）
    sampled_means = DF_MEANS.sample(n=n_samples, replace=True).reset_index(drop=True)
    for p in POMDP_PARAM_RANGE.keys():
        # 在抽到的真实人类参数基础上，加上一点高斯噪声
        noise = np.random.normal(0, DF_STDS[p] * 0.5, n_samples) 
        val = sampled_means[p].values + noise
        
        # 提取我们在开头设置的上限，防止 MAF 网络归一化时越界爆错
        _, high = POMDP_PARAM_RANGE[p] 
        
        if p.startswith('q_'):
            # 【概率类参数】：限制在 0 和 1 之间（留 1e-4 安全边距防除零）
            samples[p] = np.clip(val, 1e-4, 1.0 - 1e-4)
        else:
            # 【非概率参数】(cost_stop_error, inv_temp 等)：大于 0 即可
            samples[p] = np.clip(val, 1e-4, high)
            
    return pd.DataFrame(samples)

def load_order_stats(csv_path: Path = ORDERS_CSV_PATH) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, dtype={'order_seq': str})
    probs = df['subj_cnt'] / df['subj_cnt'].sum()
    sequences = [ast.literal_eval(seq_str) if str(seq_str).startswith('[') else [int(char) for char in str(seq_str) if char.isdigit()] for seq_str in df['order_seq']]
    return np.array(sequences, dtype=object), probs.values

def sample_task_sequence(sequences: np.ndarray, probs: np.ndarray) -> List[int]:
    return sequences[np.random.choice(len(sequences), p=probs)]

def simulate_single_dataset(args) -> Tuple[np.ndarray, np.ndarray]:
    params, sequences, probs = args 
    seq_int = sample_task_sequence(sequences, probs) 
    total_trials = len(seq_int)
    go_directions = np.random.choice([0, 1], size=total_trials)
    
    hdbm = HDBM(
        alpha_go=0.85, alpha_stop=0.85,        
        k_go=params['k_go'], rho=params['rho'],      
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
            q_d_n=params['q_d_n'], q_d=params['q_d'],
            q_s_n=params['q_s_n'], q_s=params['q_s'],
            cost_stop_error=params['cost_stop_error'], inv_temp=params['inv_temp'],
            **FIXED_PARAMS
        )
        pomdp.value_iteration_tensor()
        
        res_str, rt = simulation.simu_trial(pomdp, true_go_state=go_dir, true_stop_state=true_stop_state, ssd=ssd, verbose=False)
        
        choice_idx = RES_TO_IDX[res_str]
        actual_rt = rt if choice_idx in [0, 1, 4] else 0.0
        features_seq.append([float(is_stop), ssd / 40.0, float(go_directions[t]), choice_idx / 4.0, actual_rt / 40.0])
        
        if is_stop == 1:
            if res_str == 'SS': next_stop_ssd = min(next_stop_ssd + 2, 34)
            elif res_str == 'SE': next_stop_ssd = max(next_stop_ssd - 2, 2)
            
    return np.array(features_seq, dtype=np.float32), transform_params_for_maf(params)

def generate_data(n_samples=20000):
    print(f"\nSimulating {n_samples} E2E datasets using EMPIRICAL PRIORS on CPU...")
    sequences, probs = load_order_stats(ORDERS_CSV_PATH)
    dataset_file = BASE_DIR / f"e2e_dataset_10000_part2.pt"
    
    prior_df = sample_prior(n_samples)
    tasks = [(prior_df.iloc[i].to_dict(), sequences, probs) for i in range(n_samples)]
    
    X_data, Y_data = [], []
    num_workers = max(1, os.cpu_count() - 2)
    print(f"Dispatching tasks to {num_workers} CPU cores... (Printing every 100 datasets)", flush=True)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(simulate_single_dataset, task) for task in tasks]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            x_seq, y_param = future.result()
            X_data.append(x_seq)
            Y_data.append(y_param)
            if (i + 1) % 100 == 0:
                print(f">>> Progress: [{i + 1} / {n_samples}] datasets generated...", flush=True)
                
    dataset = TensorDataset(torch.tensor(np.array(X_data), dtype=torch.float32), torch.tensor(np.array(Y_data), dtype=torch.float32))
    torch.save(dataset, dataset_file)
    print(f"Dataset successfully saved to '{dataset_file}'. Exiting...")
    sys.exit(0)

if __name__ == "__main__":
    generate_data(n_samples=10000)
