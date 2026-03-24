# USE REAL POMDP - GPU TRAINING SCRIPT
# DEEP TRANSFORMER + FULL 8-PARAMETER JOINT INFERENCE

import sys
import os
import ast
import warnings
import math
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

def inverse_transform_from_maf(scaled_tensor: torch.Tensor) -> np.ndarray:
    scaled_array = scaled_tensor.detach().cpu().numpy() if isinstance(scaled_tensor, torch.Tensor) else scaled_tensor
    transformed_array = scaled_array * (PARAM_TRANSFORMED_MAX - PARAM_TRANSFORMED_MIN) + PARAM_TRANSFORMED_MIN
    for i, k in enumerate(FREE_PARAM):
        if k in ['cost_stop_error', 'inv_temp']:
            transformed_array[..., i] = np.exp(transformed_array[..., i])
    return transformed_array

# --- 2. EVALUATION PRIORS ---
def load_informative_priors() -> pd.DataFrame:
    df_post = pd.read_csv(POSTERIORS_CSV_PATH)
    df_means = df_post.pivot(index='subject_id', columns='index', values='mean')
    df_stds = df_post.groupby('index')['std'].mean()
    return df_means, df_stds

try:
    DF_MEANS, DF_STDS = load_informative_priors()
except Exception:
    DF_MEANS, DF_STDS = None, None

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
    
    hdbm = HDBM(alpha_go=0.85, alpha_stop=0.85, k_go=params['k_go'], rho=params['rho'], fusion_type='additive')
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
            q_d_n=params['q_d_n'], q_d=params['q_d'], q_s_n=params['q_s_n'], q_s=params['q_s'],
            cost_stop_error=params['cost_stop_error'], inv_temp=params['inv_temp'], **FIXED_PARAMS
        )
        pomdp.value_iteration_tensor()
        
        res_str, rt = simulation.simu_trial(pomdp, true_go_state=go_dir, true_stop_state=true_stop_state, ssd=ssd, verbose=False)
        choice_idx = RES_TO_IDX[res_str]
        actual_rt = rt if choice_idx in [0, 1, 4] else 0.0
        features_seq.append([float(is_stop), ssd / 40.0, float(go_directions[t]), choice_idx / 4.0, actual_rt / 40.0])
        
        if is_stop == 1:
            if res_str == 'SS': next_stop_ssd = min(next_stop_ssd + 2, 34)
            elif res_str == 'SE': next_stop_ssd = max(next_stop_ssd - 2, 2)
            
    # Dummy scaled_params (not used in inference phase, only true value matters for plotting)
    # the scaled parameter values in inference are generated elsewhere.
    return np.array(features_seq, dtype=np.float32), np.zeros(8) 

# --- 4. DEEP AMORTIZED INFERENCE NETWORK ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class AmortizedInferenceNet(nn.Module):
    # 【升级】：Deeper & Wider Transformer!
    def __init__(self, trial_feature_dim=5, d_model=128, n_heads=8, n_layers=4, param_dim=8):
        super().__init__()
        self.embedding = nn.Linear(trial_feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=400)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 【升级】：更深的 MAF 流
        self.flow = zuko.flows.MAF(features=param_dim, context=d_model, hidden_features=[256, 256, 256])
        
    def forward(self, x_seq, true_params_scaled=None):
        emb = self.pos_encoder(self.embedding(x_seq))
        context = self.transformer(emb).mean(dim=1) 
        dist = self.flow(context)
        return dist.log_prob(true_params_scaled) if true_params_scaled is not None else dist

# --- 5. EVALUATION ---
def evaluate_parameter_recovery(model, device, num_test=64): 
    print("\n" + "="*50)
    print("--- Evaluating Parameter Recovery (FAST MULTI-CORE) ---")
    model.eval()
    
    sequences, probs = load_order_stats(ORDERS_CSV_PATH)
    prior_df = sample_prior(num_test)
    
    # We must scale the true targets using the SAME function logic
    Y_test_scaled_data = []
    for i in range(num_test):
        params = prior_df.iloc[i].to_dict()
        raw_array = []
        for k in FREE_PARAM:
            val = params[k]
            raw_array.append(np.log(val) if k in ['cost_stop_error', 'inv_temp'] else val)
        transformed_array = np.array(raw_array, dtype=np.float32)
        scaled_array = (transformed_array - PARAM_TRANSFORMED_MIN) / (PARAM_TRANSFORMED_MAX - PARAM_TRANSFORMED_MIN + 1e-8)
        Y_test_scaled_data.append(scaled_array)
        
    tasks = [(prior_df.iloc[i].to_dict(), sequences, probs) for i in range(num_test)]
    print(f"Generating {num_test} new test datasets on CPU for evaluation...")
    X_test_data = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as executor:
        results = list(tqdm(executor.map(simulate_single_dataset, tasks), total=num_test))
        for x_seq, _ in results:
            X_test_data.append(x_seq)
            
    X_test_tensor = torch.tensor(np.array(X_test_data), dtype=torch.float32).to(device)
    Y_test_scaled_tensor = torch.tensor(np.array(Y_test_scaled_data), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        samples = model(X_test_tensor).sample((1000,)) 
        estimated_scaled_means = samples.mean(dim=0).cpu() 
        
    true_original = inverse_transform_from_maf(Y_test_scaled_tensor)
    estimated_original = inverse_transform_from_maf(estimated_scaled_means)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for j, (ax, param_name) in enumerate(zip(axes.flatten(), FREE_PARAM)):
        t_vals, e_vals = true_original[:, j], estimated_original[:, j]
        ax.scatter(t_vals, e_vals, alpha=0.7, color='blue', edgecolor='k')
        min_v, max_v = min(np.min(t_vals), np.min(e_vals)), max(np.max(t_vals), np.max(e_vals))
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=1.5)
        
        mae = np.mean(np.abs(t_vals - e_vals))
        corr = np.corrcoef(t_vals, e_vals)[0, 1] if np.std(t_vals)>0 and np.std(e_vals)>0 else 0
        ax.set_title(f"{param_name}\nMAE: {mae:.4f} | r: {corr:.2f}")
        ax.set(xlabel="True", ylabel="Estimated")
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    plot_out_path = BASE_DIR / "parameter_recovery_scatter_DEEP_8PARAMS.png"
    plt.savefig(plot_out_path, dpi=300)
    plt.close()
    print(f"Saved scatter plot matrix to '{plot_out_path}'.")

# # --- 6. MAIN TRAINING PIPELINE ---
# def train_pipeline(n_samples=20000, epochs=500, batch_size=256, patience=50):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"--- Deep Amortized Inference Initialization ---")
#     print(f"Using device: {device.type.upper()}")
    
#     # dataset_file = BASE_DIR / f"e2e_dataset_{n_samples}_REAL_POMDP.pt"
#     # if dataset_file.exists():
#     #     print(f"\n[Fast Track] Loading {n_samples} Empirical Prior dataset...")
#     #     dataset = torch.load(dataset_file, weights_only=False)
#     # else:
#     #     print(f"\n[ERROR] Dataset '{dataset_file}' not found! Run CPU generation first.")
#     #     sys.exit(1)
    
#     dataset_file_1 = BASE_DIR / "e2e_dataset_10000_part1.pt"
#     dataset_file_2 = BASE_DIR / "e2e_dataset_10000_part2.pt"
    
#     if dataset_file_1.exists() and dataset_file_2.exists():
#         print(f"\n[Fast Track] Found two dataset parts. Loading and merging...")
#         d1 = torch.load(dataset_file_1, weights_only=False)
#         d2 = torch.load(dataset_file_2, weights_only=False)
        
#         X_all = torch.cat([d1.tensors[0], d2.tensors[0]], dim=0)
#         Y_all = torch.cat([d1.tensors[1], d2.tensors[1]], dim=0)
        
#         dataset = TensorDataset(X_all, Y_all)
#         print(f"Successfully merged! Total dataset size: {len(dataset)}")
#     else:
#         print(f"\n[ERROR] Dataset parts not found! Please run CPU generation scripts first.")
#         sys.exit(1)
    
#     val_size = int(0.2 * len(dataset))
#     train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
#     model = AmortizedInferenceNet(param_dim=len(FREE_PARAM)).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
#     best_val_loss = float('inf')
#     best_model_weights = copy.deepcopy(model.state_dict())
#     epochs_no_improve = 0
    
#     print(f"\n--- Starting Deep Training Loop ---")
#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0.0
#         for batch_x, batch_y in train_loader:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#             optimizer.zero_grad()
#             loss = -model(batch_x, true_params_scaled=batch_y).mean()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#             optimizer.step()
#             train_loss += loss.item()
            
#         avg_train_loss, avg_val_loss = train_loss / len(train_loader), 0.0
        
#         model.eval()
#         with torch.no_grad():
#             for batch_x, batch_y in val_loader:
#                 avg_val_loss += -model(batch_x.to(device), true_params_scaled=batch_y.to(device)).mean().item()
#         avg_val_loss /= len(val_loader)
        
#         if (epoch+1) % 5 == 0 or epoch == 0:
#             print(f"Epoch [{epoch+1:03d}/{epochs}] | LR: {optimizer.param_groups[0]['lr']:.6f} | Train NLL: {avg_train_loss:.4f} | Val NLL: {avg_val_loss:.4f}")
        
#         scheduler.step(avg_val_loss)
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_model_weights = copy.deepcopy(model.state_dict())
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 print(f"\nEarly stopping at epoch {epoch+1}! Best Val Loss: {best_val_loss:.4f}")
#                 break

#     model.load_state_dict(best_model_weights)
#     torch.save(model.state_dict(), BASE_DIR / "amortized_inference_net_DEEP_8PARAMS.pth")
#     evaluate_parameter_recovery(model, device, num_test=64)

# if __name__ == "__main__":
#     train_pipeline(n_samples=20000, epochs=500, batch_size=256, patience=50)

# --- 6. MAIN TRAINING PIPELINE (MODIFIED FOR PLOTTING ONLY) ---
def train_pipeline(n_samples=20000, epochs=500, batch_size=256, patience=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Fast Track Plotting Mode ---")
    print(f"Using device: {device.type.upper()}")
    
    # 1. 确认能读到 CSV，防止再次报错
    if DF_MEANS is None:
        print("\n[ERROR] 依然没有读到 params_posteriors.csv！请确保它和本脚本在同一个文件夹！")
        sys.exit(1)
        
    # 2. 初始化网络结构
    print(f"Initializing Amortized Inference Network...")
    model = AmortizedInferenceNet(param_dim=len(FREE_PARAM)).to(device)
    
    # 3. 直接加载刚才训练好的神仙权重
    model_path = BASE_DIR / "amortized_inference_net_DEEP_8PARAMS.pth"
    if model_path.exists():
        print(f"Loading trained weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"[ERROR] 找不到模型权重文件 {model_path}！")
        sys.exit(1)
        
    # 4. 直接进入画图评估阶段！
    evaluate_parameter_recovery(model, device, num_test=64)

if __name__ == "__main__":
    # 直接调用画图，不需要给 epochs 等参数了
    train_pipeline()