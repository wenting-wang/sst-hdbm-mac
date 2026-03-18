import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# 复用之前的路径和参数配置
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
sys.path.append(str(BASE_DIR))

from core.hdbm_v2 import HDBM

# --- 参数还原配置 ---
HDBM_PARAM_RANGE = {
    "alpha_go": (0.0, 1.0), "alpha_stop": (0.0, 1.0), 
    "k_go": (0.0, 10.0), "rho": (0.0, 1.0)
}
POMDP_PARAM_RANGE = {
    "q_d_n": (0.0, 1.0), "q_d": (0.5, 1.0),
    "q_s_n": (0.0, 1.0), "q_s": (0.5, 1.0),
    "cost_stop_error": (0.3, 2.0), "inv_temp": (10, 100)
}
FREE_PARAM = list(HDBM_PARAM_RANGE.keys()) + list(POMDP_PARAM_RANGE.keys())
PARAM_RANGE = {**HDBM_PARAM_RANGE, **POMDP_PARAM_RANGE}

_min, _max = [], []
for k in FREE_PARAM:
    low, high = PARAM_RANGE[k]
    if k in ['cost_stop_error', 'inv_temp']:
        _min.append(np.log(low))
        _max.append(np.log(high))
    else:
        _min.append(low)
        _max.append(high)
P_MIN = np.array(_min, dtype=np.float32)
P_MAX = np.array(_max, dtype=np.float32)

def inverse_transform(scaled_array):
    """把 [0, 1] 的缩放参数还原为真实物理值"""
    arr = scaled_array * (P_MAX - P_MIN) + P_MIN
    for i, k in enumerate(FREE_PARAM):
        if k in ['cost_stop_error', 'inv_temp']:
            arr[:, i] = np.exp(arr[:, i])
    return arr

# --- 1. 行为学指标提取 ---
def extract_behaviors_from_tensor(x_seq):
    """
    从 trial 特征张量中提取宏观行为指标。
    x_seq 维度: (n_trials, 5) -> [is_stop, ssd_scaled, go_dir, choice_scaled, rt_scaled]
    """
    is_stop = x_seq[:, 0].numpy()
    ssd = x_seq[:, 1].numpy() * 40.0
    choice = np.round(x_seq[:, 3].numpy() * 4.0) # 0:GS, 1:GE, 2:GM, 3:SS, 4:SE
    rt = x_seq[:, 4].numpy() * 40.0
    
    go_mask = (is_stop == 0)
    stop_mask = (is_stop == 1)
    
    # Go 试次指标
    go_responses = rt[go_mask & ((choice == 0) | (choice == 1))]
    mean_go_rt = np.mean(go_responses) if len(go_responses) > 0 else np.nan
    p_go_miss = np.mean(choice[go_mask] == 2)
    
    # Stop 试次指标
    p_stop_error = np.mean(choice[stop_mask] == 4)
    mean_ssd = np.mean(ssd[stop_mask])
    
    # 失败的 Stop (Signal Respond RT)
    sr_responses = rt[stop_mask & (choice == 4)]
    mean_sr_rt = np.mean(sr_responses) if len(sr_responses) > 0 else np.nan
    
    return [mean_go_rt, p_go_miss, p_stop_error, mean_ssd, mean_sr_rt]

# --- 2. 核心分析流程 ---
def analyze_dataset_sensitivity(dataset_path):
    print(f"Loading dataset from {dataset_path}...")
    if not Path(dataset_path).exists():
        print(f"Error: Dataset {dataset_path} not found.")
        return
        
    dataset = torch.load(dataset_path, weights_only=False)
    X_tensor, Y_tensor = dataset.tensors
    
    # 只取前 5000 个样本做敏感性分析就足够了，避免计算过慢
    n_samples = min(5000, len(X_tensor))
    X_subset = X_tensor[:n_samples]
    Y_subset = Y_tensor[:n_samples]
    
    print("Extracting behavioral metrics...")
    behaviors = []
    for i in range(n_samples):
        behaviors.append(extract_behaviors_from_tensor(X_subset[i]))
        
    # 组装为 DataFrame
    df_behaviors = pd.DataFrame(behaviors, columns=['Mean_Go_RT', 'P_Go_Miss', 'P_Stop_Error', 'Mean_SSD', 'Mean_SR_RT'])
    
    # 还原真实参数
    real_params = inverse_transform(Y_subset.numpy())
    df_params = pd.DataFrame(real_params, columns=FREE_PARAM)
    
    # 合并并计算 Spearman 秩相关系数 (捕捉非线性单调关系)
    df_all = pd.concat([df_params, df_behaviors], axis=1)
    corr_matrix = df_all.corr(method='spearman').loc[FREE_PARAM, df_behaviors.columns]
    
    # --- 绘图 1：参数-行为敏感性热力图 ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("Parameter Sensitivity: Spearman Correlation with Behaviors")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "sensitivity_heatmap.png", dpi=300)
    plt.close()
    print("Saved 'sensitivity_heatmap.png'.")

def plot_r_trajectory_variance():
    """测试不同 HDBM 参数是否能拉开主观预测概率 r 的差距"""
    print("Simulating r-trajectories to check HDBM variance...")
    
    # 构造一个极端的测试序列：20次Go，然后密集交替
    test_seq = [0]*20 + [1, 0, 0, 1, 0, 1, 0, 0, 0, 1] 
    
    plt.figure(figsize=(12, 5))
    
    # 采样 50 组不同的参数跑 r
    for _ in range(50):
        alpha_go = np.random.uniform(*HDBM_PARAM_RANGE['alpha_go'])
        alpha_stop = np.random.uniform(*HDBM_PARAM_RANGE['alpha_stop'])
        k_go = np.random.uniform(*HDBM_PARAM_RANGE['k_go'])
        rho = np.random.uniform(*HDBM_PARAM_RANGE['rho'])
        
        hdbm = HDBM(alpha_go=alpha_go, alpha_stop=alpha_stop, k_go=k_go, rho=rho, fusion_type='additive')
        r_traj = hdbm.simu_task(test_seq)
        
        plt.plot(r_traj, color='blue', alpha=0.1) # 叠画
        
    plt.title("Variance of Subjective Stop Probability (r) across 50 random parameter sets")
    plt.xlabel("Trial Number")
    plt.ylabel("Predicted P(Stop) -> r")
    plt.ylim(0, 1)
    
    # 标记 Stop 试次的位置
    stop_idx = [i for i, x in enumerate(test_seq) if x == 1]
    for idx in stop_idx:
        plt.axvline(x=idx, color='red', linestyle='--', alpha=0.5, ymax=0.1)
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / "r_trajectory_variance.png", dpi=300)
    plt.close()
    print("Saved 'r_trajectory_variance.png'.")

if __name__ == "__main__":
    # 请确保这里的名字与你实际生成的 .pt 数据集名字一致
    DATASET_NAME = "/Users/w/Desktop/e2e_dataset_50000_additive.pt" 
    
    plot_r_trajectory_variance()
    analyze_dataset_sensitivity(DATASET_NAME)
    print("\nAnalysis complete! Please check the two generated PNG images.")