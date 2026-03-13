import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 确保路径正确
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

np.random.seed(42)

from train_surrogate import load_surrogate
from core.hdbm import HDBM

# 1. 设定固定的 POMDP 参数 (取一个典型的中间值)
POMDP_PARAMS = {
    "q_d_n": 0.5, "q_d": 0.8,
    "q_s_n": 0.5, "q_s": 0.8,
    "cost_stop_error": 1.0, 
    "inv_temp": 10.0,
    "cost_time": 0.001, 
    "cost_go_error": 1.0,
    "cost_go_missing": 1.0}

OUTCOME_MAP = {0: 'GS', 1: 'GE', 2: 'GM', 3: 'SS', 4: 'SE'}

def sample_task_sequence(sequences: np.ndarray, probs: np.ndarray, n_blocks: int = 2):
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

def simulate_single_dataset(args):
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
            
    return np.array(features_seq, dtype=np.float32), seq_int



# --- 4. SANITY CHECK: Effect of Alpha/Rho on r and RT (Surrogate) ---
def sanity_check():
    print("Running sanity check with Surrogate for 3 alpha values...")
    
    # 1. 准备 Surrogate 模型
    surrogate_path = BASE_DIR / "pomdp_surrogate.pth"
    try:
        surr_model, X_min, X_max = load_surrogate(str(surrogate_path))
        surr_model.eval()
    except Exception as e:
        print(f"Error loading surrogate model: {e}")
        return

    # 2. 生成固定的 360 trials 序列 (增加 Seed 控制)
    my_seed = 42
    # 确保你的 sample_task_sequence 已经按之前的建议加上了 seed 参数
    seq_int = sample_task_sequence(None, None, n_blocks=2) 
    total_trials = len(seq_int)
    
    # 统一生成 Go 信号方向 (使用相同的 seed)
    go_directions = np.random.choice([0, 1], size=total_trials)
    
    # === 3. 设定三组参数对比 (固定 rho=0, 改变 alpha) ===
    # 注：如果你之前用的是 POMDP_PARAMS，请自行替换 FIXED_POMDP_PARAMS
    params_1 = {**POMDP_PARAMS, "alpha": 0.1, "rho": 0}
    params_2 = {**POMDP_PARAMS, "alpha": 0.5, "rho": 0}
    params_3 = {**POMDP_PARAMS, "alpha": 0.9, "rho": 0}
    
    # 4. 内部模拟函数
    def run_sim(params):
        hdbm = HDBM(alpha=params['alpha'], rho=params['rho'])
        r_preds = hdbm.simu_task(seq_int, block_size=180)
        
        next_stop_ssd = 2
        rts = []
        outcome_counts = {k: 0 for k in OUTCOME_MAP.keys()}
        
        for t in range(total_trials):
            is_stop = seq_int[t]
            go_dir = 'right' if go_directions[t] == 1 else 'left'
            r_pred = float(r_preds[t])
            
            true_stop_state = 'stop' if is_stop == 1 else 'nonstop'
            ssd = next_stop_ssd if is_stop == 1 else -1
            
            choice_idx, rt = simulate_trial_surrogate(
                r_pred, POMDP_PARAMS, ssd, go_dir, true_stop_state, surr_model, X_min, X_max
            )
            
            outcome_counts[choice_idx] += 1
            
            if choice_idx in [2, 3]: 
                actual_rt = np.nan
            else:
                actual_rt = rt
                
            rts.append(actual_rt)
            
            if is_stop == 1:
                res_str = OUTCOME_MAP[choice_idx]
                if res_str == 'SS': 
                    next_stop_ssd = min(next_stop_ssd + 2, 34)
                elif res_str == 'SE': 
                    next_stop_ssd = max(next_stop_ssd - 2, 2)
                
        return np.array(r_preds), np.array(rts), outcome_counts
    
    def print_outcomes(counts, label):
        print(f"\n--- Outcomes for {label} ---")
        for idx, name in OUTCOME_MAP.items():
            pct = (counts[idx] / total_trials) * 100
            print(f"{name}: {pct:>5.1f}%  ({counts[idx]}/{total_trials})")
        print("-" * 30)


    # 5. 执行模拟
    label_1 = rf"alpha={params_1['alpha']}, rho={params_1['rho']}"
    print(f"\nSimulating with param set 1: {label_1}...")
    r_preds_1, rts_1, counts_1 = run_sim(params_1)
    print_outcomes(counts_1, label_1)
    
    label_2 = rf"alpha={params_2['alpha']}, rho={params_2['rho']}"
    print(f"Simulating with param set 2: {label_2}...")
    r_preds_2, rts_2, counts_2 = run_sim(params_2)
    print_outcomes(counts_2, label_2)

    label_3 = rf"alpha={params_3['alpha']}, rho={params_3['rho']}"
    print(f"Simulating with param set 3: {label_3}...")
    r_preds_3, rts_3, counts_3 = run_sim(params_3)
    print_outcomes(counts_3, label_3)
    
    # === 6. 可视化 (动态图例) ===
    label_1 = rf"$\alpha={params_1['alpha']}, \rho={params_1['rho']}$"
    label_2 = rf"$\alpha={params_2['alpha']}, \rho={params_2['rho']}$"
    label_3 = rf"$\alpha={params_3['alpha']}, \rho={params_3['rho']}$"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # --- Subplot 1: r_pred 变化 ---
    ax1.plot(r_preds_1, label=label_1, color='blue', alpha=0.8, linewidth=1.5)
    ax1.plot(r_preds_2, label=label_2, color='green', alpha=0.8, linewidth=1.5)
    ax1.plot(r_preds_3, label=label_3, color='red', alpha=0.8, linewidth=1.5)
    
    # 标记实际的 Stop trials
    stop_trials = np.where(np.array(seq_int) == 1)[0]
    ax1.scatter(stop_trials, [-0.02] * len(stop_trials), color='black', marker='|', s=50, label='Stop Trials', zorder=3)
    
    ax1.set_ylabel('r_pred (Expected Stop Prob)')
    ax1.set_title('Changes in r_pred over 360 trials')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # --- Subplot 2: RT 变化 ---
    ax2.plot(rts_1, label=label_1, color='blue', alpha=0.6, marker='o', markersize=4, linestyle='None')
    ax2.plot(rts_2, label=label_2, color='green', alpha=0.6, marker='s', markersize=4, linestyle='None')
    ax2.plot(rts_3, label=label_3, color='red', alpha=0.6, marker='x', markersize=4, linestyle='None')
    
    # 添加简单的平滑线
    s_1 = pd.Series(rts_1).rolling(window=10, min_periods=1).mean()
    s_2 = pd.Series(rts_2).rolling(window=10, min_periods=1).mean()
    s_3 = pd.Series(rts_3).rolling(window=10, min_periods=1).mean()
    ax2.plot(s_1, color='blue', linewidth=2, alpha=0.4)
    ax2.plot(s_2, color='green', linewidth=2, alpha=0.4)
    ax2.plot(s_3, color='red', linewidth=2, alpha=0.4)
    
    ax2.set_xlabel('Trial Index')
    ax2.set_ylabel('Reaction Time (RT)')
    ax2.set_title('Changes in RT over 360 trials (Points: Raw, Lines: 10-trial moving average)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = BASE_DIR / 'sanity_check_surrogate.png'
    plt.savefig(save_path, dpi=300)
    print(f"Sanity check plot saved to '{save_path}'.")
    plt.show()
    
    # plot histogram of rts for each condition
    plt.figure(figsize=(12, 6))
    plt.hist(rts_1[~np.isnan(rts_1)], bins=30, alpha=0.5, label=label_1, color='blue')
    plt.hist(rts_2[~np.isnan(rts_2)], bins=30, alpha=0.5, label=label_2, color='green')
    plt.hist(rts_3[~np.isnan(rts_3)], bins=30, alpha=0.5, label=label_3, color='red')
    plt.xlabel('Reaction Time (RT)')
    plt.ylabel('Frequency')
    plt.title('Histogram of RTs for Different Alpha Values') 
    plt.legend()
    save_path_hist = BASE_DIR / 'sanity_check_surrogate_hist.png'
    plt.savefig(save_path_hist, dpi=300)
    print(f"Sanity check histogram saved to '{save_path_hist}'.")
    plt.show()

if __name__ == "__main__":
    sanity_check()

    

# --- Outcomes for alpha=0.1, rho=0 ---
# GS:  74.2%  (267/360)
# GE:   7.5%  (27/360)
# GM:   0.0%  (0/360)
# SS:   5.0%  (18/360)
# SE:  13.3%  (48/360)
# ------------------------------
# Simulating with param set 2: alpha=0.5, rho=0...

# --- Outcomes for alpha=0.5, rho=0 ---
# GS:  77.2%  (278/360)
# GE:   4.2%  (15/360)
# GM:   0.6%  (2/360)
# SS:   4.2%  (15/360)
# SE:  13.9%  (50/360)
# ------------------------------
# Simulating with param set 3: alpha=0.9, rho=0...

# --- Outcomes for alpha=0.9, rho=0 ---
# GS:  73.3%  (264/360)
# GE:   8.1%  (29/360)
# GM:   0.3%  (1/360)
# SS:   5.6%  (20/360)
# SE:  12.8%  (46/360)
# ------------------------------