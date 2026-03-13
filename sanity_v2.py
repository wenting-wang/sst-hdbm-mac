import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 确保路径正确
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

np.random.seed(42)

from core.hdbm import HDBM
from core.pomdp import POMDP
from core import simulation

POMDP_PARAMS = {
    "q_d_n": 0.5, "q_d": 0.8,
    "q_s_n": 0.5, "q_s": 0.8,
    "cost_stop_error": 1.0, 
    "inv_temp": 15.0,
    "cost_time": 0.001, 
    "cost_go_error": 1.0,
    "cost_go_missing": 1.0}

OUTCOME_MAP = {0: 'GS', 1: 'GE', 2: 'GM', 3: 'SS', 4: 'SE'}

def sample_task_sequence(sequences: np.ndarray, probs: np.ndarray, n_blocks: int = 2):
    if sequences is None:
        # seed
        return [1 if np.random.rand() < 1/6 else 0 for _ in range(180 * n_blocks)]
    sampled_idx = np.random.choice(len(sequences), size=n_blocks, p=probs)
    full_seq = []
    for idx in sampled_idx:
        full_seq.extend(sequences[idx])
    return full_seq

# --- 核心：使用真实的 POMDP 替代 Surrogate ---
def simulate_trial_exact(r_pred, pomdp_params, ssd, true_go_state, true_stop_state):
    """
    这是一个 Wrapper 函数。你需要在这里调用你写好的 Exact POMDP 求解器。
    """
    
    pomdp = POMDP(rate_stop_trial=r_pred, **pomdp_params)
    pomdp.value_iteration_tensor()
    res, rt = simulation.simu_trial(
                pomdp, true_go_state=true_go_state, 
                true_stop_state=true_stop_state, 
                ssd=ssd, verbose=False)
    
    # 返回格式需要和 surrogate 保持一致：
    # choice_idx: 0(GS), 1(GE), 2(GM), 3(SS), 4(SE)
    # rt: 反应时间 (ms)
    choice_idx = {'GS': 0, 'GE': 1, 'GM': 2, 'SS': 3, 'SE': 4}[res]
    
    return choice_idx, rt


def sanity_check_exact_pomdp():
    print("Running Exact POMDP sanity check for 3 alpha values...")
    
    # === 控制模拟长度和随机种子 ===
    # HDBM 很快，我们生成 360 个试次的序列看完整的 r 变化
    # 但 Exact POMDP 很慢，这里设置只计算前多少个 trial 的 RT
    MAX_POMDP_TRIALS = 100  # 你可以根据能忍受的等待时间调整这个值，比如改成 20 或 50
    
    # 1. 生成固定的序列 (使用 seed 控制)
    seq_int = sample_task_sequence(None, None, n_blocks=2) 
    total_trials = len(seq_int)
    
    # 统一生成 Go 信号方向 (使用相同的 seed)
    go_directions = np.random.choice([0, 1], size=total_trials)
    
    # 2. 设定三组参数对比 (固定 rho=0, 改变 alpha)
    params_1 = {**POMDP_PARAMS, "alpha": 0.1, "rho": 0}
    params_2 = {**POMDP_PARAMS, "alpha": 0.5, "rho": 0}
    params_3 = {**POMDP_PARAMS, "alpha": 0.9, "rho": 0}
    
    # 3. 模拟逻辑
    def run_sim(params):
        # HDBM 跑完整的 360 trials
        hdbm = HDBM(alpha=params['alpha'], rho=params['rho'])
        r_preds = hdbm.simu_task(seq_int, block_size=180)
        
        next_stop_ssd = 2
        rts = []
        outcome_counts = {k: 0 for k in OUTCOME_MAP.keys()}
        
        simulated_trials = min(total_trials, MAX_POMDP_TRIALS)
        
        # 只让 POMDP 跑前 MAX_POMDP_TRIALS 个
        for t in range(simulated_trials):
            is_stop = seq_int[t]
            go_dir = 'right' if go_directions[t] == 1 else 'left'
            r_pred = float(r_preds[t])
            
            true_stop_state = 'stop' if is_stop == 1 else 'nonstop'
            ssd = next_stop_ssd if is_stop == 1 else -1
            
            # ！！调用真实的 POMDP ！！
            choice_idx, rt = simulate_trial_exact(
                r_pred, POMDP_PARAMS, ssd, go_dir, true_stop_state
            )
            
            outcome_counts[choice_idx] += 1
            
            if choice_idx in [2, 3]:  # GM 或 SS 时没有 RT
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
                
        # 注意这里：r_preds 返回完整长度，rts 只有 MAX_POMDP_TRIALS 长
        return np.array(r_preds), np.array(rts), outcome_counts, simulated_trials

    def print_outcomes(counts, simulated_total, label):
        print(f"\n--- Outcomes for {label} (First {simulated_total} trials) ---")
        for idx, name in OUTCOME_MAP.items():
            pct = (counts[idx] / simulated_total) * 100
            print(f"{name}: {pct:>5.1f}%  ({counts[idx]}/{simulated_total})")
        print("-" * 45)


    # 4. 执行模拟
    label_1 = rf"alpha={params_1['alpha']}, rho={params_1['rho']}"
    print(f"\nSimulating param set 1: {label_1}...")
    r_preds_1, rts_1, counts_1, sim_tot_1 = run_sim(params_1)
    print_outcomes(counts_1, sim_tot_1, label_1)
    
    label_2 = rf"alpha={params_2['alpha']}, rho={params_2['rho']}"
    print(f"Simulating param set 2: {label_2}...")
    r_preds_2, rts_2, counts_2, sim_tot_2 = run_sim(params_2)
    print_outcomes(counts_2, sim_tot_2, label_2)

    label_3 = rf"alpha={params_3['alpha']}, rho={params_3['rho']}"
    print(f"Simulating param set 3: {label_3}...")
    r_preds_3, rts_3, counts_3, sim_tot_3 = run_sim(params_3)
    print_outcomes(counts_3, sim_tot_3, label_3)
    
    
    # 5. 可视化
    label_1 = rf"$\alpha={params_1['alpha']}, \rho={params_1['rho']}$"
    label_2 = rf"$\alpha={params_2['alpha']}, \rho={params_2['rho']}$"
    label_3 = rf"$\alpha={params_3['alpha']}, \rho={params_3['rho']}$"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False) # 取消 sharex 因为上下长度不同
    
    # --- Subplot 1: r_pred (完整 360 trials) ---
    ax1.plot(r_preds_1, label=label_1, color='blue', alpha=0.8, linewidth=1.5)
    ax1.plot(r_preds_2, label=label_2, color='green', alpha=0.8, linewidth=1.5)
    ax1.plot(r_preds_3, label=label_3, color='red', alpha=0.8, linewidth=1.5)
    
    stop_trials = np.where(np.array(seq_int) == 1)[0]
    ax1.scatter(stop_trials, [-0.02] * len(stop_trials), color='black', marker='|', s=50, label='Stop Trials', zorder=3)
    
    ax1.set_xlim(-5, total_trials + 5)
    ax1.set_ylabel('r_pred (Expected Stop Prob)')
    ax1.set_title(f'Changes in r_pred over {total_trials} trials (HDBM)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # --- Subplot 2: RT (仅前 MAX_POMDP_TRIALS) ---
    x_rt = np.arange(MAX_POMDP_TRIALS)
    ax2.plot(x_rt, rts_1, label=label_1, color='blue', alpha=0.6, marker='o', markersize=5, linestyle='None')
    ax2.plot(x_rt, rts_2, label=label_2, color='green', alpha=0.6, marker='s', markersize=5, linestyle='None')
    ax2.plot(x_rt, rts_3, label=label_3, color='red', alpha=0.6, marker='x', markersize=5, linestyle='None')
    
    # 因为点数少了，平滑窗口调小到 3
    s_1 = pd.Series(rts_1).rolling(window=3, min_periods=1).mean()
    s_2 = pd.Series(rts_2).rolling(window=3, min_periods=1).mean()
    s_3 = pd.Series(rts_3).rolling(window=3, min_periods=1).mean()
    ax2.plot(x_rt, s_1, color='blue', linewidth=2, alpha=0.4)
    ax2.plot(x_rt, s_2, color='green', linewidth=2, alpha=0.4)
    ax2.plot(x_rt, s_3, color='red', linewidth=2, alpha=0.4)
    
    ax2.set_xlim(-5, total_trials + 5)
    ax2.set_xlabel('Trial Index')
    ax2.set_ylabel('Reaction Time (RT) - Exact POMDP')
    ax2.set_title(f'Changes in RT over first {MAX_POMDP_TRIALS} trials (Points: Raw, Lines: 3-trial moving average)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = BASE_DIR / 'sanity_check_pomdp.png'
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to '{save_path}'.")
    plt.show()
    
     # plot histogram of rts for each condition
    plt.figure(figsize=(12, 6))
    plt.hist(rts_1[~np.isnan(rts_1)], bins=20, alpha=0.5, label=label_1, color='blue')
    plt.hist(rts_2[~np.isnan(rts_2)], bins=20, alpha=0.5, label=label_2, color='green')
    plt.hist(rts_3[~np.isnan(rts_3)], bins=20, alpha=0.5, label=label_3, color='red')
    plt.xlabel('Reaction Time (RT)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Reaction Times (Exact POMDP)')
    plt.legend()
    hist_save_path = BASE_DIR / 'sanity_check_pomdp_hist.png'
    plt.savefig(hist_save_path, dpi=300)
    print(f"Histogram saved to '{hist_save_path}'.")
    plt.show()

if __name__ == "__main__":
    sanity_check_exact_pomdp()
    

# --- Outcomes for alpha=0.1, rho=0 (First 100 trials) ---
# GS:  76.0%  (76/100)
# GE:   3.0%  (3/100)
# GM:   0.0%  (0/100)
# SS:   1.0%  (1/100)
# SE:  20.0%  (20/100)
# ---------------------------------------------
# Simulating param set 2: alpha=0.5, rho=0...

# --- Outcomes for alpha=0.5, rho=0 (First 100 trials) ---
# GS:  75.0%  (75/100)
# GE:   3.0%  (3/100)
# GM:   1.0%  (1/100)
# SS:   0.0%  (0/100)
# SE:  21.0%  (21/100)
# ---------------------------------------------
# Simulating param set 3: alpha=0.9, rho=0...

# --- Outcomes for alpha=0.9, rho=0 (First 100 trials) ---
# GS:  75.0%  (75/100)
# GE:   4.0%  (4/100)
# GM:   0.0%  (0/100)
# SS:   2.0%  (2/100)
# SE:  19.0%  (19/100)


