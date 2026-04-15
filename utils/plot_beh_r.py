import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import warnings
from pathlib import Path
import sys

# --- 导入模型 ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from core.hdbm_v6 import HDBM

TRENDS_FILE = BASE_DIR / 'data' / 'subject_local_trends.csv'
PARAM_FILE = BASE_DIR / 'data' / 'est_param_additive_1.csv'
ORDERS_FILE = BASE_DIR / 'data' / 'orders.csv'

TOP_N = 500
MAX_GO_RUN = 20

def get_sequence_by_sig(order_sig, df_orders):
    seq_str = df_orders.loc[df_orders['order_sig'] == order_sig, 'order_seq'].values[0]
    if seq_str.startswith('['): return ast.literal_eval(seq_str)
    return [int(char) for char in str(seq_str) if char.isdigit()]

def extract_r_streaks(is_go_array, r_array):
    streaks = []
    current_streak = []
    seen_first_stop = False 
    for go_flag, r_val in zip(is_go_array, r_array):
        if not go_flag:
            seen_first_stop = True
            if len(current_streak) > 0:
                streaks.append(current_streak)
                current_streak = []
        else:
            if seen_first_stop: current_streak.append(r_val)
    if len(current_streak) > 0: streaks.append(current_streak)
    return streaks

def main():
    print(">>> 正在加载数据...")
    df_params = pd.read_csv(PARAM_FILE)
    df_trends = pd.read_csv(TRENDS_FILE)
    df_orders = pd.read_csv(ORDERS_FILE, dtype={'order_seq': str, 'order_sig': str})
    
    # 将模型参数和行为趋势表合并
    df_merged = pd.merge(df_params, df_trends, on='subject_id', how='inner')
    
    # 核心：基于真实行为的 slope 和 curve 挑选 4 个亚组！
    # 核心修复：严谨的行为表型分离逻辑！
    
    # 定义 "接近 0" 的严格标准：取绝对值最小的前 30% 作为 "Flat" (平坦) 的准入条件
    curve_flat_threshold = df_merged['quadratic_curve'].abs().quantile(0.30)
    slope_flat_threshold = df_merged['linear_slope'].abs().quantile(0.30)
    
    # 1. 纯粹的 Speeding (加速): curve 必须平坦，slope 取最小(最负)
    df_speeding = df_merged[df_merged['quadratic_curve'].abs() < curve_flat_threshold] \
                    .sort_values('linear_slope', ascending=True).head(TOP_N)
                    
    # 2. 纯粹的 Slowing (减速): curve 必须平坦，slope 取最大(最正)
    df_slowing = df_merged[df_merged['quadratic_curve'].abs() < curve_flat_threshold] \
                    .sort_values('linear_slope', ascending=False).head(TOP_N)
                    
    # 排除已经被归入线性的被试，防止重合
    used_ids = set(df_speeding['subject_id']).union(set(df_slowing['subject_id']))
    df_remaining = df_merged[~df_merged['subject_id'].isin(used_ids)]
    
    # 3. 纯粹的 U-Shape: slope 必须平坦，curve 取最大(正U)
    df_u_shape = df_remaining[df_remaining['linear_slope'].abs() < slope_flat_threshold] \
                    .sort_values('quadratic_curve', ascending=False).head(TOP_N)
                    
    # 4. 纯粹的 Inv U-Shape: slope 必须平坦，curve 取最小(倒U)
    df_inv_u_shape = df_remaining[df_remaining['linear_slope'].abs() < slope_flat_threshold] \
                    .sort_values('quadratic_curve', ascending=True).head(TOP_N)

    archetypes = {
        "1. Pure U-Shape (Curve>0, Slope≈0)": df_u_shape,
        "2. Pure Inv U-Shape (Curve<0, Slope≈0)": df_inv_u_shape,
        "3. Pure Slowing (Slope>0, Curve≈0)": df_slowing,
        "4. Pure Speeding (Slope<0, Curve≈0)": df_speeding,
    }
    
    
    
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (group_name, df_sub) in enumerate(archetypes.items()):
        print(f"Calculating Expected 'r' for {group_name}...")
        all_streaks = []
        
        for _, row in df_sub.iterrows():
            seq_int = get_sequence_by_sig(row['order_sig'], df_orders)
            is_go_mask = [s == 0 for s in seq_int]
            
            # 初始化该被试的 HDBM
            hdbm = HDBM(
                alpha_go=0.85, alpha_stop=0.85, 
                eta=row['eta'], rho=row['rho'], k=row['k'], 
                fusion_type='additive_1'
            )
            # 零延迟算出 r 轨迹
            r_traj = hdbm.simu_task(seq_int, block_size=180)
            
            # 提取局部趋势
            streaks = extract_r_streaks(is_go_mask[:180], r_traj[:180]) + extract_r_streaks(is_go_mask[180:], r_traj[180:])
            all_streaks.extend(streaks)
            
        # ================= 数据聚合与画图 =================
        mat = np.full((len(all_streaks), MAX_GO_RUN), np.nan)
        for j, s in enumerate(all_streaks):
            length = min(len(s), MAX_GO_RUN)
            mat[j, :length] = s[:length]
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            r_means = np.nanmean(mat, axis=0)
            # 计算标准误 (SEM)
            counts = np.sum(~np.isnan(mat), axis=0)
            r_sems = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(counts, 1))
            
        x_vals = np.arange(1, MAX_GO_RUN + 1)
        ax = axes[i]
        
        # 画出均值和误差棒
        ax.errorbar(x_vals, r_means, yerr=r_sems, fmt='bo-', lw=2.5, capsize=4, label='Predicted Prob (r)')
        
        ax.set_title(f"{group_name} (N={TOP_N})")
        ax.set_xlabel("Consecutive Go Trials")
        ax.set_ylabel("Predicted Probability of Stop (r)")
        
        # 固定 Y 轴，方便跨组对比
        ax.set_ylim(0, 0.5) 
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()

    plt.tight_layout()
    out_path = BASE_DIR / 'behavioral_group_predicted_r.png'
    plt.savefig(out_path, dpi=300)
    print(f"\n>>> 成功！行为亚组的预测概率 r 已保存至: {out_path}")

if __name__ == "__main__":
    main()