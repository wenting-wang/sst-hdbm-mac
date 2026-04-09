import sys
import argparse
import tempfile
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast
import warnings
from joblib import Parallel, delayed
import torch

# 限制 PyTorch 内部线程，防止多核计算时冲突死锁
torch.set_num_threads(1)
# 屏蔽各种无害的 NaN 均值警告
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Mean of empty slice")

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from core.hdbm_v4 import HDBM
from core.pomdp import POMDP
from core import simulation

TRENDS_FILE = BASE_DIR / 'subject_local_trends.csv'
POMDP_PRIOR_FILE = BASE_DIR / 'pomdp_posterior.csv'
ORDERS_FILE = BASE_DIR / 'orders.csv'
DATA_ROOT = Path("/u/wenwang/data/sst_valid_base")

FIXED_PARAMS = {"cost_time": 0.001, "cost_go_error": 1.0, "cost_go_missing": 1.0}
N_SIMULATIONS = 50  

def extract_real_rts(subject_id):
    zip_filename = f"{subject_id}_baseline_year_1_arm_1_sst.csv.zip"
    zip_path = DATA_ROOT / zip_filename
    
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing ZIP file: {zip_path}")
        
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
            
        csv_filename = f"{subject_id}_baseline_year_1_arm_1_sst.csv"
        csv_path = Path(tmpdir) / "SST" / "baseline_year_1_arm_1" / csv_filename
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV inside ZIP: {csv_path}")
            
        df = pd.read_csv(csv_path, usecols=['sst_expcon', 'sst_go_rt'])
        is_go = df['sst_expcon'].astype(str).str.strip() == 'GoTrial'
        rts = pd.to_numeric(df['sst_go_rt'], errors='coerce').to_numpy()
        
        rts = np.where(rts <= 200, np.nan, rts)
        
    return is_go.to_numpy(), rts

def get_sequence_by_sig(order_sig):
    df = pd.read_csv(ORDERS_FILE, dtype={'order_seq': str, 'order_sig': str})
    seq_str = df.loc[df['order_sig'] == order_sig, 'order_seq'].values[0]
    if seq_str.startswith('['):
        return ast.literal_eval(seq_str)
    return [int(char) for char in str(seq_str) if char.isdigit()]

def get_subject_params(subject_id, hdbm_param_file):
    short_sub_id = subject_id.replace('NDAR_', '')
    
    df_pomdp = pd.read_csv(POMDP_PRIOR_FILE)
    df_sub = df_pomdp[df_pomdp['subject_id'].str.contains(short_sub_id, na=False)]
    if df_sub.empty:
        raise ValueError(f"Subject {short_sub_id} not found in POMDP priors.")
    pomdp_params = {row['index']: row['mean'] for _, row in df_sub.iterrows()}
    
    df_hdbm = pd.read_csv(hdbm_param_file)
    df_hdbm_sub = df_hdbm[df_hdbm['subject_id'].str.contains(short_sub_id, na=False)]
    if df_hdbm_sub.empty:
        raise ValueError(f"Subject {short_sub_id} not found in HDBM parameters.")
    hdbm_params = df_hdbm_sub.iloc[0].to_dict()
    
    return pomdp_params, hdbm_params

def simulate_behavior(seq_int, pomdp_p, hdbm_p=None, fusion_mode='additive_2'):
    total_trials = len(seq_int)
    go_directions = np.random.choice([0, 1], size=total_trials)
    sim_rts = np.full(total_trials, np.nan)
    next_stop_ssd = 2
    
    if hdbm_p is None:
        r_static = pomdp_p.get('rate_stop_trial', 0.1666)
        pomdp = POMDP(
            rate_stop_trial=r_static,  q_d_n=pomdp_p['q_d_n'], q_d=pomdp_p['q_d'],
            q_s_n=pomdp_p['q_s_n'], q_s=pomdp_p['q_s'],
            cost_stop_error=pomdp_p['cost_stop_error'], inv_temp=pomdp_p['inv_temp'], **FIXED_PARAMS
        )
        pomdp.value_iteration_tensor()
        
        for t in range(total_trials):
            is_stop = seq_int[t]
            go_dir = 'right' if go_directions[t] == 1 else 'left'
            true_stop_state = 'stop' if is_stop == 1 else 'nonstop'
            ssd = next_stop_ssd if is_stop == 1 else -1
            
            res_str, rt = simulation.simu_trial(pomdp, true_go_state=go_dir, true_stop_state=true_stop_state, ssd=ssd, verbose=False)
            if is_stop == 0 and res_str in ['GS', 'GE']: sim_rts[t] = rt * 25.0 
            if is_stop == 1:
                if res_str == 'SS': next_stop_ssd = min(next_stop_ssd + 2, 34)
                elif res_str == 'SE': next_stop_ssd = max(next_stop_ssd - 2, 2)
    else:
        hdbm_kwargs = {'alpha_go': 0.85, 'alpha_stop': 0.85, 'eta': hdbm_p['eta'], 'fusion_type': fusion_mode}
        if fusion_mode in ['additive_1', 'additive_2']: hdbm_kwargs['rho'] = hdbm_p['rho']
        elif fusion_mode == 'multiplicative': hdbm_kwargs['gamma'] = hdbm_p['gamma']
            
        hdbm = HDBM(**hdbm_kwargs)
        r_preds = hdbm.simu_task(seq_int, block_size=180)

        for t in range(total_trials):
            is_stop = seq_int[t]
            go_dir = 'right' if go_directions[t] == 1 else 'left'
            r_pred = float(r_preds[t])
            true_stop_state = 'stop' if is_stop == 1 else 'nonstop'
            ssd = next_stop_ssd if is_stop == 1 else -1
            
            pomdp = POMDP(
                rate_stop_trial=r_pred, q_d_n=hdbm_p['q_d_n'], q_d=hdbm_p['q_d'],
                q_s_n=hdbm_p['q_s_n'], q_s=hdbm_p['q_s'],
                cost_stop_error=hdbm_p['cost_stop_error'], inv_temp=hdbm_p['inv_temp'], **FIXED_PARAMS
            )
            pomdp.value_iteration_tensor()
            
            res_str, rt = simulation.simu_trial(pomdp, true_go_state=go_dir, true_stop_state=true_stop_state, ssd=ssd, verbose=False)
            if is_stop == 0 and res_str in ['GS', 'GE']: sim_rts[t] = rt * 25.0 
            if is_stop == 1:
                if res_str == 'SS': next_stop_ssd = min(next_stop_ssd + 2, 34)
                elif res_str == 'SE': next_stop_ssd = max(next_stop_ssd - 2, 2)
                
    return sim_rts

def extract_mean_local_trend(is_go_array, rts_array):
    def extract_streaks(is_go_array, rt_array):
        streaks = []
        current_streak = []
        seen_first_stop = False 
        for go_flag, rt in zip(is_go_array, rt_array):
            if not go_flag:
                seen_first_stop = True
                if len(current_streak) > 0:
                    streaks.append(current_streak)
                    current_streak = []
            else:
                if seen_first_stop: current_streak.append(rt)
        if len(current_streak) > 0: streaks.append(current_streak)
        return streaks
    
    streaks = extract_streaks(is_go_array[:180], rts_array[:180]) + extract_streaks(is_go_array[180:], rts_array[180:])
    if not streaks: return []
    
    max_len = max(len(s) for s in streaks)
    mat = np.full((len(streaks), max_len), np.nan)
    for i, s in enumerate(streaks): mat[i, :len(s)] = s
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(mat, axis=0)

def run_single_simulation_pair(seq_int, pomdp_p, hdbm_p, is_go_mask, fusion_mode):
    rts_pomdp = simulate_behavior(seq_int, pomdp_p, hdbm_p=None)
    local_pomdp = extract_mean_local_trend(is_go_mask, rts_pomdp)
    
    rts_hdbm = simulate_behavior(seq_int, pomdp_p, hdbm_p=hdbm_p, fusion_mode=fusion_mode)
    local_hdbm = extract_mean_local_trend(is_go_mask, rts_hdbm)
    
    return local_pomdp, rts_pomdp, local_hdbm, rts_hdbm

def append_plot_data(data_list, arch, sub_id, metric, model_name, x_vals, mean_vals, std_vals=None):
    if std_vals is None:
        std_vals = [0.0] * len(x_vals) 
    for x, m, s in zip(x_vals, mean_vals, std_vals):
        data_list.append({
            'Archetype': arch, 'Subject': sub_id, 'Metric': metric, 
            'Model': model_name, 'X_Index': x, 'Mean_RT': m, 'Std_RT': s
        })

def main():
    parser = argparse.ArgumentParser(description="Simulate and Compare HDBM vs Static POMDP")
    parser.add_argument("--fusion_mode", type=str, default="additive_2", choices=['additive_1', 'additive_2', 'multiplicative'])
    args = parser.parse_args()
    
    print(f"--- Running Simulation Comparison for {args.fusion_mode.upper()} ---")
    hdbm_param_file = BASE_DIR / f'est_param_{args.fusion_mode}.csv'
    df_trends = pd.read_csv(TRENDS_FILE)
    
    archetypes = {
        "1. Extreme U-Shape": df_trends.sort_values('quadratic_curve', ascending=False).iloc[0],
        "2. Extreme Inv U-Shape": df_trends.sort_values('quadratic_curve', ascending=True).iloc[0],
        "3. Continuous Slowing": df_trends.sort_values('linear_slope', ascending=False).iloc[0],
        "4. Continuous Speeding": df_trends.sort_values('linear_slope', ascending=True).iloc[0],
    }
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    export_data_list = [] 
    param_export_list = [] # 用于保存被试的参数
    
    for row_idx, (title, subj_data) in enumerate(archetypes.items()):
        sub_id = subj_data['subject_id']
        print(f"Simulating {title} (Subj: {sub_id})...")
        
        seq_int = get_sequence_by_sig(subj_data['order_sig'])
        is_go_mask = np.array([s == 0 for s in seq_int])
        
        real_is_go, real_rts = extract_real_rts(sub_id)
        real_local = extract_mean_local_trend(real_is_go, real_rts)
        pomdp_p, hdbm_p = get_subject_params(sub_id, hdbm_param_file)
        
        # --- 收集并打印被试的参数 ---
        merged_params = {
            'Archetype': title,
            'Subject_ID': sub_id,
            'Order_Sig': subj_data['order_sig']
        }
        for k, v in pomdp_p.items():
            merged_params[f'Static_POMDP_{k}'] = v
        for k, v in hdbm_p.items():
            if k != 'subject_id': 
                merged_params[f'E2E_HDBM_{k}'] = v
        param_export_list.append(merged_params)
        
        # --- Parallel Execution ---
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(run_single_simulation_pair)(seq_int, pomdp_p, hdbm_p, is_go_mask, args.fusion_mode)
            for _ in range(N_SIMULATIONS)
        )
        
        pomdp_local_acc = [res[0] for res in results]
        pomdp_global_acc = [res[1] for res in results]
        hdbm_local_acc = [res[2] for res in results]
        hdbm_global_acc = [res[3] for res in results]

        # Calculate Means and Standard Deviations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            mat_p = np.full((N_SIMULATIONS, max(len(s) for s in pomdp_local_acc)), np.nan)
            for i, s in enumerate(pomdp_local_acc): mat_p[i, :len(s)] = s
            pomdp_local_mean = np.nanmean(mat_p, axis=0)
            pomdp_local_std = np.nanstd(mat_p, axis=0)
            
            mat_h = np.full((N_SIMULATIONS, max(len(s) for s in hdbm_local_acc)), np.nan)
            for i, s in enumerate(hdbm_local_acc): mat_h[i, :len(s)] = s
            hdbm_local_mean = np.nanmean(mat_h, axis=0)
            hdbm_local_std = np.nanstd(mat_h, axis=0)

            pomdp_global_mean = np.nanmean(np.array(pomdp_global_acc), axis=0)
            pomdp_global_std = np.nanstd(np.array(pomdp_global_acc), axis=0)
            
            hdbm_global_mean = np.nanmean(np.array(hdbm_global_acc), axis=0)
            hdbm_global_std = np.nanstd(np.array(hdbm_global_acc), axis=0)

        # --- DATA RECORDING FOR CSV EXPORT ---
        x_local_real = range(1, len(real_local)+1)
        x_local_p = range(1, len(pomdp_local_mean)+1)
        x_local_h = range(1, len(hdbm_local_mean)+1)
        
        append_plot_data(export_data_list, title, sub_id, 'Local', 'Real Data', x_local_real, real_local)
        append_plot_data(export_data_list, title, sub_id, 'Local', 'Static POMDP', x_local_p, pomdp_local_mean, pomdp_local_std)
        append_plot_data(export_data_list, title, sub_id, 'Local', 'E2E HDBM', x_local_h, hdbm_local_mean, hdbm_local_std)
        
        window = 15
        real_smooth = pd.Series(real_rts).interpolate().rolling(window, min_periods=1).mean().values
        pomdp_smooth = pd.Series(pomdp_global_mean).interpolate().rolling(window, min_periods=1).mean().values
        pomdp_std_sm = pd.Series(pomdp_global_std).interpolate().rolling(window, min_periods=1).mean().values
        hdbm_smooth = pd.Series(hdbm_global_mean).interpolate().rolling(window, min_periods=1).mean().values
        hdbm_std_sm = pd.Series(hdbm_global_std).interpolate().rolling(window, min_periods=1).mean().values
        x_trials = np.arange(1, len(seq_int)+1)
        
        append_plot_data(export_data_list, title, sub_id, 'Global', 'Real Data', x_trials, real_smooth)
        append_plot_data(export_data_list, title, sub_id, 'Global', 'Static POMDP', x_trials, pomdp_smooth, pomdp_std_sm)
        append_plot_data(export_data_list, title, sub_id, 'Global', 'E2E HDBM', x_trials, hdbm_smooth, hdbm_std_sm)

        # --- PLOTTING WITH ERROR BANDS ---
        ax_local = axes[row_idx, 0]
        ax_local.plot(x_local_real, real_local, 'ko-', lw=2, label="Real Data")
        
        ax_local.plot(x_local_p, pomdp_local_mean, 'r--', lw=2, label="Static POMDP Baseline")
        ax_local.fill_between(x_local_p, pomdp_local_mean - pomdp_local_std, pomdp_local_mean + pomdp_local_std, color='r', alpha=0.15)
        
        ax_local.plot(x_local_h, hdbm_local_mean, 'b-', lw=2.5, label="E2E HDBM+POMDP")
        ax_local.fill_between(x_local_h, hdbm_local_mean - hdbm_local_std, hdbm_local_mean + hdbm_local_std, color='b', alpha=0.15)
        
        ax_local.set_title(f"{title}: Local RT Trend")
        ax_local.set_xlabel("Consecutive Go Trials")
        ax_local.set_ylabel("RT (ms)")
        ax_local.grid(True, alpha=0.3)
        if row_idx == 0: ax_local.legend()

        ax_global = axes[row_idx, 1]
        ax_global.plot(x_trials, real_smooth, 'k-', alpha=0.5, lw=2, label="Real Data (Smoothed)")
        
        ax_global.plot(x_trials, pomdp_smooth, 'r--', lw=2, label="Static POMDP Baseline")
        ax_global.fill_between(x_trials, pomdp_smooth - pomdp_std_sm, pomdp_smooth + pomdp_std_sm, color='r', alpha=0.15)
        
        ax_global.plot(x_trials, hdbm_smooth, 'b-', lw=2.5, label="E2E HDBM+POMDP")
        ax_global.fill_between(x_trials, hdbm_smooth - hdbm_std_sm, hdbm_smooth + hdbm_std_sm, color='b', alpha=0.15)
        
        ax_global.set_title(f"{title}: Global Trial Trajectory")
        ax_global.set_xlabel("Trial Number")
        ax_global.grid(True, alpha=0.3)
        if row_idx == 0: ax_global.legend()

    # --- SAVE PLOTS AND DATA ---
    plt.tight_layout()
    out_dir = BASE_DIR / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_img = out_dir / f'model_comparison_{args.fusion_mode}.png'
    plt.savefig(out_img, dpi=300)
    print(f"\n>>> Saved comparison plot to {out_img}")
    
    df_export = pd.DataFrame(export_data_list)
    out_csv = out_dir / f'plot_data_{args.fusion_mode}.csv'
    df_export.to_csv(out_csv, index=False)
    print(f">>> Saved raw plotting data to {out_csv}")
    
    # --- PRINT AND SAVE ARCHETYPE PARAMS ---
    df_params = pd.DataFrame(param_export_list)
    out_param_csv = out_dir / f'archetype_params_{args.fusion_mode}.csv'
    df_params.to_csv(out_param_csv, index=False)
    
    print("\n==================================================")
    print("🎯 ARCHETYPE SUBJECT PARAMETERS EXPORTED")
    print("==================================================")
    for p in param_export_list:
        print(f"\n🔹 {p['Archetype']} (Subj: {p['Subject_ID']})")
        
        hdbm_core_str = f"eta: {p.get('E2E_HDBM_eta', 'N/A'):.4f}"
        if args.fusion_mode in ['additive_1', 'additive_2']:
            hdbm_core_str += f" | rho: {p.get('E2E_HDBM_rho', 'N/A'):.4f}"
        else:
            hdbm_core_str += f" | gamma: {p.get('E2E_HDBM_gamma', 'N/A'):.4f}"
            
        print(f"   [HDBM Core]   {hdbm_core_str}")
        print(f"   [POMDP q_s]   Static: {p.get('Static_POMDP_q_s', 'N/A'):.4f} | E2E: {p.get('E2E_HDBM_q_s', 'N/A'):.4f}")
        print(f"   [POMDP q_d]   Static: {p.get('Static_POMDP_q_d', 'N/A'):.4f} | E2E: {p.get('E2E_HDBM_q_d', 'N/A'):.4f}")
        print(f"   [POMDP cost]  Static: {p.get('Static_POMDP_cost_stop_error', 'N/A'):.4f} | E2E: {p.get('E2E_HDBM_cost_stop_error', 'N/A'):.4f}")
        
    print(f"\n>>> Saved full Archetype parameters to {out_param_csv}")

if __name__ == "__main__":
    main()