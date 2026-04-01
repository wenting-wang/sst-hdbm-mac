import sys
import tempfile
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import ast
import warnings
warnings.filterwarnings("ignore")

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from core.hdbm_v4 import HDBM
from core.pomdp import POMDP
from core import simulation

TRENDS_FILE = BASE_DIR / 'subject_local_trends.csv'
POMDP_PRIOR_FILE = BASE_DIR / 'pomdp_posterior.csv'
HDBM_PARAM_FILE = BASE_DIR / 'est_param_additive_2.csv'
ORDERS_FILE = BASE_DIR / 'orders.csv'
DATA_ROOT = Path("/u/wenwang/data/sst_valid_base")

FIXED_PARAMS = {"cost_time": 0.001, "cost_go_error": 1.0, "cost_go_missing": 1.0}
FUSION_MODE = 'additive_2'
N_SIMULATIONS = 50  # Number of simulations per subject to average out POMDP noise

def extract_real_rts(subject_id):
    """
    Extract real reaction times using exact, deterministic file paths.
    This is extremely fast as it skips all file searching.
    """
    # 1. Exact zip file path
    zip_filename = f"{subject_id}_baseline_year_1_arm_1_sst.csv.zip"
    zip_path = DATA_ROOT / zip_filename
    
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing ZIP file: {zip_path}")
        
    # 2. Extract and read using a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
            
        # 3. Exact extracted CSV path
        csv_filename = f"{subject_id}_baseline_year_1_arm_1_sst.csv"
        csv_path = Path(tmpdir) / "SST" / "baseline_year_1_arm_1" / csv_filename
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV inside ZIP: {csv_path}")
            
        # 4. Process the data
        df = pd.read_csv(csv_path, usecols=['sst_expcon', 'sst_go_rt'])
        is_go = df['sst_expcon'].astype(str).str.strip() == 'GoTrial'
        rts = pd.to_numeric(df['sst_go_rt'], errors='coerce').to_numpy()
        
        # Filter out anticipatory noise / missing responses (<= 200ms)
        rts = np.where(rts <= 200, np.nan, rts)
        
    return is_go.to_numpy(), rts

def get_sequence_by_sig(order_sig):
    """Retrieve the binary sequence list for a given order signature."""
    # FIX: Explicitly tell Pandas to treat these columns as strings to prevent OverflowError
    df = pd.read_csv(ORDERS_FILE, dtype={'order_seq': str, 'order_sig': str})
    
    seq_str = df.loc[df['order_sig'] == order_sig, 'order_seq'].values[0]
    if seq_str.startswith('['):
        return ast.literal_eval(seq_str)
    return [int(char) for char in str(seq_str) if char.isdigit()]

def get_subject_params(subject_id):
    """Extract parameters from both CSVs, handling the 'NDAR_' prefix discrepancy."""
    short_sub_id = subject_id.replace('NDAR_', '')
    
    # 1. Extract Pure POMDP params
    df_pomdp = pd.read_csv(POMDP_PRIOR_FILE)
    df_sub = df_pomdp[df_pomdp['subject_id'].str.contains(short_sub_id, na=False)]
    
    if df_sub.empty:
        raise ValueError(f"Subject {short_sub_id} not found in POMDP priors.")
    pomdp_params = {row['index']: row['mean'] for _, row in df_sub.iterrows()}
    
    # 2. Extract full HDBM+POMDP 8 parameters
    df_hdbm = pd.read_csv(HDBM_PARAM_FILE)
    df_hdbm_sub = df_hdbm[df_hdbm['subject_id'].str.contains(short_sub_id, na=False)]
    
    if df_hdbm_sub.empty:
        raise ValueError(f"Subject {short_sub_id} not found in HDBM parameters.")
    hdbm_params = df_hdbm_sub.iloc[0].to_dict()
    
    return pomdp_params, hdbm_params

def simulate_behavior(seq_int, pomdp_p, hdbm_p=None):
    """
    Core Simulation Function:
    - If hdbm_p is None: Runs Static POMDP (1 Value Iteration).
    - If hdbm_p exists: Runs Dynamic HDBM + POMDP (Value Iteration per trial).
    """
    total_trials = len(seq_int)
    go_directions = np.random.choice([0, 1], size=total_trials)
    sim_rts = np.full(total_trials, np.nan)
    next_stop_ssd = 2
    
    if hdbm_p is None:
        # MODE 1: PURE POMDP (Static Prior)
        r_static = pomdp_p.get('rate_stop_trial', 0.1666)
        
        pomdp = POMDP(
            rate_stop_trial=r_static,  
            q_d_n=pomdp_p['q_d_n'], q_d=pomdp_p['q_d'],
            q_s_n=pomdp_p['q_s_n'], q_s=pomdp_p['q_s'],
            cost_stop_error=pomdp_p['cost_stop_error'], inv_temp=pomdp_p['inv_temp'],
            **FIXED_PARAMS
        )
        pomdp.value_iteration_tensor()
        
        for t in range(total_trials):
            is_stop = seq_int[t]
            go_dir = 'right' if go_directions[t] == 1 else 'left'
            true_stop_state = 'stop' if is_stop == 1 else 'nonstop'
            ssd = next_stop_ssd if is_stop == 1 else -1
            
            res_str, rt = simulation.simu_trial(
                pomdp, true_go_state=go_dir, true_stop_state=true_stop_state, ssd=ssd, verbose=False
            )
            
            if is_stop == 0 and res_str in ['GS', 'GE']:
                sim_rts[t] = rt * 40.0 
                
            if is_stop == 1:
                if res_str == 'SS': next_stop_ssd = min(next_stop_ssd + 2, 34)
                elif res_str == 'SE': next_stop_ssd = max(next_stop_ssd - 2, 2)
                
    else:
        # MODE 2: HDBM + POMDP (Dynamic Prior)
        hdbm = HDBM(
            alpha_go=0.85, alpha_stop=0.85, 
            eta=hdbm_p['eta'], rho=hdbm_p['rho'], 
            fusion_type=FUSION_MODE
        )
        r_preds = hdbm.simu_task(seq_int, block_size=180)

        for t in range(total_trials):
            is_stop = seq_int[t]
            go_dir = 'right' if go_directions[t] == 1 else 'left'
            r_pred = float(r_preds[t])
            true_stop_state = 'stop' if is_stop == 1 else 'nonstop'
            ssd = next_stop_ssd if is_stop == 1 else -1
            
            pomdp = POMDP(
                rate_stop_trial=r_pred,  
                q_d_n=hdbm_p['q_d_n'], q_d=hdbm_p['q_d'],
                q_s_n=hdbm_p['q_s_n'], q_s=hdbm_p['q_s'],
                cost_stop_error=hdbm_p['cost_stop_error'], inv_temp=hdbm_p['inv_temp'],
                **FIXED_PARAMS
            )
            pomdp.value_iteration_tensor()
            
            res_str, rt = simulation.simu_trial(
                pomdp, true_go_state=go_dir, true_stop_state=true_stop_state, ssd=ssd, verbose=False
            )
            
            if is_stop == 0 and res_str in ['GS', 'GE']:
                sim_rts[t] = rt * 40.0 
                
            if is_stop == 1:
                if res_str == 'SS': next_stop_ssd = min(next_stop_ssd + 2, 34)
                elif res_str == 'SE': next_stop_ssd = max(next_stop_ssd - 2, 2)
                
    return sim_rts

def extract_mean_local_trend(is_go_array, rts_array):
    """Extract and average consecutive Go trial streaks."""
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
                if seen_first_stop:
                    current_streak.append(rt)
        if len(current_streak) > 0:
            streaks.append(current_streak)
        return streaks
    
    streaks = extract_streaks(is_go_array[:180], rts_array[:180]) + extract_streaks(is_go_array[180:], rts_array[180:])
    if not streaks: 
        return []
    
    max_len = max(len(s) for s in streaks)
    mat = np.full((len(streaks), max_len), np.nan)
    for i, s in enumerate(streaks): 
        mat[i, :len(s)] = s
        
    return np.nanmean(mat, axis=0)

def main():
    df_trends = pd.read_csv(TRENDS_FILE)
    
    archetypes = {
        "1. Extreme U-Shape": df_trends.sort_values('quadratic_curve', ascending=False).iloc[0],
        "2. Extreme Inv U-Shape": df_trends.sort_values('quadratic_curve', ascending=True).iloc[0],
        "3. Continuous Slowing": df_trends.sort_values('linear_slope', ascending=False).iloc[0],
        "4. Continuous Speeding": df_trends.sort_values('linear_slope', ascending=True).iloc[0],
    }
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    
    for row_idx, (title, subj_data) in enumerate(archetypes.items()):
        sub_id = subj_data['subject_id']
        print(f"Simulating {title} (Subj: {sub_id})...")
        
        seq_int = get_sequence_by_sig(subj_data['order_sig'])
        is_go_mask = np.array([s == 0 for s in seq_int])
        
        real_is_go, real_rts = extract_real_rts(sub_id)
        real_local = extract_mean_local_trend(real_is_go, real_rts)
        
        pomdp_p, hdbm_p = get_subject_params(sub_id)
        
        pomdp_local_acc, pomdp_global_acc = [], []
        hdbm_local_acc, hdbm_global_acc = [], []
        
        for sim_i in range(N_SIMULATIONS):
            if (sim_i + 1) % 10 == 0:
                print(f"  -> Simulation {sim_i + 1}/{N_SIMULATIONS}...")
                
            rts_pomdp = simulate_behavior(seq_int, pomdp_p, hdbm_p=None)
            pomdp_local_acc.append(extract_mean_local_trend(is_go_mask, rts_pomdp))
            pomdp_global_acc.append(rts_pomdp)
            
            rts_hdbm = simulate_behavior(seq_int, pomdp_p, hdbm_p=hdbm_p)
            hdbm_local_acc.append(extract_mean_local_trend(is_go_mask, rts_hdbm))
            hdbm_global_acc.append(rts_hdbm)
            
        mat_p = np.full((N_SIMULATIONS, max(len(s) for s in pomdp_local_acc)), np.nan)
        for i, s in enumerate(pomdp_local_acc): mat_p[i, :len(s)] = s
        pomdp_local_mean = np.nanmean(mat_p, axis=0)
        
        mat_h = np.full((N_SIMULATIONS, max(len(s) for s in hdbm_local_acc)), np.nan)
        for i, s in enumerate(hdbm_local_acc): mat_h[i, :len(s)] = s
        hdbm_local_mean = np.nanmean(mat_h, axis=0)

        pomdp_global_mean = np.nanmean(np.array(pomdp_global_acc), axis=0)
        hdbm_global_mean = np.nanmean(np.array(hdbm_global_acc), axis=0)

        # --- PLOTTING: Local Trend (Left Column) ---
        ax_local = axes[row_idx, 0]
        ax_local.plot(range(1, len(real_local)+1), real_local, 'ko-', lw=2, label="Real Data")
        ax_local.plot(range(1, len(pomdp_local_mean)+1), pomdp_local_mean, 'r--', lw=2, label="Static POMDP")
        ax_local.plot(range(1, len(hdbm_local_mean)+1), hdbm_local_mean, 'b-', lw=2.5, label="HDBM + POMDP")
        ax_local.set_title(f"{title}: Local RT Trend")
        ax_local.set_xlabel("Consecutive Go Trials")
        ax_local.set_ylabel("RT (ms)")
        ax_local.grid(True, alpha=0.3)
        if row_idx == 0: ax_local.legend()

        # --- PLOTTING: Global Trend (Right Column) ---
        ax_global = axes[row_idx, 1]
        window = 15
        real_smooth = pd.Series(real_rts).interpolate().rolling(window, min_periods=1).mean()
        pomdp_smooth = pd.Series(pomdp_global_mean).interpolate().rolling(window, min_periods=1).mean()
        hdbm_smooth = pd.Series(hdbm_global_mean).interpolate().rolling(window, min_periods=1).mean()
        
        x_trials = np.arange(1, len(seq_int)+1)
        ax_global.plot(x_trials, real_smooth, 'k-', alpha=0.5, lw=2, label="Real Data (Smoothed)")
        ax_global.plot(x_trials, pomdp_smooth, 'r--', lw=2, label="Static POMDP")
        ax_global.plot(x_trials, hdbm_smooth, 'b-', lw=2.5, label="HDBM + POMDP")
        ax_global.set_title(f"{title}: Global Trial Trajectory")
        ax_global.set_xlabel("Trial Number")
        ax_global.grid(True, alpha=0.3)
        if row_idx == 0: ax_global.legend()

    plt.tight_layout()
    out_file = BASE_DIR / 'outputs' / 'model_comparison_archetypes.png'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    print(f"\n>>> Saved comparison plot to {out_file}")

if __name__ == "__main__":
    main()