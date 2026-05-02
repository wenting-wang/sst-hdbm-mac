import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from pathlib import Path

# ==========================================
# 1. DRM 模拟器与损失函数 (5参数升级版)
# ==========================================
def generate_wald_rts(v, a, t_er, n_trials, seed=None):
    if seed is not None: np.random.seed(seed)
    v = max(v, 0.001); a = max(a, 0.001)
    return np.random.wald(a / v, a**2, n_trials) + t_er

def simulate_drm_fast_for_fitting(params, ssd_list, n_go=2000, n_stop_per_ssd=500, seed=42):
    v_go_c, v_go_e, v_stop, a, t_er = params
    
    go_c_rts = generate_wald_rts(v_go_c, a, t_er, n_go, seed=seed)
    go_e_rts = generate_wald_rts(v_go_e, a, t_er, n_go, seed=seed+1)
    
    go_rts = np.minimum(go_c_rts, go_e_rts)
    is_error = go_e_rts < go_c_rts
    is_miss = go_rts > 1.0 
    
    sim_p_ge = np.mean(is_error & ~is_miss)
    sim_p_gm = np.mean(is_miss)
    
    valid_gs = go_rts[~is_error & ~is_miss]
    if len(valid_gs) >= 5:
        go_quantiles = np.percentile(valid_gs, [10, 30, 50, 70, 90])
    else:
        go_quantiles = np.array([1.0]*5)
        
    stop_results = {}
    for ssd in ssd_list:
        sim_go_c = generate_wald_rts(v_go_c, a, t_er, n_stop_per_ssd, seed=seed+int(ssd*1000))
        sim_go_e = generate_wald_rts(v_go_e, a, t_er, n_stop_per_ssd, seed=seed+int(ssd*2000))
        sim_go = np.minimum(sim_go_c, sim_go_e)
        
        sim_stop = generate_wald_rts(v_stop, a, t_er, n_stop_per_ssd, seed=seed+int(ssd*3000)) + ssd
        p_respond = np.mean((sim_go < sim_stop) & (sim_go <= 1.0))
        stop_results[ssd] = p_respond
        
    return go_quantiles, sim_p_ge, sim_p_gm, stop_results

def simulate_drm_abcd_format(params, n_trials=400, n_repeat=30, step_size_ms=25):
    v_go_c, v_go_e = params['v_go_c'], params['v_go_e']
    v_stop, a, t_er = params['v_stop'], params['a'], params['t_er']
    
    all_rows = []
    deadline_steps = int(1000 / step_size_ms)
    
    for i in range(n_repeat):
        np.random.seed(42 + i) 
        current_ssd_steps = 200 / step_size_ms 
        
        for t in range(n_trials):
            is_stop_trial = np.random.rand() < 0.20
            
            mean_go_c = a / max(v_go_c, 0.001)
            mean_go_e = a / max(v_go_e, 0.001)
            go_c_rt_raw = np.random.wald(mean_go_c, a**2) + t_er
            go_e_rt_raw = np.random.wald(mean_go_e, a**2) + t_er
            
            if go_c_rt_raw < go_e_rt_raw:
                go_rt_raw, is_correct = go_c_rt_raw, True
            else:
                go_rt_raw, is_correct = go_e_rt_raw, False
                
            go_rt_steps = int(go_rt_raw * (1000 / step_size_ms)) 
            
            if not is_stop_trial:
                if go_rt_steps > deadline_steps:
                    result, rt_val = 'GM', np.nan
                else:
                    result = 'GS' if is_correct else 'GE'
                    rt_val = go_rt_steps
                ssd_val = np.nan
            else:
                ssd_val = current_ssd_steps  # <--- 修复: 之前漏了这句，导致画图时找不到 SSD 的值
                mean_stop = a / max(v_stop, 0.001)
                stop_rt_raw = np.random.wald(mean_stop, a**2) + t_er
                stop_rt_steps = int(stop_rt_raw * (1000 / step_size_ms))
                
                # Go 与 Stop 赛跑 (如果在死线之前跑赢了Stop，算SE；否则全算作SS)
                if go_rt_steps < (stop_rt_steps + current_ssd_steps) and go_rt_steps <= deadline_steps:
                    result, rt_val = 'SE', go_rt_steps
                    current_ssd_steps = max(0, current_ssd_steps - (50 / step_size_ms))
                else:
                    result, rt_val = 'SS', np.nan
                    current_ssd_steps = current_ssd_steps + (50 / step_size_ms)
                    
            all_rows.append({'result': result, 'rt': rt_val, 'ssd': ssd_val, 'sim_id': i, 'trial': t + 1})
    return pd.DataFrame(all_rows)

def loss_function(params, empirical_data, w_rt=1.0, w_inhibition=2.0, w_ge=5.0, w_gm=5.0):
    v_go_c, v_go_e, v_stop, a, t_er = params
    penalty = 0.0
    if v_go_c <= 0.1: penalty += (0.1 - v_go_c) * 1000 + 100
    if v_go_e <= 0.001: penalty += (0.001 - v_go_e) * 1000 + 100
    if v_stop <= 0.1: penalty += (0.1 - v_stop) * 1000 + 100
    if a <= 0.1: penalty += (0.1 - a) * 1000 + 100
    if t_er <= 0.05: penalty += (0.05 - t_er) * 1000 + 100
    if t_er >= 0.6: penalty += (t_er - 0.6) * 1000 + 100
    if penalty > 0: return 1000 + penalty 
        
    ssd_list = list(empirical_data['p_respond'].keys())
    sim_go_q, sim_p_ge, sim_p_gm, sim_p_resp = simulate_drm_fast_for_fitting(params, ssd_list)
    
    emp_go_q = np.array(empirical_data['go_quantiles'])
    error_rt = np.mean((np.array(sim_go_q) - emp_go_q)**2) * w_rt
    
    error_ge = (sim_p_ge - empirical_data['p_ge'])**2 * w_ge
    error_gm = (sim_p_gm - empirical_data['p_gm'])**2 * w_gm
    
    emp_p_array = np.array([empirical_data['p_respond'][ssd] for ssd in ssd_list])
    sim_p_array = np.array([sim_p_resp[ssd] for ssd in ssd_list])
    error_inhibition = np.mean((sim_p_array - emp_p_array)**2) * w_inhibition
        
    return error_rt + error_inhibition + error_ge + error_gm

# ==========================================
# 2. 数据处理与特征提取
# ==========================================
def extract_empirical_data(df):
    go_trials = df[df['sequence'] == 0]
    if go_trials.empty: return None
    
    gs_trials = go_trials[go_trials['result'] == 'GS']
    if gs_trials.empty: return None
    
    rts = gs_trials['rt_real'].dropna().values / 1000.0
    go_quantiles = np.percentile(rts, [10, 30, 50, 70, 90])
    
    p_ge = (go_trials['result'] == 'GE').mean()
    p_gm = (go_trials['result'] == 'GM').mean()
    
    stop_trials = df[df['result'].isin(['SS', 'SE'])]
    p_respond = {}
    if not stop_trials.empty:
        for ssd_real, group in stop_trials.groupby('ssd_real'):
            if pd.isna(ssd_real) or ssd_real == 0: continue
            p_respond[ssd_real / 1000.0] = (group['result'] == 'SE').mean()
            
    if not p_respond: return None
    return {'go_quantiles': go_quantiles, 'p_ge': p_ge, 'p_gm': p_gm, 'p_respond': p_respond}

from preprocessing import preprocessing 

def get_subject_id(file_name):
    name = Path(file_name).name
    if 'NDAR' in name:
        parts = name.split('_')
        for i, part in enumerate(parts):
            if part == 'NDAR' and i + 1 < len(parts):
                return parts[i+1] 
    return Path(file_name).stem.split('_')[0].replace('NDAR_', '')

def load_and_preprocess_file(file_path):
    fp = Path(file_path)
    subject_id = get_subject_id(fp)

    if fp.suffix == '.zip':
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(fp, 'r') as zr:
                zr.extractall(tmpdir)
            csvs = list(Path(tmpdir).rglob(f"*{subject_id}*.csv")) or list(Path(tmpdir).rglob("*.csv"))
            if csvs: return subject_id, preprocessing(str(csvs[0]))
            else: raise FileNotFoundError(f"No CSV found inside {fp.name}")
    else:
        return subject_id, preprocessing(str(fp))

# ==========================================
# 3. 绘图配置与函数
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

STYLE = {
    "label_fontsize": 9, "title_fontsize": 10, "tick_fontsize": 8,
    "legend_fontsize": 10, "panel_label_size": 12,
    "mean_lw": 1.5, "fine_lw": 0.6, "sim_trace_lw": 0.6,
    "axis_lw": 0.8, "marker_size": 3.0,     
}
COLOR_OBS = '#404040'   
COLOR_SIM = '#d95f02'   

def _clean_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(STYLE['axis_lw'])
    ax.spines['bottom'].set_linewidth(STYLE['axis_lw'])
    ax.tick_params(width=STYLE['axis_lw'], labelsize=STYLE['tick_fontsize'])

def get_outcome_stats(df_sim):
    outcomes = ['GS', 'GE', 'GM', 'SS', 'SE']
    rates_per_sim = df_sim.groupby('sim_id')['result'].value_counts(normalize=True).unstack(fill_value=0)
    rates_per_sim = rates_per_sim.reindex(columns=outcomes, fill_value=0)
    return rates_per_sim.mean(), rates_per_sim.std()

def get_rt_dist_stats(df_sim, outcome, bins):
    sub_all = df_sim[df_sim['result'] == outcome]
    if sub_all.empty: return np.zeros(len(bins)-1), np.zeros(len(bins)-1)
    
    densities = []
    grouped = sub_all.groupby('sim_id')
    for i in df_sim['sim_id'].unique():
        if i in grouped.groups:
            hist, _ = np.histogram(grouped.get_group(i)['rt'] * 25, bins=bins, density=True)
        else:
            hist = np.zeros(len(bins)-1)
        densities.append(hist)
    return np.mean(densities, axis=0), np.std(densities, axis=0)

def plot_drm_ppc(subjects_data, output_path="fig_drm_ppc.png", step_size_ms=25):
    n_subjects = len(subjects_data)
    fig, axs = plt.subplots(5, n_subjects, figsize=(2.5 * n_subjects, 8.5), constrained_layout=True)
    if n_subjects == 1: axs = np.expand_dims(axs, axis=1)
    panel_labels = ['A', 'B', 'C', 'D', 'E']

    for col_idx, subj_info in enumerate(subjects_data):
        label, df_obs, df_sim = subj_info['label'], subj_info['df_obs'], subj_info['df_sim']
        
        # --- Row 1 (Panel A): Outcomes ---
        ax = axs[0, col_idx]
        outcomes = ['GS', 'GE', 'GM', 'SS', 'SE']
        rates_obs = [df_obs['result'].value_counts(normalize=True).get(o, 0) for o in outcomes]
        mean_sim, std_sim = get_outcome_stats(df_sim)
        
        x = np.arange(len(outcomes))
        width = 0.35
        ax.bar(x - width/2, rates_obs, width, label='Obs', color=COLOR_OBS, alpha=0.7)
        ax.bar(x + width/2, mean_sim, width, yerr=std_sim, capsize=2, color=COLOR_SIM, alpha=0.7, error_kw={'elinewidth': 0.8})
        ax.set_xticks(x); ax.set_xticklabels(outcomes, fontsize=STYLE['tick_fontsize']); ax.set_ylim(0, 1.05)
        ax.set_title(label, fontsize=STYLE['title_fontsize'], pad=4)
        if col_idx == 0: 
            ax.set_ylabel("Probability Mass", fontsize=STYLE['label_fontsize'])
            ax.text(-0.35, 1.05, panel_labels[0], transform=ax.transAxes, fontsize=STYLE['panel_label_size'], fontweight='bold', va='top', ha='right')
        _clean_spines(ax)

        # --- Row 2 & 3 (Panel B & C): GS & SE RT Distributions ---
        bins_ms = np.linspace(0, 1000, 41)
        bin_centers = (bins_ms[:-1] + bins_ms[1:]) / 2
        
        for row_offset, result_type in enumerate(['GS', 'SE']):
            row_idx = 1 + row_offset
            ax = axs[row_idx, col_idx]
            
            obs_data = df_obs[df_obs['result'] == result_type]['rt_real'].dropna()
            if not obs_data.empty:
                ax.hist(obs_data, bins=bins_ms, density=True, alpha=0.3, color=COLOR_OBS, ec='white', linewidth=0.3)
            
            for sim_i in range(min(3, df_sim['sim_id'].nunique())):
                sim_data_i = df_sim[(df_sim['sim_id'] == sim_i) & (df_sim['result'] == result_type)]['rt'].dropna() * step_size_ms
                if not sim_data_i.empty:
                    hist_i, _ = np.histogram(sim_data_i, bins=bins_ms, density=True)
                    smooth_sim_i = gaussian_filter1d(hist_i, sigma=1.0)
                    ax.plot(bin_centers, smooth_sim_i, color=COLOR_SIM, linewidth=STYLE['fine_lw'], alpha=0.5)

            mean_dens, std_dens = get_rt_dist_stats(df_sim, result_type, bins_ms)
            if np.any(mean_dens > 0): 
                smooth_mean = gaussian_filter1d(mean_dens, sigma=1.0)
                ax.plot(bin_centers, smooth_mean, color=COLOR_SIM, linewidth=STYLE['mean_lw'], zorder=5)
                smooth_upper = gaussian_filter1d(mean_dens + std_dens, sigma=1.0)
                smooth_lower = gaussian_filter1d(np.maximum(0, mean_dens - std_dens), sigma=1.0)
                ax.fill_between(bin_centers, smooth_lower, smooth_upper, color=COLOR_SIM, alpha=0.15, linewidth=0)
                
            ax.set_xlim(0, 1000); ax.set_xlabel("Time (ms)", fontsize=STYLE['label_fontsize'])
            if col_idx == 0: 
                ax.set_ylabel(f"{result_type} RT Density", fontsize=STYLE['label_fontsize'])
                ax.text(-0.35, 1.05, panel_labels[row_idx], transform=ax.transAxes, fontsize=STYLE['panel_label_size'], fontweight='bold', va='top', ha='right')
            _clean_spines(ax)

        # --- Row 4 (Panel D): SSD Tracking ---
        ax = axs[3, col_idx]
        ax.scatter(df_obs.index + 1, df_obs['ssd_real'], color=COLOR_OBS, s=STYLE['marker_size'], alpha=0.6, zorder=10)
        # 修复: 加入 bfill() 确保如果最初几次是 Go 也能画出线来
        df_sim['ssd_ms'] = df_sim.groupby('sim_id')['ssd'].ffill().bfill() * step_size_ms
        for i, sub in df_sim.groupby('sim_id'):
            if i < 3: ax.step(sub['trial'], sub['ssd_ms'], color=COLOR_SIM, alpha=0.6, linewidth=STYLE['sim_trace_lw'], where='post')
        ax.set_xlim(0, 400); ax.set_yticks([0, 200, 400, 600]); ax.set_ylim(-25, 750); ax.set_xlabel("Trial", fontsize=STYLE['label_fontsize'])
        if col_idx == 0: 
            ax.set_ylabel("SSD (ms)", fontsize=STYLE['label_fontsize'])
            ax.text(-0.35, 1.05, panel_labels[3], transform=ax.transAxes, fontsize=STYLE['panel_label_size'], fontweight='bold', va='top', ha='right')
        _clean_spines(ax)

        # --- Row 5 (Panel E): Inhibition Function ---
        ax = axs[4, col_idx]
        stop_trials = df_sim[df_sim['result'].isin(['SS', 'SE'])].copy()
        if not stop_trials.empty:
            stop_trials['is_ss'] = (stop_trials['result'] == 'SS').astype(int)
            prob_per_sim = stop_trials.pivot_table(index='sim_id', columns='ssd', values='is_ss', aggfunc='mean')
            mean_p, std_p = prob_per_sim.mean(axis=0), prob_per_sim.std(axis=0).fillna(0)
            ssd_vals_ms = mean_p.index * step_size_ms
            ax.fill_between(ssd_vals_ms, np.clip(mean_p - std_p, 0, 1), np.clip(mean_p + std_p, 0, 1), color=COLOR_SIM, alpha=0.15, linewidth=0)
            ax.plot(ssd_vals_ms, mean_p, 'o-', color=COLOR_SIM, linewidth=STYLE['mean_lw'], markersize=4, markeredgecolor='white', markeredgewidth=0.5)
            
        stop_df_obs = df_obs[df_obs['result'].isin(['SS', 'SE'])].copy()
        if not stop_df_obs.empty:
            stop_df_obs['is_ss'] = (stop_df_obs['result'] == 'SS').astype(int)
            prob_obs = stop_df_obs.groupby('ssd')['is_ss'].mean()
            ax.plot(prob_obs.index * step_size_ms, prob_obs.values, 'o-', color=COLOR_OBS, linewidth=1.0, markersize=4, markeredgecolor='white', markeredgewidth=0.5)
            
        ax.set_ylim(-0.05, 1.05); ax.set_xlim(-25, 650); ax.set_xlabel("SSD (ms)", fontsize=STYLE['label_fontsize'])
        if col_idx == 0: 
            ax.set_ylabel("P(Stop Success)", fontsize=STYLE['label_fontsize'])
            ax.text(-0.35, 1.05, panel_labels[4], transform=ax.transAxes, fontsize=STYLE['panel_label_size'], fontweight='bold', va='top', ha='right')
        _clean_spines(ax)

    legend_elements = [Patch(facecolor=COLOR_OBS, alpha=0.7, label='Empirical Observation'), Patch(facecolor=COLOR_SIM, alpha=0.8, label='DRM Prediction')]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False, fontsize=STYLE['legend_fontsize'])
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Final DRM PPC plot saved to: {output_path}")

# ==========================================
# 4. 主执行流 (Pipeline)
# ==========================================
def run_pipeline(data_dir, output_csv="drm_fit_results.csv", output_img="fig_drm_ppc_comparison.png", n_fit=100, target_plot_sids=None):
    folder_path = Path(data_dir)
    files = list(folder_path.glob("*.zip")) or list(folder_path.rglob("*.csv"))
    target_files = files[:n_fit]
    
    print(f"\n--- STEP 1: FITTING {len(target_files)} SUBJECTS ---")
    results = []
    initial_guess = [2.5, 0.2, 3.0, 1.0, 0.2] 
    
    for fp in target_files:
        try:
            subject_id, df_clean = load_and_preprocess_file(fp)
            empirical_data = extract_empirical_data(df_clean)
            if not empirical_data:
                print(f"[{subject_id}] Skipped: Not enough valid trials.")
                continue
                
            res = minimize(loss_function, initial_guess, args=(empirical_data,), method='Nelder-Mead', options={'maxiter': 1000})
            results.append({
                'subject_id': subject_id, 'file_path': str(fp), 'df_clean': df_clean, 
                'v_go_c': res.x[0], 'v_go_e': res.x[1], 'v_stop': res.x[2], 'a': res.x[3], 't_er': res.x[4], 'gof': res.fun
            })
            print(f"[{subject_id}] Fitted. Cost: {res.fun:.2f}")
        except Exception as e:
            print(f"[{fp.name}] Error: {e}")
            
    df_results = pd.DataFrame(results)
    df_results[['subject_id', 'file_path', 'v_go_c', 'v_go_e', 'v_stop', 'a', 't_er', 'gof']].to_csv(output_csv, index=False)
    print(f"\n=> Fitting completed. Parameters saved to {output_csv}")
    
    print(f"\n--- STEP 2: GENERATING PLOT ---")
    df_sorted = df_results.sort_values(by='gof').reset_index(drop=True)
    n = len(df_sorted)
    selection = []
    
    if target_plot_sids:
        print(f"Looking for specific target subjects: {target_plot_sids}")
        for sid in target_plot_sids:
            matches = df_sorted[df_sorted['subject_id'] == sid]
            if not matches.empty:
                idx = matches.index[0]
                cost_val = matches.iloc[0]['gof']
                selection.append((idx, f"Target Subject\nCost: {cost_val:.2f}"))
            else:
                print(f"[Warning] Target Subject '{sid}' not found among fitted subjects.")
    else:
        if n < 3:
            print("Not enough successful fits to select 3 percentiles. Aborting plot generation.")
            return
        selection = [
            (int(n * 0.05), "Good Fit (5th %ile Cost)"),
            (int(n * 0.50), "Moderate Fit (50th %ile Cost)"),
            (int(n * 0.95), "Poor Fit (95th %ile Cost)")
        ]
    
    if not selection: return

    subjects_data_for_plot = []
    for idx, label_text in selection:
        subj_info = df_sorted.iloc[idx]
        sid = subj_info['subject_id']
        print(f"Preparing data for: {sid} ({label_text.replace(chr(10), ' - ')})")
        
        df_obs = subj_info['df_clean']
        params_dict = {
            'v_go_c': subj_info['v_go_c'], 'v_go_e': subj_info['v_go_e'], 
            'v_stop': subj_info['v_stop'], 'a': subj_info['a'], 't_er': subj_info['t_er']
        }
        df_sim = simulate_drm_abcd_format(params_dict, n_trials=400, n_repeat=30, step_size_ms=25)
        
        subjects_data_for_plot.append({'label': f"{sid}\n{label_text}", 'df_obs': df_obs, 'df_sim': df_sim})
        
    plot_drm_ppc(subjects_data_for_plot, output_path=output_img, step_size_ms=25)

if __name__ == "__main__":
    DATA_DIR = "/Users/w/Desktop/data/sst_valid_base"
    NUM_TO_FIT = 100 
    TARGET_SUBJECTS = None 
    
    run_pipeline(DATA_DIR, n_fit=NUM_TO_FIT, target_plot_sids=TARGET_SUBJECTS)