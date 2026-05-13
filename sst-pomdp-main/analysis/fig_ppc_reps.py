import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import traceback

# --- Directory Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Add project root to sys.path to import local modules
sys.path.append(str(PROJECT_ROOT)) 

from core.models import POMDP
from core import simulation

# --- PLOS CB STYLING SETUP ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

STYLE = {
    "label_fontsize": 9,
    "title_fontsize": 10,
    "tick_fontsize": 8,
    "legend_fontsize": 10,   
    "panel_label_size": 12,
    "text_color": 'black',
    
    # Line Styles
    "mean_lw": 1.2,        
    "sim_trace_lw": 0.6,
    "axis_lw": 0.8,
    "marker_size": 2.5,     
}

# --- Configuration & Paths ---
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"
RAW_DATA_DIR = DATA_DIR / "example_processed_data"  

# Ensure directories exist
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Inputs
FILTER_CSV = DATA_DIR / "example_clinical_behavior.csv"
PARAMS_CSV = DATA_DIR / "example_params_posteriors.csv"
PPC_METRICS_CSV = DATA_DIR / "example_ppc_metrics.csv" 

# Output
OUTPUT_IMAGE = OUT_DIR / "fig_ppc_reps.png"

# Plot Settings
COLOR_OBS = '#404040'      # Dark Grey 
COLOR_SIM = '#008080'      # Teal
N_REPEAT = 30              
N_TRACES_HIGHLIGHT = 3     

# --- Core Functions ---

def select_representative_subjects(ppc_file, filter_file):
    """
    Selects 3 representative subjects (Good, Moderate, Poor fit) based on composite PPC cost.
    """
    if not ppc_file.exists() or not filter_file.exists():
        raise FileNotFoundError(
            f"Missing required summary files in {DATA_DIR}. "
            "Ensure clinical_behavior and ppc_metrics CSVs are present."
        )

    df_dist = pd.read_csv(ppc_file)
    df_filter = pd.read_csv(filter_file)
    
    # Safely handle 'year' format differences
    df_filter['year'] = df_filter['year'].astype(str)
    valid_ids = df_filter[df_filter['year'] == 'baseline']['subject_id'].values
    df_dist = df_dist[df_dist['subject_id'].isin(valid_ids)].copy()

    if df_dist.empty:
        raise ValueError("No baseline subjects found in the PPC metrics file.")

    # Calculate Composite Cost (Z-scored distances)
    dis_cols = [col for col in df_dist.columns if col.startswith('dis_')]
    df_z = df_dist.copy()
    for col in dis_cols:
        mean = df_dist[col].mean()
        std = df_dist[col].std()
        df_z[col] = (df_dist[col] - mean) / std if std != 0 else 0
            
    df_dist['composite_cost'] = df_z[dis_cols].sum(axis=1)
    df_sorted = df_dist.sort_values(by='composite_cost').reset_index(drop=True)
    n = len(df_sorted)
    
    selection = [
        (int(n * 0.05), "Good Fit (95th Percentile)"),
        (int(n * 0.50), "Moderate Fit (50th Percentile)"),
        (int(n * 0.95), "Poor Fit (5th Percentile)")
    ]
    
    subjects = []
    for idx, label in selection:
        subjects.append({
            "sid": df_sorted.iloc[idx]['subject_id'],
            "label": label,
            "cost": df_sorted.iloc[idx]['composite_cost']
        })
        print(f"Selected {label}: {df_sorted.iloc[idx]['subject_id']}")
        
    return subjects


def get_subject_params(params_file, subject_id, year='baseline'):
    """Extracts POMDP parameters for a specific subject."""
    df = pd.read_csv(params_file)
    year_col = 'subject_year' if 'subject_year' in df.columns else 'year'
    df[year_col] = df[year_col].astype(str)
    
    mask = (df["subject_id"] == subject_id) & (df[year_col] == str(year))
    if not mask.any():
        return None
    
    idx_col = 'index' if 'index' in df.columns else 'param'
    return df.loc[mask, [idx_col, "mean"]].set_index(idx_col)["mean"].to_dict()


def run_simulation_multi(params, n_repeat=30):
    """Runs the POMDP simulation multiple times for a subject (Optimized list generation)."""
    pomdp = POMDP(**params)
    pomdp.value_iteration_tensor()

    # Create a single list of dictionaries instead of 30 isolated DataFrames
    all_rows = []
    for i in range(n_repeat):
        out = simulation.simu_task(pomdp)
        # Assuming out provides row tuples mapping to [result, rt, ssd]
        for t, row in enumerate(out):
            all_rows.append({
                'result': row[0], 'rt': row[1], 'ssd': row[2], 
                'sim_id': i, 'trial': t + 1
            })
            
    return pd.DataFrame(all_rows)


# --- Helper Stats Functions (Heavily Optimized) ---

def get_outcome_stats(df_sim):
    outcomes = ['GS', 'GE', 'GM', 'SS', 'SE']
    rates_per_sim = df_sim.groupby('sim_id')['result'].value_counts(normalize=True).unstack(fill_value=0)
    rates_per_sim = rates_per_sim.reindex(columns=outcomes, fill_value=0)
    return rates_per_sim.mean(), rates_per_sim.std()

def get_rt_dist_stats(df_sim, outcome, bins):
    """Optimized using groupby to avoid boolean masking in a loop."""
    sub_all = df_sim[df_sim['result'] == outcome]
    densities = []
    
    if sub_all.empty:
        return np.zeros(len(bins)-1), np.zeros(len(bins)-1)

    grouped = sub_all.groupby('sim_id')
    sim_ids = df_sim['sim_id'].unique()

    for i in sim_ids:
        if i in grouped.groups:
            grp = grouped.get_group(i)
            hist, _ = np.histogram(grp['rt'] * 25, bins=bins, density=True)
        else:
            hist = np.zeros(len(bins)-1)
        densities.append(hist)

    densities = np.array(densities)
    return np.mean(densities, axis=0), np.std(densities, axis=0)

def get_inhibition_stats(df_sim):
    """Optimized using pivot_table to calculate stop probabilities instantly."""
    stop_trials = df_sim[df_sim['result'].isin(['SS', 'SE'])].copy()
    if stop_trials.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    stop_trials['is_ss'] = (stop_trials['result'] == 'SS').astype(int)
    
    # Automatically calculates means across simulations matching the same SSD
    prob_per_sim = stop_trials.pivot_table(index='sim_id', columns='ssd', values='is_ss', aggfunc='mean')
    
    return prob_per_sim.mean(axis=0), prob_per_sim.std(axis=0).fillna(0)

def _clean_spines(ax):
    """Applies PLOS style cleaning to axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(STYLE['axis_lw'])
    ax.spines['bottom'].set_linewidth(STYLE['axis_lw'])
    ax.tick_params(width=STYLE['axis_lw'], labelsize=STYLE['tick_fontsize'])

def _get_prob_ss(df):
    """Calculates empirical probability of stop success by SSD."""
    stop_df = df[df['result'].isin(['SS', 'SE'])].copy()
    if stop_df.empty:
        return pd.Series(dtype=float)
    stop_df['is_ss'] = (stop_df['result'] == 'SS').astype(int)
    return stop_df.groupby('ssd')['is_ss'].mean()


# --- Plotting Functions ---

def plot_model_fit(subjects, raw_data_dir, params_file, output_path):
    """Generates the 5x3 Posterior Predictive Check plot grid."""
    fig, axs = plt.subplots(5, 3, figsize=(7.5, 8.5), constrained_layout=True)
    
    step_size_ms = 25
    panel_labels = ['A', 'B', 'C', 'D', 'E']

    for col_idx, subj_info in enumerate(subjects):
        sid = subj_info['sid']
        print(f"\nProcessing Subject: {sid}...")
        
        # 1. Prepare Data
        params = get_subject_params(params_file, sid)
        file_path = raw_data_dir / f"{sid}.csv"
            
        if not file_path.exists():
            print(f"  -> WARNING: Missing raw trial data for {sid} in {raw_data_dir}.")
            continue
            
        if not params:
            print(f"  -> WARNING: Missing posterior parameters for {sid}.")
            continue
            
        df_obs = pd.read_csv(file_path)
        df_sim = run_simulation_multi(params, n_repeat=N_REPEAT)

        # -----------------------------------------------------------
        # Row 0 (A): Outcome Rates
        # -----------------------------------------------------------
        ax = axs[0, col_idx]
        outcomes = ['GS', 'GE', 'GM', 'SS', 'SE']
        rates_obs = [df_obs['result'].value_counts(normalize=True).get(o, 0) for o in outcomes]
        mean_sim, std_sim = get_outcome_stats(df_sim)
        
        x = np.arange(len(outcomes))
        width = 0.35
        
        ax.bar(x - width/2, rates_obs, width, label='Obs', color=COLOR_OBS, alpha=0.7)
        ax.bar(x + width/2, mean_sim, width, yerr=std_sim, capsize=2, 
               label='Sim', color=COLOR_SIM, alpha=0.7, 
               error_kw={'ecolor': '#004d40', 'elinewidth': 0.8})
        
        ax.set_xticks(x)
        ax.set_xticklabels(outcomes, fontsize=STYLE['tick_fontsize'])
        ax.set_ylim(0, 1.05)
        ax.set_title(subj_info['label'], fontsize=STYLE['title_fontsize'], pad=4)
        
        if col_idx == 0: 
            ax.set_ylabel("Probability Mass", fontsize=STYLE['label_fontsize'])
            ax.text(-0.35, 1.05, panel_labels[0], transform=ax.transAxes, 
                    fontsize=STYLE['panel_label_size'], fontweight='bold', va='top', ha='right')
        _clean_spines(ax)

        # -----------------------------------------------------------
        # Row 1 & 2 (B & C): RT Distributions
        # -----------------------------------------------------------
        bins_ms = np.linspace(0, 1000, 41)
        bin_centers = (bins_ms[:-1] + bins_ms[1:]) / 2
        
        for row_offset, result_type in enumerate(['GS', 'SE']):
            row_idx = 1 + row_offset
            ax = axs[row_idx, col_idx]
            
            obs_data = df_obs[df_obs['result'] == result_type]['rt'] * step_size_ms
            
            ax.hist(obs_data, bins=bins_ms, density=True, 
                    alpha=0.3, color=COLOR_OBS, ec='white', linewidth=0.3)
            
            one_sim_rt = df_sim[(df_sim['sim_id'] == 0) & (df_sim['result'] == result_type)]['rt'] * step_size_ms
            ax.hist(one_sim_rt, bins=bins_ms, density=True, 
                    histtype='step', linewidth=0.6, color=COLOR_SIM, alpha=0.6)

            mean_dens, std_dens = get_rt_dist_stats(df_sim, result_type, bins_ms)
            smooth_mean = gaussian_filter1d(mean_dens, sigma=1.0)
            smooth_upper = gaussian_filter1d(mean_dens + std_dens, sigma=1.0)
            smooth_lower = gaussian_filter1d(np.maximum(0, mean_dens - std_dens), sigma=1.0)

            ax.plot(bin_centers, smooth_mean, color=COLOR_SIM, linewidth=STYLE['mean_lw'])
            ax.fill_between(bin_centers, smooth_lower, smooth_upper, 
                            color=COLOR_SIM, alpha=0.15, linewidth=0)
            
            ax.set_xlim(0, 1000)
            ax.set_xlabel("Time (ms)", fontsize=STYLE['label_fontsize'])

            if col_idx == 0: 
                ax.set_ylabel(f"{result_type} RT Density", fontsize=STYLE['label_fontsize'])
                ax.text(-0.35, 1.05, panel_labels[row_idx], transform=ax.transAxes, 
                        fontsize=STYLE['panel_label_size'], fontweight='bold', va='top', ha='right')
            _clean_spines(ax)

        # -----------------------------------------------------------
        # Row 3 (D): SSD Trajectory
        # -----------------------------------------------------------
        ax = axs[3, col_idx]
        
        ax.scatter(df_obs.index + 1, df_obs['ssd'] * step_size_ms, 
                   color=COLOR_OBS, s=STYLE['marker_size'], alpha=0.6, zorder=10)
        
        # Pre-fill SSDs across the whole grouped dataset ONCE (Massive speed up)
        df_sim['ssd_filled'] = df_sim.groupby('sim_id')['ssd'].ffill() * step_size_ms

        for i, sub in df_sim.groupby('sim_id'):
            is_highlight = i < N_TRACES_HIGHLIGHT
            alpha_val = 0.8 if is_highlight else 0.2
            lw_val = STYLE['sim_trace_lw'] if is_highlight else 0.2
            ax.step(sub['trial'], sub['ssd_filled'], 
                    color=COLOR_SIM, alpha=alpha_val, linewidth=lw_val, where='post')

        ax.set_xlim(0, 360)
        ax.set_xticks([1, 120, 240, 360])
        ax.set_yticks([0, 400, 800]) 
        ax.set_ylim(-25, 850)
        ax.set_xlabel("Trial", fontsize=STYLE['label_fontsize'])
        
        if col_idx == 0: 
            ax.set_ylabel("SSD (ms)", fontsize=STYLE['label_fontsize'])
            ax.text(-0.35, 1.05, panel_labels[3], transform=ax.transAxes, 
                    fontsize=STYLE['panel_label_size'], fontweight='bold', va='top', ha='right')
        _clean_spines(ax)

        # -----------------------------------------------------------
        # Row 4 (E): Inhibition Function
        # -----------------------------------------------------------
        ax = axs[4, col_idx]
        mean_p, std_p = get_inhibition_stats(df_sim)
        ssd_vals_ms = mean_p.index * step_size_ms
        
        ms = 4  
        mew = 0.5 
        
        ax.fill_between(ssd_vals_ms, 
                        np.clip(mean_p - std_p, 0, 1), 
                        np.clip(mean_p + std_p, 0, 1),
                        color=COLOR_SIM, alpha=0.15, linewidth=0)
        ax.plot(ssd_vals_ms, mean_p, 'o-', color=COLOR_SIM, 
                linewidth=STYLE['mean_lw'], markersize=ms, 
                markeredgecolor='white', markeredgewidth=mew)
        
        prob_obs = _get_prob_ss(df_obs)
        if not prob_obs.empty:
            ax.plot(prob_obs.index * step_size_ms, prob_obs.values, 'o-', 
                    color=COLOR_OBS, linewidth=1.0, markersize=ms, 
                    markeredgecolor='white', markeredgewidth=mew)
        
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-25, 850)
        ax.set_xlabel("SSD (ms)", fontsize=STYLE['label_fontsize'])
        
        if col_idx == 0: 
            ax.set_ylabel("P(Stop Success)", fontsize=STYLE['label_fontsize'])
            ax.text(-0.35, 1.05, panel_labels[4], transform=ax.transAxes, 
                    fontsize=STYLE['panel_label_size'], fontweight='bold', va='top', ha='right')
        _clean_spines(ax)

    # -----------------------------------------------------------
    # Global Legend
    # -----------------------------------------------------------
    legend_elements = [
        Patch(facecolor=COLOR_OBS, alpha=0.7, label='Observation'),
        Patch(facecolor=COLOR_SIM, alpha=0.8, label='Simulation')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 1.04), ncol=2, 
               frameon=False, fontsize=STYLE['legend_fontsize'], 
               columnspacing=2.0, handlelength=1.5)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    print(f"\nGrid plot successfully saved to: {output_path}")


def main():
    try:
        subjects = select_representative_subjects(PPC_METRICS_CSV, FILTER_CSV)
        plot_model_fit(subjects, RAW_DATA_DIR, PARAMS_CSV, OUTPUT_IMAGE)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()