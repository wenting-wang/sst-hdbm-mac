import sys
import os
import csv
import math
import multiprocessing as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# --- PATH SETUP ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Add project root to sys.path to import local modules
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Custom Imports ---
from utils.metrics import get_stats_mean_sim
from core.models import POMDP
from core import simulation

# =============================================================================
# PLOS CB STYLE CONFIGURATION
# =============================================================================
# Font setup
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Style dictionary
STYLE = {
    "label_fontsize": 11,
    "legend_fontsize": 9,
    "tick_fontsize": 8,
    "line_width": 1.5,
    "axis_width": 0.8,
}

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"

# Ensure output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Input / Output Files
PARAMS_CSV = DATA_DIR / "example_params_posteriors.csv"
OUTPUT_CSV = DATA_DIR / "example_sensitivity_1d.csv"  # Intermediate simulated data
OUTPUT_IMAGE = OUT_DIR / "fig_sensitivity.png"

# Default subject to analyze (will fallback to first available if not found)
SUBJECT_ID = 'EXAMPLE_SUB_001'

# Simulation Parameters
N_REPEAT = 10
BINS = 10
MAX_WORKERS = max(1, mp.cpu_count() - 2)
EPSILON = 1e-6

FIXED_PARAMS = {
    'cost_go_error': 1.0,
    'cost_go_missing': 1.0,
    'cost_time': 0.001,
    'rate_stop_trial': 1/6
}

PARAM_RANGES = {
    "q_d":   (0.5, 1.0),
    "q_s":   (0.5, 1.0),
    "cost_stop_error": (0.0, 2.0),
    "q_d_n": (0.0, 1.0),
    "q_s_n": (0.0, 1.0),
    "inv_temp":       (20, 100),
}

DYNAMIC_PARAMS_KEYS = [
    "q_d", "q_s", "cost_stop_error",
    "q_d_n", "q_s_n", "inv_temp",
]

METRIC_NAMES = ["mrt_gs", "ssrt"]

PARAM_DISPLAY_NAMES = {
    "q_d_n": r"$\chi'$",
    "q_d": r"$\chi$",
    "q_s_n": r"$\delta'$",
    "q_s": r"$\delta$",
    "cost_stop_error": r"$c_{\mathrm{se}}$",
    "inv_temp": r"$\varphi$",
}

METRIC_DISPLAY_NAMES = {
    "mrt_gs": "GSRT",
    "ssrt": "SSRT",
}

# =============================================================================
# DATA LOADING & SIMULATION
# =============================================================================

def get_subject_params(target_id, csv_path, fixed_params):
    """Loads baseline parameters for a subject, with fallback logic for mock data."""
    if not csv_path.exists(): 
        raise FileNotFoundError(f"Missing data file: {csv_path.name}. Please generate mock data first.")
    
    df = pd.read_csv(csv_path)
    
    # Fallback logic for example/mock datasets
    if target_id not in df['subject_id'].values:
        available_id = df['subject_id'].iloc[0]
        print(f"Warning: Subject '{target_id}' not found. Using '{available_id}' instead.")
        target_id = available_id
        
    subject_df = df[df['subject_id'] == target_id]
    
    # Handle different possible column names for the parameter index
    idx_col = 'index' if 'index' in subject_df.columns else 'param'
    if idx_col not in subject_df.columns:
        raise KeyError("Could not find 'index' or 'param' column in posterior data.")
        
    param_dict = subject_df.set_index(idx_col)['mean'].to_dict()
    clean_params = {k: param_dict[k] for k in DYNAMIC_PARAMS_KEYS if k in param_dict}
    
    return {**clean_params, **fixed_params}, target_id

def safe_linspace(a, b, bins, eps=1e-3):
    lo, hi = min(a, b), max(a, b)
    return np.linspace(lo + eps, hi - eps, bins)

def simulate_and_extract_stat(param_dict, baseline_params, n_repeat, metric_names):
    stats_buffer = {m: [] for m in metric_names}
    try:
        # 1. Update Parameters
        test_params = baseline_params.copy()
        test_params.update(param_dict)
        
        # 2. Initialize Model
        pomdp = POMDP(**test_params)
        pomdp.value_iteration_tensor()

        # 3. Run Repeats
        for _ in range(n_repeat):
            out = simulation.simu_task(pomdp) 
            df_out = pd.DataFrame(out, columns=["result", "rt", "ssd"])
            stats = get_stats_mean_sim(df_out)
            stat_dict = {"mrt_gs": stats[4], "ssrt": stats[7]}
            
            for m in metric_names: 
                stats_buffer[m].append(stat_dict.get(m, np.nan))
                
    except Exception as e:
        print(f"Error for params {param_dict}: {e}") 
        for m in metric_names: 
            stats_buffer[m] = [np.nan] * n_repeat

    # Aggregate results
    rows = []
    key, val = list(param_dict.items())[0]
    for m in metric_names:
        arr = np.asarray(stats_buffer[m], dtype=float)
        rows.append({
            "param_1": key, 
            "value_1": val, 
            "metric": m,
            "mean": float(np.nanmean(arr)), 
            "std": float(np.nanstd(arr)),
        })
    return rows

def run_sensitivity_analysis(baseline_params):
    print("Starting sensitivity simulations...")
    value_grids = {k: safe_linspace(v[0], v[1], BINS, eps=EPSILON) for k, v in PARAM_RANGES.items()}
    header = ["param_1", "value_1", "metric", "mean", "std"]
    
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(simulate_and_extract_stat, {p: float(v)}, baseline_params, N_REPEAT, METRIC_NAMES)
                       for p, vals in value_grids.items() for v in vals]
            
            for i, fut in enumerate(as_completed(futures)):
                for row in fut.result(): 
                    writer.writerow(row)
                    
    print(f"Results saved to {OUTPUT_CSV}")

# =============================================================================
# PLOTTING
# =============================================================================

def plot_sensitivity_merged(df, baseline_params, save_path=None, ncols=3):
    """
    Plots GSRT and SSRT in the same subplots.
    Formatted for PLOS Computational Biology (width ~7.5 inches).
    """
    print("Generating sensitivity plots...")
    
    # Define colors
    metric_colors = {
        "mrt_gs": "#006d77", # Teal
        "ssrt":   "#e07a5f"  # Terra Cotta
    }
    
    params_in_data = list(df["param_1"].dropna().unique())
    param_order = [p for p in PARAM_RANGES.keys() if p in params_in_data]

    n_plots = len(param_order)
    nrows = math.ceil(n_plots / ncols)
    
    # PLOS Sizing: Width 7.5 inches is standard for full-width figures
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5, 2.5 * nrows), dpi=300)
    
    if nrows > 1 or ncols > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]
        
    legend_handles = []
    legend_labels = []
    legend_collected = False

    for idx, p in enumerate(param_order):
        ax = axes_flat[idx]
        
        # --- Plotting ---
        for m in METRIC_NAMES:
            sub = df[(df["param_1"] == p) & (df["metric"] == m)].copy().sort_values("value_1")
            if sub.empty: continue
            
            x = sub["value_1"].to_numpy()
            y = sub["mean"].to_numpy()
            yerr = sub["std"].to_numpy()
            color = metric_colors.get(m, "black")
            label = METRIC_DISPLAY_NAMES.get(m, m)
            
            # Plot Line
            line, = ax.plot(x, y, color=color, linewidth=STYLE['line_width'], label=label)
            # Error shading
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.15, linewidth=0)
            
            # Data points/error bars
            ax.errorbar(x, y, yerr=yerr, linestyle="none", capsize=2, color=color, alpha=0.5, marker='o', markersize=3)
            
            # Collect legend (collect only once)
            if not legend_collected:
                legend_handles.append(line)
                legend_labels.append(label)
        
        if not legend_collected:
            legend_collected = True

        # Plot Subject Baseline Vertical Line
        if p in baseline_params:
            v_line = ax.axvline(baseline_params[p], linestyle="--", color="gray", alpha=0.7, linewidth=1.0)
            if idx == 0:
                legend_handles.append(v_line)
                legend_labels.append("Benchmark")

        # --- Style & Formatting ---
        
        # Axis labels
        ax.set_xlabel(PARAM_DISPLAY_NAMES.get(p, p), fontsize=STYLE['label_fontsize'])
        
        # Show Y-axis label only for the first column
        if idx % ncols == 0:
            ax.set_ylabel("Response Time (ms)", fontsize=STYLE['label_fontsize'])
        
        # Scientific notation
        if p == "cost_time":
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        
        ax.set_ylim(0, 750)
        
        # Tick style
        ax.tick_params(axis='both', which='major', labelsize=STYLE['tick_fontsize'])
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(STYLE['axis_width'])
        ax.spines['bottom'].set_linewidth(STYLE['axis_width'])

    # Remove extra empty subplots
    for k in range(n_plots, len(axes_flat)):
        axes_flat[k].axis("off")

    # --- Top Global Legend ---
    fig.legend(
        handles=legend_handles, 
        labels=legend_labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.02), # Offset slightly upwards to avoid overlap
        ncol=3, 
        fontsize=STYLE['legend_fontsize'],
        frameon=False
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Reserve top space for legend
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        print(f"Saved plot to: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    try:
        subject_params, final_subject_id = get_subject_params(SUBJECT_ID, PARAMS_CSV, FIXED_PARAMS)
        print(f"Using parameters from subject: {final_subject_id}")
        
        # Run simulations if the intermediate data file does not exist yet
        if not os.path.exists(OUTPUT_CSV):
            run_sensitivity_analysis(subject_params)

        # Generate plot from the data
        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            df.replace(0, np.nan, inplace=True)
            
            plot_sensitivity_merged(
                df, 
                baseline_params=subject_params,
                save_path=OUTPUT_IMAGE
            )
            
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        traceback.print_exc()