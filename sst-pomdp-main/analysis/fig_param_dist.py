import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import traceback

# --- Configuration & Paths ---

# --- Directory Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"

# Ensure output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Input Files ---
PARAMS_CSV = DATA_DIR / "example_params_posteriors.csv"
STATS_CSV = DATA_DIR / "example_clinical_behavior.csv"

# --- Output ---
OUTPUT_PLOT = OUT_DIR / "fig_param_dist.png"

PLOT_PARAMS = [
    'q_d',              # Chi
    'q_s',              # Delta
    'cost_stop_error',  # C_se
    'q_d_n',            # Chi'
    'q_s_n',            # Delta'
    'inv_temp'          # Varphi
]

PARAM_DISPLAY_NAMES = {
    'q_d_n': r"$\chi'$ (Go Null)",
    'q_d': r"$\chi$ (Go Precision)",
    'q_s_n': r"$\delta'$ (Stop Null)",
    'q_s': r"$\delta$ (Stop Precision)",
    'cost_stop_error': r"$c_{\mathrm{se}}$ (Stop Error Cost)",
    'inv_temp': r"$\varphi$ (Inverse Temperature)",
}

# Plot Colors
EDGE_COLOR = "#ffffff"
FILL_COLOR = "#534F57"   # Original Grey
COLOR_MEAN = "#B03A2E"   # Deep Carmine (Red)
COLOR_MEDIAN = "#21618C" # Steel Blue (Blue)
COLOR_MODE = "#B7950B"   # Dark Ochre (Gold)

BINS = 40 

# PLOS CB Style Settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10         
plt.rcParams['axes.labelsize'] = 10    
plt.rcParams['xtick.labelsize'] = 9   
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8    

# --- Core Functions ---

def load_and_merge_data(params_path, stats_path, params_to_keep):
    """
    Loads and merges parameter data with behavioral stats, safely filtering for baseline.
    """
    missing_files = [p.name for p in [params_path, stats_path] if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing input files in 'data' folder: {', '.join(missing_files)}")

    # 1. Load Parameters
    df_param = pd.read_csv(params_path)
    if 'subject_year' in df_param.columns:
        df_param = df_param.rename(columns={"subject_year": "year"})
    
    # Force string for merging
    df_param['year'] = df_param['year'].astype(str)
    
    # Pivot logic using pivot_table to handle duplicates gracefully
    if 'index' in df_param.columns:
        df_pivot = df_param.pivot_table(index=['subject_id', 'year'], columns='index', values='mean').reset_index()
    elif 'param' in df_param.columns:
        df_pivot = df_param.pivot_table(index=['subject_id', 'year'], columns='param', values='mean').reset_index()
    else:
        raise ValueError("Cannot figure out pivot column (expected 'index' or 'param').")
        
    df_pivot.columns.name = None
    
    # 2. Load Stats
    df_stats = pd.read_csv(stats_path)
    df_stats['year'] = df_stats['year'].astype(str)
    
    # 3. Merge
    df_merged = pd.merge(df_stats, df_pivot, on=['subject_id', 'year'], how='inner')
    
    # 4. Filter for Baseline
    df_baseline = df_merged[df_merged['year'] == 'baseline'].copy()
    
    # 5. Drop NA only across parameters we actually want to plot
    avail_cols = [c for c in params_to_keep if c in df_baseline.columns]
    df_final = df_baseline.dropna(subset=avail_cols).copy()
    
    print(f"Data Loaded. Subjects (Baseline): {len(df_final)}")
    return df_final


def plot_parameter_distributions(df, params, output_path):
    """
    Generates a 2x3 grid of parameter distributions with mean, median, and mode lines.
    """
    nrows, ncols = 2, 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5, 4))
    axes = axes.flatten()

    print("\n" + "="*40)
    print("PARAMETER STATISTICS")
    print("="*40)

    for i, param in enumerate(params):
        ax = axes[i]
        
        if param not in df.columns:
            ax.text(0.5, 0.5, f"{param} missing", ha='center', va='center')
            ax.axis('off')
            continue
            
        data = df[param].dropna()
        if data.empty:
            ax.axis('off')
            continue
        
        # --- Calculate Statistics ---
        mean_val = data.mean()
        median_val = data.median()
        # Mode calculation (taking the first mode if multiple exist)
        mode_val = data.round(3).mode()
        mode_val = mode_val[0] if not mode_val.empty else median_val

        # --- Print to Console ---
        label = PARAM_DISPLAY_NAMES.get(param, param)
        print(f"Variable: {label}")
        print(f"  Mean:   {mean_val:.4f}")
        print(f"  Median: {median_val:.4f}")
        print(f"  Mode:   {mode_val:.4f}")
        print("-" * 20)

        # --- Handle Bin Widths ---
        # Lock bin_range to [0,1] for parameters starting with 'q_'
        if param.startswith('q_'):
            current_bin_range = (0, 1)
        else:
            current_bin_range = None # Default: use data min/max
        
        # --- Plot Histogram ---
        sns.histplot(data, bins=BINS, binrange=current_bin_range,
                     color=FILL_COLOR, edgecolor=EDGE_COLOR, 
                     linewidth=0.3, stat="density", ax=ax, alpha=0.4)

        # --- Add Vertical Lines ---
        ax.axvline(mean_val, color=COLOR_MEAN, linestyle='--', linewidth=1.2, zorder=5)
        ax.axvline(median_val, color=COLOR_MEDIAN, linestyle='-', linewidth=1.2, zorder=5)
        ax.axvline(mode_val, color=COLOR_MODE, linestyle='-.', linewidth=1.2, zorder=5)

        # --- Formatting ---
        ax.set_xlabel(label, labelpad=5)
        
        if param.startswith('q_'):
            ax.set_xlim(0, 1)
        
        # Y-Axis Label (Only show on left column)
        if i % ncols == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("")
            
        # --- Legend ---
        custom_lines = [
            Line2D([0], [0], color=COLOR_MEAN, linestyle='--', lw=1.2),
            Line2D([0], [0], color=COLOR_MEDIAN, linestyle='-', lw=1.2),
            Line2D([0], [0], color=COLOR_MODE, linestyle='-.', lw=1.2)
        ]
        
        ax.legend(custom_lines, 
                  [f'Mean: {mean_val:.2f}', f'Med: {median_val:.2f}', f'Mode: {mode_val:.2f}'],
                  frameon=False, loc='best')

        sns.despine(ax=ax)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.55)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

# --- Main Execution ---

def main():
    try:
        df = load_and_merge_data(PARAMS_CSV, STATS_CSV, PLOT_PARAMS)
        if df is not None and not df.empty:
            plot_parameter_distributions(df, PLOT_PARAMS, OUTPUT_PLOT)
        else:
            print("Warning: Dataset is empty after processing. No plot generated.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()