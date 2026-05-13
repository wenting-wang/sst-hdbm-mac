import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import string
import traceback

# --- Directory Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Add project root to sys.path to import local modules if needed
sys.path.append(str(PROJECT_ROOT))

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"

# Ensure output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Input Files
PPC_METRICS_CSV = DATA_DIR / "example_ppc_metrics.csv"
SUBJECT_FILTER_CSV = DATA_DIR / "example_clinical_behavior.csv"

# Output File
OUTPUT_IMAGE = OUT_DIR / "fig_ppc_population.png"

# Plotting Configuration
PLOT_COLS = [
    "dis_perc_gs", 
    "dis_ws_rt_gs", 
    "dis_ks_rt_gs",
    "dis_perc_ss",
    "dis_ws_rt_se",
    "dis_ks_rt_se"
]

# LaTeX formatted titles for X-axis
PLOT_TITLES = {
    "dis_perc_gs": r"Abs. Diff. $|P(\text{GS})_{\text{obs}} - P(\text{GS})_{\text{sim}}|$",
    "dis_perc_ss": r"Abs. Diff. $|P(\text{SS})_{\text{obs}} - P(\text{SS})_{\text{sim}}|$",
    "dis_ws_rt_gs": "Wasserstein Distance (GS RT)", 
    "dis_ws_rt_se": "Wasserstein Distance (SE RT)",
    "dis_ks_rt_gs": "K-S Distance (GS RT)",
    "dis_ks_rt_se": "K-S Distance (SE RT)",
}

# ==========================================
# 2. PLOS STYLING SETUP
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

STYLE = {
    "label_fontsize": 10,
    "title_fontsize": 11,
    "tick_fontsize": 9,
    "panel_label_size": 12,
    
    # Histogram specific style matching the recovery plot theme
    "hist_face_color": "#2F2F2F",  # Dark Charcoal
    "hist_alpha": 0.3,             # Light transparency
    "hist_edge_color": "white",    # White edges for contrast
    "line_width": 1.0,
}

# ==========================================
# 3. CORE FUNCTIONS
# ==========================================

def load_and_filter_data(ppc_path, filter_path):
    """
    Loads PPC metrics and filters for subjects present in the validated dataset (baseline only).
    """
    missing_files = [p.name for p in [ppc_path, filter_path] if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required data files in 'data' folder: {', '.join(missing_files)}")

    print("Loading data...")
    df_ppc = pd.read_csv(ppc_path)
    df_filter = pd.read_csv(filter_path, index_col=False)

    # 1. Select only baseline subjects from the filter file
    if 'year' in df_filter.columns:
        df_filter['year'] = df_filter['year'].astype(str)
        baseline_subjects = df_filter[df_filter['year'] == 'baseline']['subject_id'].unique()
    else:
        print("Warning: 'year' column not found in clinical behavior data. Proceeding with all subjects.")
        baseline_subjects = df_filter['subject_id'].unique()
    
    # 2. Filter PPC data to include only these subjects
    df_clean = df_ppc[df_ppc['subject_id'].isin(baseline_subjects)].copy()
    
    print(f"Loaded {len(df_ppc)} PPC records.")
    print(f"Filtered down to {len(df_clean)} valid baseline subjects.")
    
    return df_clean


def plot_ppc_histograms(df, output_path):
    """
    Generates and saves a 2x3 grid of density histograms for PPC metrics.
    Compliant with PLOS CB standards (7.5 inch width, panel labels, fonts).
    """
    print("Generating plots...")
    
    # Setup Figure: PLOS width is ~7.5 inches. 
    # 2 rows, 3 columns. Height adjusted to maintain good aspect ratio.
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7.5, 5), dpi=300)
    axes_flat = axes.flatten()
    
    panel_labels = list(string.ascii_uppercase) # A, B, C...

    for i, col in enumerate(PLOT_COLS):
        ax = axes_flat[i]
        
        # 0. Panel Label (A, B, C...)
        # Placed in the upper-left, outside the plot area
        ax.text(-0.25, 1.1, panel_labels[i], transform=ax.transAxes,
                fontsize=STYLE['panel_label_size'], fontweight='bold', va='top', ha='right')

        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. Skipping.")
            ax.axis('off')
            continue

        # 1. Data
        data = df[col].dropna()
        if data.empty:
            ax.axis('off')
            continue

        # 2. Define Explicit Bins based on X-axis limits
        if col in ["dis_perc_gs", "dis_perc_ss"]:
            x_max = 0.4
        elif col in ["dis_ws_rt_gs", "dis_ws_rt_se"]:
            x_max = 0.5
        elif col in ["dis_ks_rt_gs", "dis_ks_rt_se"]:
            x_max = 1.0
        else:
            x_max = data.max() # Fallback

        # Create 40 equally spaced bin edges from 0 to x_max
        fixed_bins = np.linspace(0, x_max, 40)

        # 3. Plot Histogram (Density)
        ax.hist(
            data,
            bins=fixed_bins,  
            density=True,  
            color=STYLE['hist_face_color'],
            alpha=STYLE['hist_alpha'],
            edgecolor=STYLE['hist_edge_color'],
            linewidth=STYLE['line_width']
        )

        # 4. Styling
        
        # X-Axis: The Metric Name (Title moved here)
        label_text = PLOT_TITLES.get(col, col)
        ax.set_xlabel(label_text, fontsize=STYLE['label_fontsize'])
        
        # X-Axis Limits
        if col in ["dis_perc_gs", "dis_perc_ss"]:
            ax.set_xlim(0, 0.4)
        elif col in ["dis_ws_rt_gs", "dis_ws_rt_se"]:
            ax.set_xlim(0, 0.6)
        elif col in ["dis_ks_rt_gs", "dis_ks_rt_se"]:
            ax.set_xlim(0, 0.9)
        
        # Y-Axis
        ax.set_ylabel("Probability Density", fontsize=STYLE['label_fontsize'])
        
        # Axis cleanup (Tufte/PLOS style)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

        # Ticks
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4)) # Limit ticks
        ax.tick_params(axis='both', which='major', labelsize=STYLE['tick_fontsize'])
        
        ax.grid(False)

    # Hide any unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.45, hspace=0.6) 
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    print(f"Plot saved to: {output_path}")


# --- Main Execution ---

def main():
    try:
        # 1. Load Data
        df = load_and_filter_data(PPC_METRICS_CSV, SUBJECT_FILTER_CSV)
        
        if df.empty:
            print("Warning: No valid data found after filtering. Plot will not be generated.")
            return

        # 2. Plot
        plot_ppc_histograms(df, OUTPUT_IMAGE)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()