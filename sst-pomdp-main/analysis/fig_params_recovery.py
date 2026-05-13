import sys
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd
import traceback

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================

# --- Directory Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"

# Ensure output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Input/Output Files
RECOVERY_CSV = DATA_DIR / "example_params_recovery.csv"
OUTPUT_PLOT = OUT_DIR / "fig_params_recovery.png" 

# Parameters Config
PARAMS = [
    "q_d", "q_s", "cost_stop_error",
    "q_d_n", "q_s_n", "inv_temp",
]

# LaTeX formatted names
PARAM_DISPLAY_NAMES = {
    'q_d_n': r"$\chi'$",
    'q_d': r"$\chi$",
    'q_s_n': r"$\delta'$",
    'q_s': r"$\delta$",
    'cost_stop_error': r"$c_{\mathrm{se}}$",
    "inv_temp": r"$\varphi$",
}

# Plot Settings
N_COLS = 3 
MAX_POINTS = 2000 # Downsample for plotting speed if N is huge

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
    
    "scatter_color": "#2F2F2F", # Dark Charcoal
    "scatter_alpha": 0.4,
    "scatter_size": 12,
    
    "line_color": "black",
    "line_width": 0.8,
    "line_alpha": 0.5,
    "line_style": "--",
}

# ==========================================
# 3. CORE FUNCTIONS
# ==========================================

def load_data(csv_path):
    """Loads the recovery details CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Recovery file not found: {csv_path.name}. "
            "Please ensure the mock data is generated in the 'data' folder."
        )
    df = pd.read_csv(csv_path)
    return df

def get_stats(x, y):
    """Computes Pearson r and RMSE."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 2:
        return np.nan, np.nan
    
    r, p = pearsonr(x, y)
    rmse = np.sqrt(np.mean((x - y)**2))
    return r, rmse

def plot_recovery_grid(df, params, output_path):
    """
    Generates a publication-quality grid of True vs. Recovered parameters.
    """
    n_params = len(params)
    n_rows = int(np.ceil(n_params / N_COLS))

    # Calculate figure size: PLOS width is ~7.5 inches. 
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(7.5, n_rows * 2.6), dpi=300)
    axes = axes.flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        
        # 1. Extract Data
        col_true = f"gt_{param}"
        col_rec = f"mu_{param}"

        if col_true not in df.columns or col_rec not in df.columns:
            print(f"Warning: Missing columns for parameter {param}. Skipping.")
            ax.axis('off')
            continue

        x = df[col_true].to_numpy(float)
        y = df[col_rec].to_numpy(float)

        # 2. Stats & Filtering
        r_val, rmse_val = get_stats(x, y)
        
        # Print stats to console
        print(f"Parameter: {param:15s} | r = {r_val:.3f} | RMSE = {rmse_val:.3f}")

        # Downsample for visualization if too dense
        if MAX_POINTS is not None and len(x) > MAX_POINTS:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(x), size=MAX_POINTS, replace=False)
            x_plot, y_plot = x[idx], y[idx]
        else:
            x_plot, y_plot = x, y

        # 3. Determine Axis Limits & Ticks
        # =========================================================
        
        # Default calculations
        all_vals = np.concatenate([x, y])
        v_min, v_max = np.min(all_vals), np.max(all_vals)
        pad = (v_max - v_min) * 0.05
        if pad == 0: pad = 0.1
        lims = [v_min - pad, v_max + pad]
        current_ticks = None

        # --- Special Handling for Chi and Delta ---
        if param in ['q_d', 'q_s']:
            # Force range [0.48, 1.02] and explicit ticks including 0.5
            lims = [0.48, 1.02]
            current_ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # --- Special Handling for Chi' and Delta' ---
        elif param in ['q_d_n', 'q_s_n']:
            # Force range [-0.02, 1.02] for cleaner look
            lims = [-0.02, 1.02]
            current_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # Apply settings
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        if current_ticks is not None:
            ax.set_xticks(current_ticks)
            ax.set_yticks(current_ticks)

        # 4. Plot Identity Line (Diagonal)
        ax.plot(lims, lims, linestyle=STYLE['line_style'], color=STYLE['line_color'], 
                linewidth=STYLE['line_width'], alpha=STYLE['line_alpha'], zorder=1)

        # 5. Plot Scatter
        ax.scatter(x_plot, y_plot, 
                   s=STYLE['scatter_size'], 
                   color=STYLE['scatter_color'], 
                   alpha=STYLE['scatter_alpha'], 
                   edgecolors='none', 
                   zorder=2)

        # 6. Annotation (Stats)
        stats_text = f"$r = {r_val:.3f}$\n$RMSE = {rmse_val:.3f}$"
        at = AnchoredText(stats_text, prop=dict(size=STYLE['tick_fontsize']), 
                          frameon=False, loc='upper left')
        ax.add_artist(at)

        # 7. Formatting
        display_name = PARAM_DISPLAY_NAMES.get(param, param)
        
        ax.set_xlabel(f"True {display_name}", fontsize=STYLE['label_fontsize'])
        ax.set_ylabel(f"Recovered {display_name}", fontsize=STYLE['label_fontsize'])
        
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize=STYLE['tick_fontsize'])
        
        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35, hspace=0.4) 

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    try:
        df_rec = load_data(RECOVERY_CSV)
        plot_recovery_grid(df_rec, PARAMS, OUTPUT_PLOT)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()