"""
fig_params_recovery_10p.py
Generates the parameter recovery grid plot for the 10-parameter model.
"""
import sys
import argparse
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd
import traceback

# ==========================================
# 1. CONFIGURATION
# ==========================================
# 10 Parameters (Added tau)
PARAMS = [
    "q_d_n", "q_d", "q_s_n", "q_s", "tau",
    "inv_temp", "cost_stop_error", 
    "cost_time", "cost_go_error", "cost_go_missing"
]

LOG_PARAMS = [
    "inv_temp", "cost_stop_error", 
    "cost_time", "cost_go_error", "cost_go_missing"
]

PARAM_DISPLAY_NAMES = {
    'q_d_n': r"$\chi'$",
    'q_d': r"$\chi$",
    'q_s_n': r"$\delta'$",
    'q_s': r"$\delta$",
    'tau': r"$\tau$",
    'inv_temp': r"$\varphi$",
    'cost_stop_error': r"$c_{\mathrm{se}}$",
    'cost_time': r"$c_{\mathrm{time}}$",
    'cost_go_error': r"$c_{\mathrm{ge}}$",
    'cost_go_missing': r"$c_{\mathrm{gm}}$",
}

N_COLS = 3 
MAX_POINTS = 2000

# ==========================================
# 2. PLOT STYLING SETUP
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

STYLE = {
    "label_fontsize": 10,
    "title_fontsize": 11,
    "tick_fontsize": 9,
    "panel_label_size": 12,
    
    "scatter_color": "#2F2F2F", 
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
    if not csv_path.exists():
        raise FileNotFoundError(f"Recovery file not found: {csv_path}")
    return pd.read_csv(csv_path)

def remove_outliers(x, y, lower_pct=1.0, upper_pct=99.0):
    mask_valid = np.isfinite(x) & np.isfinite(y)
    x = x[mask_valid]
    y = y[mask_valid]
    
    if len(y) < 2:
        return x, y
        
    y_low = np.percentile(y, lower_pct)
    y_high = np.percentile(y, upper_pct)
    
    mask_outliers = (y >= y_low) & (y <= y_high)
    
    return x[mask_outliers], y[mask_outliers]

def get_stats(x, y, is_log=False):
    if len(x) < 2:
        return np.nan, np.nan
        
    if is_log:
        mask = (x > 0) & (y > 0)
        x_log = np.log10(x[mask])
        y_log = np.log10(y[mask])
        if len(x_log) < 2:
            return np.nan, np.nan
        r, _ = pearsonr(x_log, y_log)
        rmse = np.sqrt(np.mean((x_log - y_log)**2))
    else:
        r, _ = pearsonr(x, y)
        rmse = np.sqrt(np.mean((x - y)**2))
        
    return r, rmse

def plot_recovery_grid(df, params, output_path):
    n_params = len(params)
    n_rows = int(np.ceil(n_params / N_COLS))

    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(7.5, n_rows * 2.6), dpi=300)
    axes = axes.flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        
        col_true = f"gt_{param}"
        col_rec = f"mu_{param}"

        if col_true not in df.columns or col_rec not in df.columns:
            print(f"Warning: Missing columns for {param}. Skipping.")
            ax.axis('off')
            continue

        x_raw = df[col_true].to_numpy(float)
        y_raw = df[col_rec].to_numpy(float)

        x, y = remove_outliers(x_raw, y_raw, lower_pct=5.0, upper_pct=95.0)

        is_log_param = param in LOG_PARAMS
        r_val, rmse_val = get_stats(x, y, is_log=is_log_param)
        
        print(f"Parameter: {param:15s} | Retained: {len(x)}/{len(x_raw)} | r = {r_val:.3f} | RMSE = {rmse_val:.3f}")

        if MAX_POINTS is not None and len(x) > MAX_POINTS:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(x), size=MAX_POINTS, replace=False)
            x_plot, y_plot = x[idx], y[idx]
        else:
            x_plot, y_plot = x, y

        all_vals = np.concatenate([x_plot, y_plot])
        
        if is_log_param:
            valid_vals = all_vals[all_vals > 0]
            v_min = np.min(valid_vals) if len(valid_vals) > 0 else 1e-4
            v_max = np.max(valid_vals) if len(valid_vals) > 0 else 1.0
            lims = [v_min / 1.5, v_max * 1.5]
            ax.set_xscale('log')
            ax.set_yscale('log')
            current_ticks = None
        else:
            v_min, v_max = np.min(all_vals), np.max(all_vals)
            pad = (v_max - v_min) * 0.05
            if pad == 0: pad = 0.1
            lims = [v_min - pad, v_max + pad]
            current_ticks = None

            if param in ['q_d', 'q_s']:
                lims = [0.48, 1.02]
                current_ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            elif param in ['q_d_n', 'q_s_n']:
                lims = [-0.02, 1.02]
                current_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            elif param == 'tau':
                lims = [2.0, 18.0]
                current_ticks = [4, 8, 12, 16]

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        if current_ticks is not None:
            ax.set_xticks(current_ticks)
            ax.set_yticks(current_ticks)

        ax.plot(lims, lims, linestyle=STYLE['line_style'], color=STYLE['line_color'], 
                linewidth=STYLE['line_width'], alpha=STYLE['line_alpha'], zorder=1)

        ax.scatter(x_plot, y_plot, 
                   s=STYLE['scatter_size'], 
                   color=STYLE['scatter_color'], 
                   alpha=STYLE['scatter_alpha'], 
                   edgecolors='none', 
                   zorder=2)

        stats_text = f"$r = {r_val:.3f}$\n$RMSE = {rmse_val:.3f}$"
        if is_log_param:
            stats_text += "\n(log scale)"
            
        at = AnchoredText(stats_text, prop=dict(size=STYLE['tick_fontsize']), 
                          frameon=False, loc='upper left')
        ax.add_artist(at)

        display_name = PARAM_DISPLAY_NAMES.get(param, param)
        
        ax.set_xlabel(f"True {display_name}", fontsize=STYLE['label_fontsize'])
        ax.set_ylabel(f"Recovered {display_name}", fontsize=STYLE['label_fontsize'])
        
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', which='major', labelsize=STYLE['tick_fontsize'])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35, hspace=0.4) 

    if output_path:
        # Create parent directories if they don't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
        print(f"\nPlot saved to: {output_path}")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 10-Parameter Recovery Grid")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory containing the recovery CSV")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    recovery_csv = out_dir / "params_recovery_10p.csv"
    output_plot = out_dir / "fig_params_recovery_10p.png"

    try:
        df_rec = load_data(recovery_csv)
        plot_recovery_grid(df_rec, PARAMS, output_plot)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()