import importlib
import sys
import argparse
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd
import traceback

temp_parser = argparse.ArgumentParser(add_help=False)
temp_parser.add_argument("--model_config", type=str, required=True)
temp_args, _ = temp_parser.parse_known_args()

m = importlib.import_module(temp_args.model_config)

PARAM_ORDER = m.PARAM_ORDER
LOG_PARAMS = m.LOG_PARAMS
PARAM_RANGES = m.PARAM_RANGES
MODEL_TAG = m.MODEL_TAG

PARAM_DISPLAY_NAMES = {
    'q_d_n': r"$\chi'$", 'q_d': r"$\chi$", 'q_s_n': r"$\delta'$", 'q_s': r"$\delta$",
    'tau': r"$\tau$", 'inv_temp': r"$\varphi$", 'cost_stop_error': r"$c_{\mathrm{se}}$",
    'cost_time': r"$c_{\mathrm{time}}$", 'cost_go_error': r"$c_{\mathrm{ge}}$", 'cost_go_missing': r"$c_{\mathrm{gm}}$",
}

STYLE = {"label_fontsize": 10, "title_fontsize": 11, "tick_fontsize": 9, "scatter_color": "#2F2F2F", "scatter_alpha": 0.4, "scatter_size": 12, "line_color": "black", "line_width": 0.8, "line_style": "--"}


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


def plot_recovery_grid(df, output_path):
    n_params = len(PARAM_ORDER)
    n_rows = int(np.ceil(n_params / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(7.5, n_rows * 2.6), dpi=300)
    axes = axes.flatten()

    for i, param in enumerate(PARAM_ORDER):
        ax = axes[i]
        col_true, col_rec = f"gt_{param}", f"mu_{param}"
        if col_true not in df.columns or col_rec not in df.columns:
            ax.axis('off'); continue

        x_raw, y_raw = df[col_true].to_numpy(float), df[col_rec].to_numpy(float)
        x, y = remove_outliers(x_raw, y_raw, lower_pct=5.0, upper_pct=95.0)

        is_log_param = param in LOG_PARAMS
        r_val, rmse_val = get_stats(x, y, is_log=is_log_param)

        x_plot, y_plot = x, y
        if len(x) > 2000:
            idx = np.random.default_rng(42).choice(len(x), size=2000, replace=False)
            x_plot, y_plot = x[idx], y[idx]

        # [FIX]: 完全使用 Prior 范围作为坐标轴
        lo, hi = PARAM_RANGES[param]
        lims = [lo, hi]
        ax.set_xlim(lims); ax.set_ylim(lims)
        
        if is_log_param:
            ax.set_xscale('log'); ax.set_yscale('log')

        ax.plot(lims, lims, linestyle=STYLE['line_style'], color=STYLE['line_color'], linewidth=STYLE['line_width'], zorder=1)
        ax.scatter(x_plot, y_plot, s=STYLE['scatter_size'], color=STYLE['scatter_color'], alpha=STYLE['scatter_alpha'], edgecolors='none', zorder=2)

        stats_text = f"$r = {r_val:.3f}$\n$RMSE = {rmse_val:.3f}$"
        if is_log_param: stats_text += "\n(log scale)"
        at = AnchoredText(stats_text, prop=dict(size=STYLE['tick_fontsize']), frameon=False, loc='upper left')
        ax.add_artist(at)

        display_name = PARAM_DISPLAY_NAMES.get(param, param)
        ax.set_xlabel(f"True {display_name}", fontsize=STYLE['label_fontsize'])
        ax.set_ylabel(f"Recovered {display_name}", fontsize=STYLE['label_fontsize'])
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', which='major', labelsize=STYLE['tick_fontsize'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    for j in range(i + 1, len(axes)): axes[j].axis("off")
    plt.tight_layout(); plt.subplots_adjust(wspace=0.35, hspace=0.4) 

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--model_config", type=str, required=True)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    recovery_csv = out_dir / f"params_recovery_{MODEL_TAG}.csv"
    output_plot = out_dir / f"fig_params_recovery_{MODEL_TAG}.png"

    df_rec = pd.read_csv(recovery_csv)
    plot_recovery_grid(df_rec, output_plot)