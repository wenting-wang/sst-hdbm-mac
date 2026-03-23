import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from pathlib import Path
import sys

# ================= CONFIGURATION =================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

INPUT_FILE = BASE_DIR / 'data' / 'orders.csv'

# Output Data
GLOBAL_HAZARD_CSV = BASE_DIR / 'data' / 'fitted_hazard_global.csv'
ORDER_HAZARD_CSV = BASE_DIR / 'data' / 'fitted_hazard_per_order.csv'

# Output Plots
GLOBAL_IMG = BASE_DIR / 'outputs' / 'hazard_global.png'
ORDER_GRID_IMG = BASE_DIR / 'outputs' / 'hazard_order.png'

# --- PLOS Computational Biology Formatting ---
# Max Width: 7.5 inches (19.05 cm)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

LABEL_SIZE = 10       
TICK_SIZE = 8         
LEGEND_SIZE = 9       
PANEL_LABEL_SIZE = 12 

# ===============================================

# --- Discrete Weibull Functions ---
def get_dw_pmf(q, beta, x_vals):
    term1 = np.power(q, np.power(x_vals - 1, beta))
    term2 = np.power(q, np.power(x_vals, beta))
    return term1 - term2

def get_dw_hazard(q, beta, x_vals):
    x = x_vals - 1 
    term = np.power(x + 1, beta) - np.power(x, beta)
    return 1 - np.power(q, term)

def fit_discrete_weibull(data):
    def dw_neg_log_likelihood(params, data):
        q, beta = params
        if q <= 0.001 or q >= 0.999 or beta <= 0.01: return 1e10
        term1 = np.power(data, beta)
        term2 = np.power(data + 1, beta)
        q_delta = np.power(q, term2 - term1)
        term_inside = 1 - q_delta
        term_inside[term_inside < 1e-10] = 1e-10
        return -np.sum(term1 * np.log(q) + np.log(term_inside))

    initial_guess = [0.8, 1.5]
    bounds = [(0.001, 0.999), (0.1, 10.0)]
    try:
        res = minimize(dw_neg_log_likelihood, initial_guess, args=(data,), 
                       bounds=bounds, method='L-BFGS-B')
        return res.x
    except:
        return [0.5, 1.0]

# --- Data Processing Functions ---
def extract_runs_from_seq(seq):
    segments = seq.split('1')[:-1] 
    return [len(s) for s in segments if len(s) > 0]

def calculate_empirical_stats(runs, max_x=20):
    runs = np.array(runs)
    stats = []
    for x in range(1, max_x + 1):
        n_at_risk = np.sum(runs >= x)
        n_events = np.sum(runs == x)
        if n_at_risk > 0:
            h = n_events / n_at_risk
            se = np.sqrt(h * (1 - h) / n_at_risk) if n_at_risk > 1 else 0
        else:
            h, se = np.nan, np.nan
        stats.append({
            'go_run_length': x, 
            'n_at_risk': n_at_risk, 
            'n_events': n_events, 
            'empirical_hazard': h, 
            'se': se
        })
    return pd.DataFrame(stats)

def main():
    print(f">>> 1. Loading data from {INPUT_FILE}...")
    try:
        df_orders = pd.read_csv(INPUT_FILE, dtype={'order_seq': str})
    except FileNotFoundError:
        print(f"Error: Cannot find '{INPUT_FILE}'.")
        return

    all_global_runs = []
    order_info_dict = {}

    for _, row in df_orders.iterrows():
        order_id = row['order_sig']
        seq = row['order_seq']
        subj_cnt = int(row['subj_cnt'])
        
        base_runs = extract_runs_from_seq(seq)
        repeated_runs = base_runs * subj_cnt
        
        all_global_runs.extend(repeated_runs)
        
        if order_id not in order_info_dict:
            order_info_dict[order_id] = {'runs': [], 'n_subj': subj_cnt}
        order_info_dict[order_id]['runs'].extend(repeated_runs)

    all_global_runs = np.array(all_global_runs)
    
    # ================= FITTING =================
    print(">>> 2. Fitting Discrete Weibull Models...")
    q_global, beta_global = fit_discrete_weibull(all_global_runs - 1)
    
    order_fits = {}
    order_empirical = {}
    order_hazard_records = []
    max_x = 20
    x_plot = np.arange(1, max_x + 1)
    
    for oid, info in order_info_dict.items():
        runs_arr = np.array(info['runs'])
        if len(runs_arr) < 10: continue
            
        q_f, beta_f = fit_discrete_weibull(runs_arr - 1)
        order_fits[oid] = (q_f, beta_f)
        
        df_emp = calculate_empirical_stats(runs_arr, max_x)
        order_empirical[oid] = df_emp
        
        h_vals = get_dw_hazard(q_f, beta_f, x_plot)
        for x, h in zip(x_plot, h_vals):
            order_hazard_records.append({
                'order_id': oid,
                'go_run_length': x,
                'fitted_hazard': h,
                'q_param': q_f,
                'beta_param': beta_f
            })

    # ================= SAVING DATA =================
    print(">>> 3. Exporting Data...")
    GLOBAL_HAZARD_CSV.parent.mkdir(parents=True, exist_ok=True)
    ORDER_GRID_IMG.parent.mkdir(parents=True, exist_ok=True)
    
    df_global_stats = calculate_empirical_stats(all_global_runs, max_x)
    df_global_stats['fitted_hazard'] = get_dw_hazard(q_global, beta_global, x_plot)
    df_global_stats['fitted_pmf'] = get_dw_pmf(q_global, beta_global, x_plot)
    df_global_stats.to_csv(GLOBAL_HAZARD_CSV, index=False)
    pd.DataFrame(order_hazard_records).to_csv(ORDER_HAZARD_CSV, index=False)

    # ================= PLOTTING =================
    print(">>> 4. Generating PLOS-compliant Figures...")
    
    # ----- Plot 1: Global Panel (A & B) -----
    fig1, axes1 = plt.subplots(1, 2, figsize=(7.5, 4.0))
    plt.subplots_adjust(wspace=0.25, bottom=0.15, left=0.08, right=0.98, top=0.88)
    
    ax0 = axes1[0]
    hist_data = all_global_runs[all_global_runs <= max_x]
    bins = np.arange(0.5, max_x + 1.5, 1)
    
    ax0.hist(hist_data, bins=bins, density=True, color='#e0e0e0', edgecolor='white', linewidth=0.5, label='Empirical Density')
    line_pdf, = ax0.plot(x_plot, df_global_stats['fitted_pmf'], color="#333333", linewidth=1.5, label='Fitted Density')
    
    ax0.set_xlabel('Go Run', fontsize=LABEL_SIZE)
    ax0.set_ylabel('Probability Density', fontsize=LABEL_SIZE)
    ax0.set_xlim(0, 21)
    ax0.set_ylim(0, ax0.get_ylim()[1]) 
    ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax0.set_xticks([1, 5, 10, 15, 20])
    ax0.legend(frameon=False, fontsize=LEGEND_SIZE, loc='upper right')

    ax1 = axes1[1]
    for oid, (q_f, beta_f) in order_fits.items():
        h_vals = get_dw_hazard(q_f, beta_f, x_plot)
        ax1.plot(x_plot, h_vals, color='#b51f1f', alpha=0.3, linewidth=0.8)
    
    line_haz, = ax1.plot(x_plot, df_global_stats['fitted_hazard'], color="#b51f1f", linewidth=1.5, label='Fitted Hazard')
    mask = (df_global_stats['n_at_risk'] > 0) & (df_global_stats['go_run_length'] <= 20)
    err_emp = ax1.errorbar(df_global_stats[mask]['go_run_length'], df_global_stats[mask]['empirical_hazard'], 
                           yerr=df_global_stats[mask]['se'], fmt='o', color='black', 
                           markersize=3.5, capsize=1.5, elinewidth=0.8, label='Empirical Hazard')
    
    ax1.set_xlabel('Go Run', fontsize=LABEL_SIZE)
    ax1.set_ylabel('Hazard', fontsize=LABEL_SIZE)
    # Added negative buffer to left primary axis
    ax1.set_xlim(0, 21)
    ax1.set_ylim(-0.05, 1.05) 
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xticks([1, 5, 10, 15, 20])
    # Legend update 
    # ax1.legend([err_emp, line_haz], ['Empirical Hazard', 'Fitted Hazard'], frameon=False, fontsize=LEGEND_SIZE, loc='lower right')
    ax1.legend([err_emp, line_haz], ['Empirical Hazard', 'Fitted Hazard'], 
           frameon=False, fontsize=LEGEND_SIZE, 
           loc='lower right', bbox_to_anchor=(1.02, -0.02))

    for ax, label in zip(axes1, ['A', 'B']):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(width=0.8, labelsize=TICK_SIZE)
        ax.text(-0.15, 1.05, label, transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='right')

    fig1.savefig(GLOBAL_IMG, dpi=300, bbox_inches='tight')

    # ----- Plot 2: Per-Order Grid (4x4) -----
    fig2, axes2 = plt.subplots(4, 4, figsize=(7.5, 7.5))
    
    # Increased wspace slightly to 0.25 to make subplots wider
    plt.subplots_adjust(wspace=0.25, hspace=0.45, bottom=0.08, left=0.08, right=0.92, top=0.90)
    axes2 = axes2.flatten()
    
    for idx, (oid, (q_f, beta_f)) in enumerate(order_fits.items()):
        ax = axes2[idx]
        df_emp = order_empirical[oid]
        runs_arr = np.array(order_info_dict[oid]['runs'])
        n_subj = order_info_dict[oid]['n_subj']
        
        order_num = int(oid.split('_')[1]) + 1
        title_str = f"Order {order_num} (N={n_subj})"
        
        # 1. Twin axis for histogram (Bottom Layer)
        ax2 = ax.twinx()
        hist_data = runs_arr[runs_arr <= max_x]
        ax2.hist(hist_data, bins=bins, density=True, color='#e0e0e0', edgecolor='none', zorder=1)
        
        # Secondary Y-axis styling (Density) - Zero buffer at the bottom
        ax2.set_ylim(0, 0.45) 
        ax2.set_yticks([0, 0.2, 0.4])
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        
        ax2.spines['right'].set_visible(True)
        ax2.spines['right'].set_color('grey')
        ax2.spines['right'].set_linewidth(0.6)
        
        if idx % 4 == 3:
            ax2.tick_params(axis='y', right=True, labelright=True, labelsize=7, colors='grey', width=0.6)
        else:
            ax2.tick_params(axis='y', right=True, labelright=False, colors='grey', width=0.6)

        # 2. Hazard line and scatter (Top Layer)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        
        h_vals = get_dw_hazard(q_f, beta_f, x_plot)
        ax.plot(x_plot, h_vals, color="#b51f1f", linewidth=1.5, zorder=3)
        
        mask = (df_emp['n_at_risk'] > 0) & (df_emp['go_run_length'] <= max_x)
        ax.errorbar(df_emp[mask]['go_run_length'], df_emp[mask]['empirical_hazard'],
                    yerr=df_emp[mask]['se'], fmt='o', color='black',
                    markersize=2.5, capsize=1, elinewidth=0.6, zorder=4)
        
        ax.set_title(title_str, fontsize=10, pad=5)
        
        # Buffer added below 0 to prevent cutting off scatter points
        ax.set_xlim(0, 21)
        ax.set_ylim(-0.05, 1.05)
        
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.tick_params(axis='x', bottom=True, labelbottom=True, labelsize=TICK_SIZE, width=0.6)
        
        ax.set_yticks([0.0, 0.5, 1.0])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False) 
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        if idx % 4 == 0:
            ax.tick_params(axis='y', left=True, labelleft=True, labelsize=TICK_SIZE, width=0.6, colors='black')
        else:
            ax.tick_params(axis='y', left=True, labelleft=False, width=0.6, colors='black')

    for idx in range(len(order_fits), len(axes2)):
        axes2[idx].set_visible(False)

    # 3. Create a centralized top legend
    hist_patch = mpatches.Patch(color='#e0e0e0', label='Empirical Density')
    emp_dots = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='black', markeredgecolor='black', markersize=4, label='Empirical Hazard')
    fit_line = mlines.Line2D([], [], color='#b51f1f', linewidth=1.5, label='Fitted Hazard')
    
    # Updated labels and order
    fig2.legend(handles=[hist_patch, emp_dots, fit_line], 
                labels=['Empirical Density', 'Empirical Hazard', 'Fitted Hazard'],
                loc='upper center', bbox_to_anchor=(0.5, 0.98), 
                ncol=3, frameon=False, fontsize=LEGEND_SIZE)

    # Changed Go Run Length to Go Run
    fig2.text(0.5, 0.02, 'Go Run', ha='center', fontsize=LABEL_SIZE)
    fig2.text(0.015, 0.5, 'Hazard', va='center', rotation='vertical', fontsize=LABEL_SIZE)
    fig2.text(0.97, 0.5, 'Probability Density', va='center', rotation=-90, fontsize=LABEL_SIZE, color='grey')
    
    fig2.savefig(ORDER_GRID_IMG, dpi=300, bbox_inches='tight')
    print(f"Done! Plots generated:\n  - {GLOBAL_IMG}\n  - {ORDER_GRID_IMG}")

if __name__ == "__main__":
    main()