import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.ticker import MaxNLocator

# ================= CONFIGURATION =================
GLOBAL_FILE = 'go_run_hazard_combined.csv'
ORDER_FILE = 'go_run_hazard_per_order.csv'
OUTPUT_IMG = 'hazard.png'

# --- PLOS Computational Biology Formatting ---
# Width: 7.5 inches (19.05 cm) matches the full page width standard.
# Fonts: Arial/Helvetica, Size 8-12pt.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

LABEL_SIZE = 10       # Axis Labels (10-12pt)
TICK_SIZE = 9         # Tick Labels (8-10pt)
LEGEND_SIZE = 8       # Legend Text (Smaller as requested: 8pt)
PANEL_LABEL_SIZE = 12 # A/B Tags (12pt Bold)

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

def reconstruct_data_from_df(df):
    reconstructed_data = []
    df = df.sort_values('go_run_length')
    for _, row in df.iterrows():
        count = int(row['n_events'])
        length = int(row['go_run_length'])
        if count > 0:
            reconstructed_data.extend([length] * count)
    return np.array(reconstructed_data)

def main():
    print(">>> 1. Loading Data...")
    try:
        df_global = pd.read_csv(GLOBAL_FILE)
        df_orders = pd.read_csv(ORDER_FILE)
    except FileNotFoundError:
        print(f"Error: Please ensure '{GLOBAL_FILE}' and '{ORDER_FILE}' are in the current directory.")
        return

    global_runs = reconstruct_data_from_df(df_global)
    q_global, beta_global = fit_discrete_weibull(global_runs - 1)
    
    # Print Parameters for Manuscript
    print("="*40)
    print(f"Global Discrete Weibull Fit Parameters:")
    print(f"  q (parameter) = {q_global:.4f}")
    print(f"  beta (shape)  = {beta_global:.4f}")
    print("="*40)

    # Order Fits
    order_fits = []
    for oid in sorted(df_orders['order_id'].unique()):
        sub_df = df_orders[df_orders['order_id'] == oid]
        data = reconstruct_data_from_df(sub_df)
        if len(data) < 10: continue
        q_f, beta_f = fit_discrete_weibull(data - 1)
        order_fits.append((oid, q_f, beta_f))

    # ================= PLOTTING =================
    print(">>> 2. Plotting PLOS Compliant Figure (v6)...")
    
    # Size: 7.5 inches wide, 3.5 inches high
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    
    # Adjust spacing: ample bottom/left space for labels
    plt.subplots_adjust(wspace=0.25, bottom=0.15, left=0.08, right=0.98, top=0.9)
    
    max_x = 20
    x_plot = np.arange(1, max_x + 1)
    
    # --- PANEL A: PDF (Distribution) ---
    ax0 = axes[0]
    hist_data = global_runs[global_runs <= 20]
    bins = np.arange(0.5, max_x + 1.5, 1)
    
    # Histogram
    ax0.hist(hist_data, bins=bins, density=True, 
             color='lightgray', alpha=0.6, edgecolor='white', label='Empirical Data')
    
    # Fit Curve (Dark Gray, Smooth Line)
    pmf_vals = get_dw_pmf(q_global, beta_global, x_plot)
    line_pdf, = ax0.plot(x_plot, pmf_vals, color="#404040", linewidth=1.8, 
                         label='Discrete Weibull Fit')

    ax0.set_xlabel('Go Run Length', fontsize=LABEL_SIZE)
    ax0.set_ylabel('Probability Density', fontsize=LABEL_SIZE)
    ax0.set_xlim(0.5, 20.5)
    
    # Integer Ticks Only
    ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax0.set_xticks([1, 5, 10, 15, 20])
    
    ax0.legend(frameon=False, fontsize=LEGEND_SIZE, loc='upper right')

    # --- PANEL B: Hazard Function ---
    ax1 = axes[1]
    
    # 13 Order Lines (Thin, Faint, No Dots)
    for _, q_f, beta_f in order_fits:
        h_vals = get_dw_hazard(q_f, beta_f, x_plot)
        ax1.plot(x_plot, h_vals, color='#b51f1f', alpha=0.3, linewidth=1)
        
    # Global Fit (Thick Red Line, No Dots)
    h_global = get_dw_hazard(q_global, beta_global, x_plot)
    line_haz, = ax1.plot(x_plot, h_global, color="#b51f1f", linewidth=1.8, 
                         label='Derived Hazard')

    # Empirical Data (Black Dots with Errorbars)
    mask = (df_global['n_at_risk'] > 0) & (df_global['go_run_length'] <= 20)
    h_emp = df_global[mask]['hazard_prob'].values
    err_emp = ax1.errorbar(df_global[mask]['go_run_length'], h_emp, 
                 yerr=df_global[mask]['se'], fmt='o', color='black', 
                 markersize=4, capsize=2, elinewidth=1, markeredgewidth=1,
                 label='Empirical Hazard')

    # save
    df_hazard = pd.DataFrame({'go_run_length': x_plot, 
                              'fitted_hazard': h_global,
                                'empirical_hazard': h_emp})
    df_hazard.to_csv('fitted_hazard.csv', index=False)
    

    ax1.set_xlabel('Go Run Length', fontsize=LABEL_SIZE)
    # Changed Label Here
    ax1.set_ylabel('Hazard Value', fontsize=LABEL_SIZE)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0.5, 20.5)
    
    # Integer Ticks Only
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xticks([1, 5, 10, 15, 20])
    
    # Legend: Smaller size
    ax1.legend([line_haz, err_emp], ['Derived Hazard', 'Empirical Hazard'],
               frameon=False, fontsize=LEGEND_SIZE, loc='lower right')

    # --- COMMON STYLING ---
    for ax, label in zip(axes, ['A', 'B']):
        # Clean Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        # Ticks
        ax.tick_params(width=0.8, labelsize=TICK_SIZE)
        
        # Panel Labels (A, B)
        ax.text(-0.15, 1.05, label, transform=ax.transAxes, 
                fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='right')

    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"Done! Plot saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()

# Global Discrete Weibull Fit Parameters:
#   q (parameter) = 0.9277
#   beta (shape)  = 1.6186
