from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from matplotlib.ticker import MaxNLocator

# ================= CONFIGURATION =================
ORDER_FILE = 'go_run_hazard_per_order.csv'
RAW_FILE = 'go_run_hazard_raw_data.csv'
OUTPUT_IMG = 'S1_Fig_Individual_Orders.png'

# --- PLOS Computational Biology Formatting ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Font Sizes
TITLE_SIZE = 9      
LABEL_SIZE = 9       
TICK_SIZE = 8        
TEXT_SIZE = 8        
LEGEND_SIZE = 9

# ===============================================

# --- Discrete Weibull Functions (Unchanged) ---
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
    print(">>> 1. Loading Order Data...")
    try:
        df_orders = pd.read_csv(ORDER_FILE)
    except FileNotFoundError:
        print(f"Error: '{ORDER_FILE}' not found. Please place it in the directory.")
        return

    order_ids = sorted(df_orders['order_id'].unique())
    n_orders = len(order_ids)
    print(f"Found {n_orders} orders.")
    
    try:
        df_raw = pd.read_csv(RAW_FILE)
        subject_counts = df_raw.groupby('order_id')['filename'].nunique().to_dict()
        print(f"Subject counts loaded: {subject_counts}")
    except FileNotFoundError:
        print(f"Warning: '{RAW_FILE}' not found. N will be 0.")
        subject_counts = {}
    
    # ================= PLOTTING GRID =================
    print(">>> 2. Plotting 4x4 Grid (Swapped Axes)...")
    
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(7, 7)) 
    axes_flat = axes.flatten()
    
    # Adjust spacing
    plt.subplots_adjust(wspace=0.2, hspace=0.4, left=0.08, right=0.92, top=0.90, bottom=0.08)
    
    max_x = 20
    x_plot = np.arange(1, max_x + 1)
    
    for i, ax_left in enumerate(axes_flat):
        if i < n_orders:
            oid = order_ids[i]
            sub_df = df_orders[df_orders['order_id'] == oid]
            data_raw = reconstruct_data_from_df(sub_df)
            
            # --- Fit Model ---
            if len(data_raw) >= 5:
                q_fit, beta_fit = fit_discrete_weibull(data_raw - 1)
            else:
                q_fit, beta_fit = 0.5, 1.0 
            
            # =========================================================
            # Plot Density on Right Axis (Secondary)
            # =========================================================
            
            ax_right = ax_left.twinx()
            bins = np.arange(0.5, max_x + 1.5, 1)
            ax_right.hist(data_raw[data_raw <= 20], bins=bins, density=True, 
                         color='lightgray', alpha=0.6, edgecolor='none')
            
            ax_right.set_ylim(0, 0.4)
            ax_right.spines['right'].set_color('gray')
            
            if (i + 1) % cols == 0:
                ax_right.set_ylabel('Density', fontsize=LABEL_SIZE, color='gray')
                ax_right.tick_params(axis='y', labelsize=TICK_SIZE, length=2, colors='gray', labelright=True)
            else:
                ax_right.tick_params(axis='y', length=2, colors='gray', labelright=False)
            
            # =========================================================
            # Plot Hazard on Left Axis (Primary)
            # =========================================================
            # 1. Derived Hazard (Line)
            h_vals = get_dw_hazard(q_fit, beta_fit, x_plot)
            ax_left.plot(x_plot, h_vals, color='#b51f1f', linewidth=1.5, alpha=0.9)
            
            # 2. Empirical Hazard (Dots)
            mask = (sub_df['n_at_risk'] > 0) & (sub_df['go_run_length'] <= 20)
            ax_left.errorbar(sub_df[mask]['go_run_length'], sub_df[mask]['hazard_prob'], 
                              yerr=sub_df[mask]['se'], fmt='o', color='black', 
                              markersize=2.5, capsize=1, elinewidth=0.8, markeredgewidth=0.8)

            ax_left.set_ylim(-0.05, 1.05)
            ax_left.tick_params(axis='y', labelsize=TICK_SIZE, length=2, color='black')

            # CONTROL VISIBILITY: Only show Hazard labels/ticks on FIRST column
            if i % cols == 0:
                ax_left.set_ylabel('Hazard', fontsize=LABEL_SIZE, color='black')
            else:
                ax_left.tick_params(labelleft=False) # Hide numbers on inner plots

            # -----------------------------------------------------------
            # FIX Z-ORDER (Crucial Step)
            # -----------------------------------------------------------
            # Put ax_left (Hazard) ON TOP of ax_right (Density)
            ax_left.set_zorder(ax_right.get_zorder() + 1)
            
            # Make ax_left background transparent so we can see ax_right through it
            ax_left.patch.set_visible(False)
            ax_right.patch.set_visible(True)

            # =========================================================
            # Common Styling
            # =========================================================
            ax_left.set_title(f"Order {oid} (N={subject_counts.get(oid, 0)})", fontsize=TITLE_SIZE, fontweight='regular', pad=4)
            ax_left.set_xlim(0, 21)
            
            # X Axis Formatting
            ax_left.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax_left.tick_params(axis='x', labelsize=TICK_SIZE, length=2)
            ax_left.set_xticks([1, 5, 10, 15, 20])
            ax_left.set_xticklabels([1, 5, 10, 15, 20], fontsize=TICK_SIZE)
            
            # Only label X-Axis on the bottom row
            if i // cols == rows - 1:
                ax_left.set_xlabel('Go Run Length', fontsize=LABEL_SIZE)
            elif i // cols == rows - 2 and (i + 1) % cols != 1:
                ax_left.set_xlabel('Go Run Length', fontsize=LABEL_SIZE)
            
            # Handle Spines (Clean look)
            ax_left.spines['top'].set_visible(False)
            ax_right.spines['top'].set_visible(False)
            
            # Left Axis Spines (Hazard) - Hide right side
            ax_left.spines['right'].set_visible(False) 
            ax_left.spines['left'].set_visible(True)
            
            # Right Axis Spines (Density) - Hide left side
            ax_right.spines['left'].set_visible(False)
            ax_right.spines['right'].set_visible(True) 
            
        else:
            # Hide empty subplots
            ax_left.axis('off')

        # ================= LEGEND =================
        # Create custom legend handles
        legend_elements = [
            Line2D([0], [0], color='#b51f1f', lw=1.5, label='Derived Hazard'),
            Line2D([0], [0], marker='o', color='w', label='Empirical Hazard',
                markerfacecolor='black', markersize=5),
            Patch(facecolor='lightgray', edgecolor='none', alpha=0.6, label='Density')
        ]

        # Add legend to the figure (Global)
        fig.legend(handles=legend_elements, 
                loc='upper center',          # Position anchor
                bbox_to_anchor=(0.5, 0.98),  # Coordinate (x=0.5 center, y=0.98 top)
                ncol=3,                      # 3 items in one row
                frameon=False,               # No box around legend
                fontsize=LEGEND_SIZE)



    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"Done! Modified figure saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()