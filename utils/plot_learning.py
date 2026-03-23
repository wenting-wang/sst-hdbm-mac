import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import matplotlib.lines as mlines
from pathlib import Path
import sys

# ================= CONFIGURATION =================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import the HDBM model from the core directory
try:
    from core.hdbm_v2 import HDBM
except ImportError:
    print("Error: Could not import HDBM. Ensure 'core/hdbm_v2.py' exists.")
    sys.exit()

# Output Plot
OUTPUT_IMG = BASE_DIR / 'outputs' / 'learning.png'

# --- Formatting to match previous PLOS style ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

LABEL_SIZE = 10       
TICK_SIZE = 8         
LEGEND_SIZE = 9       
# ===============================================

def main():
    print(">>> 1. Initializing Sequence and Parameters...")
    # Define the sequence: 20 Go trials, 1 Stop trial
    seq = [0] * 20 + [1]
    eta_values = [1.0, 5.0, 10.0]
    
    print(">>> 2. Generating the 1x3 Comparison Plot...")
    # Create 1x3 subplots, max width 7.5 inches
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    plt.subplots_adjust(wspace=0.25, bottom=0.18, left=0.08, right=0.98, top=0.82)
    
    x = np.linspace(0.001, 0.999, 1000)
    cmap = plt.cm.Blues
    
    # Pre-calculate the stop index for styling
    stop_idx = seq.index(1) + 1 
    
    # We will track the global maximum Y value to sync all y-axes
    global_max_y = 0.0

    for idx, eta in enumerate(eta_values):
        ax = axes[idx]
        
        # Initialize model with current eta (replacing k_go)
        model = HDBM(alpha_go=0.7, alpha_stop=0.8, k_go=eta, a0=5.0, b0=1.0)
        
        # Track the (a, b) parameters
        a, b = model.a0, model.b0
        ab_traj = [(a, b)]  # Index 0 is the Prior
        
        # Run parameter update logic
        for trial in seq:
            if trial == 0:  # Go Trial
                a_new = (1 - model.alpha_go) * model.a0 + model.alpha_go * (a + model.k_go)
                b_new = (1 - model.alpha_go) * model.b0 + model.alpha_go * (b + 0)
            else:           # Stop Trial
                a_new = (1 - model.alpha_stop) * model.a0 + model.alpha_stop * (a + 0)
                b_new = (1 - model.alpha_stop) * model.b0 + model.alpha_stop * (b + 1)
            
            a, b = a_new, b_new
            ab_traj.append((a, b))

        n_lines = len(ab_traj)
        
        # Plot each step in the trajectory
        for i, (a_val, b_val) in enumerate(ab_traj):
            # b_val (Stop) is first argument, a_val (Go) is second argument
            y = beta.pdf(x, b_val, a_val)
            
            # Update global max y (filtering out infinities near edges)
            valid_y = y[np.isfinite(y)]
            if len(valid_y) > 0:
                global_max_y = max(global_max_y, np.max(valid_y))
            
            # Map intensity (0.3 to 1.0)
            color_intensity = 0.3 + 0.7 * (i / (n_lines - 1))
            c = cmap(color_intensity)
            
            # Uniform line width
            lw = 1.2
            
            # Line styling
            if i == 0:
                ls = '-'
                zorder = 5
            elif i < stop_idx:
                ls = '-'
                zorder = 3
            else:
                ls = '--'  # Dashed for the Stop trial
                zorder = 4
                
            ax.plot(x, y, color=c, linestyle=ls, linewidth=lw, zorder=zorder)
            
            # Fill the prior
            if i == 0:
                ax.fill_between(x, 0, y, color=c, alpha=0.1)

        # --- Subplot Aesthetics ---
        ax.set_title(rf'$\eta$ = {int(eta)}', fontsize=11, pad=8)
        ax.set_xlabel('Stop Prior', fontsize=LABEL_SIZE)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=0.8)
        
        # Only add Y label to the leftmost plot
        if idx == 0:
            ax.set_ylabel('Probability Density', fontsize=LABEL_SIZE)

    # Apply the exact same Y-axis limits to all subplots
    for ax in axes:
        ax.set_ylim(0, global_max_y * 1.05)

    # --- Centralized Custom Legend ---
    prior_line = mlines.Line2D([], [], color=cmap(0.3), linestyle='-', linewidth=1.2, label='Prior')
    go_lines = mlines.Line2D([], [], color=cmap(0.6), linestyle='-', linewidth=1.2, label='After Go Trials (0)')
    stop_lines = mlines.Line2D([], [], color=cmap(0.9), linestyle='--', linewidth=1.2, label='After Stop Trial (1)')
    
    # Place legend above the subplots
    fig.legend(handles=[prior_line, go_lines, stop_lines], frameon=False, 
               fontsize=LEGEND_SIZE, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)

    # Ensure output directory exists before saving
    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f">>> Done! Saved as '{OUTPUT_IMG}'.")

if __name__ == "__main__":
    main()