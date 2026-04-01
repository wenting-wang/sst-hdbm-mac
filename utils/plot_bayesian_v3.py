import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import beta
from pathlib import Path
import sys

# ================= CONFIGURATION =================
BASE_DIR = Path(__file__).resolve().parent.parent

# Output Plot
OUTPUT_IMG = BASE_DIR / 'outputs' / 'bayesian.png'

# --- Formatting to match Nature/PLOS style ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']

LABEL_SIZE = 10       
TICK_SIZE = 8         
LEGEND_SIZE = 9       
# ===============================================

def get_ab_trajectory(seq, alpha_go=0.85, alpha_stop=0.85, eta=1.0, a0=5.0, b0=1.0):
    """
    Calculates the PREDICTIVE trial-by-trial parameters of the Beta distribution.
    The values at index `i` represent the prior BEFORE observing trial `i`.
    """
    a, b = a0, b0
    # Trial 1 starts with the pure initial prior
    ab_traj = [(a, b)] 
    
    # We process up to the second-to-last trial to get the prior for the last trial
    for trial in seq[:-1]:
        if trial == 0:  # Go Trial
            a_new = (1 - alpha_go) * a0 + alpha_go * (a + eta)
            b_new = (1 - alpha_go) * b0 + alpha_go * (b + 0)
        else:           # Stop Trial
            a_new = (1 - alpha_stop) * a0 + alpha_stop * (a + 0)
            b_new = (1 - alpha_stop) * b0 + alpha_stop * (b + 1)
        
        a, b = a_new, b_new
        ab_traj.append((a, b))
        
    return ab_traj

def main():
    print(">>> 1. Initializing Sequence and Parameters...")
    # Sequence: 3 Go, 1 Stop, 10 Go, 1 Stop, 20 Go (35 trials total)
    seq = [0]*3 + [1] + [0]*10 + [1] + [0]*20
    
    # Model base parameters
    alpha_go, alpha_stop = 0.85, 0.85
    a0, b0 = 5.0, 1.0
    
    # Define etas to plot (Low, Medium, High)
    etas = [0.5, 1.0, 2.0]
    mid_eta_idx = 1 # We use eta = 1.0 for the Heatmap background
    
    # Calculate predictive trajectories for all etas
    trajectories = []
    for eta in etas:
        traj = get_ab_trajectory(seq, alpha_go, alpha_stop, eta, a0, b0)
        trajectories.append(traj)

    print(">>> 2. Generating the Plot...")
    fig, ax = plt.subplots(figsize=(7.5, 3))
    plt.subplots_adjust(bottom=0.15, left=0.12, right=0.98, top=0.90) 
    
    trials = np.arange(1, len(seq) + 1)
    y_vals = np.linspace(0.001, 0.999, 500)
    X, Y = np.meshgrid(trials, y_vals)
    Z = np.zeros_like(X, dtype=float)
    
    # --- Calculate Heatmap (Z) based ONLY on the middle eta ---
    mid_traj = trajectories[mid_eta_idx]
    for idx, t in enumerate(trials):
        a_val, b_val = mid_traj[idx]
        # scipy beta(a,b) usually represents successes/failures. 
        # Our E(r) = b/(a+b), so we pass (b_val, a_val) to pdf.
        pdf = beta.pdf(y_vals, b_val, a_val) 
        Z[:, idx] = pdf
        
    # --- Colormap Truncation ---
    base_cmap = plt.get_cmap('Greys')
    custom_gray_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc_Greys', base_cmap(np.linspace(0.0, 0.75, 256))
    )
    vmax_val = np.max(Z) * 0.85
    mesh = ax.pcolormesh(X, Y, Z, cmap=custom_gray_cmap, shading='nearest', vmin=0, vmax=vmax_val)
    
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, aspect=40)
    cbar.set_label(r'Probability Density ($\eta=1.0$)', fontsize=LABEL_SIZE, rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=TICK_SIZE, length=2, width=0.6)
    cbar.outline.set_visible(False)
    
    # --- Overlay the predictive mean trajectory lines for all etas ---
    # Use different shades of blue, all solid lines
    colors = [plt.cm.Blues(0.5), plt.cm.Blues(0.75), plt.cm.Blues(0.95)]
    
    for i, eta in enumerate(etas):
        traj = trajectories[i]
        means = [b / (a + b) for a, b in traj]
        ax.plot(trials, means, color=colors[i], linestyle='-', 
                linewidth=1.3, label=rf'Expected Stop Prior ($\eta={eta}$)')
    
    # Add vertical lines for the Stop trials
    stop_indices = [i + 1 for i, val in enumerate(seq) if val == 1]
    for idx, st in enumerate(stop_indices):
        label = 'Stop Trial' if idx == 0 else None
        ax.axvline(x=st, color='#777777', linestyle='--', linewidth=1.0, alpha=0.8, label=label, zorder=1)        
        ax.text(st, 1.03, 'Stop Trial', color='#555555', fontsize=9, ha='center', va='bottom')
        
    # --- Minimalist Aesthetics ---
    ax.set_xlabel('Trial', fontsize=LABEL_SIZE)
    ax.set_ylabel('Stop Prior', fontsize=LABEL_SIZE)
    ax.set_xlim(1, len(trials))
    ax.set_ylim(0, 1.0)
    
    max_trial = len(trials)
    dynamic_xticks = [1] + list(range(5, max_trial + 1, 5))
    ax.set_xticks(dynamic_xticks)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=0.8, length=4)
    
    ax.legend(frameon=False, fontsize=LEGEND_SIZE, loc='upper right')

    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f">>> Done! Saved as '{OUTPUT_IMG}'.")

if __name__ == "__main__":
    main()