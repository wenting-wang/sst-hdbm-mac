import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import beta
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
OUTPUT_IMG = BASE_DIR / 'outputs' / 'learning_heatmap.png'

# --- Formatting to match Nature/PLOS style ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']

LABEL_SIZE = 10       
TICK_SIZE = 8         
LEGEND_SIZE = 9       
# ===============================================

def main():
    print(">>> 1. Initializing Sequence and Parameters...")
    # Sequence: 3 Go, 1 Stop, 10 Go, 1 Stop, 20 Go, 1 Stop (36 trials total)
    seq = [0]*3 + [1] + [0]*10 + [1] + [0]*20
    # seq = '100100000000000000000001000100000'
    # seq = [int(char) for char in seq]
    
    # Initialize model
    model = HDBM(alpha_go=0.85, alpha_stop=0.85, k_go=1, a0=5.0, b0=1.0)
    
    # Track the (a, b) parameters
    a, b = model.a0, model.b0
    ab_traj = [] 
    
    # Run parameter update logic (ignoring the 0th prior for the plot)
    for trial in seq:
        if trial == 0:  # Go Trial
            a_new = (1 - model.alpha_go) * model.a0 + model.alpha_go * (a + model.k_go)
            b_new = (1 - model.alpha_go) * model.b0 + model.alpha_go * (b + 0)
        else:           # Stop Trial
            a_new = (1 - model.alpha_stop) * model.a0 + model.alpha_stop * (a + 0)
            b_new = (1 - model.alpha_stop) * model.b0 + model.alpha_stop * (b + 1)
        
        a, b = a_new, b_new
        ab_traj.append((a, b))

    print(">>> 2. Generating the Plot...")
    # Scaled down figsize as requested
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    plt.subplots_adjust(bottom=0.15, left=0.12, right=0.98, top=0.90) 
    
    # Trials strictly from 1 to N
    trials = np.arange(1, len(ab_traj) + 1)
    y_vals = np.linspace(0.001, 0.999, 500)
    X, Y = np.meshgrid(trials, y_vals)
    Z = np.zeros_like(X, dtype=float)
    
    means = []
    
    # Calculate density and mean
    for idx, t in enumerate(trials):
        a_val, b_val = ab_traj[idx]
        pdf = beta.pdf(y_vals, b_val, a_val)
        Z[:, idx] = pdf
        means.append(b_val / (a_val + b_val))
        
    # --- Colormap Truncation ---
    base_cmap = plt.get_cmap('Greys')
    custom_gray_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc_Greys', base_cmap(np.linspace(0.0, 0.55, 256))
    )

    # Plot the density heatmap
    mesh = ax.pcolormesh(X, Y, Z, cmap=custom_gray_cmap, shading='nearest', vmin=0, vmax=np.max(Z))
    
    # Add minimal Colorbar
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, aspect=40)
    cbar.set_label('Probability Density', fontsize=LABEL_SIZE, rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=TICK_SIZE, length=2, width=0.6)
    cbar.outline.set_visible(False)
    
    # Overlay the mean trajectory line
    ax.plot(trials, means, color=plt.cm.Blues(0.9), linewidth=1.2, label='Expected Stop Prior')
    
    # Add vertical lines for the Stop trials
    stop_indices = [i + 1 for i, val in enumerate(seq) if val == 1]
    for idx, st in enumerate(stop_indices):
        label = 'Stop Trial' if idx == 0 else None
        ax.axvline(x=st, color='#888888', linestyle='--', linewidth=0.8, alpha=0.7, label=label, zorder=5)
        # Subtle text at the top
        ax.text(st, 1.02, 'Stop', color='#666666', fontsize=8, ha='center', va='bottom')

    # --- Minimalist Aesthetics ---
    ax.set_xlabel('Trial', fontsize=LABEL_SIZE)
    ax.set_ylabel('Expected Stop Prior', fontsize=LABEL_SIZE)
    
    ax.set_xlim(1, len(trials)+1)
    ax.set_ylim(0, 1.0)
    
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    
    max_trial = len(trials)
    dynamic_xticks = [1] + list(range(5, max_trial + 1, 5))
    ax.set_xticks(dynamic_xticks)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Delicate ticks
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=0.8, length=4)
    
    # Clean Legend (Capitalized)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE, loc='upper right')

    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f">>> Done! Saved as '{OUTPUT_IMG}'.")

if __name__ == "__main__":
    main()