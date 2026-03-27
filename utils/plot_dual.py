import numpy as np
import matplotlib.pyplot as plt
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
OUTPUT_IMG = BASE_DIR / 'outputs' / 'dual.png'

# --- Formatting to match Nature/PLOS style ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.major.size'] = 4.0
plt.rcParams['ytick.major.size'] = 4.0

LABEL_SIZE = 10       
TICK_SIZE = 8         
LEGEND_SIZE = 9       
PANEL_LABEL_SIZE = 12
# ===============================================

def main():
    print(">>> 1. Initializing Sequence and Parameters...")
    # Sequence: 3 Go, 1 Stop, 10 Go, 1 Stop, 20 Go, 1 Stop (36 trials total)
    seq = [0]*20 + [1]
    trials = np.arange(1, len(seq) + 1)
    
    # Define our 2x2 factorial parameter grid
    eta_vals = [0.1, 10.0]  
    rho_vals = [0.1, 0.9]   
    
    print(">>> 2. Generating the 2x2 Grid Plot...")
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
    
    # Slightly increased hspace so the "Trial" label on the top row doesn't clash with the title below it
    plt.subplots_adjust(wspace=0.1, hspace=0.45, bottom=0.12, left=0.1, right=0.98, top=0.85)
    
    # Pre-calculate Stop trial indices for plotting vertical lines
    stop_indices = [i + 1 for i, val in enumerate(seq) if val == 1]
    
    # X-axis ticks generation
    max_trial = len(trials)
    dynamic_xticks = [1] + list(range(5, max_trial + 1, 5))

    for i, eta in enumerate(eta_vals):
        for j, rho in enumerate(rho_vals):
            ax = axes[i, j]
            
            # Initialize model with current eta (k_go) and rho
            model = HDBM(alpha_go=0.85, alpha_stop=0.85, k_go=eta, rho=rho, 
                         fusion_type='additive', a0=5.0, b0=1.0)
            
            # Run simulation requesting detailed trajectories
            r_traj, Er_traj, h_traj = model.simu_task(seq, return_details=True)
            
            # Calculate the explicit hazard component being added
            hazard_component = model.rho * h_traj
            
            # 1. Plot the Learning Component (Blue, solid, thin)
            ax.plot(trials, Er_traj, color=plt.cm.Blues(0.9), linestyle='-', linewidth=1.2, zorder=3)
            
            # 2. Plot the Hazard Component (Red, solid, thin)
            ax.plot(trials, hazard_component, color='#b51f1f', linestyle='-', linewidth=1.2, zorder=3)
            
            # 3. Plot the Final Stop Prior (Purple, solid, thin)
            ax.plot(trials, r_traj, color='#6a51a3', linestyle='-', linewidth=1.2, zorder=4)
            
            # Add vertical dashed lines for Stop trials
            for st in stop_indices:
                ax.axvline(x=st, color='#888888', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
                
            # --- Aesthetics per subplot ---
            title_str = rf'$\eta$ = {eta:.1f},  $\rho$ = {rho:.1f}'
            ax.set_title(title_str, fontsize=11, pad=8)
            
            ax.set_xlim(1, max_trial+1)
            ax.set_ylim(0, 1.05)
            
            ax.set_xticks(dynamic_xticks)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=TICK_SIZE)
            
            # Show X labels and ticks on ALL rows now
            ax.set_xlabel('Trial', fontsize=LABEL_SIZE)
                
            # Only show Y labels on the left column to avoid clutter
            if j == 0:
                ax.set_ylabel('Stop Prior', fontsize=LABEL_SIZE) # Title Case
            else:
                ax.set_yticklabels([])

    # --- Centralized Global Legend (Title Case) ---
    line_er = mlines.Line2D([], [], color=plt.cm.Blues(0.9), linestyle='-', linewidth=1.2, label='Learning Component')
    line_haz = mlines.Line2D([], [], color='#b51f1f', linestyle='-', linewidth=1.2, label='Hazard Component')
    line_r = mlines.Line2D([], [], color='#6a51a3', linestyle='-', linewidth=1.2, label='Final Stop Prior')
    
    fig.legend(handles=[line_er, line_haz, line_r], frameon=False, 
               fontsize=LEGEND_SIZE, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3)

    # Output Saving
    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f">>> Done! Saved as '{OUTPUT_IMG}'.")

if __name__ == "__main__":
    main()