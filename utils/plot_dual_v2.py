import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
import sys

# ================= CONFIGURATION =================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from core.hdbm_v3 import HDBM
except ImportError:
    print("Error: Could not import HDBM. Ensure 'core/hdbm_v2.py' exists.")
    sys.exit()

OUTPUT_IMG = BASE_DIR / 'outputs' / 'dual.png'

# --- Formatting ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.major.size'] = 6.0
plt.rcParams['ytick.major.size'] = 6.0

LABEL_SIZE = 10       
TICK_SIZE = 8         
LEGEND_SIZE = 9       
# ===============================================

def main():
    print(">>> 1. Initializing Sequence and Parameters...")
    seq = [0]*20 + [1]
    trials = np.arange(1, len(seq) + 1)
    
    # Define 4 specific parameter combinations to show the 3D parameter space
    # Feel free to change k_go, w_h (w_hazard), and w_f (w_fatigue) here!
    param_sets = [
        {'title': 'Inverse-U (High Hazard, High Fatigue)', 'k_go': 2.0,  'w_h': 0.7, 'w_f': -0.4},
        {'title': 'U-Shape (High Learning, Med Hazard)',   'k_go': 6.0, 'w_h': 0.5, 'w_f': 0.0},
        {'title': 'Linear Slowing (Hazard Only)',          'k_go': 1,  'w_h': 0.9, 'w_f': 0.0},
        {'title': 'Linear Rushing (Learning + Fatigue)',   'k_go': 5.0,  'w_h': 0.0, 'w_f': 0.4}
    ]
    
    print(">>> 2. Generating the 2x2 Grid Plot...")
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5))
    plt.subplots_adjust(wspace=0.15, hspace=0.5, bottom=0.12, left=0.1, right=0.98, top=0.82)
    
    stop_indices = [i + 1 for i, val in enumerate(seq) if val == 1]
    max_trial = len(trials)
    dynamic_xticks = [1] + list(range(5, max_trial + 1, 5))

    for idx, params in enumerate(param_sets):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Init model with the specific parameter set
        model = HDBM(alpha_go=0.85, alpha_stop=0.85, 
                     k_go=params['k_go'], w_hazard=params['w_h'], w_fatigue=params['w_f'], 
                     fatigue_shape=2.5, a0=5.0, b0=1.0)
        
        r_traj, Er_traj, h_traj, f_traj = model.simu_task(seq, return_details=True)
        
        # Calculate actual weighted contributions
        comp_learning = model.w_learning * Er_traj
        comp_hazard = model.w_hazard * h_traj
        comp_fatigue = -model.w_fatigue * f_traj  # Plotted as negative
        
        # 1. Learning Component (Blue)
        ax.plot(trials, comp_learning, color=plt.cm.Blues(0.8), linestyle='-', linewidth=1.2, zorder=3)
        # 2. Hazard Component (Red)
        ax.plot(trials, comp_hazard, color='#b51f1f', linestyle='-', linewidth=1.2, zorder=3)
        # 3. Fatigue Component (Orange, downward)
        ax.plot(trials, comp_fatigue, color='#e66101', linestyle='-', linewidth=1.2, zorder=3)
        # 4. Final Stop Prior (Purple, thick)
        ax.plot(trials, r_traj, color='#6a51a3', linestyle='-', linewidth=2.0, zorder=4)
        
        for st in stop_indices:
            ax.axvline(x=st, color='#888888', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
            
        # Display the parameters in the title
        title_str = f"{params['title']}\n$k_{{go}}$={params['k_go']}, $w_h$={params['w_h']}, $w_f$={params['w_f']}"
        ax.set_title(title_str, fontsize=10, pad=8)
        
        ax.set_xlim(1, max_trial+1)
        ax.set_ylim(-0.45, 1.05)
        ax.axhline(0, color='black', linewidth=0.5, zorder=2)
        
        ax.set_xticks(dynamic_xticks)
        ax.set_yticks([-0.4, 0.0, 0.5, 1.0])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=TICK_SIZE)
        
        # Bottom row gets X label
        if row == 1:
            ax.set_xlabel('Trial', fontsize=LABEL_SIZE)
            
        # Left col gets Y label
        if col == 0:
            ax.set_ylabel('Weighted Components / Prior', fontsize=LABEL_SIZE)
        else:
            ax.set_yticklabels([])

    # Centralized Global Legend
    l_er = mlines.Line2D([], [], color=plt.cm.Blues(0.8), linestyle='-', label='Learning (+)')
    l_haz = mlines.Line2D([], [], color='#b51f1f', linestyle='-', label='Hazard (+)')
    l_fat = mlines.Line2D([], [], color='#e66101', linestyle='-', label='Fatigue (-)')
    l_r = mlines.Line2D([], [], color='#6a51a3', linestyle='-', linewidth=2.0, label='Final Stop Prior')
    
    fig.legend(handles=[l_er, l_haz, l_fat, l_r], frameon=False, 
               fontsize=LEGEND_SIZE, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4)

    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f">>> Done! Saved as '{OUTPUT_IMG}'.")

if __name__ == "__main__":
    main()