import sys
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# --- Your existing setup ---
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

np.random.seed(42)

from core.hdbm import HDBM
from core.pomdp import POMDP
from core import simulation

# Updated parameters
POMDP_PARAMS = {
    "q_d_n": 0.518, "q_d": 0.725,
    "q_s_n": 0.018, "q_s": 0.839,
    "cost_stop_error": 1.649, 
    "inv_temp": 36.174,
    "cost_time": 0.001, 
    "cost_go_error": 1.0,
    "cost_go_missing": 1.0}

def gen(r_pred, ssd, batch):
    pomdp = POMDP(rate_stop_trial=r_pred, **POMDP_PARAMS)
    pomdp.value_iteration_tensor()
    res_lst = []
    rt_lst = []
    for _ in range(batch):
        # res in [GS, GE, GM, SS, SE]
        res, rt = simulation.simu_trial(
                    pomdp, true_go_state=true_go_state, 
                    true_stop_state=true_stop_state, 
                    ssd=ssd, verbose=False)
        res_lst.append(res)
        rt_lst.append(rt)
    return res_lst, rt_lst

BATCH_SIZE = 500

# ==========================================
# GO TRIALS
# ==========================================
true_go_state = 'right'
true_stop_state = 'nonstop'
r_pred_lst = [0.1, 0.3, 0.5, 0.7, 0.9]
ssd = -1  # -1 for Go trials

go_data_file = BASE_DIR / 'go_sim_data.pkl'

if os.path.exists(go_data_file):
    print(f"Loading Go trial data from {go_data_file}...")
    with open(go_data_file, 'rb') as f:
        go_percentages, go_rts = pickle.load(f)
else:
    print("Simulating Go Trials...")
    go_percentages = []
    go_rts = {}
    
    for r_pred in r_pred_lst:
        res, rt = gen(r_pred, ssd, BATCH_SIZE)
        
        # Calculate percentages
        pct_gs = res.count('GS') / BATCH_SIZE
        pct_ge = res.count('GE') / BATCH_SIZE
        pct_gm = res.count('GM') / BATCH_SIZE
        go_percentages.append({'GS': pct_gs, 'GE': pct_ge, 'GM': pct_gm})
        
        # Store valid RTs
        valid_rts = [r for r, outcome in zip(rt, res) if outcome in ['GS', 'GE'] and r is not None]
        go_rts[r_pred] = valid_rts
        
    print(f"Saving Go trial data to {go_data_file}...")
    with open(go_data_file, 'wb') as f:
        pickle.dump((go_percentages, go_rts), f)

# 2. Visualize Go trials
fig_go, axes_go = plt.subplots(1, 2, figsize=(14, 5))

# Percentage of GS, GE, GM
df_go = pd.DataFrame(go_percentages, index=r_pred_lst)
df_go.plot(kind='bar', stacked=True, ax=axes_go[0], colormap='viridis')
axes_go[0].set_title('Go Trial Outcomes by r_pred')
axes_go[0].set_xlabel('r_pred (Probability of Stop Trial)')
axes_go[0].set_ylabel('Percentage')
axes_go[0].legend(title='Outcome')

# RT Histograms
for r_pred, rts in go_rts.items():
    if len(rts) > 0:
        axes_go[1].hist(rts, bins=20, alpha=0.5, density=True, label=f'r_pred={r_pred}')
axes_go[1].set_title('Reaction Time (RT) Distributions')
axes_go[1].set_xlabel('Reaction Time')
axes_go[1].set_ylabel('Density')
axes_go[1].legend()

plt.tight_layout()
plt.savefig('go_trials_visualization.png', dpi=300)
plt.close(fig_go)


# ==========================================
# STOP TRIALS
# ==========================================
true_go_state = 'right'
true_stop_state = 'stop'
ssd_lst = [2, 10, 20, 30, 34]

stop_data_file = BASE_DIR / 'stop_sim_data.pkl'

if os.path.exists(stop_data_file):
    print(f"Loading Stop trial data from {stop_data_file}...")
    with open(stop_data_file, 'rb') as f:
        stop_ss_pct, stop_ss_err, stop_rts = pickle.load(f)
        
    # Print out a summary of loaded SE counts
    print("\n--- Loaded Stop Error (SE) Counts ---")
    for i, r_pred in enumerate(r_pred_lst):
        for j, s in enumerate(ssd_lst):
            print(f"r_pred={r_pred}, SSD={s}: {len(stop_rts[(r_pred, s)])} SE trials out of {BATCH_SIZE}")
else:
    print("\nSimulating Stop Trials...")
    stop_ss_pct = np.zeros((len(r_pred_lst), len(ssd_lst)))
    stop_ss_err = np.zeros((len(r_pred_lst), len(ssd_lst)))
    stop_rts = {}
    
    for i, r_pred in enumerate(r_pred_lst):
        for j, s in enumerate(ssd_lst):
            res, rt = gen(r_pred, s, BATCH_SIZE)
            
            # --- Stop Success (SS) calculations ---
            ss_array = np.array([1 if r == 'SS' else 0 for r in res])
            stop_ss_pct[i, j] = np.mean(ss_array)
            stop_ss_err[i, j] = stats.sem(ss_array)
            
            # --- Stop Error (SE) RT raw data ---
            valid_se_rts = [r for r, outcome in zip(rt, res) if outcome == 'SE' and r is not None]
            stop_rts[(r_pred, s)] = valid_se_rts
            
            # Print diagnostic info
            print(f"r_pred={r_pred}, SSD={s}: {len(valid_se_rts)} SE trials out of {BATCH_SIZE}")

    print(f"Saving Stop trial data to {stop_data_file}...")
    with open(stop_data_file, 'wb') as f:
        pickle.dump((stop_ss_pct, stop_ss_err, stop_rts), f)

# 2. Visualize Stop trials (SS Percentage)
fig_stop_pct, ax_stop = plt.subplots(figsize=(8, 6))

for i, r_pred in enumerate(r_pred_lst):
    ax_stop.errorbar(ssd_lst, stop_ss_pct[i, :], yerr=stop_ss_err[i, :], 
                     marker='o', capsize=4, label=f'r_pred={r_pred}')

ax_stop.set_title('Stop Success (SS) vs SSD')
ax_stop.set_xlabel('Stop Signal Delay (SSD)')
ax_stop.set_ylabel('SS Percentage')
ax_stop.legend(title='r_pred')
ax_stop.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stop_trials_ss_percentage.png', dpi=300)
plt.close(fig_stop_pct)

# 3. Visualize Stop trials (SE RT Distributions)
fig_stop_rt, axes_stop_rt = plt.subplots(len(r_pred_lst), len(ssd_lst), 
                                         figsize=(16, 12), sharex=True, sharey=True)

for i, r_pred in enumerate(r_pred_lst):
    for j, s in enumerate(ssd_lst):
        ax = axes_stop_rt[i, j]
        rts = stop_rts[(r_pred, s)]
        
        if len(rts) > 0:
            # Add text inside the plot showing how many samples are in the histogram
            ax.text(0.5, 0.9, f"n={len(rts)}", transform=ax.transAxes, 
                    ha='center', va='top', fontsize=9, alpha=0.7)
            ax.hist(rts, bins=15, color='coral', alpha=0.7, density=False)
        else:
            # If empty, print an indicator directly on the blank subplot
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, 
                    ha='center', va='center', fontsize=10, color='grey')
            
        # Format the grid layout
        if i == 0:
            ax.set_title(f'SSD = {s}')
        if j == 0:
            ax.set_ylabel(f'r_pred = {r_pred}\nFrequency')
        if i == len(r_pred_lst) - 1:
            ax.set_xlabel('RT')

fig_stop_rt.suptitle('Stop Error (SE) Reaction Time Distributions', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig('stop_trials_se_rt_distributions.png', dpi=300, bbox_inches='tight')
plt.close(fig_stop_rt)

print("Visualizations complete. All figures saved.")