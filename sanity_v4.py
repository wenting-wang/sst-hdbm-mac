import sys
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Highly recommended for this type of visualization
from scipy import stats

# --- Your existing setup ---
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

np.random.seed(42)

from core.hdbm import HDBM
from core.pomdp import POMDP
from core import simulation

seq_int = [1 if np.random.rand() < 1/6 else 0 for _ in range(180 * 2)]

alpha_lst = [0.1, 0.3, 0.5, 0.7, 0.9]
rho_lst = [0.1, 0.3, 0.5, 0.7, 0.9]
param_combinations = [(alpha, rho) for alpha in alpha_lst for rho in rho_lst]

# Store all results in a list to convert to a DataFrame
all_results = []

for alpha, rho in param_combinations: # FIXED: Unpacking the tuple directly
    hdbm = HDBM(alpha=alpha, rho=rho)
    r_preds = hdbm.simu_task(seq_int, block_size=180)
    
    # Store the time series data
    for t, r_val in enumerate(r_preds):
        all_results.append({
            'alpha': alpha,
            'rho': rho,
            'time_step': t,
            'r': r_val
        })

df = pd.DataFrame(all_results)

############
# Create a grid where columns are alpha, rows are rho
g = sns.FacetGrid(df, col="alpha", row="rho", margin_titles=True, height=1.5, aspect=1.8)
g.map_dataframe(sns.lineplot, x="time_step", y="r", color="blue")

g.set_axis_labels("Time", "r")
g.fig.suptitle('Simulated r for all Alpha and Rho Combinations', y=1.02)
# save
plt.savefig(BASE_DIR / "hdbm_sanity_check.png", dpi=300, bbox_inches='tight')
plt.show()
