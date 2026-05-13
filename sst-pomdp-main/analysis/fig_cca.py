import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from pathlib import Path
from matplotlib.gridspec import GridSpec

# =============================================================================
# 1. CONFIGURATION & PATHS
# =============================================================================

# 1. Get the directory of this script (the 'analysis' folder)
current_dir = Path(__file__).resolve().parent
# 2. Go one level up to get the repository root
repo_root = current_dir.parent

# 3. Define paths relative to the repository root
OUT_DIR = repo_root / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True) 

EMBED_CSV = repo_root / 'data' / 'example_embeddings.csv'
STATS_CSV = repo_root / 'data' / 'example_clinical_behavior.csv'

OUTPUT_PLOT = OUT_DIR / "fig_cca.png"

# =============================================================================
# 2. PLOTTING STYLE
# =============================================================================

# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 1.0    

# =============================================================================
# 3. DATA LOADING & PREPROCESSING
# =============================================================================

print("Loading data...")

if not EMBED_CSV.exists() or not STATS_CSV.exists():
    raise FileNotFoundError(f"Example data files not found.\n"
                            f"Expected to find them at:\n"
                            f" - {EMBED_CSV}\n"
                            f" - {STATS_CSV}")

df_emb = pd.read_csv(EMBED_CSV)
df_stats = pd.read_csv(STATS_CSV)

df_emb['subject_id'] = df_emb['subject_id'].astype(str)
df_stats['subject_id'] = df_stats['subject_id'].astype(str)

df = pd.merge(df_emb, df_stats, on=['subject_id', 'year'], how='inner')
print(f"Data merged. Total subjects: {len(df)}")

emb_cols = [f'emb_{i}' for i in range(64)]
target_stats = [
    'mrt_gs', 'perc_gs', 'perc_gm', 'mrt_ge', 
    'ssrt', 'perc_ss', 'mrt_se',
    'rate_perc_ss_ssd', 'rate_rt_se_ssd',
    'pes', 'pss', 'pges', 'rt_acf_1', 'rt_acf_2', 'rt_slope'
]

available_stats = [c for c in target_stats if c in df.columns]

X = df[emb_cols].values
Y = df[available_stats].values

mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
X_clean = X[mask]
Y_clean = Y[mask]

std_y = Y_clean.std(axis=0)
valid_cols_idx = np.where(std_y > 1e-6)[0]
Y_clean = Y_clean[:, valid_cols_idx]
final_stats_names = [available_stats[i] for i in valid_cols_idx]

X_std = StandardScaler().fit_transform(X_clean)
Y_std = StandardScaler().fit_transform(Y_clean)

# =============================================================================
# 4. RUN CCA
# =============================================================================

print("Running CCA...")
cca = CCA(n_components=1)
cca.fit(X_std, Y_std)
X_c, Y_c = cca.transform(X_std, Y_std)

corr, p_val = pearsonr(X_c[:, 0], Y_c[:, 0])

print("\nCCA RESULTS:")
print(f"Canonical Correlation (r): {corr:.4f}")
print(f"P-value: {p_val:.4e}\n")

# =============================================================================
# 5. VISUALIZATION
# =============================================================================

print("Generating final publication-quality plot...")

# Calculate Cross-Loadings
y_variate = Y_c[:, 0]
x_variate = X_c[:, 0]

y_cross_loadings = np.array([np.corrcoef(Y_std[:, i], x_variate)[0, 1] for i in range(Y_std.shape[1])])
x_cross_loadings = np.array([np.corrcoef(X_std[:, i], y_variate)[0, 1] for i in range(X_std.shape[1])])

# Sort Features for Both Sides
sort_idx_Y = np.argsort(y_cross_loadings)[::-1]
sort_idx_X = np.argsort(x_cross_loadings)[::-1]

y_cross_sorted = y_cross_loadings[sort_idx_Y]
x_cross_sorted = x_cross_loadings[sort_idx_X]

name_map = {
    'mrt_gs': 'Mean Go Success RT', 'mrt_se': 'Mean Stop Error RT', 'mrt_ge': 'Mean Go Error RT',
    'perc_gs': 'Go Success Rate', 'perc_gm': 'Go Missing Rate', 'perc_ss': 'Stop Success Rate',
    'ssrt': 'Stop Signal RT (SSRT)', 'rate_perc_ss_ssd': 'Stop Succ. Rate across SSD', 'rate_rt_se_ssd': 'Stop Error RT across SSD',
    'pes': 'Post-Stop Error Slowing', 'pss': 'Post-Stop Success Slowing', 'pges': 'Post-Go Error Slowing',
    'rt_acf_1': 'RT Autocorrelation (Lag-1)', 'rt_acf_2': 'RT Autocorrelation (Lag-2)', 'rt_slope': 'Fatigue Slope (RTs)'
}
sorted_Y_names = [name_map.get(final_stats_names[i], final_stats_names[i]) for i in sort_idx_Y]

X_std_sorted = X_std[:, sort_idx_X]
Y_std_sorted = Y_std[:, sort_idx_Y]

cross_corr_matrix = np.dot(Y_std_sorted.T, X_std_sorted) / (X_std_sorted.shape[0] - 1)

# Plot Layout (Top: Heatmap, Bottom: Two Bar Charts)
fig = plt.figure(figsize=(7.5, 7.5)) 
gs = GridSpec(2, 2, height_ratios=[1.2, 1], width_ratios=[1, 1.2], hspace=0.35, wspace=0.3)
COLOR = '#4d4d4d'

# Panel A: Heatmap
axA = fig.add_subplot(gs[0, :])
cax = axA.imshow(cross_corr_matrix, cmap='BrBG', vmin=-0.6, vmax=0.6, aspect='auto')

axA.set_yticks(np.arange(len(sorted_Y_names)))
axA.set_yticklabels(sorted_Y_names)
axA.text(-0.205, 1.03, 'Behavioral Metrics', transform=axA.transAxes, ha='left', va='bottom', fontsize=9, weight='regular')

axA.set_xticks([0, 15, 31, 47, 63])
axA.set_xticklabels(['1', '16', '32', '48', '64'])
axA.set_xlabel('Latent Behavioral Embeddings (Sorted)', weight='regular')

cbar = fig.colorbar(cax, ax=axA, orientation='vertical', fraction=0.03, pad=0.02)
cbar.ax.set_title('Pearson r', pad=10, weight='regular', fontsize=9) 
axA.text(-0.3, 1.05, 'A', transform=axA.transAxes, size=14, weight='bold')

# Panel B: Behavior Cross-Loadings
axB = fig.add_subplot(gs[1, 0])
y_pos = np.arange(len(sorted_Y_names))

axB.barh(y_pos, y_cross_sorted[::-1], color=COLOR, height=0.6)
axB.set_yticks(y_pos)
axB.set_yticklabels(sorted_Y_names[::-1])
axB.set_xlabel('Cross-Loading with Embeddings', weight='regular')

axB.spines['top'].set_visible(False)
axB.spines['right'].set_visible(False)
axB.axvline(0, color='black', linewidth=1)
axB.text(-1.1, 1.05, 'B', transform=axB.transAxes, size=14, weight='bold')

# Panel C: Embedding Cross-Loadings
axC = fig.add_subplot(gs[1, 1])
x_pos = np.arange(len(x_cross_sorted))

axC.bar(x_pos, x_cross_sorted, color=COLOR, width=0.6)
axC.set_ylabel('Cross-Loading with Behavioral Metrics', weight='regular')
axC.set_xlabel('Latent Behavioral Embeddings (Sorted)', weight='regular')

axC.spines['top'].set_visible(False)
axC.spines['right'].set_visible(False)

axC.set_xticks([0, 15, 31, 47, 63])
axC.set_xticklabels(['1', '16', '32', '48', '64'])
axC.axhline(0, color='black', linewidth=1)
axC.text(-0.17, 1.05, 'C', transform=axC.transAxes, size=14, weight='bold')

# Final Output & Save
print(f"Saving figure to: {OUTPUT_PLOT}")
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight', format='png')