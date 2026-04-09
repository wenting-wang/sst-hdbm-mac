import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Read the dataset
# Options: 'additive_1', 'additive_2', 'multiplicative'
mode = 'multiplicative'
df = pd.read_csv(f'/Users/w/sst-hdbm-mac/data/est_param_{mode}.csv')

# 2. Drop the subject_id column
if 'subject_id' in df.columns:
    df_params = df.drop(columns=['subject_id'])
else:
    df_params = df.copy()

# 3. Define the custom order of the parameters
# Rearrange the items in this list to change the plot order!
if mode == 'additive_1' or mode == 'additive_2':
    custom_order = [
        'eta', 
        'q_d_n',
        'q_s_n', 
        'cost_stop_error', 
        'rho', 
        'q_d', 
        'q_s', 
    ]
elif mode == 'multiplicative':
    custom_order = [
        'eta', 
        'q_d_n',
        'q_s_n', 
        'cost_stop_error', 
        'gamma', 
        'q_d', 
        'q_s', 
        'inv_temp', 
    ]

# Keep only the columns present in the dataframe in the specific order
param_order = [p for p in custom_order if p in df_params.columns]
df_params = df_params[param_order]

# 4. Set up the figure for 4 columns
num_params = len(param_order)
cols = 4
rows = (num_params + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
axes = axes.flatten()

# 5. Plot each parameter in the specified order
for i, col in enumerate(param_order):
    sns.histplot(df_params[col], kde=True, ax=axes[i], color='mediumseagreen')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# 6. Hide any unused empty subplots
for j in range(num_params, len(axes)):
    fig.delaxes(axes[j])

# 7. Adjust layout and save/show the plot
plt.tight_layout()
plt.savefig(f'/Users/w/sst-hdbm-mac/outputs/param_{mode}.png', dpi=300)