import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import warnings

# Ensure paths are correct and allow imports from the project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# ================= CONFIGURATION =================
LIST_FILE = BASE_DIR / 'data' / 'clinical_behavior.csv'
DATA_ROOT = Path('/Users/w/Desktop/data/sst_valid_base/')
ORDERS_FILE = BASE_DIR / 'data' / 'orders.csv'

# Outputs
SUBJECT_ORDER_MAP_FILE = BASE_DIR / 'data' / 'subject_order_map.csv'

# Target Orders as a List
# You can put multiple orders here, e.g., ['order_0', 'order_1', 'order_2']
# Or use ['ALL'] to process every order found in orders.csv
# TARGET_ORDER_SIGS = ['order_0', 'order_1'] 
TARGET_ORDER_SIGS = ['ALL']  # Process all orders

OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ===============================================

def extract_data(file_path: Path):
    try:
        # Changed column to sst_go_rt
        df = pd.read_csv(file_path, usecols=['sst_expcon', 'sst_go_rt'])
        is_go = df['sst_expcon'].astype(str).str.strip() == 'GoTrial'
        sig = "".join(np.where(is_go, '0', '1'))
        
        # Convert to numeric
        rts = pd.to_numeric(df['sst_go_rt'], errors='coerce').to_numpy()
        
        # Replace values <= 200 (which includes 0) with NaN 
        # to filter out missed trials and anticipatory noise, while keeping array length 360
        rts = np.where(rts <= 200, np.nan, rts)
        
        # Replace 0 with NaN so it's ignored in mean calculations but keeps the array length 360
        # rts = np.where(rts == 0, np.nan, rts)
        
        return sig, rts
    except Exception:
        return None, None

def extract_streaks(is_go_array, rt_array):
    streaks = []
    current_streak = []
    seen_first_stop = False 
    for go_flag, rt in zip(is_go_array, rt_array):
        if not go_flag:
            seen_first_stop = True
            if len(current_streak) > 0:
                streaks.append(current_streak)
                current_streak = []
        else:
            if seen_first_stop:
                current_streak.append(rt)
    if len(current_streak) > 0:
        streaks.append(current_streak)
    return streaks

def analyze_and_plot_order(order_name, target_subject_trends):
    """Handles the 5-way clustering and plotting for a specific order."""
    num_valid_subjects = len(target_subject_trends)
    print(f"\n--- Analyzing {num_valid_subjects} subjects in {order_name} ---")

    metrics = []
    max_streak_length = max(len(subj['trend']) for subj in target_subject_trends)
    x_full = np.arange(1, max_streak_length + 1)

    for i, subj in enumerate(target_subject_trends):
        trend = subj['trend']
        valid_idx = ~np.isnan(trend)
        x_valid = x_full[valid_idx]
        y_valid = trend[valid_idx]
        
        # Need at least 3 points to fit a quadratic curve
        if len(x_valid) >= 3:
            z1 = np.polyfit(x_valid, y_valid, 1) # Linear
            z2 = np.polyfit(x_valid, y_valid, 2) # Quadratic
            
            padded_trend = np.full(max_streak_length, np.nan)
            padded_trend[:len(trend)] = trend
            
            metrics.append({'idx': i, 'slope': z1[0], 'curve': z2[0], 'trend': padded_trend})

    df_metrics = pd.DataFrame(metrics)
    num_analyzed = len(df_metrics)
    
    if num_analyzed == 0:
        print(f"Not enough valid subjects to analyze {order_name}.")
        return

    # --- 1. Clustering (20-20-30-30-Rest split) ---
    n_top_20 = int(num_analyzed * 0.20)
    
    df_metrics = df_metrics.sort_values(by='slope', ascending=False)
    idx_slowing = df_metrics.iloc[:n_top_20]['idx'].tolist()  # Top 20% Pos Slope
    idx_speeding = df_metrics.iloc[-n_top_20:]['idx'].tolist() # Bottom 20% Neg Slope
    
    # The remaining 60%
    rest_df = df_metrics.iloc[n_top_20:-n_top_20].copy()
    n_rest = len(rest_df)
    n_top_30_rest = int(n_rest * 0.30)
    
    # Sort the rest by curvature
    rest_df = rest_df.sort_values(by='curve', ascending=False)
    idx_ushape = rest_df.iloc[:n_top_30_rest]['idx'].tolist()     # Top 30% Pos Curvature
    idx_inv_ushape = rest_df.iloc[-n_top_30_rest:]['idx'].tolist() # Bottom 30% Neg Curvature
    
    # The final leftover (~24% of total) is the baseline
    idx_baseline = rest_df.iloc[n_top_30_rest:-n_top_30_rest]['idx'].tolist()

    clusters = [
        ('1. Continuous Slowing (Top 20%)', idx_slowing, '#d62728'),      # Red
        ('2. Continuous Speeding (Bottom 20%)', idx_speeding, '#2ca02c'), # Green
        ('3. U-Shape (Top 30% of Rest)', idx_ushape, '#1f77b4'),          # Blue
        ('4. Inv-U Shape (Bottom 30% of Rest)', idx_inv_ushape, '#ff7f0e'),# Orange
        ('5. Baseline / Flat (Remaining)', idx_baseline, '#9467bd')       # Purple
    ]

    # --- 2. Plotting a 2x3 Grid ---
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, (title, indices, color) in enumerate(clusters):
        ax = axes[i]
        if not indices:
            ax.set_title(f"{title} (n=0)")
            continue
            
        cluster_trends = [metrics[idx]['trend'] for idx in range(len(metrics)) if metrics[idx]['idx'] in indices]
        cluster_mat = np.array(cluster_trends)
        
        # Suppress warnings for empty slices in nanmean/nanstd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_cluster_trend = np.nanmean(cluster_mat, axis=0)
            sem_cluster_trend = np.nanstd(cluster_mat, axis=0) / np.sqrt(len(indices))
        
        for trend in cluster_trends:
            ax.plot(x_full, trend, color='gray', alpha=0.10, linewidth=1)
            
        valid_mask = ~np.isnan(mean_cluster_trend)
        if np.any(valid_mask):
            ax.errorbar(x_full[valid_mask], mean_cluster_trend[valid_mask], yerr=sem_cluster_trend[valid_mask], 
                        fmt='o-', color=color, ecolor='black', elinewidth=1.5, capsize=4,
                        linewidth=3, markersize=8, label='Cluster Mean')
            
            # Baseline reference
            first_val = mean_cluster_trend[valid_mask][0]
            ax.axhline(y=first_val, color=color, linestyle=':', alpha=0.4)
            
        ax.set_title(f"{title}\n(n={len(indices)} | {len(indices)/num_analyzed:.1%})", fontsize=13, fontweight='bold')
        ax.set_xticks(x_full)
        ax.set_ylim(200, 1000)  # Fixed Y-axis range to 0-1000
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    # Remove the empty 6th subplot
    fig.delaxes(axes[5])

    # Global labels
    fig.text(0.5, 0.02, 'Consecutive Go Trial Number (Full Streak Length)', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.02, 0.5, 'Reaction Time (ms)', va='center', rotation='vertical', fontsize=16, fontweight='bold')
    fig.suptitle(f'Heterogeneous Local RT Strategies ({order_name} | n={num_analyzed})', fontsize=22, fontweight='bold', y=0.96)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.94])
    output_plot = OUTPUT_DIR / f'local_trend_{order_name}_groups.png'
    plt.savefig(output_plot, dpi=300)
    print(f"Plot saved to: {output_plot}")


def main():
    print(">>> Loading Order references...")
    try:
        # Prevent "int too large" error by reading sequence as string
        df_orders = pd.read_csv(ORDERS_FILE, dtype={'order_seq': str}) 
        seq_to_sig = dict(zip(df_orders['order_seq'], df_orders['order_sig']))
        
        target_seqs = {}
        if 'ALL' in TARGET_ORDER_SIGS:
            target_seqs = {row['order_sig']: row['order_seq'] for _, row in df_orders.iterrows()}
            print(f"Will process ALL {len(target_seqs)} orders.")
        else:
            for sig_name in TARGET_ORDER_SIGS:
                seqs = df_orders.loc[df_orders['order_sig'] == sig_name, 'order_seq'].values
                if len(seqs) > 0:
                    target_seqs[sig_name] = seqs[0]
            print(f"Target orders to process: {list(target_seqs.keys())}")
            
    except Exception as e:
        print(f"Failed to read orders.csv: {e}")
        return

    print(">>> Scanning subjects (Single Pass)...")
    try:
        df_list = pd.read_csv(LIST_FILE)
        file_list = df_list['filename'].tolist() if 'filename' in df_list.columns else df_list.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Failed to read the list file: {e}")
        return

    subject_mapping = []
    
    # Dictionary to store extracted trend data for each target order
    data_by_order = defaultdict(list)
    
    total_files = len(file_list)
    for i, fname in enumerate(file_list):
        full_path = DATA_ROOT / fname
        if not full_path.exists():
            full_path = DATA_ROOT / 'SST' / 'baseline_year_1_arm_1' / fname
            if not full_path.exists(): 
                continue
                
        sig, rts = extract_data(full_path)
        
        if sig and len(sig) == 360:
            order_sig_name = seq_to_sig.get(sig, "Unknown")
            subject_mapping.append({'filename': fname, 'order_sig': order_sig_name})
            
            # If this subject belongs to one of our TARGET orders, extract their trend
            if order_sig_name in target_seqs and len(rts) == 360:
                is_go = np.array([char == '0' for char in sig])
                streaks_b1 = extract_streaks(is_go[:180], rts[:180])
                streaks_b2 = extract_streaks(is_go[180:], rts[180:])
                all_subj_streaks = streaks_b1 + streaks_b2
                
                if all_subj_streaks:
                    max_len = max(len(s) for s in all_subj_streaks)
                    mat = np.full((len(all_subj_streaks), max_len), np.nan)
                    for idx_s, s in enumerate(all_subj_streaks):
                        mat[idx_s, :len(s)] = s
                    
                    # Suppress runtime warnings locally just for this mean calculation
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        mean_trend = np.nanmean(mat, axis=0)
                        
                    data_by_order[order_sig_name].append({'filename': fname, 'trend': mean_trend})

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{total_files}...")

    # Save Subject-Order Mapping
    df_map = pd.DataFrame(subject_mapping)
    df_map.to_csv(SUBJECT_ORDER_MAP_FILE, index=False)
    print(f"\nSaved subject->order mapping to: {SUBJECT_ORDER_MAP_FILE}")

    # Process each order and plot
    for order_name in target_seqs.keys():
        trends = data_by_order.get(order_name, [])
        if trends:
            analyze_and_plot_order(order_name, trends)
        else:
            print(f"No valid data found for {order_name}.")

    print("\n>>> All processing complete.")

if __name__ == "__main__":
    main()