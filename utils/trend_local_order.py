import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Ensure paths are correct and allow imports from the project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# ================= CONFIGURATION =================
LIST_FILE = BASE_DIR / 'data' / 'clinical_behavior.csv'
DATA_ROOT = Path('/Users/w/Desktop/data/sst_valid_base/')

# --- TARGET ORDER ---
# 0 = Most frequent order, 1 = 2nd most frequent, etc.
TARGET_ORDER_INDEX = 1

OUTPUT_PLOT = BASE_DIR / 'outputs' / f'local_trend_order_{TARGET_ORDER_INDEX}_aggregate.png'
# ===============================================

def extract_data(file_path: Path):
    """ Reads a file and extracts both the sequence signature and the RT sequence. """
    try:
        df = pd.read_csv(file_path, usecols=['sst_expcon', 'sst_primaryrt'])
        is_go = df['sst_expcon'].astype(str).str.strip() == 'GoTrial'
        sig = "".join(np.where(is_go, '0', '1'))
        rts = pd.to_numeric(df['sst_primaryrt'], errors='coerce').to_numpy()
        return sig, rts
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        return None, None
    except Exception as e:
        print(f"Error parsing {file_path.name}: {e}")
        return None, None

def extract_streaks(is_go_array, rt_array):
    """ Extracts streaks of continuous Go trials that occur AFTER a Stop trial. """
    streaks = []
    current_streak = []
    seen_first_stop = False 

    for go_flag, rt in zip(is_go_array, rt_array):
        if not go_flag: # Stop Trial
            seen_first_stop = True
            if len(current_streak) > 0:
                streaks.append(current_streak)
                current_streak = []
        else: # Go Trial
            if seen_first_stop:
                current_streak.append(rt)
                
    if len(current_streak) > 0:
        streaks.append(current_streak)
        
    return streaks

def main():
    print(f">>> Scanning data to aggregate Local Trend for Order Index {TARGET_ORDER_INDEX}...")
    
    try:
        df_list = pd.read_csv(LIST_FILE)
        file_list = df_list['filename'].tolist() if 'filename' in df_list.columns else df_list.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Failed to read the list file at {LIST_FILE}: {e}")
        return

    order_groups = defaultdict(list)
    total_files = len(file_list)
    
    for i, fname in enumerate(file_list):
        full_path = DATA_ROOT / fname
        if not full_path.exists():
            full_path = DATA_ROOT / 'SST' / 'baseline_year_1_arm_1' / fname
            if not full_path.exists(): 
                continue
        
        sig, rts = extract_data(full_path)
        if sig and rts is not None and len(sig) == 360 and len(rts) == 360:
            order_groups[sig].append(rts)
            
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{total_files}...")

    if not order_groups:
        print("No valid data extracted.")
        return

    # Sort orders by frequency
    sorted_orders = sorted(order_groups.items(), key=lambda item: len(item[1]), reverse=True)
    
    if TARGET_ORDER_INDEX >= len(sorted_orders):
        print(f"Error: Requested Order Index {TARGET_ORDER_INDEX} is out of bounds (max {len(sorted_orders)-1}).")
        return
        
    target_sig, rts_list = sorted_orders[TARGET_ORDER_INDEX]
    num_subjects = len(rts_list)
    print(f"\nSelected Order {TARGET_ORDER_INDEX} (n={num_subjects} subjects).")
    
    # Reconstruct boolean Go array from signature ('0' = Go, '1' = Stop)
    is_go = np.array([char == '0' for char in target_sig])

    # --- 1. Panel A Data: Average Continuous RT for this Order ---
    rt_matrix_continuous = np.array(rts_list)
    mean_continuous_rt = np.nanmean(rt_matrix_continuous, axis=0)

    # --- 2. Panel B Data: Extract all streaks from all subjects ---
    all_order_streaks = []
    
    for rt_array in rts_list:
        # Respect the Block Split (0-180, 180-360)
        is_go_b1, rts_b1 = is_go[:180], rt_array[:180]
        is_go_b2, rts_b2 = is_go[180:], rt_array[180:]
        
        streaks_b1 = extract_streaks(is_go_b1, rts_b1)
        streaks_b2 = extract_streaks(is_go_b2, rts_b2)
        
        all_order_streaks.extend(streaks_b1 + streaks_b2)

    # Prepare data for Panel B plotting
    max_streak_len = max(len(s) for s in all_order_streaks) if all_order_streaks else 0
    streak_matrix = np.full((len(all_order_streaks), max_streak_len), np.nan)
    
    for i, s in enumerate(all_order_streaks):
        streak_matrix[i, :len(s)] = s

    mean_local_rt = np.nanmean(streak_matrix, axis=0)
    std_local_rt = np.nanstd(streak_matrix, axis=0)
    valid_n = np.sum(~np.isnan(streak_matrix), axis=0)
    sem_local_rt = np.divide(std_local_rt, np.sqrt(valid_n), out=np.zeros_like(std_local_rt), where=valid_n!=0)

    # --- 3. Plotting ---
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))
    
    # === Panel A: Average Continuous View ===
    trials = np.arange(1, 361)
    ax1.plot(trials, mean_continuous_rt, 'o-', color='#1f77b4', markersize=4, linewidth=1.5, label='Mean Go Trial RT')
    
    stop_indices = np.where(~is_go)[0]
    for idx in stop_indices:
        ax1.axvline(x=idx + 1, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    ax1.axvline(x=-10, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Stop Trial')
    
    ax1.axvline(x=180.5, color='black', linestyle=':', linewidth=2, label='Block Split')
    
    ax1.set_xlim(1, 360)
    ax1.set_title(f'Order {TARGET_ORDER_INDEX} Aggregate: Average RT over 360 Trials (n = {num_subjects} subjects)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trial Number', fontsize=12)
    ax1.set_ylabel('Reaction Time (ms)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # === Panel B: Average Local Streaks ===
    positions = np.arange(1, max_streak_len + 1)
    
    # Due to potentially huge number of streaks, we only plot the individual streaks
    # with a very low alpha to form a "density cloud", or you can comment this out if it's too heavy.
    for s in all_order_streaks:
        ax2.plot(np.arange(1, len(s) + 1), s, color='gray', alpha=0.01, linewidth=1)
        
    # Plot the mean trend with Error Bars
    # Require at least 5% of subjects to have reached this streak length to plot it, to avoid noisy tails
    min_n_required = max(5, int(num_subjects * 0.05)) 
    valid_mask = valid_n >= min_n_required 
    
    ax2.errorbar(positions[valid_mask], mean_local_rt[valid_mask], 
                 yerr=sem_local_rt[valid_mask], 
                 fmt='o-', color='darkorange', ecolor='darkred', elinewidth=2, capsize=5,
                 linewidth=3, markersize=8, label='Average Local Trend ± SEM')
    
    for pos, mean_val, sem_val, n in zip(positions[valid_mask], mean_local_rt[valid_mask], sem_local_rt[valid_mask], valid_n[valid_mask]):
        if not np.isnan(mean_val):
            ax2.text(pos, mean_val + sem_val + 5, f'n={n}', ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')

    ax2.set_title('Aggregated Local RT Trend: Continuous "Go" Trials AFTER a "Stop" Trial', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Consecutive Go Trial Number (1 = First Go after a Stop)', fontsize=12)
    ax2.set_ylabel('Reaction Time (ms)', fontsize=12)
    ax2.set_xticks(positions)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Order aggregate plot saved successfully to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()