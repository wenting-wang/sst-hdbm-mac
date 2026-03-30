import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure paths are correct and allow imports from the project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# ================= CONFIGURATION =================
LIST_FILE = BASE_DIR / 'data' / 'clinical_behavior.csv'
DATA_ROOT = Path('/Users/w/Desktop/data/sst_valid_base/')

# --- TARGET SUBJECT ---
# 0 = 1st valid subject, 1 = 2nd valid subject, 2 = 3rd, etc.
TARGET_SUBJECT_INDEX = 300

OUTPUT_PLOT = BASE_DIR / 'outputs' / f'local_trend_subject_{TARGET_SUBJECT_INDEX}.png'
# ===============================================

def load_single_subject_data(file_path: Path):
    """ Reads a single subject's file and extracts ExpCon and RT. """
    try:
        df = pd.read_csv(file_path, usecols=['sst_expcon', 'sst_primaryrt'])
        is_go = df['sst_expcon'].astype(str).str.strip() == 'GoTrial'
        rts = pd.to_numeric(df['sst_primaryrt'], errors='coerce').to_numpy()
        return is_go.to_numpy(), rts
    except Exception as e:
        print(f"Error parsing {file_path.name}: {e}")
        return None, None

def extract_streaks(is_go_array, rt_array):
    """ 
    Extracts streaks of continuous Go trials that occur AFTER a Stop trial.
    e.g., Stop -> Go(1) -> Go(2) -> Go(3) -> Stop
    Returns: A list of arrays, e.g., [[RT1, RT2, RT3], [RT4], ...]
    """
    streaks = []
    current_streak = []
    seen_first_stop = False 

    for go_flag, rt in zip(is_go_array, rt_array):
        if not go_flag: # Stop Trial
            seen_first_stop = True
            if len(current_streak) > 0:
                streaks.append(current_streak)
                current_streak = [] # Reset for the next streak
        else: # Go Trial
            if seen_first_stop:
                current_streak.append(rt) # Add this Go trial's RT to the current streak
                
    # If the block ends on a Go streak, add the last one
    if len(current_streak) > 0:
        streaks.append(current_streak)
        
    return streaks

def main():
    print(f">>> Searching for the valid subject at index {TARGET_SUBJECT_INDEX}...")
    
    try:
        df_list = pd.read_csv(LIST_FILE)
        file_list = df_list['filename'].tolist() if 'filename' in df_list.columns else df_list.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Failed to read the list file at {LIST_FILE}: {e}")
        return

    is_go, rts = None, None
    subj_filename = ""
    valid_count = 0
    
    for fname in file_list:
        full_path = DATA_ROOT / fname
        if not full_path.exists():
            full_path = DATA_ROOT / 'SST' / 'baseline_year_1_arm_1' / fname
            if not full_path.exists(): 
                continue
        
        temp_is_go, temp_rts = load_single_subject_data(full_path)
        
        if temp_is_go is not None and len(temp_is_go) == 360:
            if valid_count == TARGET_SUBJECT_INDEX:
                is_go = temp_is_go
                rts = temp_rts
                subj_filename = fname
                print(f"Found it! Selected subject file: {subj_filename}")
                break
            valid_count += 1

    if is_go is None:
        print(f"Could not find a valid subject at index {TARGET_SUBJECT_INDEX}. Total valid found: {valid_count}")
        return

    # --- 1. Extract Go Streaks with Block Split Logic ---
    # We split the 360 trials into Block 1 (0-180) and Block 2 (180-360)
    # This ensures a streak is strictly cut off at trial 180 and resets.
    is_go_b1, rts_b1 = is_go[:180], rts[:180]
    is_go_b2, rts_b2 = is_go[180:], rts[180:]
    
    streaks_b1 = extract_streaks(is_go_b1, rts_b1)
    streaks_b2 = extract_streaks(is_go_b2, rts_b2)
    
    # Combine streaks from both blocks for the local trend analysis
    all_streaks = streaks_b1 + streaks_b2

    # --- 2. Prepare data for plotting aligned streaks ---
    max_streak_len = max(len(s) for s in all_streaks) if all_streaks else 0
    streak_matrix = np.full((len(all_streaks), max_streak_len), np.nan)
    
    for i, s in enumerate(all_streaks):
        streak_matrix[i, :len(s)] = s

    # Calculate statistics
    mean_local_rt = np.nanmean(streak_matrix, axis=0)
    std_local_rt = np.nanstd(streak_matrix, axis=0)
    valid_n = np.sum(~np.isnan(streak_matrix), axis=0)
    
    # Calculate Standard Error (SEM) for the error bars
    sem_local_rt = np.divide(std_local_rt, np.sqrt(valid_n), out=np.zeros_like(std_local_rt), where=valid_n!=0)
    
    # --- 3. Plotting ---
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))
    
    # === Panel A: Continuous Raw View ===
    trials = np.arange(1, 361)
    ax1.plot(trials, rts, 'o-', color='#1f77b4', markersize=4, linewidth=1.2, label='Go Trial RT')
    
    stop_indices = np.where(~is_go)[0]
    for idx in stop_indices:
        ax1.axvline(x=idx + 1, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.axvline(x=-10, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Stop Trial')
    
    # Add a visual block split line
    ax1.axvline(x=180.5, color='black', linestyle=':', linewidth=2, label='Block Split')
    
    ax1.set_xlim(1, 360)
    ax1.set_title(f'Continuous RT over 360 Trials (Subject Index: {TARGET_SUBJECT_INDEX} | File: {subj_filename})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trial Number', fontsize=12)
    ax1.set_ylabel('Reaction Time (ms)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # === Panel B: Aligned Local Streaks ===
    positions = np.arange(1, max_streak_len + 1)
    
    # Plot individual streaks as faint lines in the background
    for s in all_streaks:
        ax2.plot(np.arange(1, len(s) + 1), s, color='gray', alpha=0.25, linewidth=1)
        
    # Plot the mean trend with Error Bars
    valid_mask = valid_n >= 2 # Only plot points that have at least 2 samples
    
    ax2.errorbar(positions[valid_mask], mean_local_rt[valid_mask], 
                 yerr=sem_local_rt[valid_mask], 
                 fmt='o-', color='darkorange', ecolor='darkred', elinewidth=2, capsize=5,
                 linewidth=3, markersize=8, label='Average Local Trend ± SEM')
    
    # Add text to show sample size (n)
    for pos, mean_val, sem_val, n in zip(positions[valid_mask], mean_local_rt[valid_mask], sem_local_rt[valid_mask], valid_n[valid_mask]):
        if not np.isnan(mean_val):
            # Place the text slightly above the top of the error bar to avoid overlap
            ax2.text(pos, mean_val + sem_val + 10, f'n={n}', ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

    ax2.set_title('Local RT Trend: Continuous "Go" Trials AFTER a "Stop" Trial', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Consecutive Go Trial Number (1 = First Go after a Stop)', fontsize=12)
    ax2.set_ylabel('Reaction Time (ms)', fontsize=12)
    ax2.set_xticks(positions)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Local trend plot saved successfully to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()