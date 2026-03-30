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

# Output paths
OUTPUT_PLOT = BASE_DIR / 'outputs' / 'rt_trend_order.png'
# ===============================================

def extract_data(file_path: Path) -> tuple[str, np.ndarray] | tuple[None, None]:
    """
    Reads a file and extracts both the sequence signature and the sst_primaryrt sequence.
    Returns: (signature_string, rt_array)
    """
    try:
        df = pd.read_csv(file_path, usecols=['sst_expcon', 'sst_primaryrt'])
        
        # 1. Extract Order Signature
        is_go = df['sst_expcon'].astype(str).str.strip() == 'GoTrial'
        sig_array = np.where(is_go, '0', '1')
        signature = "".join(sig_array)
        
        # 2. Extract Reaction Times
        rts = pd.to_numeric(df['sst_primaryrt'], errors='coerce').to_numpy()
        
        return signature, rts
        
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        return None, None
    except Exception as e:
        print(f"Error parsing {file_path.name}: {e}")
        return None, None

def main():
    print(">>> Starting Order-specific Reaction Time (RT) trend analysis (with Blocks)...")
    
    try:
        df_list = pd.read_csv(LIST_FILE)
        file_list = df_list['filename'].tolist() if 'filename' in df_list.columns else df_list.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Failed to read the list file at {LIST_FILE}: {e}")
        return

    order_groups = defaultdict(list)
    total_files = len(file_list)
    
    print(f"Scanning {total_files} files...")
    
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
        print("No valid data extracted. Exiting program.")
        return

    # Sort orders by the number of subjects
    sorted_orders = sorted(order_groups.items(), key=lambda item: len(item[1]), reverse=True)
    top_13_orders = sorted_orders[:13]
    
    print(f"\nData extraction complete. Found {len(order_groups)} unique orders.")
    print(f"Plotting the top {len(top_13_orders)} most frequent orders with Block trends...")

    # --- Plotting the Grid ---
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a 4x4 grid of subplots
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(24, 18), sharex=True, sharey=True)
    axes = axes.flatten()
    trials = np.arange(1, 361)
    
    for idx, (sig, rts_list) in enumerate(top_13_orders):
        ax = axes[idx]
        rt_matrix = np.array(rts_list)
        
        # Calculate statistics
        mean_rt = np.nanmean(rt_matrix, axis=0)
        std_rt = np.nanstd(rt_matrix, axis=0)
        valid_n_per_trial = np.sum(~np.isnan(rt_matrix), axis=0) 
        sem_rt = np.divide(std_rt, np.sqrt(valid_n_per_trial), out=np.zeros_like(std_rt), where=valid_n_per_trial!=0)
        
        # Plot Mean and SEM
        line_mean, = ax.plot(trials, mean_rt, color='#1f77b4', linewidth=1.5, alpha=0.9, label='Mean RT')
        fill_sem = ax.fill_between(trials, mean_rt - sem_rt, mean_rt + sem_rt, color='#1f77b4', alpha=0.3, label='SEM')
        
        valid_idx = ~np.isnan(mean_rt)
        slope_all, slope_b1, slope_b2 = 0.0, 0.0, 0.0
        
        # 1. Overall Trendline (1-360)
        if np.any(valid_idx):
            z_all = np.polyfit(trials[valid_idx], mean_rt[valid_idx], 1)
            p_all = np.poly1d(z_all)
            line_all, = ax.plot(trials, p_all(trials), "r--", linewidth=2.5, label='Overall Trend')
            slope_all = z_all[0]
            
        # 2. Block 1 Trendline (1-180)
        mask_b1 = (trials <= 180) & valid_idx
        if np.any(mask_b1):
            z_b1 = np.polyfit(trials[mask_b1], mean_rt[mask_b1], 1)
            p_b1 = np.poly1d(z_b1)
            line_b1, = ax.plot(trials[mask_b1], p_b1(trials[mask_b1]), "g--", linewidth=2.5, label='Block 1 Trend')
            slope_b1 = z_b1[0]
            
        # 3. Block 2 Trendline (181-360)
        mask_b2 = (trials > 180) & valid_idx
        if np.any(mask_b2):
            z_b2 = np.polyfit(trials[mask_b2], mean_rt[mask_b2], 1)
            p_b2 = np.poly1d(z_b2)
            line_b2, = ax.plot(trials[mask_b2], p_b2(trials[mask_b2]), "m--", linewidth=2.5, label='Block 2 Trend')
            slope_b2 = z_b2[0]
            
        # Add Block Split Line
        line_split = ax.axvline(x=180.5, color='black', linestyle=':', linewidth=1.5, alpha=0.6, label='Block Split')
        
        # Subplot aesthetics - Show all 3 slopes in the title concisely
        title_str = (f'Order {idx+1} (n={len(rts_list)})\n'
                     f'Slopes -> All: {slope_all:.4f} | B1: {slope_b1:.4f} | B2: {slope_b2:.4f}')
        ax.set_title(title_str, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlim(1, 360)
        ax.set_xticks([0, 90, 180, 270, 360])

    # Hide the unused subplots
    for j in range(len(top_13_orders), len(axes)):
        fig.delaxes(axes[j])

    # Add global labels
    fig.text(0.5, 0.02, 'Trial Number', ha='center', fontsize=18, fontweight='bold')
    fig.text(0.02, 0.5, 'Reaction Time (sst_primaryrt)', va='center', rotation='vertical', fontsize=18, fontweight='bold')
    fig.suptitle('RT Trends Over 360 Trials by Top 13 Orders', fontsize=24, fontweight='bold', y=0.98)
    
    # Add a global legend at the top of the figure to avoid cluttering subplots
    handles = [line_mean, line_all, line_b1, line_b2, line_split]
    labels = ['Mean Reaction Time & SEM', 'Overall Trend (Red)', 'Block 1 Trend (Green)', 'Block 2 Trend (Magenta)', 'Block Split (180)']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=5, fontsize=14, frameon=False)

    # Adjust layout to make room for global labels and legend
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.92]) 
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Grid plot saved successfully to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()