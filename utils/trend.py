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

# Output paths
OUTPUT_PLOT = BASE_DIR / 'outputs' / 'rt_trend.png'
OUTPUT_CSV_STATS = BASE_DIR / 'data' / 'rt_trend_stats.csv'
# ===============================================

def extract_rt_data(file_path: Path) -> np.ndarray | None:
    """
    Reads a file and extracts the sst_primaryrt sequence.
    Returns: A numpy array of floats representing reaction times.
    """
    try:
        df = pd.read_csv(file_path, usecols=['sst_primaryrt'])
        rts = pd.to_numeric(df['sst_primaryrt'], errors='coerce').to_numpy()
        return rts
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        return None
    except Exception as e:
        print(f"Error parsing {file_path.name}: {e}")
        return None

def main():
    print(">>> Starting Reaction Time (RT) trend analysis...")
    
    # 1. Load or Calculate Data
    if OUTPUT_CSV_STATS.exists():
        print(f"Found existing stats file at {OUTPUT_CSV_STATS}. Loading data directly...")
        df_stats = pd.read_csv(OUTPUT_CSV_STATS)
        trials = df_stats['trial'].to_numpy()
        mean_rt = df_stats['mean_rt'].to_numpy()
        sem_rt = df_stats['sem_rt'].to_numpy()
    else:
        print("Stats file not found. Calculating from raw data...")
        try:
            df_list = pd.read_csv(LIST_FILE)
            file_list = df_list['filename'].tolist() if 'filename' in df_list.columns else df_list.iloc[:, 0].tolist()
        except Exception as e:
            print(f"Failed to read the list file at {LIST_FILE}: {e}")
            return

        all_rts = []
        total_files = len(file_list)
        print(f"Scanning {total_files} files...")
        
        for i, fname in enumerate(file_list):
            full_path = DATA_ROOT / fname
            if not full_path.exists():
                full_path = DATA_ROOT / 'SST' / 'baseline_year_1_arm_1' / fname
                if not full_path.exists(): 
                    continue
            
            rts = extract_rt_data(full_path)
            if rts is not None and len(rts) == 360:
                all_rts.append(rts)
            
            if (i + 1) % 500 == 0: 
                print(f"Processed {i + 1}/{total_files}...")

        if not all_rts:
            print("No valid RT sequences extracted. Exiting program.")
            return

        print(f"\nScan complete. Valid files with 360 trials: {len(all_rts)}")
        
        rt_matrix = np.array(all_rts)
        mean_rt = np.nanmean(rt_matrix, axis=0)
        std_rt = np.nanstd(rt_matrix, axis=0)
        valid_n_per_trial = np.sum(~np.isnan(rt_matrix), axis=0) 
        
        sem_rt = np.divide(std_rt, np.sqrt(valid_n_per_trial), out=np.zeros_like(std_rt), where=valid_n_per_trial!=0)

        # Save aggregated data
        df_stats = pd.DataFrame({
            'trial': np.arange(1, 361),
            'mean_rt': mean_rt,
            'sem_rt': sem_rt,
            'valid_subjects': valid_n_per_trial
        })
        OUTPUT_CSV_STATS.parent.mkdir(parents=True, exist_ok=True)
        df_stats.to_csv(OUTPUT_CSV_STATS, index=False)
        print(f"Statistics saved to: {OUTPUT_CSV_STATS}")
        
        trials = df_stats['trial'].to_numpy()

    # 2. Plotting the trend
    print(">>> Plotting trends with error bars...")
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(16, 8))
    
    # Plot Mean RT and Error Bars
    plt.plot(trials, mean_rt, color='#1f77b4', linewidth=1.5, label='Mean Reaction Time', alpha=0.8)
    plt.fill_between(trials, mean_rt - sem_rt, mean_rt + sem_rt, color='#1f77b4', alpha=0.2, label='Standard Error (SEM)')
    
    # Add a vertical line to separate Block 1 and Block 2
    plt.axvline(x=180.5, color='black', linestyle=':', linewidth=2, label='Block 1 / Block 2 split')
    
    # Calculate Trendlines
    valid_idx = ~np.isnan(mean_rt)
    
    if np.any(valid_idx):
        # Overall Trend (1-360)
        z_all = np.polyfit(trials[valid_idx], mean_rt[valid_idx], 1)
        p_all = np.poly1d(z_all)
        plt.plot(trials, p_all(trials), "r--", linewidth=2.5, label=f'Overall Trend (Slope: {z_all[0]:.4f})')
        
        # Block 1 Trend (1-180)
        mask_b1 = (trials <= 180) & valid_idx
        if np.any(mask_b1):
            z_b1 = np.polyfit(trials[mask_b1], mean_rt[mask_b1], 1)
            p_b1 = np.poly1d(z_b1)
            plt.plot(trials[mask_b1], p_b1(trials[mask_b1]), "g--", linewidth=2.5, label=f'Block 1 Trend (Slope: {z_b1[0]:.4f})')
        
        # Block 2 Trend (181-360)
        mask_b2 = (trials > 180) & valid_idx
        if np.any(mask_b2):
            z_b2 = np.polyfit(trials[mask_b2], mean_rt[mask_b2], 1)
            p_b2 = np.poly1d(z_b2)
            plt.plot(trials[mask_b2], p_b2(trials[mask_b2]), "m--", linewidth=2.5, label=f'Block 2 Trend (Slope: {z_b2[0]:.4f})')

        # Print slopes to console
        print("\n--- Trend Analysis Results ---")
        print(f"Overall Slope: {z_all[0]:.4f} units/trial")
        if np.any(mask_b1): print(f"Block 1 Slope: {z_b1[0]:.4f} units/trial")
        if np.any(mask_b2): print(f"Block 2 Slope: {z_b2[0]:.4f} units/trial")
        print("------------------------------\n")

    # Aesthetics
    plt.title('Average sst_primaryrt Trend Over 360 Trials (Block 1 vs Block 2)', fontsize=16)
    plt.xlabel('Trial Number', fontsize=14)
    plt.ylabel('Reaction Time (sst_primaryrt)', fontsize=14)
    plt.xlim(1, 360)
    
    # Add minor ticks and grid for better readability
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Position legend outside or intelligently inside
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.tight_layout()

    # Save the plot
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Plot saved successfully to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()