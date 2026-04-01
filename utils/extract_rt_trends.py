import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
LIST_FILE = BASE_DIR / 'data' / 'clinical_behavior.csv'
DATA_ROOT = Path('/Users/w/Desktop/data/sst_valid_base/')
ORDERS_FILE = BASE_DIR / 'data' / 'orders.csv'
OUTPUT_FILE = BASE_DIR / 'data' / 'subject_local_trends.csv'

def extract_data(file_path: Path):
    try:
        df = pd.read_csv(file_path, usecols=['sst_expcon', 'sst_go_rt'])
        is_go = df['sst_expcon'].astype(str).str.strip() == 'GoTrial'
        sig = "".join(np.where(is_go, '0', '1'))
        rts = pd.to_numeric(df['sst_go_rt'], errors='coerce').to_numpy()
        rts = np.where(rts <= 200, np.nan, rts)
        return sig, rts
    except Exception:
        return None, None

def extract_streaks(is_go_array, rt_array):
    streaks, current_streak = [], []
    seen_first_stop = False 
    for go_flag, rt in zip(is_go_array, rt_array):
        if not go_flag:
            seen_first_stop = True
            if current_streak:
                streaks.append(current_streak)
                current_streak = []
        else:
            if seen_first_stop:
                current_streak.append(rt)
    if current_streak:
        streaks.append(current_streak)
    return streaks

def main():
    print(">>> Scanning subjects to calculate Linear and Quadratic trends...")
    df_orders = pd.read_csv(ORDERS_FILE, dtype={'order_seq': str}) 
    seq_to_sig = dict(zip(df_orders['order_seq'], df_orders['order_sig']))
    
    df_list = pd.read_csv(LIST_FILE)
    file_list = df_list['filename'].tolist() if 'filename' in df_list.columns else df_list.iloc[:, 0].tolist()

    results = []
    
    for i, fname in enumerate(file_list):
        full_path = DATA_ROOT / fname
        if not full_path.exists():
            # Try subfolder structure
            full_path = DATA_ROOT / 'SST' / 'baseline_year_1_arm_1' / fname
            if not full_path.exists(): continue
                
        sig, rts = extract_data(full_path)
        
        if sig and len(sig) == 360:
            order_sig = seq_to_sig.get(sig, "Unknown")
            is_go = np.array([char == '0' for char in sig])
            
            # Split into two blocks to reset streaks
            streaks_b1 = extract_streaks(is_go[:180], rts[:180])
            streaks_b2 = extract_streaks(is_go[180:], rts[180:])
            all_subj_streaks = streaks_b1 + streaks_b2
            
            if all_subj_streaks:
                max_len = max(len(s) for s in all_subj_streaks)
                mat = np.full((len(all_subj_streaks), max_len), np.nan)
                for idx_s, s in enumerate(all_subj_streaks):
                    mat[idx_s, :len(s)] = s
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_trend = np.nanmean(mat, axis=0)
                
                valid_idx = ~np.isnan(mean_trend)
                x_valid = np.arange(1, max_len + 1)[valid_idx]
                y_valid = mean_trend[valid_idx]
                
                # Need at least 4 points to fit a reliable curve
                if len(x_valid) >= 4:
                    z1 = np.polyfit(x_valid, y_valid, 1) # Linear [slope, intercept]
                    z2 = np.polyfit(x_valid, y_valid, 2) # Quadratic [curve, slope, intercept]
                    
                    subject_id = fname.split('_baseline_')[0]
                    results.append({
                        'subject_id': subject_id,
                        'order_sig': order_sig,
                        'linear_slope': z1[0],
                        'quadratic_curve': z2[0],
                        'max_streak': len(x_valid)
                    })

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(file_list)}...")

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_FILE, index=False)
    print(f"\n>>> Done! Saved {len(df_results)} subjects to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()