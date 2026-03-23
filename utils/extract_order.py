import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# Ensure paths are correct and allow imports from the project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# ================= CONFIGURATION =================
# Input file list path (relative to the project root)
LIST_FILE = BASE_DIR / 'data' / 'clinical_behavior.csv'

# Data root directory 
DATA_ROOT = Path('/Users/w/Desktop/data/sst_valid_base/')

OUTPUT_CSV = BASE_DIR / 'data' / 'orders.csv'
# ===============================================

def get_file_signature(file_path: Path) -> str | None:
    """
    Reads a file and extracts the sst_expcon sequence as a unique signature.
    Returns: A string composed of '0' (Go) and '1' (Stop).
    """
    try:
        # Only read the sst_expcon column to speed up I/O
        df = pd.read_csv(file_path, usecols=['sst_expcon'])
        
        # Use vectorized operations instead of .apply() to significantly boost speed
        is_go = df['sst_expcon'].astype(str).str.strip() == 'GoTrial'
        sig_array = np.where(is_go, '0', '1')
        
        return "".join(sig_array)
        
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        # Silently ignore missing files, missing columns, or empty files
        return None
    except Exception as e:
        print(f"Error parsing {file_path.name}: {e}")
        return None

def main():
    print(">>> Starting to calculate Order distribution...")
    
    try:
        df_list = pd.read_csv(LIST_FILE)
        # Compatible with different column names: prioritize 'filename', otherwise use the first column
        file_list = df_list['filename'].tolist() if 'filename' in df_list.columns else df_list.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Failed to read the list file at {LIST_FILE}: {e}")
        return

    signature_counts = Counter()
    total_files = len(file_list)
    print(f"Scanning {total_files} files...")
    
    valid_count = 0
    
    for i, fname in enumerate(file_list):
        # Elegantly handle paths using pathlib
        full_path = DATA_ROOT / fname
        if not full_path.exists():
            full_path = DATA_ROOT / 'SST' / 'baseline_year_1_arm_1' / fname
            if not full_path.exists(): 
                continue
        
        sig = get_file_signature(full_path)
        
        if sig:
            signature_counts[sig] += 1
            valid_count += 1
        
        if (i + 1) % 500 == 0: 
            print(f"Processed {i + 1}/{total_files}...")

    print(f"\nScan complete. Valid files: {valid_count}")
    print(f"Found {len(signature_counts)} different Orders.")

    if not signature_counts:
        print("No valid sequences extracted, exiting program.")
        return

    # --- Convert to DataFrame and save ---
    # Build data dictionary using list comprehension for conciseness
    data = [
        {'order_sig': f'order_{i}', 'order_seq': seq, 'subj_cnt': count}
        for i, (seq, count) in enumerate(signature_counts.items())
    ]
    
    df_out = pd.DataFrame(data)
    
    # Sort by subject count in descending order
    df_out = df_out.sort_values(by='subj_cnt', ascending=False)
    
    # Ensure the output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # len of seq should be 360, remove any rows that don't match this length (data quality check)
    df_out = df_out[df_out['order_seq'].str.len() == 360]
    
    # Sort and save
    df_out = df_out.sort_values(
    by='order_sig', 
    key=lambda col: col.str.replace('order_', '').astype(int),
    ascending=True)
    
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()