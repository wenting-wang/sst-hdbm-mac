"""
filter.py

Data Filtering and Quality Control (QC) for ABCD Stop-Signal Task (SST) Data.

This script filters the raw ABCD SST dataset based on strict metadata criteria 
and behavioral quality checks (e.g., excessively fast reaction times). Valid 
subject zip files are safely copied to a new directory for downstream processing.

Reference Documentation:
- NDA Data Structure: https://nda.nih.gov/data-structure/abcd_sst02
- Task Details: https://docs.abcdstudy.org/latest/documentation/imaging/type_trial.html#stop-signal-task-sst
"""

import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
# --- Base Directory Setup ---
# Users should update this path to point to their local data folder
# BASE_DIR = Path("/kyb/agpd/wwang/data_abcd/")  # <-- UPDATE THIS PATH
BASE_DIR = Path("/Users/w/Desktop/data/")  # <-- UPDATE THIS PATH

# --- Input Files & Directories ---
INPUT_CSV = BASE_DIR / "mri_y_tfmr_sst_beh.csv"
RAW_SOURCE_DIR = BASE_DIR / "abcd_sst_tlb01"

# --- Output Files & Directories ---
CLEAN_DATA_DIR = BASE_DIR / "sst_valid_base"
FOUND_FILES_TXT = BASE_DIR / "sst_valid_filenames.txt"


# ==============================================================================
# QUALITY CONTROL (QC) THRESHOLDS
# ==============================================================================
RT_FAST_THRESHOLD = 200  # Reaction time threshold in milliseconds
MAX_FAST_TRIALS = 3      # Maximum allowed fast trials (0, 1, 2, 3 are OK. 4 is a fail)


# ==============================================================================
# FILTERING FUNCTIONS
# ==============================================================================

def filter_metadata(file_path):
    """
    Loads the behavioral metadata CSV and applies strict inclusion criteria.
    
    Criteria enforced:
      - Baseline event only
      - Performance flag is valid (1)
      - No switch flags (0)
      - No glitch flags (0)
      - Violator flag is clear (0)
      - 0 SSD count is <= 3
      - Total trials (total_nt) exactly 360
      - Stop trials (s_nt) exactly 60
      
    Args:
        file_path (Path): Path to the `mri_y_tfmr_sst_beh.csv` file.
        
    Returns:
        pd.DataFrame: A filtered DataFrame containing only valid subjects.
    """
    print(f"Loading metadata from {file_path}...")
    df = pd.read_csv(file_path, index_col=False, low_memory=False)

    mask = (
        (df['eventname'].str.contains('baseline', case=False, na=False)) & 
        (df['tfmri_sst_beh_performflag'] == 1) &
        (df['tfmri_sst_beh_switchflag'] == 0) & 
        (df['tfmri_sst_beh_glitchflag'] == 0) &
        (df['tfmri_sst_beh_0ssdcount'] <= 3) &
        (df['tfmri_sst_beh_violatorflag'] == 0) &
        (df['tfmri_sst_all_beh_total_nt'] == 360) &
        (df['tfmri_sst_all_beh_s_nt'] == 60)
    )
    return df[mask].copy()


def check_rt_quality(zip_path):
    """
    Checks the raw trial data inside the subject's zip file to ensure reaction 
    times (RT) are reliable.
    
    Args:
        zip_path (Path): Path to the subject's raw data zip file.
        
    Returns:
        bool: True (Pass) if fast RTs are within acceptable limits, False (Fail) otherwise.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Find the target CSV, ignoring macOS hidden system files
            csv_files = [f for f in z.namelist() if f.endswith('sst.csv') and '__MACOSX' not in f]
            if not csv_files: 
                return False
            
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f, usecols=['sst_primaryrt'])
            
            # Extract valid Reaction Times
            valid_rts = df[df['sst_primaryrt'] > 0]['sst_primaryrt'].values
            
            # Count how many trials violate the fast threshold (strictly < 200ms)
            n_fast = np.sum(valid_rts < RT_FAST_THRESHOLD) 
            
            return n_fast <= MAX_FAST_TRIALS

    except Exception:
        # Fails safely if the zip file is corrupt or unreadable
        return False


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    # 1. Metadata Filter
    df_filtered = filter_metadata(INPUT_CSV)
    print(f"Subjects passing metadata: {len(df_filtered)}")

    # 2. File Copy & Content Check
    if not CLEAN_DATA_DIR.exists():
        CLEAN_DATA_DIR.mkdir(parents=True)
    
    stats = {'copied': 0, 'missing_file': 0, 'bad_rt': 0}
    
    print(f"Processing files from {RAW_SOURCE_DIR}...")
    
    for _, row in df_filtered.iterrows():
        # Construct expected ABCD filename format
        filename = f"{row['src_subject_id']}_{row['eventname']}_sst.csv.zip"
        src = RAW_SOURCE_DIR / filename
        dest = CLEAN_DATA_DIR / filename
        
        # Check if the raw file exists for this subject
        if not src.exists():
            stats['missing_file'] += 1
            continue
            
        # Check RT quality and copy if passed
        if check_rt_quality(src):
            shutil.copy2(src, dest)
            stats['copied'] += 1
        else:
            stats['bad_rt'] += 1

    # 3. Final Report
    print("\n" + "="*40)
    print("FINAL FILTERING REPORT")
    print("="*40)
    print(f"Metadata Valid Subjects : {len(df_filtered)}")
    print(f" - Missing Raw Files    : {stats['missing_file']}")
    print(f" - Failed RT Quality    : {stats['bad_rt']}")
    print(f" - Successfully Copied  : {stats['copied']}")
    print("="*40)

    # 4. Save Execution Log
    with open(FOUND_FILES_TXT, 'w') as f:
        for file_path in sorted(CLEAN_DATA_DIR.glob('*.zip')):
            f.write(f"{file_path.name}\n")
    print(f"List saved to {FOUND_FILES_TXT}")


if __name__ == "__main__":
    main()

