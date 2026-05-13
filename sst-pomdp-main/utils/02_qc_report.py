"""
qc_stats.py

Quality Control (QC) Statistics Generator for ABCD Stop-Signal Task (SST).

This script calculates the dropout/exclusion rates for each QC criterion applied 
to the ABCD SST dataset. It evaluates both the behavioral metadata flags and 
the raw trial-level data (reaction times) inside the subject zip files. 

The script outputs a summary table to the console and saves the exact exclusion 
counts and percentages to a CSV file.

================================================================================
USAGE & CONFIGURATION
================================================================================
Note: This script requires access to the RAW uncompressed dataset from the NDA 
(`abcd_sst_tlb01` zip files) and the metadata CSV (`mri_y_tfmr_sst_beh.csv`). 
It cannot be run on the pre-processed example data.

Users must update the `BASE_DIR` in the Configuration section below to point to 
their local or HPC data directory.
================================================================================
"""

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
# --- Base Directory Setup ---
# Update this path to where your raw ABCD data and metadata CSV are stored.
# BASE_DIR = Path("/kyb/agpd/wwang/data_abcd/")  # <-- UPDATE THIS PATH
BASE_DIR = Path("/Users/w/Desktop/data/")  # <-- UPDATE THIS PATH

# --- Input Files & Directories ---
INPUT_CSV = BASE_DIR / "mri_y_tfmr_sst_beh.csv"
RAW_DATA_DIR = BASE_DIR / "abcd_sst_tlb01"

# --- Output Files ---
OUTPUT_STATS_CSV = BASE_DIR / "sst_final_qc_stats.csv"

# --- Quality Control (QC) Thresholds ---
RT_THRESHOLD = 200       # Reaction time threshold in ms (Strictly < 200)
MAX_ALLOWED_COUNT = 3    # Max allowed fast trials (0, 1, 2, 3 are OK. 4 is a fail)


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def check_rt_status(zip_path: Path) -> str:
    """
    Checks the status of the raw zip file and evaluates reaction time quality.
    
    Args:
        zip_path (Path): Path to the subject's raw data zip file.
        
    Returns:
        str: Status code indicating file quality:
            - 'MISSING': File does not exist.
            - 'BAD_ZIP': File exists but cannot be read or contains no CSV.
            - 'FAIL_RT': Exists, readable, but fails RT check (> 3 fast trials).
            - 'PASS': File exists, is readable, and passes all checks.
    """
    if not zip_path.exists():
        return 'MISSING'

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Find the target CSV, ignoring macOS hidden system files
            csv_files = [f for f in z.namelist() if f.endswith('sst.csv') and '__MACOSX' not in f]
            if not csv_files:
                return 'BAD_ZIP'
            
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f, usecols=['sst_primaryrt'])
            
            # Extract valid Reaction Times
            valid_rts = df[df['sst_primaryrt'] > 0]['sst_primaryrt'].values
            
            # Count fast trials (Strictly < 200ms)
            fast_count = np.sum(valid_rts < RT_THRESHOLD)
            
            if fast_count > MAX_ALLOWED_COUNT:
                return 'FAIL_RT'
            else:
                return 'PASS'

    except Exception:
        # Fails safely if the zip file is corrupt
        return 'BAD_ZIP'


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print(f"Loading metadata from {INPUT_CSV}...")
    if not INPUT_CSV.exists():
        print(f"Error: Input CSV not found at {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, low_memory=False)

    # --- 1. Define Baseline Population ---
    df_baseline = df[df['eventname'].str.contains('baseline', case=False, na=False)].copy()
    total_baseline = len(df_baseline)
    print(f"Total Baseline Subjects: {total_baseline}")

    # --- 2. Metadata Criteria (Rules 1-6) ---
    criteria_masks = {
        "(1) Trial Counts": df_baseline['tfmri_sst_beh_performflag'] == 1,
        "(2) No Switching": df_baseline['tfmri_sst_beh_switchflag'] == 0,
        "(3) No Glitch": df_baseline['tfmri_sst_beh_glitchflag'] == 0,
        "(4) 0ms SSD <= 3": df_baseline['tfmri_sst_beh_0ssdcount'] <= 3,
        "(5) No Violators": df_baseline['tfmri_sst_beh_violatorflag'] == 0,
        "(6) Complete Session": (df_baseline['tfmri_sst_all_beh_total_nt'] == 360) & 
                                (df_baseline['tfmri_sst_all_beh_s_nt'] == 60)
    }

    # --- 3. Check Raw Data (Rules 7 & 8) ---
    print(f"Checking {total_baseline} subjects for data availability and RT quality...")
    
    missing_data_mask = []
    rt_fail_mask = [] # Tracks IF data exists, does it fail RT?
    
    for i, (idx, row) in enumerate(df_baseline.iterrows()):
        if i % 1000 == 0:
            print(f"  Processed {i}/{total_baseline}...")

        subject_id = row['src_subject_id']
        event_name = row['eventname']
        filename = f"{subject_id}_{event_name}_sst.csv.zip"
        file_path = RAW_DATA_DIR / filename
        
        status = check_rt_status(file_path)
        
        # Criterion 7: Data Availability
        if status == 'MISSING':
            missing_data_mask.append(False) # Failed availability
        else:
            missing_data_mask.append(True)  # Passed availability
            
        # Criterion 8: RT Quality 
        # Note: If missing, we mark as False for "Passing", but distinct from 7.
        # To keep "Independent" counts clean, we count "RT Failures" strictly 
        # as files that exist BUT fail the content check.
        if status == 'FAIL_RT':
            rt_fail_mask.append(False) # Failed RT
        elif status == 'PASS':
            rt_fail_mask.append(True)  # Passed RT
        else:
            # MISSING or BAD_ZIP -> These are data availability issues, not RT 
            # content failures per se. But for final intersection, they must be False.
            rt_fail_mask.append(False) 

    # Convert lists to Pandas Series
    s_missing = pd.Series(missing_data_mask, index=df_baseline.index)
    s_rt_pass = pd.Series(rt_fail_mask, index=df_baseline.index) 

    # --- 4. Generate Statistics ---
    stats_rows = []
    
    # Process Metadata Rules (1-6)
    for name, mask in criteria_masks.items():
        n_failed = (~mask).sum()
        stats_rows.append({
            "Criteria": name, 
            "Excluded_N": n_failed, 
            "Excluded_Pct": (n_failed / total_baseline) * 100
        })

    # Process Criterion 7: Missing Raw Data
    n_missing = (~s_missing).sum()
    stats_rows.append({
        "Criteria": "(7) Missing Raw Data", 
        "Excluded_N": n_missing, 
        "Excluded_Pct": (n_missing / total_baseline) * 100
    })

    # Process Criterion 8: RT Quality Check
    # "Excluded by RT Check" = File Exists AND Fails Check.
    n_rt_fail_only = ((s_missing) & (~s_rt_pass)).sum()
    stats_rows.append({
        "Criteria": "(8) RT < 200ms Check (Found Files)", 
        "Excluded_N": n_rt_fail_only, 
        "Excluded_Pct": (n_rt_fail_only / total_baseline) * 100
    })

    # Calculate Final Intersection (Must pass all rules)
    final_mask = pd.Series([True] * total_baseline, index=df_baseline.index)
    for mask in criteria_masks.values():
        final_mask = final_mask & mask
    
    final_mask = final_mask & s_missing & s_rt_pass
    final_valid_count = final_mask.sum()

    # --- 5. Output Results ---
    print("\n" + "="*60)
    print(f"{'Criteria ID':<30} | {'Excluded (N)':<12} | {'Excluded (%)':<10}")
    print("-" * 60)
    for row in stats_rows:
        print(f"{row['Criteria']:<30} | {row['Excluded_N']:<12} | {row['Excluded_Pct']:.1f}%")
    print("-" * 60)
    print(f"{'FINAL VALID DATASET':<30} | {final_valid_count:<12} | {(final_valid_count / total_baseline) * 100:.1f}%")
    print("="*60)

    # Save to disk
    pd.DataFrame(stats_rows).to_csv(OUTPUT_STATS_CSV, index=False)
    print(f"QC Statistics saved to {OUTPUT_STATS_CSV}")


if __name__ == "__main__":
    main()

# ==============================================================================
# EXAMPLE EXPECTED OUTPUT
# ==============================================================================
# Criteria ID                    | Excluded (N) | Excluded (%)
# ------------------------------------------------------------
# (1) Trial Counts               | 1392         | 12.1%
# (2) No Switching               | 188          | 1.6%
# (3) No Glitch                  | 308          | 2.7%
# (4) 0ms SSD <= 3               | 3817         | 33.2%
# (5) No Violators               | 854          | 7.4%
# (6) Complete Session           | 252          | 2.2%
# (7) Missing Raw Data           | 1299         | 11.3%
# (8) RT < 200ms Check (Found Files) | 3058         | 26.6%
# ------------------------------------------------------------
# FINAL VALID DATASET            | 5345         | 46.5%
# ==============================================================================