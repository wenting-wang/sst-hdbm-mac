"""
04_attrition_report.py

Full Data Attrition Report for the ABCD SST Dataset.

This script calculates independent exclusion counts for all QC criteria. 
Note: Exclusion categories are NOT mutually exclusive. A single subject 
may fail multiple criteria and be counted in several rows.
"""

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# BASE_DIR = Path('/kyb/agpd/wwang/data_abcd/')
BASE_DIR = Path("/Users/w/Desktop/data/")  # <-- UPDATE THIS PATH

INPUT_CSV = BASE_DIR / "mri_y_tfmr_sst_beh.csv"
RAW_DATA_DIR = BASE_DIR / "abcd_sst_tlb01"
VALID_TXT = BASE_DIR / "sst_valid_filenames.txt"
FINAL_DATA_CSV = BASE_DIR / "clinical_behavior.csv"
UNFILTERED_STATS_CSV = BASE_DIR / "unfiltered_behavioral_stats.csv"

RT_THRESHOLD = 200
MAX_ALLOWED_COUNT = 3
THRESHOLD = 0.1

def check_rt_status(zip_path: Path) -> str:
    """Checks file availability and RT quality directly from raw zip."""
    if not zip_path.exists():
        return 'MISSING'

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_files = [f for f in z.namelist() if f.endswith('sst.csv') and '__MACOSX' not in f]
            if not csv_files:
                return 'BAD_ZIP'
            
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f, usecols=['sst_primaryrt'])
            
            valid_rts = df[df['sst_primaryrt'] > 0]['sst_primaryrt'].values
            fast_count = np.sum(valid_rts < RT_THRESHOLD)
            
            if fast_count > MAX_ALLOWED_COUNT:
                return 'FAIL_RT'
            else:
                return 'PASS'
    except Exception:
        return 'BAD_ZIP'


def main():
    print("Loading metadata and evaluating attrition criteria...")
    if not INPUT_CSV.exists():
        print(f"Error: Input CSV not found at {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df_baseline = df[df['eventname'].str.contains('baseline', case=False, na=False)].copy()
    total_baseline = len(df_baseline)

    # --- Rules 1-6 (Independent Masks) ---
    criteria_masks = {
        "(1) Trial Counts": df_baseline['tfmri_sst_beh_performflag'] == 1,
        "(2) No Switching": df_baseline['tfmri_sst_beh_switchflag'] == 0,
        "(3) No Glitch": df_baseline['tfmri_sst_beh_glitchflag'] == 0,
        "(4) 0ms SSD <= 3": df_baseline['tfmri_sst_beh_0ssdcount'] <= 3,
        "(5) No Violators": df_baseline['tfmri_sst_beh_violatorflag'] == 0,
        "(6) Complete Session": (df_baseline['tfmri_sst_all_beh_total_nt'] == 360) & 
                                (df_baseline['tfmri_sst_all_beh_s_nt'] == 60)
    }

    # --- Rules 7-8 (Independent Checks) ---
    missing_data_mask = []
    rt_fail_mask = []
    
    print("Checking raw data availability and RT quality (this may take a minute)...")
    for i, (idx, row) in enumerate(df_baseline.iterrows()):
        subject_id = row['src_subject_id']
        event_name = row['eventname']
        file_path = RAW_DATA_DIR / f"{subject_id}_{event_name}_sst.csv.zip"
        
        status = check_rt_status(file_path)
        
        missing_data_mask.append(status != 'MISSING')
        rt_fail_mask.append(status == 'PASS')

    s_missing = pd.Series(missing_data_mask, index=df_baseline.index)
    s_rt_pass = pd.Series(rt_fail_mask, index=df_baseline.index)

    # --- Consolidate Statistics ---
    stats_rows = []
    
    # 1. Rules 1-6
    for name, mask in criteria_masks.items():
        n_failed = (~mask).sum()
        stats_rows.append({"Criteria": name, "Excluded_N": n_failed, "Excluded_Pct": (n_failed / total_baseline) * 100})

    # 2. Rule 7
    n_missing = (~s_missing).sum()
    stats_rows.append({"Criteria": "(7) Missing Raw Data", "Excluded_N": n_missing, "Excluded_Pct": (n_missing / total_baseline) * 100})

    # 3. Rule 8
    n_rt_fail_only = ((s_missing) & (~s_rt_pass)).sum()
    stats_rows.append({"Criteria": "(8) RT < 200ms Check (Found Files)", "Excluded_N": n_rt_fail_only, "Excluded_Pct": (n_rt_fail_only / total_baseline) * 100})

    # --- Rules 9-10 (Independent Behavioral Checks) ---
    valid_after_behavioral = 0
    if UNFILTERED_STATS_CSV.exists():
        df_unfiltered_stats = pd.read_csv(UNFILTERED_STATS_CSV)
        
        # Rule 9
        n_fail_gm = (df_unfiltered_stats['perc_gm'] > THRESHOLD).sum()
        stats_rows.append({"Criteria": f"(9) Go Misses > {THRESHOLD*100:.0f}%", "Excluded_N": n_fail_gm, "Excluded_Pct": (n_fail_gm / total_baseline) * 100})
        
        # Rule 10
        n_fail_ge = (df_unfiltered_stats['perc_ge'] > THRESHOLD).sum()
        stats_rows.append({"Criteria": f"(10) Go Errors > {THRESHOLD*100:.0f}%", "Excluded_N": n_fail_ge, "Excluded_Pct": (n_fail_ge / total_baseline) * 100})
        
        # To calculate how many proceed to clinical merge, we need the logical AND of passing both
        valid_after_behavioral = len(df_unfiltered_stats[(df_unfiltered_stats['perc_gm'] <= THRESHOLD) & 
                                                         (df_unfiltered_stats['perc_ge'] <= THRESHOLD)])
    else:
        print("Warning: unfiltered_behavioral_stats.csv not found. Run 03_build_dataset.py first.")
        if VALID_TXT.exists():
            with open(VALID_TXT, 'r') as f:
                valid_after_behavioral = sum(1 for line in f if line.strip())

    # --- Rule 11 (Clinical Data Missing) ---
    final_count = 0
    if FINAL_DATA_CSV.exists():
        df_final = pd.read_csv(FINAL_DATA_CSV)
        final_count = len(df_final)
    else:
        print("Warning: clinical_behavior.csv not found.")

    # The difference between subjects entering the clinical merge step and the final dataset size
    missing_demo = valid_after_behavioral - final_count

    stats_rows.append({
        "Criteria": "(11) Incomplete Demographic/Clinical", 
        "Excluded_N": missing_demo, 
        "Excluded_Pct": (missing_demo / total_baseline) * 100
    })

    # --- Print Formatting ---
    print("\n" + "="*75)
    print("QC ATTRITION REPORT (Independent Counts)")
    print("="*75)
    print(f"{'Criteria ID':<38} | {'Excluded (N)':<12} | {'Excluded (%)':<10}")
    print("-" * 75)
    for row in stats_rows:
        print(f"{row['Criteria']:<38} | {row['Excluded_N']:<12} | {row['Excluded_Pct']:.1f}%")
    print("-" * 75)
    print(f"{'FINAL VALID DATASET':<38} | {final_count:<12} | {(final_count / total_baseline) * 100:.1f}%")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()

# ===========================================================================
# QC ATTRITION REPORT (Independent Counts)
# ===========================================================================
# Criteria ID                            | Excluded (N) | Excluded (%)
# ---------------------------------------------------------------------------
# (1) Trial Counts                       | 1392         | 12.1%
# (2) No Switching                       | 188          | 1.6%
# (3) No Glitch                          | 308          | 2.7%
# (4) 0ms SSD <= 3                       | 3817         | 33.2%
# (5) No Violators                       | 854          | 7.4%
# (6) Complete Session                   | 252          | 2.2%
# (7) Missing Raw Data                   | 1299         | 11.3%
# (8) RT < 200ms Check (Found Files)     | 3058         | 26.6%
# (9) Go Misses > 10%                    | 1443         | 12.5%
# (10) Go Errors > 10%                   | 307          | 2.7%
# (11) Incomplete Demographic/Clinical   | 136          | 1.2%
# ---------------------------------------------------------------------------
# FINAL VALID DATASET                    | 3567         | 31.0%
# ===========================================================================


