"""
stats.py

Behavioral and Clinical Statistics Aggregator for ABCD Stop-Signal Task (SST).

This script performs two main functions:
1. Iterates through valid SST raw data files to compute comprehensive behavioral 
   statistics (e.g., Go/Stop accuracies, Reaction Times, Post-Error Slowing).
2. Merges these behavioral metrics with external clinical and demographic data 
   (CBCL ADHD scores, Demographics, ADHD Medication flags, and IQ scores).

The final output is a clean, merged CSV dataset ready for downstream analysis 
or model training.
"""

import sys
import os
import re
import pandas as pd
from pathlib import Path

# --- Path Setup to allow importing local modules ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.preprocessing import preprocessing
from utils.metrics import get_stats_mean

pd.set_option('future.no_silent_downcasting', True)
# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================

# --- Base Directories ---
# BASE_DIR = Path('/kyb/agpd/wwang/data_abcd/')
BASE_DIR = Path("/Users/w/Desktop/data/")  # <-- UPDATE THIS PATH

# --- Input Data Directories & Files ---
FILE_LIST_PATH = BASE_DIR / 'sst_valid_filenames.txt'
DATA_DIR = BASE_DIR / 'sst_valid_base'

# --- Clinical / Demographic Input Files ---
CBCL_FILE = BASE_DIR / 'mh_p_cbcl.csv'
DEMO_FILE = BASE_DIR / 'abcd_p_demo.csv'
MEDS_FILE = BASE_DIR / 'ph_p_meds.csv'
IQ_FILE = BASE_DIR / 'nc_y_wisc.csv'

# --- Final Merged Output ---
OUT_FINAL_CSV = BASE_DIR / 'clinical_behavior.csv'


# ==============================================================================
# CONSTANTS & MAPPINGS
# ==============================================================================

YEAR_MAP = {
    'baseline_year_1_arm_1': 'baseline',
    '1_year_follow_up_y_arm_1': '1',
    '2_year_follow_up_y_arm_1': '2',
    '3_year_follow_up_y_arm_1': '3',
    '4_year_follow_up_y_arm_1': '4'
}

ADHD_MEDS = [
    "Adderall", "Concerta", "Methylphenidate", "Ritalin", "Focalin", "Strattera", "Amphetamine",
    "Quillivant", "Guanfacine", "Evekeo", "Atomoxetine", "Lisdexamfetamine", "Dexedrine", "Dynavel",
    "Adzenys", "Metadate", "Kapvay", "Clonidine", "Intuniv", "Daytrana", "Methylin", "Dextrostat",
    "Zenzedi", "Tenex", "Catapres", "Aptensio", "Cotempla", "Quillichew", "Bupropion", "Wellbutrin",
    "Norpramin", "Desipramine", "Impiprmine", "Tofranil", "Nortriptyline", "Aventyl", "Pamelor"
]


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def clean_subject_id(raw_id: str) -> str:
    """Extracts the base subject ID (e.g., 'INV...') from 'NDAR_INV...' format."""
    return raw_id.split('_')[1] if '_' in raw_id else raw_id

def clean_year(raw_year: str) -> str:
    """Maps ABCD raw event names to simplified string years (e.g., 'baseline')."""
    return YEAR_MAP.get(raw_year, raw_year.split('_')[0])


# ==============================================================================
# DATA LOADING & PROCESSING FUNCTIONS
# ==============================================================================

def get_behavioral_stats(file_list_path: Path, data_dir: Path) -> pd.DataFrame:
    """
    Iterates through raw subject files to compute comprehensive behavioral statistics.
    
    Returns:
        pd.DataFrame: A dataframe containing behavioral metrics for all successfully 
        processed subjects.
    """
    results = []
    failures = []

    with file_list_path.open('r') as f:
        filenames = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(filenames)} behavioral files...")

    for fname in filenames:
        # Expected format: NDAR_INVXJK60DCE_baseline_year_1_arm_1_sst.csv.zip
        parts = fname.split('_')
        if len(parts) < 3 or parts[0] != 'NDAR':
            failures.append((fname, 'unexpected filename pattern'))
            continue

        subject_id = parts[1]
        year = parts[2] # Typically 'baseline'
        
        file_path = data_dir / fname
        
        if not file_path.exists():
            failures.append((fname, 'file not found'))
            continue

        try:
            # Load and preprocess the raw data
            df_obs = preprocessing(str(file_path))
            
            # Compute advanced statistics
            stats = get_stats_mean(df_obs)
            
            # Unpack the 16 computed values
            (perc_gs, perc_ge, perc_gm, perc_ss,
             mrt_gs, mrt_ge, mrt_se, mssd, ssrt,
             rate_perc_ss_ssd, rate_rt_se_ssd,
             pes, pss, pges, acf1, acf2, slope) = stats

            results.append({
                'subject_id': subject_id,
                'year': year,
                'filename': fname,
                # Basic Metrics
                'perc_gs': perc_gs, 'perc_ge': perc_ge, 'perc_gm': perc_gm, 'perc_ss': perc_ss,
                'mrt_gs': mrt_gs, 'mrt_ge': mrt_ge, 'mrt_se': mrt_se, 'mssd': mssd, 'ssrt': ssrt,
                # SSD Slopes
                'rate_perc_ss_ssd': rate_perc_ss_ssd, 'rate_rt_se_ssd': rate_rt_se_ssd,
                # Sequential / Temporal Effects
                'pes': pes,             # Post-Error Slowing
                'pss': pss,             # Post-Stop Slowing
                'pges': pges,           # Post-Go Error Slowing
                'rt_acf_1': acf1,       # Lag-1 Autocorrelation
                'rt_acf_2': acf2,       # Lag-2 Autocorrelation
                'rt_slope': slope       # Global Fatigue/Practice Trend
            })

        except Exception as e:
            failures.append((fname, f'processing error: {e}'))

    if failures:
        print(f"Encountered {len(failures)} failures.")
    
    return pd.DataFrame(results)


def get_cbcl_data(path: Path) -> pd.DataFrame:
    """Loads and cleans Child Behavior Checklist (CBCL) ADHD scores."""
    df = pd.read_csv(path, low_memory=False)
    df['subject_id'] = df['src_subject_id'].apply(clean_subject_id)
    df['year'] = df['eventname'].apply(clean_year)
    df = df[['subject_id', 'year', 'cbcl_scr_dsm5_adhd_r']]
    df = df.rename(columns={'cbcl_scr_dsm5_adhd_r': 'adhd'})
    return df


def get_demo_data(path: Path) -> pd.DataFrame:
    """Loads and cleans demographic data, forward/backward filling missing Sex values."""
    df = pd.read_csv(path, low_memory=False)
    df['subject_id'] = df['src_subject_id'].apply(clean_subject_id)
    df['year'] = df['eventname'].apply(clean_year)
    
    # Filter out unspecified sex (3) and map 1/2 to string labels
    df = df[df['demo_sex_v2'] != 3] 
    df['sex'] = df['demo_sex_v2'].replace({1: 'Male', 2: 'Female'})
    df = df[['subject_id', 'year', 'sex']]
    
    # Fill missing sex values for a subject across their longitudinal visits
    df['sex'] = df.groupby('subject_id')['sex'].transform(lambda s: s.ffill().bfill()).infer_objects(copy=False)
    return df


def get_med_data(path: Path) -> pd.DataFrame:
    """Loads medication logs and flags subjects currently taking ADHD meds."""
    df = pd.read_csv(path, low_memory=False)
    med_cols = [col for col in df.columns if "rxnorm_p" in col]
    
    # Search for known ADHD medication names using Regex
    pattern = re.compile("|".join(ADHD_MEDS), flags=re.IGNORECASE)
    df["adhd_med_flag"] = df[med_cols].apply(
        lambda row: int(any(bool(pattern.search(str(v))) for v in row)), axis=1
    )
    
    df['subject_id'] = df['src_subject_id'].apply(clean_subject_id)
    df['year'] = df['eventname'].apply(clean_year)
    return df[['subject_id', 'year', 'adhd_med_flag']]


def get_iq_data(path: Path) -> pd.DataFrame:
    """Loads and cleans WISC-V IQ (Intelligence Quotient) scores."""
    df = pd.read_csv(path, low_memory=False)
    df['subject_id'] = df['src_subject_id'].apply(clean_subject_id)
    df['year'] = df['eventname'].apply(clean_year)
    df = df[['subject_id', 'year', 'pea_wiscv_trs']]
    df = df.rename(columns={'pea_wiscv_trs': 'iq'})
    return df


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("Computing behavioral stats from raw files...")
    df_stats = get_behavioral_stats(FILE_LIST_PATH, DATA_DIR)

    UNFILTERED_STATS_CSV = BASE_DIR / 'unfiltered_behavioral_stats.csv'
    df_stats.to_csv(UNFILTERED_STATS_CSV, index=False)
    print(f"Unfiltered behavioral stats saved to: {UNFILTERED_STATS_CSV}")

    THRESHOLD = 0.1
    initial_stats_len = len(df_stats)
    df_stats = df_stats[(df_stats['perc_gm'] <= THRESHOLD) & (df_stats['perc_ge'] <= THRESHOLD)]
    print(f"Dropped {initial_stats_len - len(df_stats)} rows due to perc_gm or perc_ge > {THRESHOLD}%.")

    print("Loading Demographics, CBCL, Meds, and IQ...")
    df_cbcl = get_cbcl_data(CBCL_FILE)
    df_demo = get_demo_data(DEMO_FILE)
    df_meds = get_med_data(MEDS_FILE)
    df_iq   = get_iq_data(IQ_FILE)

    print("Merging datasets...")
    # Inner join on foundational data
    df_merged = pd.merge(df_stats, df_cbcl, on=['subject_id', 'year'], how='inner')
    df_merged = pd.merge(df_merged, df_demo, on=['subject_id', 'year'], how='inner')
    
    # Left join on supplementary data (Meds and IQ)
    df_merged = pd.merge(df_merged, df_meds, on=['subject_id', 'year'], how='left')
    df_merged = pd.merge(df_merged, df_iq, on=['subject_id', 'year'], how='left')

    initial_len = len(df_merged)
    df_final = df_merged.dropna()
    print(f"Dropped {initial_len - len(df_final)} rows with missing data.")
    
    # Save Final Dataset
    df_final.to_csv(OUT_FINAL_CSV, index=False)
    print(f"Done. Final dataset saved to: {OUT_FINAL_CSV}")


if __name__ == "__main__":
    main()