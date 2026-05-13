"""
ppc.py

Posterior Predictive Checks (PPC) for the POMDP-SST simulator.

This script loads the maximum a posteriori (MAP) or mean parameter estimates 
for each subject, simulates the task using these parameters, and computes 
discrepancy metrics (distance) between the simulated data and the actual 
observed data.

================================================================================
USAGE & CONFIGURATION
================================================================================
This script is designed to run efficiently in parallel. 

--- OPTION A: HPC EXECUTION (RECOMMENDED FOR FULL RUNS) ---
For processing the full dataset, it is highly recommended to run this on an 
HPC cluster. You will need to update the `RAW_DATA_DIR` and `PARAMS_POSTERIOR_CSV` 
paths in the configuration section below to point to your cluster directories.
Ensure `USE_PREPROCESSING = True` if using ABCD SST raw data.

--- OPTION B: LOCAL TESTING (SMOKE TEST) ---
The active configuration below is set up for local testing using the public 
example dataset. This is strictly for verifying that the pipeline works without 
needing access to the private dataset. `USE_PREPROCESSING` is set to `False` 
because the example data is already processed.
================================================================================
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

# --- Custom Modules ---
from utils.preprocessing import preprocessing
from utils.metrics import get_distance
from core.models import POMDP
from core import simulation

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- OPTION A: HPC Production Configuration (Recommended) ---
# Uncomment and update these paths when running on your cluster.
# DATA_DIR = Path('/kyb/agpd/wwang/data/')
# OUT_DIR = Path('/kyb/agpd/wwang/outputs/')
# RAW_DATA_DIR = DATA_DIR / "sst_valid_base"
# PARAMS_POSTERIOR_CSV = OUT_DIR / "params_posteriors.csv"
# OUTPUT_PPC_CSV = OUT_DIR / "ppc_metrics.csv"
# USE_PREPROCESSING = True  # Set to True if using ABCD SST raw data

# --- OPTION B: Local Testing Configuration ---
# Uses the provided example data for testing.
# Dynamically resolve the project root based on the location of this script
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"

# Ensure the output directory exists before saving any files
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_DIR = DATA_DIR / "example_processed_data"
PARAMS_POSTERIOR_CSV = DATA_DIR / "params_posteriors.csv"
OUTPUT_PPC_CSV = DATA_DIR / "ppc_metrics.csv"
USE_PREPROCESSING = False  # Set to False because the example data is already processed

# ==============================================================================
# MODEL CONSTANTS
# ==============================================================================
FIXED_PARAMS = {
    "cost_go_error":   1.0,
    "cost_go_missing": 1.0,
    "cost_time":       0.001,
    "rate_stop_trial": 1.0 / 6.0,
}

LINEAR_PARAMS = ["q_d_n", "q_d", "q_s_n", "q_s", "inv_temp"]
LOG_PARAMS = ["cost_stop_error"]
PARAM_ORDER = LINEAR_PARAMS + LOG_PARAMS

@dataclass
class SimConfig:
    """Configuration settings for the PPC Simulation."""
    n_rep: int = 1                  # Number of repetitions (Single point estimate simulation)
    seed: int = 2025                # Base seed for reproducibility
    use_preprocessing: bool = False # Toggle to apply raw data preprocessing


# ==============================================================================
# CORE HELPER FUNCTIONS
# ==============================================================================

def load_posterior_mean(summary_csv: Path, subject_id: str) -> Dict[str, float]:
    """
    Loads the MAP (Mean) parameter estimates for a specific subject from the 
    posterior summary CSV.
    
    Args:
        summary_csv (Path): Path to the CSV containing posterior summaries.
        subject_id (str): The unique identifier for the target subject.
        
    Returns:
        Dict[str, float]: Dictionary mapping parameter names to their estimated values.
    """
    col_names = [
        "param", "mean", "std", "hdi_5", "hdi_50", "hdi_95", 
        "subject_id", "subject_year"
    ]
    df = pd.read_csv(summary_csv, header=None, names=col_names)
    
    # Filter for the target subject
    subset = df[df["subject_id"] == subject_id]
    
    if subset.empty:
        raise ValueError(f"No parameters found for subject {subject_id}")

    theta = {}
    for param in PARAM_ORDER:
        row = subset[subset["param"] == param]
        if not row.empty:
            theta[param] = float(row["mean"].iloc[0])
            
    # Include fixed parameters required by the simulator
    theta.update(FIXED_PARAMS)
    return theta


def simulate_task(params: Dict[str, float], config: SimConfig, seed: int) -> Optional[pd.DataFrame]:
    """
    Executes a single simulation of the POMDP task using the provided parameters.
    
    Args:
        params (Dict[str, float]): Model parameters for the simulation.
        config (SimConfig): Simulation configuration object.
        seed (int): Random seed to ensure deterministic output.
        
    Returns:
        Optional[pd.DataFrame]: Simulated trial data (result, rt, ssd), or None if failed.
    """
    try:
        np.random.seed(seed)
        
        # Initialize and Run Model
        agent = POMDP(**params)
        agent.value_iteration_tensor()
        sim_data = simulation.simu_task(agent)
        
        return pd.DataFrame(sim_data, columns=["result", "rt", "ssd"])
        
    except Exception as e:
        print(f"Simulation failed for params {params}: {e}")
        return None


def run_ppc_single_subject(subject_id: str, filename: str, config: SimConfig) -> dict:
    """
    Worker function for parallel processing: 
    Loads observed data, runs the simulator using posterior means, and calculates 
    the distance/discrepancy metrics between real and simulated data.
    
    Args:
        subject_id (str): The target subject's ID.
        filename (str): Name of the raw data file.
        config (SimConfig): Global simulation configuration.
        
    Returns:
        dict: A dictionary containing PPC discrepancy metrics or error logs.
    """
    try:
        file_path = RAW_DATA_DIR / filename
        
        # 1. Load MAP Parameters
        theta = load_posterior_mean(PARAMS_POSTERIOR_CSV, subject_id)

        # 2. Load Observed Data
        if not file_path.exists():
            return {"subject_id": subject_id, "error": "Raw file not found"}
            
        # Toggle preprocessing based on configuration
        if config.use_preprocessing:
            df_obs = preprocessing(str(file_path))
        else:
            df_obs = pd.read_csv(file_path)
        
        # 3. Simulate Data (PPC)
        # Create a deterministic seed based on subject ID to ensure reproducibility
        sid_hash = (abs(hash(subject_id)) % 1_000_000)
        sim_seed = config.seed + sid_hash
        
        df_sim = simulate_task(theta, config, seed=sim_seed)
        
        if df_sim is None or df_sim.empty:
            return {"subject_id": subject_id, "error": "Simulation returned empty"}

        # 4. Compute Metrics (Distance between Observed and Simulated)
        (
            d_p_gs, d_p_ge, d_p_gm, d_p_ss,
            d_ws_gs, d_ws_ge, d_ws_se,
            d_ks_gs, d_ks_se,
            d_ssd_mean
        ) = get_distance(df_obs, df_sim)

        return {
            "subject_id": subject_id,
            "filename": filename,
            "dis_perc_gs": d_p_gs,
            "dis_perc_ge": d_p_ge,
            "dis_perc_gm": d_p_gm,
            "dis_perc_ss": d_p_ss,
            "dis_ws_rt_gs": d_ws_gs,
            "dis_ws_rt_ge": d_ws_ge,
            "dis_ws_rt_se": d_ws_se,
            "dis_ks_rt_gs": d_ks_gs,
            "dis_ks_rt_se": d_ks_se,
            "dis_ssd_mean": d_ssd_mean,
            "error": None
        }

    except Exception as e:
        return {"subject_id": subject_id, "filename": filename, "error": str(e)}


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    # 1. Setup & Validation
    config = SimConfig(use_preprocessing=USE_PREPROCESSING)
    
    if not RAW_DATA_DIR.exists():
        print(f"Error: Data directory not found at {RAW_DATA_DIR}")
        return

    # 2. File Discovery
    # Iterate over the directory and grab all non-hidden files
    file_paths = [p for p in RAW_DATA_DIR.iterdir() if p.is_file() and not p.name.startswith('.')]
    filenames = [p.name for p in file_paths]

    print(f"Found {len(filenames)} files in {RAW_DATA_DIR}")
    print(f"Using Posterior Summary: {PARAMS_POSTERIOR_CSV}")
    print(f"Preprocessing Enabled: {config.use_preprocessing}")

    results = []
    
    # 3. Parallel Processing
    with ProcessPoolExecutor() as executor:
        futures = {}
        for fname in filenames:
            # Parse filename to extract Subject ID
            # Note: Assumes format like 'PREFIX_SUBJECTID_...' (e.g., 'NDAR_INV123_...' or 'EXAMPLE_SUB_001')
            parts = fname.split('_')
            
            # Safety check to ensure filename format is parsable
            if len(parts) > 1:
                sid = parts[1]
                futures[executor.submit(run_ppc_single_subject, sid, fname, config)] = sid
            else:
                print(f"Skipping file with unexpected format: {fname}")

        # 4. Collect Results
        for future in as_completed(futures):
            sid = futures[future]
            try:
                res = future.result()
                if res.get("error"):
                    print(f"[{sid}] Failed: {res['error']}")
                else:
                    pass # Execution succeeded
                results.append(res)
            except Exception as e:
                print(f"[{sid}] Critical Exception: {e}")

    # 5. Output Management
    if not results:
        print("No results generated. Please check the input directory and file formats.")
        return

    df_results = pd.DataFrame(results)
    
    # Isolate successful runs by filtering out errors
    if 'error' in df_results.columns:
        df_clean = df_results[df_results['error'].isna()].drop(columns=['error'])
        print(f"Simulation completed. Success: {len(df_clean)}/{len(filenames)}")
    else:
        df_clean = df_results
        
    # Save to disk
    OUTPUT_PPC_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_PPC_CSV, index=False)
    print(f"Saved PPC metrics to: {OUTPUT_PPC_CSV}")


if __name__ == "__main__":
    main()