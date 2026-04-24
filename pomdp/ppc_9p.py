"""
ppc.py
"""
import pandas as pd
import numpy as np
import sys
import argparse
import zipfile
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.preprocessing import preprocessing
from utils.metrics import get_distance
from tesbi_e2e_9param import (
    scaled_to_dict, PARAM_ORDER, FIXED_PARAMS, simulate_data, 
    USE_PREPROCESSING, PARAM_RANGES
)

# Configuration
PARAMS_CSV = Path("./outputs/params_posteriors_9p.csv")
OUTPUT_PPC = Path("./outputs/ppc_metrics_9p.csv")
OUTPUT_TOP5 = Path("./outputs/top_5_percent_subjects.csv")

def run_ppc(fp: Path, df_params: pd.DataFrame):
    filename = fp.name
    matched_sid = None
    df_obs = None
    
    for sid in df_params["subject_id"].unique():
        if str(sid) in filename:
            matched_sid = str(sid)
            break
            
    if not matched_sid:
        if fp.suffix == '.zip':
            alt_sid = filename.split('_baseline_')[0]
            if alt_sid in df_params["subject_id"].unique():
                matched_sid = alt_sid

    if not matched_sid:
        return {"filename": filename, "error": "No matching subject_id found in params CSV"}

    try:
        if fp.suffix == '.zip':
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(fp, 'r') as zr:
                    zr.extractall(tmpdir)
                csvs = list(Path(tmpdir).rglob(f"*{matched_sid}*.csv"))
                if not csvs:
                    csvs = list(Path(tmpdir).rglob("*.csv")) 
                if not csvs:
                    return {"filename": filename, "error": "No CSV found inside ZIP"}
                df_obs = preprocessing(str(csvs[0])) if USE_PREPROCESSING else pd.read_csv(csvs[0])
        else:
            df_obs = preprocessing(str(fp)) if USE_PREPROCESSING else pd.read_csv(fp)
    except Exception as e:
        return {"filename": filename, "error": f"Data load error: {e}"}

    subset = df_params[df_params["subject_id"] == matched_sid]
    param_col = "index" if "index" in subset.columns else "param"
    
    theta = {}
    for _, row in subset.iterrows():
        if row[param_col] in PARAM_ORDER:
            theta[row[param_col]] = float(row["mean"])
            
    # BOUNDARY CLIPPING: Prevent probabilities from being negative or > 1
    # Prevent costs/temperatures from being <= 0
    for k in PARAM_ORDER:
        if k in theta:
            lo, hi = PARAM_RANGES[k]
            if k.startswith("q_"):
                # Strict clipping for probabilities to avoid 0.0 or 1.0 edge cases in Numpy
                theta[k] = float(np.clip(theta[k], max(lo, 1e-5), min(hi, 1.0 - 1e-5)))
            else:
                theta[k] = float(np.clip(theta[k], lo, hi))
                
    theta.update(FIXED_PARAMS)
    
    if not theta:
        return {"filename": filename, "error": "Parameters dictionary is empty"}
        
    # WRAP SIMULATOR IN TRY-EXCEPT to prevent pool crash
    try:
        df_sim = simulate_data(theta, seed=2026)
    except Exception as e:
        return {"filename": filename, "error": f"Simulation crashed: {e}"}
        
    if df_sim is None or df_sim.empty:
        return {"filename": filename, "error": "Simulation returned empty data"}
        
    try:
        metrics = get_distance(df_obs, df_sim)
    except Exception as e:
        return {"filename": filename, "error": f"Distance calc crashed: {e}"}
        
    keys = ["dis_perc_gs", "dis_perc_ge", "dis_perc_gm", "dis_perc_ss", 
            "dis_ws_rt_gs", "dis_ws_rt_ge", "dis_ws_rt_se", "dis_ks_rt_gs", 
            "dis_ks_rt_se", "dis_ssd_mean"]
    
    res = {"subject_id": matched_sid, "filename": filename}
    total_distance = 0
    for k, v in zip(keys, metrics):
        res[k] = v
        total_distance += v
    res["total_distance"] = total_distance
    
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_folder", type=str, required=True, help="Folder containing real CSVs")
    args = parser.parse_args()
    
    data_dir = Path(args.sst_folder)
    
    if not PARAMS_CSV.exists():
        print(f"Error: Parameter file {PARAMS_CSV} not found. Ensure Step 1 completed successfully.")
        sys.exit(1)

    df_params = pd.read_csv(PARAMS_CSV, dtype={"subject_id": str})
    
    files = list(data_dir.rglob("*.csv")) + list(data_dir.glob("*.zip"))
    files = list(set(files)) 

    results = []
    errors = []
    
    print(f"Starting PPC for {len(files)} files found in {data_dir}...")
    
    if len(files) == 0:
        print(f"CRITICAL ERROR: No .csv or .zip files found in {data_dir}!")
        sys.exit(1)
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_ppc, f, df_params): f for f in files}
        for future in as_completed(futures):
            res = future.result()
            if "error" not in res:
                results.append(res)
            else:
                errors.append(res)
                
    if not results:
        print("CRITICAL ERROR: ALL subjects failed during the PPC stage.")
        print("First 5 error logs for debugging:")
        for err in errors[:5]:
            print(f"  - {err.get('filename')}: {err.get('error')}")
        sys.exit(1)
        
    if errors:
        print(f"Warning: {len(errors)} subjects failed and were skipped.")
                
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by="total_distance", ascending=True)
    df_res.to_csv(OUTPUT_PPC, index=False)
    
    top_5_count = max(1, int(len(df_res) * 0.05))
    df_top5 = df_res.head(top_5_count)
    df_top5.to_csv(OUTPUT_TOP5, index=False)
    
    best_subject = df_top5.iloc[0]['subject_id']
    print(f"Top 5% threshold calculated. {top_5_count} subjects saved to {OUTPUT_TOP5}.")
    print(f"Absolute Best Fitting Subject: {best_subject} (Distance: {df_top5.iloc[0]['total_distance']:.4f})")