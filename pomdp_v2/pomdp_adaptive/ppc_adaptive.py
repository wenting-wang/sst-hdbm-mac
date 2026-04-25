import argparse
import importlib
import sys
import pandas as pd
import numpy as np
import zipfile
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.preprocessing import preprocessing
from utils.metrics import get_distance
from core.models import POMDP
from core import simulation

temp_parser = argparse.ArgumentParser(add_help=False)
temp_parser.add_argument("--model_config", type=str, required=True)
temp_args, _ = temp_parser.parse_known_args()

m = importlib.import_module(temp_args.model_config)

PARAM_ORDER = m.PARAM_ORDER
FIXED_PARAMS = m.FIXED_PARAMS
USE_PREPROCESSING = m.USE_PREPROCESSING
PARAM_RANGES = m.PARAM_RANGES
MODEL_TAG = m.MODEL_TAG

PARAMS_CSV = Path(f"./outputs/params_posteriors_{MODEL_TAG}.csv")
OUTPUT_PPC = Path(f"./outputs/ppc_metrics_{MODEL_TAG}.csv")
OUTPUT_SUMMARY = Path(f"./outputs/ppc_model_summary_{MODEL_TAG}.csv")


def run_simulation_multi(params, n_repeat=20, seed_base=2026):
    """Run simulation 20 times to generate a stable large sample for distance calculation."""
    pomdp = POMDP(**params)
    pomdp.value_iteration_tensor()
    
    all_rows = []
    for i in range(n_repeat):
        np.random.seed(seed_base + i)
        out = simulation.simu_task(pomdp)
        for t, row in enumerate(out):
            all_rows.append({
                'result': row[0], 'rt': row[1], 'ssd': row[2]
            })
    return pd.DataFrame(all_rows)


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
                if not csvs: csvs = list(Path(tmpdir).rglob("*.csv")) 
                if not csvs: return {"filename": filename, "error": "No CSV found inside ZIP"}
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
            
    for k in PARAM_ORDER:
        if k in theta:
            lo, hi = PARAM_RANGES[k]
            if k.startswith("q_"):
                theta[k] = float(np.clip(theta[k], max(lo, 1e-5), min(hi, 1.0 - 1e-5)))
            else:
                theta[k] = float(np.clip(theta[k], lo, hi))
                
    theta.update(FIXED_PARAMS)
    
    if not theta: return {"filename": filename, "error": "Parameters dictionary is empty"}
        
    try:
        df_sim_20x = run_simulation_multi(theta, n_repeat=20)
    except Exception as e: return {"filename": filename, "error": f"Simulation crashed: {e}"}
        
    if df_sim_20x is None or df_sim_20x.empty: return {"filename": filename, "error": "Simulation returned empty data"}
        
    try: metrics = get_distance(df_obs, df_sim_20x)
    except Exception as e: return {"filename": filename, "error": f"Distance calc crashed: {e}"}
        
    keys = ["dis_perc_gs", "dis_perc_ge", "dis_perc_gm", "dis_perc_ss", "dis_ws_rt_gs", "dis_ws_rt_ge", "dis_ws_rt_se", "dis_ks_rt_gs", "dis_ks_rt_se", "dis_ssd_mean"]
    res = {"subject_id": matched_sid, "filename": filename}
    total_distance = 0
    for k, v in zip(keys, metrics):
        res[k] = v; total_distance += v
    res["total_distance"] = total_distance
    return res

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_folder", type=str, required=True)
    parser.add_argument("--filter_csv", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.sst_folder)
    
    if not PARAMS_CSV.exists():
        print(f"Error: Parameter file {PARAMS_CSV} not found.")
        sys.exit(1)

    df_params = pd.read_csv(PARAMS_CSV, dtype={"subject_id": str})
    files = list(set(list(data_dir.rglob("*.csv")) + list(data_dir.glob("*.zip"))))

    results = []
    errors = []
    print(f"Starting PPC for {len(files)} files with model {MODEL_TAG}...")
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_ppc, f, df_params): f for f in files}
        for future in as_completed(futures):
            res = future.result()
            if "error" not in res: results.append(res)
            else: errors.append(res)
    
    df_filter = pd.read_csv(args.filter_csv)
    valid_ids = df_filter['subject_id'].astype(str).str.replace('NDAR_', '', regex=False).unique()
                
    df_res = pd.DataFrame(results)
    df_res['subject_id'] = df_res['subject_id'].astype(str).str.replace('NDAR_', '', regex=False)
    df_res = df_res[df_res['subject_id'].isin(valid_ids)]
    df_res = df_res.sort_values(by="total_distance", ascending=True)
    df_res.to_csv(OUTPUT_PPC, index=False)
    
    numeric_cols = ["total_distance"] + [col for col in df_res.columns if col.startswith("dis_")]
    df_summary = df_res[numeric_cols].mean().to_frame(name="mean_distance").T
    df_summary.insert(0, "model_name", f"{MODEL_TAG}_Filtered")
    df_summary.insert(1, "n_subjects", len(df_res))
    df_summary.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"PPC Completed. Results saved for {len(df_res)} subjects.")