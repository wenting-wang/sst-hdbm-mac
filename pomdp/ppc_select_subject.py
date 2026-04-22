"""
ppc_select_subject.py

Runs Posterior Predictive Checks on the 9-parameter outputs,
calculates aggregated distances, and selects the top 5% best fitting subjects.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.preprocessing import preprocessing
from utils.metrics import get_distance
from tesbi_e2e_9param import scaled_to_dict, PARAM_ORDER, FIXED_PARAMS, simulate_data

# Configuration
DATA_DIR = Path("./data/example_processed_data")
PARAMS_CSV = Path("./outputs/params_posteriors_9p.csv")
OUTPUT_PPC = Path("./outputs/ppc_metrics_9p.csv")
OUTPUT_TOP5 = Path("./outputs/top_5_percent_subjects.csv")

def load_mean_params(subject_id: str) -> dict:
    df = pd.read_csv(PARAMS_CSV)
    subset = df[df["subject_id"] == str(subject_id)]
    if subset.empty: return {}
    
    theta = {row["param"]: float(row["mean"]) for _, row in subset.iterrows() if row["param"] in PARAM_ORDER}
    theta.update(FIXED_PARAMS)
    return theta

def run_ppc(filename: str):
    sid = filename.split('_')[1] if '_' in filename else Path(filename).stem
    file_path = DATA_DIR / filename
    
    theta = load_mean_params(sid)
    if not theta: return {"subject_id": sid, "error": "No params"}
    
    df_obs = pd.read_csv(file_path)
    df_sim = simulate_data(theta, seed=2026)
    
    if df_sim is None or df_sim.empty: return {"subject_id": sid, "error": "Sim failed"}
    
    metrics = get_distance(df_obs, df_sim)
    keys = ["dis_perc_gs", "dis_perc_ge", "dis_perc_gm", "dis_perc_ss", 
            "dis_ws_rt_gs", "dis_ws_rt_ge", "dis_ws_rt_se", "dis_ks_rt_gs", 
            "dis_ks_rt_se", "dis_ssd_mean"]
    
    res = {"subject_id": sid, "filename": filename}
    total_distance = 0
    for k, v in zip(keys, metrics):
        res[k] = v
        total_distance += v # Aggregated distance metric
    res["total_distance"] = total_distance
    
    return res

if __name__ == "__main__":
    files = [f.name for f in DATA_DIR.iterdir() if f.is_file() and f.name.endswith('.csv')]
    results = []
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_ppc, f): f for f in files}
        for future in as_completed(futures):
            res = future.result()
            if "error" not in res:
                results.append(res)
                
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_PPC, index=False)
    
    # Select Top 5% 
    df_res = df_res.sort_values(by="total_distance", ascending=True)
    top_5_count = max(1, int(len(df_res) * 0.05))
    df_top5 = df_res.head(top_5_count)
    df_top5.to_csv(OUTPUT_TOP5, index=False)
    
    best_subject = df_top5.iloc[0]['subject_id']
    print(f"Top 5% threshold calculated. {top_5_count} subjects saved to {OUTPUT_TOP5}.")
    print(f"🥇 Absolute Best Fitting Subject: {best_subject} (Distance: {df_top5.iloc[0]['total_distance']:.4f})")