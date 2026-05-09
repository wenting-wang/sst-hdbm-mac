import argparse
import importlib
import sys
import os
import pandas as pd
import numpy as np
import zipfile
import tempfile
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.preprocessing import preprocessing
from utils.metrics import get_distance
from core.models import POMDP
from core import simulation

def run_simulation_multi_cv(params, split_idx, n_repeat=20, seed_base=2026):
    """
    Runs the POMDP simulation for the full sequence to allow internal beliefs 
    and the staircase algorithm to evolve naturally, but only returns the 
    truncated out-of-sample data (after split_idx) for distance calculation.
    """
    pomdp = POMDP(**params)
    pomdp.value_iteration_tensor()
    
    all_rows = []
    for i in range(n_repeat):
        np.random.seed(seed_base + i)
        
        # Simulate the full 360-trial sequence
        out = simulation.simu_task(pomdp)
        
        # Extract only the hold-out test set
        out_test = out[split_idx:]
        
        for row in out_test:
            all_rows.append({
                'result': row[0], 'rt': row[1], 'ssd': row[2]
            })
    return pd.DataFrame(all_rows)

def worker_cv_eval(task_data):
    """Multiprocessing worker: runs simulation and calculates distance."""
    subject_id = task_data['subject_id']
    df_obs = task_data['df_obs']
    theta = task_data['theta']
    split_idx = task_data['split_idx']
    
    # Ground truth out-of-sample data
    df_test_real = df_obs.iloc[split_idx:].reset_index(drop=True)
    
    try:
        # Model predicted out-of-sample data (repeated 20x for stability)
        df_sim_test_20x = run_simulation_multi_cv(theta, split_idx, n_repeat=20)
    except Exception as e:
        return {"subject_id": subject_id, "error": f"Simulation crashed: {e}"}
        
    if df_sim_test_20x is None or df_sim_test_20x.empty:
        return {"subject_id": subject_id, "error": "Simulation returned empty data"}
        
    try:
        # Calculate distance between real 20% and simulated 20%
        metrics = get_distance(df_test_real, df_sim_test_20x)
    except Exception as e:
        return {"subject_id": subject_id, "error": f"Distance calc crashed: {e}"}
        
    keys = ["dis_perc_gs", "dis_perc_ge", "dis_perc_gm", "dis_perc_ss", 
            "dis_ws_rt_gs", "dis_ws_rt_ge", "dis_ws_rt_se", 
            "dis_ks_rt_gs", "dis_ks_rt_se", "dis_ssd_mean"]
            
    res = {"subject_id": subject_id}
    total_distance = 0
    for k, v in zip(keys, metrics):
        res[k] = v
        total_distance += v
    res["total_distance"] = total_distance
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_folder", type=str, required=True)
    parser.add_argument("--filter_csv", type=str, required=True, help="clinical_behavior.csv path")
    parser.add_argument("--model_config", type=str, required=True, help="e.g., tesbi_e2e_5p_v3")
    parser.add_argument("--weights_path", type=str, required=True, help="Absolute path to the .pth file")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion of trials used for inference")
    args = parser.parse_args()

    # Dynamically load model configuration
    m = importlib.import_module(args.model_config)
    PARAM_ORDER = m.PARAM_ORDER
    MODEL_TAG = m.MODEL_TAG
    USE_PREPROCESSING = getattr(m, "USE_PREPROCESSING", True)
    
    OUTPUT_CV_METRICS = Path(f"./outputs/cv_metrics_{MODEL_TAG}.csv")
    OUTPUT_CV_SUMMARY = Path(f"./outputs/cv_summary_{MODEL_TAG}.csv")
    Path("./outputs").mkdir(exist_ok=True)

    data_dir = Path(args.sst_folder)

    # ==========================================
    # Phase 0: Build Highly Efficient Filter Set
    # ==========================================
    df_filter = pd.read_csv(args.filter_csv)
    valid_ids_set = set(df_filter['subject_id'].astype(str).str.replace('NDAR_', '', regex=False).tolist())
    print(f"\n[Phase 0] Loaded {len(valid_ids_set)} valid subject IDs from {args.filter_csv}")

    # ==========================================
    # Phase 1: Data Extraction & Strict Filtering
    # ==========================================
    files = list(set(list(data_dir.rglob("*.csv")) + list(data_dir.glob("*.zip"))))
    print(f"[Phase 1] Scanning {len(files)} data files in folder...")
    
    all_subjects_data = []
    
    for fp in files:
        filename = fp.name
        
        clean_filename = filename.replace('NDAR_', '')
        extracted_sid = clean_filename.split('_')[0]
        
        if extracted_sid not in valid_ids_set:
            continue
            
        try:
            if fp.suffix == '.zip':
                with tempfile.TemporaryDirectory() as tmpdir:
                    with zipfile.ZipFile(fp, 'r') as zr:
                        zr.extractall(tmpdir)
                    csvs = list(Path(tmpdir).rglob(f"*{extracted_sid}*.csv"))
                    if not csvs: csvs = list(Path(tmpdir).rglob("*.csv")) 
                    if csvs:
                        df_obs = preprocessing(str(csvs[0])) if USE_PREPROCESSING else pd.read_csv(csvs[0])
                    else: continue
            else:
                df_obs = preprocessing(str(fp)) if USE_PREPROCESSING else pd.read_csv(fp)
                
            all_subjects_data.append({"subject_id": extracted_sid, "df_obs": df_obs})
        except Exception as e:
            continue
            
    print(f"[Phase 1] Successfully loaded and filtered {len(all_subjects_data)} subjects.")

    # ==========================================
    # Phase 2: GPU Parameter Inference (Train Split)
    # ==========================================
    print(f"\n[Phase 2] Inferring parameters using Out-of-sample split (Train: {args.train_ratio*100}%) on GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize network and load weights
    net = m.EndToEndTeSBI(in_dim=18, param_dim=len(PARAM_ORDER)).to(device)
    net.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True))
    net.eval()

    inference_tasks = []
    for data in all_subjects_data:
        df_obs = data["df_obs"]
        N_total = len(df_obs)
        split_idx = int(N_total * args.train_ratio)
        
        if split_idx == 0 or split_idx == N_total:
            continue 
            
        # Extract features on full length to maintain correct time indices, then slice
        X_full = m.build_per_trial_features(df_obs)
        X_train = X_full[:split_idx, :]
        tensor_x = torch.tensor(X_train[None, ...], dtype=torch.float32).to(device)

        with torch.no_grad():
            posterior_dist = net(tensor_x)
            # Sample 1000 times to obtain a stable posterior mean
            samples_scaled = posterior_dist.sample((1000,))[:, 0, :].cpu().numpy()
            mean_scaled = samples_scaled.mean(axis=0)
            
        # Convert to physical parameter dictionary
        theta = m.scaled_to_dict(mean_scaled)
        
        inference_tasks.append({
            "subject_id": data["subject_id"],
            "df_obs": df_obs,
            "theta": theta,
            "split_idx": split_idx
        })

    # ==========================================
    # Phase 3: CPU Multiprocessing Simulation
    # ==========================================
    print(f"\n[Phase 3] Running Full POMDP simulations & calculating out-of-sample distance on CPUs...")
    results = []
    errors = []
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker_cv_eval, task): task for task in inference_tasks}
        for future in as_completed(futures):
            res = future.result()
            if "error" not in res:
                results.append(res)
            else:
                errors.append(res)
                
    # ==========================================
    # Phase 4: Aggregation and Output
    # ==========================================
    if not results:
        print("Error: No valid CV results were generated.")
        sys.exit(1)
        
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by="total_distance", ascending=True)
    df_res.to_csv(OUTPUT_CV_METRICS, index=False)
    
    numeric_cols = ["total_distance"] + [col for col in df_res.columns if col.startswith("dis_")]
    df_summary = df_res[numeric_cols].mean().to_frame(name="mean_cv_distance").T
    df_summary.insert(0, "model_name", f"{MODEL_TAG}_CV_Out_of_Sample")
    df_summary.insert(1, "n_subjects", len(df_res))
    df_summary.to_csv(OUTPUT_CV_SUMMARY, index=False)
    
    print(f"\n[Success] CV Completed for model {MODEL_TAG}. Results saved for {len(df_res)} subjects.")
    print(f"Metrics saved to: {OUTPUT_CV_METRICS}")
    print(f"Summary saved to: {OUTPUT_CV_SUMMARY}")