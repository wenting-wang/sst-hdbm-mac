"""
Locally tests pure POMDP vs HDBM+POMDP RT dynamics.
Optimized via policy pre-computation over dynamically bounded stop-signal rates.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import concurrent.futures
from pathlib import Path
import pickle

# Ensure paths are correct
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.pomdp import POMDP
from core.hdbm import HDBM
from core import simulation

USE_CACHE = True

# --- Local Test Configuration ---
N_SUBJ = 1
N_TRIALS = 100         

N_JOBS = 1            

# 1. Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FIXED_PARAMS_PATH = DATA_DIR / "params_pomdp.csv"
ORDER_CSV_PATH = DATA_DIR / "seq_orders.csv"
OUT_CSV = BASE_DIR / "rt_dyna.csv"

# Dedicated cache directory to prevent clutter
CACHE_DIR = BASE_DIR / "policy_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# New Empirical Bounds calculated from HDBM hazard & prior
# MIN_RATE = 0.0370
# MAX_RATE = 0.5222
RATES_CSV_PATH = DATA_DIR / "r_appr.csv"
RATES = pd.read_csv(RATES_CSV_PATH)['r_appr'].values
MIN_RATE = RATES.min()
MAX_RATE = RATES.max()
N_BINS = len(RATES)

# --- Global Data Loading ---
def load_order_stats(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
        probs = df['subj_cnt'] / df['subj_cnt'].sum()
        return df['order_sig'], df['order_seq'].values, probs
    except Exception:
        warnings.warn(f"Order stats not found at {csv_path}.")
        return None, None, None

FIXED_POMDP_DF = pd.read_csv(FIXED_PARAMS_PATH)
ORDER_SIGS, ORDER_SEQS, ORDER_PROBS = load_order_stats(ORDER_CSV_PATH)


# --- Simulator Functions ---
def get_task_structure(seed=None, n_trials=N_TRIALS):
    if seed is not None: np.random.seed(seed)
    
    chosen_seq = np.random.choice(ORDER_SEQS, p=ORDER_PROBS)
    
    # Safety check: Ensure the sequence is long enough for the requested trials
    if len(chosen_seq) < n_trials:
        raise ValueError(f"Chosen sequence length ({len(chosen_seq)}) is shorter than requested N_TRIALS ({n_trials}).")
        
    trial_types = np.array(['stop' if x == '1' else 'nonstop' for x in chosen_seq])
    trial_types = trial_types[:n_trials] 

    go_directions = np.random.choice(['left', 'right'], size=len(trial_types))
    order_sig = ORDER_SIGS[np.where(ORDER_SEQS == chosen_seq)[0][0]]
    return trial_types, go_directions, order_sig


def get_positions(trial_types):
    """
    Calculates the consecutive Go run length.
    example: [nonstop, nonstop, stop, nonstop, stop] -> [1, 2, 0, 1, 0]
    """
    positions = []
    go_cnt = 1
    
    for trial in trial_types:
        if trial == 'stop':          # Stop trial
            positions.append(0)
            go_cnt = 1               # Reset counter for the next trial
        elif trial == 'nonstop':     # Go trial
            positions.append(go_cnt)
            go_cnt += 1              # Increment counter
    return positions


def simulate_subject(subject_idx, subject_params, seed):
    np.random.seed(seed)
    real_sub_id = subject_params.get('subject_id', f"ID_{subject_idx}")
    
    # 1. Task Structure
    trial_types, go_directions, order_sig = get_task_structure(seed, n_trials=N_TRIALS)
    positions = get_positions(trial_types)
    
    results = []
    valid_keys = ["q_d_n", "q_d", "q_s_n", "q_s", "cost_go_error", 
                  "cost_go_missing", "cost_stop_error", "cost_time", "inv_temp"]
    agent_params = {k: subject_params[k] for k in valid_keys if k in subject_params}
    
    # # ==========================================
    # # PRE-COMPUTATION STEP
    # # ==========================================
    cache_filename = CACHE_DIR / f"policies_{real_sub_id}.pkl"
    
    # Include the pure rate in the cache so we don't recalculate it either
    pure_rate = 1.0 / 6.0
    all_rates = np.unique(np.append(RATES, pure_rate))
    
    print(f"[{real_sub_id}] Checking POMDP policies for rates...")
    

    
    def get_policy_library(rates):
        if USE_CACHE and cache_filename.exists():
            print(f"[{real_sub_id}] Loading precomputed policies from cache...")
            with open(cache_filename, "rb") as f:
                policy_library = pickle.load(f)
            # Basic validation to ensure the loaded library matches the required rates
            if all(r in policy_library for r in rates):
                return policy_library
            else:
                print(f"[{real_sub_id}] Cache incomplete. Recomputing missing rates...")

        print(f"[{real_sub_id}] Computing Value Iteration. This will take a moment...")
        policy_library = {}
        
        for r in rates:
            agent = POMDP(**agent_params, rate_stop_trial=float(r))
            agent.value_iteration_tensor()
            policy_library[r] = agent
            
        if USE_CACHE:
            print(f"[{real_sub_id}] Saving newly computed policies to {cache_filename}...")
            del agent.policy
            del agent.value
            with open(cache_filename, "wb") as f:
                pickle.dump(policy_library, f)
                
        return policy_library

    # # Initialize library
    policy_library = get_policy_library(all_rates)

    # ==========================================
    # Model 1: Pure POMDP (Constant P(stop) = 1/6)
    # ==========================================
    # Fetch from cache instead of recomputing
    agent_pure = policy_library[pure_rate]
    next_stop_ssd = 2
    
    for t, (t_type, go_dir, c_pos) in enumerate(zip(trial_types, go_directions, positions)):
        current_ssd = next_stop_ssd if t_type == 'stop' else None
        
        res, rt, _ = simulation.simu_trial(
            agent_pure, true_go_state=go_dir, true_stop_state=t_type, ssd=current_ssd, verbose=False
        )

        results.append([real_sub_id, order_sig, "Pure_POMDP", t_type, c_pos, float(rt) if rt else np.nan, 
                        float(current_ssd) if current_ssd else np.nan, pure_rate, pure_rate])    
                    
        if t_type == 'stop':
            if res == 'SS': next_stop_ssd = min(next_stop_ssd + 2, 34)
            elif res == 'SE': next_stop_ssd = max(next_stop_ssd - 2, 2)

    # ==========================================
    # Model 2, 3, 4: HDBM + POMDP
    # ==========================================
    hdbm_settings = [
        {"alpha": 0.1, "rho": 0.1, "name": "HDBM_LowA_LowR"},   
        {"alpha": 0.9, "rho": 0.1, "name": "HDBM_HighA_LowR"},
        {"alpha": 0.1, "rho": 0.9, "name": "HDBM_LowA_HighR"}, 
        {"alpha": 0.9, "rho": 0.9, "name": "HDBM_HighA_HighR"} 
    ]

    for setting in hdbm_settings:
        alpha = setting["alpha"]
        rho = setting["rho"]
        model_name = setting["name"]
        
        hdbm = HDBM(alpha=alpha, rho=rho)
        seq_int = np.where(trial_types == 'stop', 1, 0)
        
        # Simulating task structure & clipping to our new strict bounds
        raw_preds = hdbm.simu_task(seq_int, block_size=N_TRIALS)
        r_preds = np.clip(raw_preds, MIN_RATE, MAX_RATE)
        
        next_stop_ssd = 2
        
        for t, (t_type, go_dir, c_pos) in enumerate(zip(trial_types, go_directions, positions)):
            current_ssd = next_stop_ssd if t_type == 'stop' else None
            
            # Snap to closest pre-computed bin
            continuous_rate = float(r_preds[t])
            closest_rate = RATES[np.argmin(np.abs(RATES - continuous_rate))]
            agent_hdbm_cached = policy_library[closest_rate]
            
            res, rt, _ = simulation.simu_trial(
                agent_hdbm_cached, true_go_state=go_dir, true_stop_state=t_type, ssd=current_ssd, verbose=False
            )
                        
            results.append([real_sub_id, order_sig, model_name, t_type, c_pos, float(rt) if rt else np.nan,
                            float(current_ssd) if current_ssd else np.nan, continuous_rate, closest_rate])
            
            if t_type == 'stop':
                if res == 'SS': next_stop_ssd = min(next_stop_ssd + 2, 34)
                elif res == 'SE': next_stop_ssd = max(next_stop_ssd - 2, 2)
                
    cols = ["subject_id", "order_sig", "model", "trial_type", "position", "rt", "ssd", "r_preds", "closest_rate"]
    return pd.DataFrame(results, columns=cols)


def main():
    print(f"Running local test: {N_SUBJ} subjects, {N_TRIALS} trials each.")
    
    sampled_subjects = FIXED_POMDP_DF.sample(n=N_SUBJ, replace=False, random_state=42).to_dict('records')

    # Initialize empty CSV
    empty_df = pd.DataFrame(columns=["subject_id", "order_sig", "model", "trial_type", "position", "rt", "ssd", "r_preds", "closest_rate"])
    empty_df.to_csv(OUT_CSV, index=False)

    completed_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {
            executor.submit(simulate_subject, i, sampled_subjects[i], 42+i): i 
            for i in range(N_SUBJ)
        }
        
        # Note: Appending to CSV here in the main process is thread-safe.
        for future in concurrent.futures.as_completed(futures):
            subj_idx = futures[future]
            try:
                df_res = future.result()
                df_res.to_csv(OUT_CSV, mode='a', header=False, index=False)
                completed_count += 1
                print(f"Progress: [{completed_count}/{N_SUBJ}] subjects written to {OUT_CSV}")
            except Exception as e:
                print(f"Subject {subj_idx} failed with error: {e}")


if __name__ == "__main__":
    main()