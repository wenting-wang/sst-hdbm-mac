import sys
import pandas as pd
from pathlib import Path
import traceback

# --- Directory Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Add project root to sys.path to import from 'core'
sys.path.append(str(PROJECT_ROOT))

from core.models import POMDP
from core import simulation
from core import plotting

# --- Configuration ---
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"

# Ensure output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Input Files
# PARAMS_CSV = DATA_DIR / "example_params_posteriors.csv"
PARAMS_CSV = DATA_DIR / "params_posteriors_5p_v1.csv"

# Example subject to plot (will fallback to first available if not found)
# SUBJECT_ID = 'EXAMPLE_SUB_001'
SUBJECT_ID = 'NDAR_INV1R97KJ7J'

FIXED_PARAMS = {
    "rate_stop_trial": 1.0 / 6.0,
    "q_d_n": 0.05,
    "q_s_n": 0.05,
    "cost_go_error": 3.0,
    "cost_go_missing": 1.0,
    "inv_temp": 20.0
}

# Params to extract from CSV
DYNAMIC_PARAMS = [
    'q_d',
    'q_s',
    'tau',
    'cost_stop_error',
    'cost_time'
]

# --- Helper Functions ---

def get_subject_params(target_id, path):
    """
    Loads posterior parameters for a specific subject and merges them with fixed parameters.
    Includes a fallback for mock datasets where the exact target_id might not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}. Please place your dataset in the 'data' folder."
        )
        
    df = pd.read_csv(path)
    
    # Fallback logic for example/mock datasets
    if target_id not in df['subject_id'].values:
        available_id = df['subject_id'].iloc[0]
        print(f"Warning: Subject '{target_id}' not found. Using '{available_id}' instead.")
        target_id = available_id
        
    sub_df = df[df['subject_id'] == target_id]
    
    # Handle different possible column names for the parameter index
    if 'index' in sub_df.columns:
        p_dict = sub_df.set_index('index')['mean'].to_dict()
    elif 'param' in sub_df.columns:
        p_dict = sub_df.set_index('param')['mean'].to_dict()
    else:
        raise KeyError("Could not find 'index' or 'param' column in posterior data.")

    # Extract dynamic params and merge with fixed
    dynamic = {k: p_dict[k] for k in DYNAMIC_PARAMS if k in p_dict}
    merged_params = {**dynamic, **FIXED_PARAMS}
    
    return merged_params, target_id


# --- Execution ---

def main():
    try:
        print(f"Loading params for {SUBJECT_ID}...")
        params, final_subject_id = get_subject_params(SUBJECT_ID, PARAMS_CSV)
        # params['tau'] = int(params['tau']) 
        params['tau'] = 0
        
        print(f"\nParameters loaded for {final_subject_id}:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        print("\nInitializing POMDP and running Value Iteration...")
        pomdp = POMDP(**params)
        pomdp.value_iteration_tensor()
        
        print("\nRunning simulations...")

        # 1. Run simulations
        out_go = simulation.simu_trial_batch(
            model=pomdp, true_go_state='right', true_stop_state='nonstop',
            ssd=None, batch_size=2000, verbose=True
        )
        
        out_stop = simulation.simu_trial_batch(
            model=pomdp, true_go_state='right', true_stop_state='stop',
            ssd=10, batch_size=2000, verbose=True
        )

        # 2. Define mapping for outcome names, data sources, and result codes
        outcome_map = {
            "Go Success":   (out_go, "GS"),
            "Go Error":     (out_go, "GE"),
            "Go Missing":   (out_go, "GM"),
            "Stop Success": (out_stop, "SS"),
            "Stop Error":   (out_stop, "SE")
        }

        # 3. Generate the outcomes dictionary using a comprehension
        all_outcomes = {
            name: [trial for trial in data if trial['result'] == code]
            for name, (data, code) in outcome_map.items()
        }

        print("\nGenerating Plots...")

        # 4. Iterate and plot (full time steps for appendix)
        for outcome_name, (_, code) in outcome_map.items():
            save_path = OUT_DIR / f"fig_policy_{code}.png"
            plotting.plot_policy(
                model=pomdp, 
                out_dict=all_outcomes,
                outcome_name=outcome_name,
                saveto=save_path, 
                gamma=0.2
            )
            print(f"  Saved full policy plot: {save_path.name}")
        
        # 5. Plot summary for selected time steps (main text)
        summary_path = OUT_DIR / "fig_policy_summary.png"
        plotting.plot_policy_summary(
            model=pomdp,
            out_dict=all_outcomes,
            selected_timesteps=[3, 17, 33, 39],
            saveto=summary_path,
            dpi=600,
            gamma=0.2
        )
        print(f"  Saved summary policy plot: {summary_path.name}")
        
        print("\nDone.")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()