import pandas as pd
import sys
from pathlib import Path

# --- Path Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Set up paths to import local modules (e.g., from sst-pomdp-main/core/)
sys.path.append(str(PROJECT_ROOT))

from core.models import POMDP
from core import simulation
from core import plotting

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define base directories relative to the project root
DATA_DIR = PROJECT_ROOT / "data"
# DATA_DIR = Path('/Users/w/Desktop/data/sst_valid_base')

OUT_DIR = PROJECT_ROOT / "outputs"

# Ensure the output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True) 

# Input Files
# Points to the example dummy data provided in the repository
# PARAMS_CSV = DATA_DIR / "example_params_posteriors.csv"
PARAMS_CSV = DATA_DIR / "params_posteriors_5p_v1.csv"

# Simulation Settings
# Use an example subject ID that matches the dummy data
# SUBJECT_ID = 'EXAMPLE_SUB_001'
SUBJECT_ID = 'NDAR_INV1R97KJ7J'

# Fixed parameters (Constants that do not vary by subject)
FIXED_PARAMS = {
    "rate_stop_trial": 1.0 / 6.0,
    "q_d_n": 0.05,
    "q_s_n": 0.05,
    "cost_go_error": 3.0,
    "cost_go_missing": 1.0,
    "inv_temp": 20.0
}

# Parameters to extract from the CSV
DYNAMIC_PARAMS = [
    'q_d',
    'q_s',
    'tau',
    'cost_stop_error',
    'cost_time'
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_subject_params(subject_id, csv_path, fixed_params=FIXED_PARAMS):
    """
    Loads parameters for a specific subject from a CSV and merges them 
    with fixed constants.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {csv_path}\n"
                                f"Please ensure you have placed the example data in the correct directory.")

    # Load data
    df = pd.read_csv(csv_path)

    # Filter for the specific subject
    subject_df = df[df['subject_id'] == subject_id]

    if subject_df.empty:
        raise ValueError(f"Subject ID {subject_id} not found in {csv_path}")

    # Convert the subject's rows to a dictionary: {param_name: mean_value}
    param_dict = subject_df.set_index('index')['mean'].to_dict()
    clean_params = {k: param_dict[k]
                    for k in DYNAMIC_PARAMS if k in param_dict}

    # Merge with fixed parameters
    final_params = {**clean_params, **fixed_params}

    return final_params

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print(f"Loading parameters for Subject: {SUBJECT_ID}...")

    # 1. Load Parameters
    try:
        params = get_subject_params(SUBJECT_ID, PARAMS_CSV)
        # params['tau'] = int(params['tau']) 
        params['tau'] = 0
        print("Parameters loaded successfully:")
        for k, v in params.items():
            print(f"  {k}: {v}")

    except Exception as e:
        print(f"Error loading parameters: {e}")
        return

    # 2. Initialize POMDP
    print("\nInitializing POMDP Model...")

    pomdp = POMDP(**params)
    pomdp.value_iteration_tensor()

    # 3. Run Simulations
    print("Running Simulations...")
    # Go Trials
    out_go = simulation.simu_trial_batch(
        model=pomdp,
        true_go_state='right',
        true_stop_state='nonstop',
        ssd=None,
        batch_size=5000,
        verbose=True
    )

    # Stop Trials
    out_stop = simulation.simu_trial_batch(
        model=pomdp,
        true_go_state='right',
        true_stop_state='stop',
        ssd=10,
        batch_size=5000,
        verbose=True
    )

    # 4. Plotting
    print("Generating Plots...")
    plotting.plot_belief(
        out_go, out_stop, show_cnt=5, saveto=OUT_DIR / 'fig_belief_states.png'
    )

    plotting.plot_action_value(
        out_go, out_stop, show_cnt=5, saveto=OUT_DIR / 'fig_action_values.png'
    )

    # plotting.plot_stop_prior_context(
    #     model=pomdp,
    #     params=params,
    #     rates=(0.05, 1/6, 0.5, 0.75, 0.95),
    #     ssd=10,
    #     n_batch=500,     # For timecourses
    #     n_session=1000,  # For behavior
    #     saveto=OUT_DIR / "fig_stop_prior_effects.png",
    # )
    
    ############### PLOS CB TIFF ###############
    
    # plotting.plot_belief(
    #     out_go, out_stop, show_cnt=5, saveto=OUT_DIR / 'belief.tiff'
    # )
    
    # plotting.plot_action_value(
    #     out_go, out_stop, show_cnt=5, saveto=OUT_DIR / 'action_value.tiff'
    # )
    
    # plotting.plot_stop_prior_context(
    #     model=pomdp,
    #     params=params,
    #     rates=(0.05, 1/6, 0.35, 0.55, 0.80),
    #     ssd=10,
    #     n_batch=500,
    #     n_session=1000,
    #     saveto=OUT_DIR / "stop_prior_context.tiff", 
    # )

    print("Done.")


if __name__ == "__main__":
    main()