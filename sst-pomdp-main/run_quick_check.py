#!/usr/bin/env python3
"""
run_quick_check.py

Automated Quick Run for the SST-POMDP Pipeline.
This script runs a lightweight, end-to-end test of the entire repository 
using example processed data to ensure all components execute correctly without
requiring hours of computation.

Usage:
    python run_quick_check.py
"""

import subprocess
import sys
import time
from pathlib import Path

# Use the current active Python executable (e.g., from the conda env)
PYTHON_EXE = sys.executable

# Automatically get the root directory of the repository (where this script lives)
REPO_ROOT = Path(__file__).resolve().parent

def run_command(step_name, script_path, extra_args=None, expected_output=None):
    """Helper function to run a command, check for missing scripts, track time, and handle errors."""
    
    script_file = REPO_ROOT / script_path
    
    # 1. Skip if the script itself hasn't been created yet
    if not script_file.exists():
        print(f"\n{'='*60}")
        print(f"SKIPPED: {step_name}")
        print(f"Reason: Script '{script_path}' does not exist yet.")
        print(f"{'='*60}")
        return True

    # 2. Skip if the outputs already exist (Caching to save time)
    if expected_output:
        # Convert single string to list for uniform processing
        if isinstance(expected_output, str):
            expected_output = [expected_output]
            
        # Check if ALL expected outputs exist
        all_exist = True
        for out_path in expected_output:
            if not (REPO_ROOT / out_path).exists():
                all_exist = False
                break
                
        if all_exist:
            print(f"\n{'='*60}")
            print(f"SKIPPED: {step_name}")
            print("Reason: All expected output files already exist.")
            print(f"{'='*60}")
            return True

    # Build the command string
    command_list = [PYTHON_EXE, str(script_file)]
    if extra_args:
        command_list.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"CMD:  {' '.join(command_list)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        # Run the command and force it to execute inside the REPO_ROOT directory
        subprocess.run(command_list, check=True, text=True, cwd=REPO_ROOT)
        elapsed_time = time.time() - start_time
        print(f"\nSUCCESS: {step_name} completed in {elapsed_time:.2f} seconds.")
        return True
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\nFAILED: {step_name} failed after {elapsed_time:.2f} seconds.")
        print(f"Return code: {e.returncode}")
        print("Aborting the quick run.")
        sys.exit(1)

def main():
    print("Starting SST-POMDP Automated Quick Run...")
    print(f"Repository Root Directory: {REPO_ROOT}")
    print("This will test the model dynamics, run a lightweight inference pipeline, and generate analysis figures.")
    
    # ---------------------------------------------------------
    # Step 1: Model Sanity Check
    # ---------------------------------------------------------
    run_command(
        "Model Sanity Check (fig_dynamics.py)",
        "analysis/fig_dynamics.py",
        expected_output=[
            "outputs/fig_action_values.png",
            "outputs/fig_belief_states.png",
            "outputs/fig_stop_prior_effects.png"
        ]
    )

    run_command(
        "Model Sanity Check (fig_policy.py)",
        "analysis/fig_policy.py",
        expected_output=[
            "outputs/fig_policy_summary.png",
            "outputs/fig_policy_GS.png",
            "outputs/fig_policy_GE.png",
            "outputs/fig_policy_GM.png",
            "outputs/fig_policy_SS.png",
            "outputs/fig_policy_SE.png"
        ]
    )

    # ---------------------------------------------------------
    # Step 2: TeSBI Model Fitting (Lightweight Quick Run)
    # ---------------------------------------------------------
    # We use minimal epochs and simulation counts so it runs in minutes, not hours.
    tesbi_args = [
        "--stage", "all", 
        "--n1_pre", "5", 
        "--epochs", "2", 
        "--n2", "2", 
        "--density", "maf", 
        "--K", "5", 
        "--num_post", "10", 
        "--sst_folder", "./data/example_processed_data", 
        "--glob_pat", "EXAMPLE_SUB_*.csv", 
        "--num_samples", "10"
    ]
    
    run_command(
        "Model Fitting (TeSBI Quick Run)", 
        "hpc/tesbi.py", 
        extra_args=tesbi_args,
        expected_output="outputs/tesbi/posterior_final.pkl"
    )

    # ---------------------------------------------------------
    # Step 3: Results Analysis & Figure Generation
    # ---------------------------------------------------------
    # List of tuples: (Step Name, Script Path, Expected Output File(s))
    # If the script doesn't exist, or all outputs already exist, it will be skipped safely.
    analysis_scripts = [
        ("Analysis: Parameter Distributions", "analysis/fig_param_dist.py", ["outputs/fig_param_dist.png"]),
        ("Analysis: Linear Models (Params)", "analysis/tab_lm_params.py", ["outputs/tab_lm_params.tex"]),
        ("Analysis: PPC Representative Subjects", "analysis/fig_ppc_reps.py", ["outputs/fig_ppc_reps.png"]),
        ("Analysis: PPC Population", "analysis/fig_ppc_population.py", ["outputs/fig_ppc_population.png"]),
        ("Analysis: Canonical Correlation Analysis", "analysis/fig_cca.py", ["outputs/fig_cca.png"]),
        ("Analysis: PCA Embedding", "analysis/fig_pca_embed.py", ["outputs/fig_pca_embed_2d.png"]),
        ("Analysis: Parameter Recovery", "analysis/fig_params_recovery.py", ["outputs/fig_params_recovery.png"]),
        ("Analysis: Sensitivity Analysis", "analysis/fig_sensitivity.py", ["outputs/fig_sensitivity.png"]),
        ("Analysis: Demographics Counts Table", "analysis/tab_demo_counts.py", ["outputs/tab_demo_counts.tex"]),
        ("Analysis: Linear Models (Behavior)", "analysis/tab_lm_behavior.py", ["outputs/tab_lm_behavior.tex"])
    ]

    for step_name, script_path, exp_out in analysis_scripts:
        run_command(step_name, script_path, expected_output=exp_out)

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------
    print(f"\n{'*'*60}")
    print("ALL TESTS PROCESSED SUCCESSFULLY!")
    print("Check the 'outputs/' folder for the generated figures, tables, and models.")
    print(f"{'*'*60}\n")

if __name__ == "__main__":
    main()