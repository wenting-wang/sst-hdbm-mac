import torch
import numpy as np
import pandas as pd
from typing import Tuple

# IMPORTANT: Import from the updated v2 script
from train_surrogate_v2 import (
    get_or_generate_dataset, 
    load_surrogate, 
    FREE_PARAM
)

def evaluate_surrogate(df_test: pd.DataFrame, model_path: str = "pomdp_surrogate.pth"):
    # 1. Setup Device (Supports Apple Silicon MPS, Nvidia CUDA, or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for validation: {str(device).upper()}")

    print(f"Loading Surrogate Model from '{model_path}'...")
    model, X_min, X_max = load_surrogate(filepath=model_path)
    model = model.to(device)
    
    print("\nPreparing test data...")
    # 2. Re-apply the exact same preprocessing (Log transform + MinMax scaling)
    df_features = df_test.copy()
    if 'inv_temp' in df_features.columns:
        df_features['inv_temp'] = np.log(df_features['inv_temp'])
    if 'cost_stop_error' in df_features.columns:
        df_features['cost_stop_error'] = np.log(df_features['cost_stop_error'])
        
    feature_cols = FREE_PARAM + ['ssd', 'true_go_state', 'true_stop_state']
    X_raw = df_features[feature_cols].values
    
    # Use the scaling parameters loaded from the trained model!
    X_scaled = (X_raw - X_min) / (X_max - X_min + 1e-8)
    
    # Move tensor to the selected device
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    # Extract true targets
    true_choices = df_test['res'].values
    true_rts = df_test['rt'].values
    
    print("Running predictions through Surrogate Model...")
    with torch.no_grad():
        choice_logits, rt_pred_scaled = model(X_tensor)
        
        # Convert choice logits to actual predictions (argmax) and move back to CPU
        pred_choices = torch.argmax(choice_logits, dim=1).cpu().numpy()
        
        # Convert scaled RT back to actual time steps (* 40.0) and move back to CPU
        pred_rts = (rt_pred_scaled.squeeze() * 40.0).cpu().numpy()

    # 3. Calculate Metrics
    print("\n" + "="*50)
    print("SURROGATE MODEL VALIDATION RESULTS")
    print("="*50)
    
    # -- Choice Accuracy --
    correct_predictions = (pred_choices == true_choices).sum()
    accuracy = correct_predictions / len(true_choices) * 100
    print(f"Overall Choice Prediction Accuracy: {accuracy:.2f}%")
    
    # Confusion Matrix (Accuracy per class)
    print("\nClass-specific Accuracy:")
    outcome_map = {0: 'GS (Go Success)', 1: 'GE (Go Error)', 2: 'GM (Go Missing)', 3: 'SS (Stop Success)', 4: 'SE (Stop Error)'}
    for i in range(5):
        total_class = (true_choices == i).sum()
        if total_class > 0:
            correct_class = ((pred_choices == i) & (true_choices == i)).sum()
            print(f"  - {outcome_map[i]:<17}: {correct_class/total_class*100:>6.2f}% ({correct_class}/{total_class})")

    # -- RT Mean Absolute Error (MAE) --
    # Only calculate error where the trial actually had an RT (GS, GE, SE)
    valid_rt_mask = (true_choices == 0) | (true_choices == 1) | (true_choices == 4)
    if valid_rt_mask.sum() > 0:
        true_rts_valid = true_rts[valid_rt_mask]
        pred_rts_valid = pred_rts[valid_rt_mask]
        mae = np.abs(true_rts_valid - pred_rts_valid).mean()
        print(f"\nReaction Time (RT) Mean Absolute Error: {mae:.2f} time steps")
        
        # Print a few random samples for sanity check
        print("\nRandom Samples (True RT vs Pred RT):")
        sample_indices = np.random.choice(len(true_rts_valid), min(10, len(true_rts_valid)), replace=False)
        for idx in sample_indices:
            print(f"  True RT: {true_rts_valid[idx]:>4.1f} | Pred RT: {pred_rts_valid[idx]:>4.1f} | Diff: {abs(true_rts_valid[idx]-pred_rts_valid[idx]):>4.1f}")
    else:
        print("\nNo valid RTs in test set to calculate error.")
    print("="*50 + "\n")


if __name__ == "__main__":
    # Simulate e.g., 50 new POMDPs (50 * 86 = 4300 test trials)
    TEST_POMDPS = 50 
    test_filename = f"pomdp_test_dataset_{TEST_POMDPS}_solves.csv"
    
    # NOTE: Changed argument to `n_pomdp_solves` to match v2 script
    df_test = get_or_generate_dataset(n_pomdp_solves=TEST_POMDPS, filename=test_filename)
    
    # Evaluate using the model file downloaded from the cluster
    # (Make sure 'pomdp_surrogate.pth' is in the same directory)
    evaluate_surrogate(df_test, model_path="pomdp_surrogate.pth")