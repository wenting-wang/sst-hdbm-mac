import torch
import numpy as np
import pandas as pd
from typing import Tuple

from train_surrogate import (
    get_or_generate_dataset, 
    load_surrogate, 
    FREE_PARAM
)

def evaluate_surrogate(df_test: pd.DataFrame, model_path: str = "pomdp_surrogate.pth"):
    print(f"Loading Surrogate Model from '{model_path}'...")
    model, X_min, X_max = load_surrogate(filepath=model_path)
    
    print("\nPreparing test data...")
    # 1. Re-apply the exact same preprocessing (Log transform + MinMax scaling)
    df_features = df_test.copy()
    if 'inv_temp' in df_features.columns:
        df_features['inv_temp'] = np.log(df_features['inv_temp'])
    if 'cost_stop_error' in df_features.columns:
        df_features['cost_stop_error'] = np.log(df_features['cost_stop_error'])
        
    feature_cols = FREE_PARAM + ['ssd', 'true_go_state', 'true_stop_state']
    X_raw = df_features[feature_cols].values
    X_scaled = (X_raw - X_min) / (X_max - X_min + 1e-8)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # 2. Extract true targets
    true_choices = df_test['res'].values
    true_rts = df_test['rt'].values
    
    print("Running predictions through Surrogate Model...")
    with torch.no_grad():
        choice_logits, rt_pred_scaled = model(X_tensor)
        
        # Convert choice logits to actual predictions (argmax)
        pred_choices = torch.argmax(choice_logits, dim=1).numpy()
        
        # Convert scaled RT back to actual time steps (* 40.0)
        pred_rts = (rt_pred_scaled.squeeze() * 40.0).numpy()

    # 3. Calculate Metrics
    print("\n" + "="*40)
    print("SURROGATE MODEL VALIDATION RESULTS")
    print("="*40)
    
    # -- Choice Accuracy --
    correct_predictions = (pred_choices == true_choices).sum()
    accuracy = correct_predictions / len(true_choices) * 100
    print(f"Choice Prediction Accuracy: {accuracy:.2f}%")
    
    # Confusion Matrix
    outcome_map = {0: 'GS', 1: 'GE', 2: 'GM', 3: 'SS', 4: 'SE'}
    for i in range(5):
        total_class = (true_choices == i).sum()
        if total_class > 0:
            correct_class = ((pred_choices == i) & (true_choices == i)).sum()
            print(f"  - Accuracy for {outcome_map[i]}: {correct_class/total_class*100:.2f}% ({correct_class}/{total_class})")

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
            print(f"  True RT: {true_rts_valid[idx]:.1f} | Pred RT: {pred_rts_valid[idx]:.1f}")
    else:
        print("\nNo valid RTs in test set to calculate error.")
    print("="*40 + "\n")


if __name__ == "__main__":
    TEST_SAMPLES = 1000
    test_filename = f"pomdp_test_dataset_{TEST_SAMPLES}.csv"
    df_test = get_or_generate_dataset(n_samples=TEST_SAMPLES, filename=test_filename)
    
    evaluate_surrogate(df_test, model_path="pomdp_surrogate.pth")