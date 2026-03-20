import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

# IMPORTANT: Import from the updated v2 script
from train_surrogate_v2 import (
    get_or_generate_dataset, 
    load_surrogate, 
    FREE_PARAM
)

def evaluate_surrogate(df_test: pd.DataFrame, model_path: str = "pomdp_surrogate.pth"):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for validation: {str(device).upper()}")

    print(f"Loading Surrogate Model from '{model_path}'...")
    model, X_min, X_max = load_surrogate(filepath=model_path)
    model = model.to(device)
    
    print("\nPreparing test data...")
    # 2. Re-apply the exact same preprocessing
    df_features = df_test.copy()
    if 'inv_temp' in df_features.columns:
        df_features['inv_temp'] = np.log(df_features['inv_temp'])
    if 'cost_stop_error' in df_features.columns:
        df_features['cost_stop_error'] = np.log(df_features['cost_stop_error'])
        
    feature_cols = FREE_PARAM + ['ssd', 'true_go_state', 'true_stop_state']
    X_raw = df_features[feature_cols].values
    
    # Use the scaling parameters loaded from the trained model
    X_scaled = (X_raw - X_min) / (X_max - X_min + 1e-8)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    true_choices = df_test['res'].values
    true_rts = df_test['rt'].values
    
    print("Running predictions through Surrogate Model...")
    with torch.no_grad():
        choice_logits, rt_pred_scaled = model(X_tensor)
        pred_choices = torch.argmax(choice_logits, dim=1).cpu().numpy()
        pred_rts = (rt_pred_scaled.squeeze() * 40.0).cpu().numpy()

    # Add predictions back to the dataframe for grouped analysis
    df_test['pred_res'] = pred_choices
    df_test['pred_rt'] = pred_rts

    # 3. Calculate Metrics
    print("\n" + "="*50)
    print("SURROGATE MODEL VALIDATION RESULTS")
    print("="*50)
    
    # -- Choice Accuracy --
    accuracy = (pred_choices == true_choices).mean() * 100
    print(f"Overall Choice Prediction Accuracy: {accuracy:.2f}%")
    
    # -- RT Mean Absolute Error (MAE) --
    valid_rt_mask = (true_choices == 0) | (true_choices == 1) | (true_choices == 4)
    if valid_rt_mask.sum() > 0:
        true_rts_valid = true_rts[valid_rt_mask]
        pred_rts_valid = pred_rts[valid_rt_mask]
        mae = np.abs(true_rts_valid - pred_rts_valid).mean()
        print(f"\nReaction Time (RT) Mean Absolute Error: {mae:.2f} time steps")
    
    print("="*50 + "\n")

    # 4. Deep Dive: Distribution Analysis
    print("Analyzing RT Distributions for specific POMDP parameters...")
    
    # Find identical trials (same POMDP parameters, same SSD, same true states)
    # We will pick a group that has a lot of repeats (e.g., from the Go trials which you repeated 10 times)
    
    go_trials = df_test[df_test['true_stop_state'] == 0]
    
    # Group by the free parameters to find identical POMDP setups
    # We will just pick the first group that has at least 10 valid RTs
    grouped = go_trials[go_trials['res'].isin([0, 1, 4])].groupby(FREE_PARAM)
    
    analyzed_groups = 0
    for params, group in grouped:
        if len(group) >= 10:
            analyzed_groups += 1
            print(f"\n--- Sample Group {analyzed_groups} ---")
            print(f"Parameters: {dict(zip(FREE_PARAM, [round(p, 3) for p in params]))}")
            print(f"Number of identical trials simulated: {len(group)}")
            
            true_rt_dist = group['rt'].values
            pred_rt_dist = group['pred_rt'].values
            
            print(f"True RTs (Sample): {true_rt_dist[:10]}")
            print(f"True RT Mean: {true_rt_dist.mean():.2f}, Std: {true_rt_dist.std():.2f}")
            print(f"Predicted RTs (Sample): {pred_rt_dist[:10]}")
            print(f"Pred RT Mean: {pred_rt_dist.mean():.2f}, Std: {pred_rt_dist.std():.2f}")
            
            # Note: Because the current surrogate is deterministic, the predicted RTs will all be identical (Std = 0)
            
            if analyzed_groups >= 4:
                break

    if analyzed_groups == 0:
         print("No groups with enough repeated valid RT trials found for distribution analysis.")


if __name__ == "__main__":
    TEST_POMDPS = 50 
    test_filename = f"pomdp_test_dataset_{TEST_POMDPS}_solves.parquet"
    
    df_test = get_or_generate_dataset(n_pomdp_solves=TEST_POMDPS, filename=test_filename)
    evaluate_surrogate(df_test, model_path="pomdp_surrogate.pth")