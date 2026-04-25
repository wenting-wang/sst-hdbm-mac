"""
plot_corner_10p.py
"""
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import tempfile
from utils.preprocessing import preprocessing
import corner
from pathlib import Path

# CHANGED: Import from 10-param script
from tesbi_e2e_10param import (
    EndToEndTeSBI, build_per_trial_features, PARAM_ORDER, 
    unscale_params, LOG_PARAMS, USE_PREPROCESSING, PARAM_RANGES
)

def extract_and_plot(best_subject_id: str, data_filename: str, data_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating 10,000 samples for subject {best_subject_id} on {device}...")
    
    # CHANGED: param_dim=10
    model = EndToEndTeSBI(in_dim=18, d_model=128, n_heads=8, n_layers=4, param_dim=10).to(device)
    # CHANGED: Load 10p weights
    model.load_state_dict(torch.load("outputs/amortized_inference_net_10p.pth", map_location=device))
    model.eval()
    
    # Match file (supports nested CSV or ZIP)
    file_path = None
    all_files = list(data_dir.rglob("*.csv")) + list(data_dir.glob("*.zip"))
    for fp in all_files:
        if fp.name == data_filename:
            file_path = fp
            break
            
    if not file_path:
        print(f"Error: Could not find {data_filename} in {data_dir}")
        return
        
    df_obs = None
    if file_path.suffix == '.zip':
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(file_path, 'r') as zr:
                zr.extractall(tmpdir)
            csvs = list(Path(tmpdir).rglob(f"*{best_subject_id}*.csv"))
            if not csvs: 
                csvs = list(Path(tmpdir).rglob("*.csv"))
            df_obs = preprocessing(str(csvs[0])) if USE_PREPROCESSING else pd.read_csv(csvs[0])
    else:
        df_obs = preprocessing(str(file_path)) if USE_PREPROCESSING else pd.read_csv(file_path)

    X = build_per_trial_features(df_obs)
    tensor_x = torch.tensor(X[None, ...], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        posterior_dist = model(tensor_x)
        samples_scaled = posterior_dist.sample((10000,))[:, 0, :].cpu().numpy()
        
    unscaled_samples = unscale_params(samples_scaled)
    final_samples = np.zeros_like(unscaled_samples)
    
    for i, k in enumerate(PARAM_ORDER):
        if k in LOG_PARAMS:
            final_samples[:, i] = np.exp(unscaled_samples[:, i])
        else:
            final_samples[:, i] = unscaled_samples[:, i]
            
    df_samples = pd.DataFrame(final_samples, columns=PARAM_ORDER)

    # CHANGED: Added 10p suffix to outputs
    csv_out_path = Path(f"outputs/posterior_samples_10p_{best_subject_id}.csv")
    df_samples.to_csv(csv_out_path, index=False)
    print(f"Raw posterior samples saved to {csv_out_path}")
    
    plot_ranges = []
    for k in PARAM_ORDER:
        lo, hi = PARAM_RANGES[k]
        plot_ranges.append((lo, hi))
    
    print("Plotting corner matrix...")
    fig = corner.corner(
        df_samples,
        labels=PARAM_ORDER,
        range=plot_ranges,
        quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        hist_kwargs={'density': True},
        color="royalblue",
        smooth=1.0,
        plot_datapoints=False,
        fill_contours=True,
    )
    
    # CHANGED: Added 10p suffix to outputs
    out_path = Path(f"outputs/corner_10p_{best_subject_id}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Corner plot saved successfully to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_folder", type=str, required=True, help="Folder containing real CSVs")
    args = parser.parse_args()
    
    data_dir = Path(args.sst_folder)
    
    # CHANGED: Directly read ppc_metrics_10p.csv (it is already sorted by total_distance ascending in ppc.py)
    ppc_df = pd.read_csv("outputs/ppc_metrics_10p.csv")
    best_row = ppc_df.iloc[0] # Index 0 is the absolute best fitting subject
    
    extract_and_plot(str(best_row['subject_id']), best_row['filename'], data_dir)