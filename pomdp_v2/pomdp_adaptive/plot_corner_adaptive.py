import argparse
import importlib
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import tempfile
import corner
from pathlib import Path
from utils.preprocessing import preprocessing

temp_parser = argparse.ArgumentParser(add_help=False)
temp_parser.add_argument("--model_config", type=str, required=True)
temp_args, _ = temp_parser.parse_known_args()

m = importlib.import_module(temp_args.model_config)

PARAM_ORDER = m.PARAM_ORDER
LOG_PARAMS = m.LOG_PARAMS
USE_PREPROCESSING = m.USE_PREPROCESSING
PARAM_RANGES = m.PARAM_RANGES
MODEL_TAG = m.MODEL_TAG
EndToEndTeSBI = m.EndToEndTeSBI
build_per_trial_features = m.build_per_trial_features
unscale_params = m.unscale_params


def extract_and_plot(best_subject_id: str, data_filename: str, data_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating 10,000 samples for subject {best_subject_id} on {device}...")
    
    model = EndToEndTeSBI(param_dim=len(PARAM_ORDER)).to(device)
    model.load_state_dict(torch.load(f"outputs/amortized_inference_net_{MODEL_TAG}.pth", map_location=device))
    model.eval()
    
    file_path = None
    all_files = list(data_dir.rglob("*.csv")) + list(data_dir.glob("*.zip"))
    for fp in all_files:
        if fp.name == data_filename: file_path = fp; break
            
    if not file_path: return
        
    df_obs = None
    if file_path.suffix == '.zip':
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(file_path, 'r') as zr: zr.extractall(tmpdir)
            csvs = list(Path(tmpdir).rglob(f"*{best_subject_id}*.csv"))
            if not csvs: csvs = list(Path(tmpdir).rglob("*.csv"))
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
        if k in LOG_PARAMS: final_samples[:, i] = np.exp(unscaled_samples[:, i])
        else: final_samples[:, i] = unscaled_samples[:, i]
            
    df_samples = pd.DataFrame(final_samples, columns=PARAM_ORDER)
    
    
    csv_out_path = Path(f"outputs/posterior_samples_{MODEL_TAG}_{best_subject_id}.csv")
    df_samples.to_csv(csv_out_path, index=False)
    
    plot_ranges = [PARAM_RANGES[k] for k in PARAM_ORDER]
    
    fig = corner.corner(
        df_samples, labels=PARAM_ORDER, range=plot_ranges,
        quantiles=[0.05, 0.5, 0.95], show_titles=True, title_kwargs={"fontsize": 10},
        hist_kwargs={'density': True}, color="royalblue", smooth=1.0, plot_datapoints=False, fill_contours=True,
    )
    
    out_path = Path(f"outputs/corner_{MODEL_TAG}_{best_subject_id}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Corner plot saved successfully to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_folder", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    args = parser.parse_args()
    
    data_dir = Path(args.sst_folder)
    
    ppc_df = pd.read_csv(f"outputs/ppc_metrics_{MODEL_TAG}.csv")
    best_row = ppc_df.iloc[0] 
    extract_and_plot(str(best_row['subject_id']), best_row['filename'], Path(args.sst_folder))
