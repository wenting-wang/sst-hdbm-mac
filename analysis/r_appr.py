import sys
import pandas as pd
import numpy as np
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from core.hdbm import HDBM

def generate_hybrid_bins(n_bins=50, samples_per_seq=50, density_weight=0.6):
    print("Loading sequence orders from seq_orders.csv...")
    data_dir = PROJECT_ROOT / "data"
    seq_df = pd.read_csv(data_dir / "seq_orders.csv")
    
    all_r_preds = []
    np.random.seed(42)
    
    print(f"Running HDBM simulations ({samples_per_seq} random params per sequence)...")
    for idx, row in seq_df.iterrows():
        seq_str = row['order_seq']
        sequence = [int(x) for x in seq_str]
        
        for _ in range(samples_per_seq):
            alpha = np.random.uniform(0, 1)
            rho = np.random.uniform(0, 1)
            hdbm = HDBM(alpha=alpha, rho=rho)
            all_r_preds.extend(hdbm.simu_task(sequence))
            
    all_r_preds = np.array(all_r_preds)
    
    # --- 1. 获取纯密度分布的点 (Quantile) ---
    percentiles = np.linspace(0, 100, n_bins)
    quantile_rates = np.percentile(all_r_preds, percentiles)
    
    # --- 2. 获取纯均匀分布的点 (Uniform) ---
    min_r, max_r = np.min(all_r_preds), np.max(all_r_preds)
    uniform_rates = np.linspace(min_r, max_r, n_bins)
    
    # --- 3. 混合计算 (Hybrid) ---
    # 结合密度分布的细腻和均匀分布的广度
    hybrid_rates = (density_weight * quantile_rates) + ((1 - density_weight) * uniform_rates)
    
    print("-" * 50)
    print(f"Hybrid Bins generated (Density Weight: {density_weight*100}%)")
    print(f"Total r values generated: {len(all_r_preds):,}")
    print(f"Min: {hybrid_rates[0]:.4f} | Max: {hybrid_rates[-1]:.4f}")
    print("-" * 50)
    
    rates_str = ", ".join([f"{r:.6f}" for r in hybrid_rates])
    print(f"\n# Please COPY the following array into your simu_dyna.py:")
    print(f"RATES = np.array([\n    {rates_str}\n])")
    
    out_csv_path = data_dir / "r_appr.csv"
    df_rates = pd.DataFrame({'rate': hybrid_rates})
    df_rates.to_csv(out_csv_path, index=False)
    
    
    # plot bin and density and save plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(all_r_preds, bins=100, density=True, alpha=0.6, color='grey')
    plt.scatter(hybrid_rates, np.zeros_like(hybrid_rates)+0.1, color='red', marker='.', label='Bins')
    plt.title("Density of r values with Bins")
    plt.xlabel("r value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "r_appr_density.png")


if __name__ == "__main__":
    generate_hybrid_bins(density_weight=0.6)
    