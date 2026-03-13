import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FIXED_PARAMS_PATH = DATA_DIR / "params_posterior.csv"

df_raw = pd.read_csv(FIXED_PARAMS_PATH)
df = df_raw[['index', 'mean', 'subject_id', 'subject_year']]
df = df.pivot_table(
    index=['subject_id'], columns='index', values='mean'
).reset_index()
df.columns.name = None

# save
OUT_PATH = DATA_DIR / "params_pomdp.csv"
df.to_csv(OUT_PATH, index=False)