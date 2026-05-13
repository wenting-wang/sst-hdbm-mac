import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Iterable, Optional, Union, List, Tuple

# --- Configuration & Paths ---

# CURRENT_DIR is the 'analysis' folder where this script lives
CURRENT_DIR = Path(__file__).resolve().parent

# PROJECT_ROOT is the parent folder containing both 'analysis' and 'data'
PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"

# Ensure output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Input: Replace with your actual dataset filename
# Expected columns: 'year', 'adhd', 'sex', 'iq', 'adhd_med_flag'
DATA_CSV = DATA_DIR / "example_clinical_behavior.csv"

# Output: LaTeX table file
OUTPUT_TEX = OUT_DIR / "tab_demo_counts.tex"

# --- Helper Functions ---

def normalize_group_column(series: pd.Series, kind: str) -> pd.Series:
    """
    Standardizes categorical values for display in the LaTeX table.
    """
    s = series.copy()
    
    if kind == "sex":
        sex_map = {
            "F": "Female", "Female": "Female", "female": "Female", "f": "Female",
            "M": "Male", "Male": "Male", "male": "Male", "m": "Male"
        }
        return s.map(lambda x: sex_map.get(str(x), str(x)))
    
    if kind == "med":
        med_map = {
            0: "Unmedicated", 1: "Medicated", "0": "Unmedicated", "1": "Medicated",
            "no": "Unmedicated", "yes": "Medicated", 
            False: "Unmedicated", True: "Medicated"
        }
        return s.map(lambda x: med_map.get(x, med_map.get(str(x), str(x))))
        
    return s


def bin_numeric(series: pd.Series,
                labels: Iterable[str] = ("Low IQ", "Mid IQ", "High IQ"),
                bins: Optional[Iterable[Union[int, float]]] = None,
                q: Iterable[float] = (0, 1/3, 2/3, 1)) -> pd.Series:
    """
    Bins a numeric series (like IQ) into categorical labels (e.g., Tertiles).
    """
    ser = pd.to_numeric(series, errors="coerce")
    ser = ser[ser.notna()]
    
    if ser.empty:
        return pd.Categorical([np.nan] * len(series), categories=labels)

    smin, smax = float(ser.min()), float(ser.max())
    span = max(smax - smin, 1.0)
    tiny = max(1e-12, 1e-9 * span)

    if bins is not None:
        edges = list(map(float, bins))
        edges.sort()
        if edges[0] > smin: edges[0] = smin - tiny
        if edges[-1] < smax: edges[-1] = smax + tiny
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = np.nextafter(edges[i-1], float('inf'))
                
        return pd.cut(pd.to_numeric(series, errors="coerce"),
                      bins=edges, include_lowest=True, labels=labels)
        
    qs = np.array(ser.quantile(q, interpolation="linear"), dtype=float)
    
    for i in range(1, len(qs)):
        if qs[i] <= qs[i-1]:
            qs[i] = np.nextafter(qs[i-1], float('inf'))
            
    if qs[-1] < smax:  
        qs[-1] = smax + tiny
    qs[-1] = np.nextafter(qs[-1], float('inf')) 
    
    return pd.cut(pd.to_numeric(series, errors="coerce"),
                  bins=qs, include_lowest=True, labels=labels)


def generate_counts_table(
    df: pd.DataFrame,
    *,
    adhd_col: str = "adhd",
    scores: Iterable[int] = range(0, 15),
    groups: List[Tuple[str, str, str]] = None,
    caption: str = "Subgroup counts by Sex, IQ, and Medication across ADHD scores.",
    label: str = "tab:counts_sex_iq_med",
    outfile: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generates the summary dataframe and writes the LaTeX table matching the specific structure.
    """
    if groups is None:
        groups = [
            ("Sex", "sex", "sex"),
            ("IQ", "iq", "numeric"),
            ("Medication", "adhd_med_flag", "med"),
        ]

    # Select and Clean Data
    use_cols = [adhd_col] + [col for _, col, _ in groups]
    sub = df[use_cols].copy()
    
    # Ensure ADHD column is integer (round floats to handle mock data gracefully)
    sub[adhd_col] = pd.to_numeric(sub[adhd_col], errors="coerce")
    sub = sub.dropna(subset=[adhd_col])
    sub[adhd_col] = sub[adhd_col].round().astype(int)
    sub = sub[sub[adhd_col].isin(scores)]
    
    score_cols = list(scores)

    # --- Part 1: Overall "Count" Row ---
    overall_ct = sub[adhd_col].value_counts().reindex(score_cols, fill_value=0).astype(int)
    total_count = overall_ct.sum()
    
    count_df = pd.DataFrame([list(overall_ct.values) + [total_count]],
                            columns=score_cols + ["Total"])
    count_df.insert(0, "ADHD Score", "Count")
    count_df.insert(0, "Group", "")

    parts = [count_df]

    # --- Part 2: Grouped Rows (Sex, IQ, Meds) ---
    for (disp_name, col_name, kind) in groups:
        tmp = sub[[adhd_col, col_name]].copy()

        # Apply Normalization / Binning
        if kind in {"sex", "med"}:
            tmp[col_name] = normalize_group_column(tmp[col_name], kind)
        elif kind == "numeric":
            tmp[col_name] = bin_numeric(tmp[col_name], labels=("Low IQ", "Mid IQ", "High IQ"))
        
        tmp = tmp.dropna(subset=[col_name])

        # Cross-tabulation
        ct = pd.crosstab(tmp[col_name].astype(str), tmp[adhd_col])
        ct = ct.reindex(columns=score_cols, fill_value=0).astype(int)
        ct["Total"] = ct.sum(axis=1)

        # Format block
        block_df = ct.reset_index().rename(columns={col_name: "ADHD Score"})
        block_df.insert(0, "Group", "")
        if not block_df.empty:
            block_df.iloc[0, 0] = disp_name  # Set Group label on first row
            
        parts.append(block_df)

    combined = pd.concat(parts, axis=0, ignore_index=True)

    # --- Part 3: LaTeX Formatting ---
    colspec = "ll" + "r" * len(score_cols) + "r"
    header = " & ".join(["Group", "ADHD Score"] + [str(s) for s in score_cols] + ["Total"]) + r" \\"

    lines = []
    
    for i, row in combined.iterrows():
        g = str(row["Group"])
        lvl = str(row["ADHD Score"])
        vals = [str(int(row[s])) for s in score_cols] + [str(int(row["Total"]))]
        line = " & ".join([g, lvl] + vals) + r" \\"
        
        # Add a midrule before the start of any new group (Sex, IQ, Meds)
        if i > 0 and g != "":
            lines.append(r"\midrule")
            
        lines.append(line)

    # Construct the final LaTeX string matching the target format exactly
    latex_str = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\captionsetup{labelfont=bf}\n"
        f"\\caption{{\\textbf{{{caption}}}}}\n"
        f"\\label{{{label}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        "\n"
        "\\vspace{1mm}\n"
        "\\begin{minipage}{\\textwidth} % Force a container as wide as the page\n"
        "    \\footnotesize \\noindent Note: IQ groups defined by sample tertiles: "
        "Low ($<17$), Mid ($17 \\le \\text{IQ} < 20$), and High ($\\ge 20$). "
        "Due to the discrete nature of the IQ scores, the resulting groups are not exactly equal in size. "
        "Medication indicates stimulant use at assessment.\n"
        "\\end{minipage}\n"
        "\\end{table}\n"
    )

    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(latex_str)
        print(f"LaTeX table saved to: {outfile}")

    return combined

# --- Main Execution ---

def main():
    try:
        if not DATA_CSV.exists():
            raise FileNotFoundError(
                f"Data file not found at {DATA_CSV}. "
                "Please place your dataset in the 'data' folder or update the DATA_CSV path."
            )

        # 1. Load Data
        df = pd.read_csv(DATA_CSV, index_col=False)
        
        # 2. Filter for Baseline
        if 'year' in df.columns:
            df_baseline = df[df['year'].astype(str) == 'baseline'].copy()
            print(f"Processing {len(df_baseline)} baseline subjects.")
        else:
            print("Warning: 'year' column not found. Processing all subjects.")
            df_baseline = df.copy()

        # 3. Generate Table
        _ = generate_counts_table(
            df_baseline,
            outfile=OUTPUT_TEX
        )

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()