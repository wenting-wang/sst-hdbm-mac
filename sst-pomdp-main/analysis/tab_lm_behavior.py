import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from pathlib import Path

# ===== 1. Configuration & Paths =====

# --- Directory Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"

# Ensure output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Inputs: Replace with your actual dataset filename 
# Expected to have standard behavioral stats + demographic columns
STATS_CSV = DATA_DIR / "example_clinical_behavior.csv"

# Output
OUTPUT_TEX = OUT_DIR / "tab_lm_behavior.tex"

DVS = ["mrt_gs", "ssrt", "perc_gs", "perc_ss"]

SUB_CAP = {
    "mrt_gs": "Go success reaction times",
    "ssrt":   "Stop signal reaction times",
    "perc_gs": "Go success rates",
    "perc_ss": "Stop success rates",
}

# Mapping statsmodels names to readable labels
PREDICTOR_MAPPING = {
    "Intercept": "Intercept",
    "adhd": "ADHD",
    "C(sex, Treatment('Female'))[T.Male]": "Sex (Male)",
    "iq": "IQ",
    "adhd_med_flag": "Medication",
}

# ===== 2. Helper Functions =====

def format_num(x):
    """
    Formats Coef, SE, t:
    1. Default to 3 decimal places.
    2. If result is '0.000' or '-0.000', use 4 decimal places.
    """
    if pd.isna(x):
        return ""
    
    s = f"{x:.3f}"
    if s == "0.000" or s == "-0.000":
        return f"{x:.4f}"
    return s

def format_p(p):
    """
    Formats p-value:
    1. If p < 0.001, return $<0.001$$^{***}$
    2. Otherwise, return 3 decimals with stars as superscript if significant.
    """
    if pd.isna(p):
        return ""
    
    stars = ""
    if p < 0.001: stars = "***"
    elif p < 0.01: stars = "**"
    elif p < 0.05: stars = "*"
    
    if p < 0.001:
        return r"$<0.001$$^{***}$"
    else:
        s = f"{p:.3f}"
        if stars:
            return f"{s}$^{{{stars}}}$"
        return s

def fit_models(df):
    """Fits OLS models and returns a summary dictionary."""
    results = {}
    formula_template = "{dv} ~ adhd + C(sex, Treatment('Female')) + iq + adhd_med_flag"
    
    for dv in DVS:
        if dv not in df.columns:
            print(f"Warning: Dependent variable '{dv}' not found in data. Skipping.")
            continue
            
        model = smf.ols(formula=formula_template.format(dv=dv), data=df)
        fit = model.fit()
        
        summary = {}
        for predictor, label in PREDICTOR_MAPPING.items():
            if predictor in fit.params:
                summary[label] = {
                    'beta': fit.params[predictor],
                    'se': fit.bse[predictor],
                    't': fit.tvalues[predictor],
                    'p': fit.pvalues[predictor]
                }
        results[dv] = summary
    return results

# ===== 3. Table Construction =====

def build_latex_table(results_dict):
    """Generates a four-panel LaTeX table matching the threeparttable structure."""
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\captionsetup{labelfont=bf}")
    lines.append(r"\caption{\textbf{Linear model results for model-agnostic behavioral measures}}")
    lines.append(r"\label{tab:lm_all}")
    lines.append(r"")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"")
    lines.append(r"% wrap all subtables to force full width for threeparttable")
    lines.append(r"\begin{minipage}{\linewidth}")
    
    valid_dvs = [dv for dv in DVS if dv in results_dict]
    label_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    
    def get_subtable(dv, sub_label):
        sub_lines = []
        sub_lines.append(r"    \begin{subtable}[t]{0.48\linewidth}")
        sub_lines.append(r"    \centering")
        sub_lines.append(rf"    \caption{{{SUB_CAP[dv]}}}")
        sub_lines.append(rf"    \label{{tab:lm_all_{sub_label}}}")
        sub_lines.append(r"    \begin{tabular}{lcccc}")
        sub_lines.append(r"    \toprule")
        sub_lines.append(r"    \textbf{Predictor} & \textbf{Coef.} & \textbf{SE} & \textbf{t} & \textbf{p} \\")
        sub_lines.append(r"    \midrule")
        
        for _, label in PREDICTOR_MAPPING.items():
            res = results_dict[dv].get(label, {})
            
            if not res:
                beta_str, se_str, t_str, p_str = "-", "-", "-", "-"
            else:
                beta_str = format_num(res.get('beta'))
                se_str   = format_num(res.get('se'))
                t_str    = format_num(res.get('t'))
                p_str    = format_p(res.get('p'))
                
            sub_lines.append(f"    {label} & {beta_str} & {se_str} & {t_str} & {p_str} \\\\")
            
        sub_lines.append(r"    \bottomrule")
        sub_lines.append(r"    \end{tabular}")
        sub_lines.append(r"    \end{subtable}")
        return "\n".join(sub_lines)

    # Build rows dynamically
    for i in range(0, len(valid_dvs), 2):
        lines.append(get_subtable(valid_dvs[i], label_map.get(i, 'x')))
        if i + 1 < len(valid_dvs):
            lines.append(r"\hfill")
            lines.append(get_subtable(valid_dvs[i+1], label_map.get(i+1, 'y')))
        if i + 2 < len(valid_dvs):
            lines.append("\n" + r"\vspace{0.3cm}" + "\n")
    
    lines.append(r"\end{minipage}")
    
    # Bottom Note Section
    lines.append(r"")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\begin{minipage}{\textwidth} % Force a container as wide as the page")
    lines.append(r"    \footnotesize \noindent Note: $t$-tests. * $p <$ .05, ** $p <$ .01, *** $p <$ .001")
    lines.append(r"\end{minipage}")
    
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)

# ===== 4. Execution =====

def main():
    try:
        if not STATS_CSV.exists():
            raise FileNotFoundError(
                f"Data file not found at {STATS_CSV}. "
                "Please place your dataset in the 'data' folder or update the STATS_CSV path."
            )

        # Load data
        df = pd.read_csv(STATS_CSV)
        
        # Filter for baseline
        if 'year' in df.columns:
            df = df[df['year'].astype(str) == 'baseline']
            
        req_cols = DVS + ['adhd', 'sex', 'iq', 'adhd_med_flag']
        available_cols = [col for col in req_cols if col in df.columns]
        df_clean = df[available_cols].dropna().copy()
        
        print(f"Data Loaded and Cleaned. N={len(df_clean)} valid subjects.")

        if df_clean.empty:
            print("Warning: Dataset is empty after filtering for 'baseline'. Table will not be generated.")
            return

        # Fit models
        results = fit_models(df_clean)

        # Generate and Save LaTeX
        if results:
            latex_code = build_latex_table(results)
            with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
                f.write(latex_code)
            print(f"Successfully saved LaTeX table to {OUTPUT_TEX}")
        else:
            print("No models were successfully fitted.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()