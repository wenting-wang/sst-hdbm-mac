import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import traceback

# --- Configuration & Paths ---

# --- Directory Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"

# Ensure output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Input Files ---
# PARAMS_CSV = DATA_DIR / "example_params_posteriors.csv" 
# STATS_CSV = DATA_DIR / "example_clinical_behavior.csv"

PARAMS_CSV = DATA_DIR / "params_posteriors_6p_v6.csv" 
STATS_CSV = DATA_DIR / "clinical_behavior.csv"

# --- Output ---
OUTPUT_TEX = OUT_DIR / "tab_lm_params_6p_v6.tex"

# --- Analysis Config ---
PARAMS_OF_INTEREST = ['q_d_n', 
                      'q_d',
                      'q_s', 
                      'cost_stop_error',
                      'cost_time', 
                      'tau']

PARAM_DISPLAY_NAMES = {
    'q_d': r"$\chi$",
    'q_d_n': r"$\chi'$",
    'q_s': r"$\delta$",
    'cost_stop_error': r"$c_{\mathrm{se}}$",
    'cost_time': r"$c_{\mathrm{t}}$",
    'tau': r"$\tau$",
}

# --- Core Functions ---

def load_and_prep_data(params_path, stats_path, params_to_keep):
    """
    Loads posterior parameters and demographic stats, merges them, 
    and filters for baseline data.
    """
    missing_files = [p.name for p in [params_path, stats_path] if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing input files in 'data' folder: {', '.join(missing_files)}")

    # 1. Load Parameters
    df_param = pd.read_csv(params_path)
    if 'subject_year' in df_param.columns:
        df_param = df_param.rename(columns={"subject_year": "year"})
    
    # Force string for merging
    df_param['year'] = df_param['year'].astype(str)
    
    # Pivot (using pivot_table to gracefully handle any duplicate indexes)
    df_pivot = df_param.pivot_table(
        index=['subject_id', 'year'], 
        columns='index', 
        values='mean'
    ).reset_index()
    df_pivot.columns.name = None

    # 2. Load Stats
    df_stats = pd.read_csv(stats_path)
    df_stats['year'] = df_stats['year'].astype(str)
    
    df_pivot['subject_id'] = df_pivot['subject_id'].str.replace('NDAR_', '', regex=False)
    
    # 3. Merge
    df_merged = pd.merge(df_stats, df_pivot, on=['subject_id', 'year'], how='inner')
    
    # 4. Filter for Baseline
    df_baseline = df_merged[df_merged['year'] == 'baseline'].copy()
    
    # 5. Drop NA only across variables actually used in the model
    req_cols = ['adhd', 'sex', 'iq', 'adhd_med_flag'] + [p for p in params_to_keep if p in df_baseline.columns]
    avail_cols = [c for c in req_cols if c in df_baseline.columns]
    
    df_clean = df_baseline.dropna(subset=avail_cols).copy()
    
    print(f"Data Loaded. N={len(df_clean)} subjects (Baseline).")
    return df_clean


def fit_linear_models(df, params):
    """
    Fits Multiple Linear Regression (OLS) models.
    """
    rows = []
    # Force Female as the reference group for consistency
    formula_template = "{target} ~ adhd + C(sex, Treatment('Female')) + iq + adhd_med_flag"

    print(f"Fitting OLS models for: {params}...")

    for target_param in params:
        if target_param not in df.columns:
            print(f"Skipping {target_param}: not found in data.")
            continue
            
        dsub = df.dropna(subset=[target_param]).copy()
        if dsub.empty:
            continue
        
        try:
            # OLS Model
            model = smf.ols(
                formula=formula_template.format(target=target_param), 
                data=dsub
            )
            result = model.fit()

            # Helper
            def get_term(term):
                if term in result.params.index:
                    return (result.params[term], result.bse[term], result.pvalues[term])
                else:
                    return (np.nan, np.nan, np.nan)

            # Extract terms matching statsmodels' generated names
            intercept, int_se, int_p = get_term('Intercept')
            adhd, adhd_se, adhd_p = get_term('adhd')
            sex, sex_se, sex_p = get_term("C(sex, Treatment('Female'))[T.Male]")
            iq, iq_se, iq_p = get_term('iq')
            med, med_se, med_p = get_term('adhd_med_flag')

            rows.append({
                'param': target_param,
                'param_label': PARAM_DISPLAY_NAMES.get(target_param, target_param),
                'Intercept': intercept, 'Intercept_se': int_se, 'Intercept_p': int_p,
                'ADHD': adhd, 'ADHD_se': adhd_se, 'ADHD_p': adhd_p,
                'Sex': sex, 'Sex_se': sex_se, 'Sex_p': sex_p,
                'IQ': iq, 'IQ_se': iq_se, 'IQ_p': iq_p,
                'Medication': med, 'Medication_se': med_se, 'Medication_p': med_p,
            })
            
        except Exception as e:
            print(f"Error fitting model for {target_param}: {e}")

    return pd.DataFrame(rows)

def build_latex_table(summary_df, n_subjects):
    """Generates the LaTeX table string with the specified formatting layout."""
    
    def p_to_stars(p):
        if pd.isna(p): return ""
        if p < 0.001: return r"\textbf{***}"
        if p < 0.01:  return r"\textbf{**}"
        if p < 0.05:  return r"\textbf{*}"
        return ""

    def fmt(beta, se, p, is_scientific=False):
        if pd.isna(beta) or pd.isna(se):
            return "-"
        
        if is_scientific:
            val_str = f"{beta:.2e}"
            se_str = f"{se:.2e}"
        else:
            # Format Beta
            val_str = f"{beta:.3f}"
            if val_str == "0.000" or val_str == "-0.000":
                val_str = f"{beta:.4f}"
            
            # Format Standard Error
            se_str = f"{se:.3f}"
            if se_str == "0.000" or se_str == "-0.000":
                se_str = f"{se:.4f}"

        stars = p_to_stars(p)
        return f"{val_str}{stars} ({se_str})"

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\begin{adjustwidth}{-2.25in}{0in} % Comment out/remove adjustwidth environment if table fits in text column.")
    lines.append(r"\centering")
    
    # Place Caption at the top, bolded
    lines.append(r"\caption{\textbf{Multiple linear regression results of parameters with ADHD and covariates.}}")
    
    # Define Column structure
    lines.append(r"\begin{tabularx}{\linewidth}{lXXXXX}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Parameter} & \textbf{Intercept} & \textbf{ADHD} & \textbf{Sex (Male)} & \textbf{IQ} & \textbf{Medication} \\")
    lines.append(r"\midrule")

    for _, r in summary_df.iterrows():
        is_sci = (r['param'] == 'cost_time')
        row = (
            f"{r['param_label']} & "
            f"{fmt(r['Intercept'], r['Intercept_se'], r['Intercept_p'], is_sci)} & "
            f"{fmt(r['ADHD'], r['ADHD_se'], r['ADHD_p'], is_sci)} & "
            f"{fmt(r['Sex'], r['Sex_se'], r['Sex_p'], is_sci)} & "
            f"{fmt(r['IQ'], r['IQ_se'], r['IQ_p'], is_sci)} & "
            f"{fmt(r['Medication'], r['Medication_se'], r['Medication_p'], is_sci)} \\\\"
        )
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    
    # Note Section using flushleft
    lines.append(rf"\begin{{flushleft}} Note: Multiple linear regression models ($N={n_subjects:,}$). "
                 r"Entries are fixed-effect coefficients with Wald SEs in parentheses. "
                 r"Stars denote p-values: \textbf{*} $p<.05$, \textbf{**} $p<.01$, \textbf{***} $p<.001$. "
                 r"ADHD was the main predictor, with Sex, IQ, and Medication included as covariates.")
    lines.append(r"\end{flushleft}")
    
    # Labels and closing tags
    lines.append(r"\label{tab:lm_pomdp}")
    lines.append(r"\end{adjustwidth}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)

# --- Main Execution ---

def main():
    try:
        # Load and clean data
        df = load_and_prep_data(PARAMS_CSV, STATS_CSV, PARAMS_OF_INTEREST)
        n_subjects = len(df)
        
        if df.empty:
            print("Warning: Dataset is empty after filtering for 'baseline' and cleaning NaNs. Exiting.")
            return

        # Fit models
        results_df = fit_linear_models(df, PARAMS_OF_INTEREST)
        
        # Build and save table
        if not results_df.empty:
            latex_code = build_latex_table(results_df, n_subjects)
            with open(OUTPUT_TEX, "w") as f:
                f.write(latex_code)
            print(f"\nSaved LaTeX table to: {OUTPUT_TEX}")
        else:
            print("No models were successfully fit.")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()