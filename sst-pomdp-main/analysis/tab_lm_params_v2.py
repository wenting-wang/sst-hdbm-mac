import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import traceback
import warnings

warnings.filterwarnings("ignore")

# --- Configuration & Paths ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMS_CSV = DATA_DIR / "params_posteriors_5p_v1.csv" 
STATS_CSV = DATA_DIR / "clinical_behavior.csv"
OUTPUT_TEX = OUT_DIR / "tab_lm_params_5p_v1.tex"

PARAMS_OF_INTEREST = [    'q_d',
    'q_s',
    'tau',
    'cost_stop_error',
    'cost_time']

PARAM_DISPLAY_NAMES = {
    'q_d': r"$\chi$ (Go Precision)",
    'q_s': r"$\delta$ (Stop Precision)",
    'cost_stop_error': r"$c_{\mathrm{se}}$ (Stop Error Cost)",
    'cost_time': r"$c_{\mathrm{t}}$ (Time Cost)",
    'tau': r"$\tau$ (Non-decision Time)",
}


# Define predictors for model formulas and extraction
PREDICTORS_FORMULA = {
    'ADHD': 'adhd',
    'Sex': "C(sex, Treatment('Female'))",
    'IQ': 'iq',
    'Medication': 'adhd_med_flag'
}

PREDICTORS_EXTRACT = {
    'ADHD': 'adhd',
    'Sex': "C(sex, Treatment('Female'))[T.Male]",
    'IQ': 'iq',
    'Medication': 'adhd_med_flag'
}

# --- Core Functions ---
def load_and_prep_data(params_path, stats_path, params_to_keep):
    try:
        df_param = pd.read_csv(params_path)
        df_stats = pd.read_csv(stats_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing file: {e.filename}")

    if 'subject_year' in df_param.columns:
        df_param = df_param.rename(columns={"subject_year": "year"})
    
    df_param['year'] = df_param['year'].astype(str)
    df_pivot = df_param.pivot_table(index=['subject_id', 'year'], columns='index', values='mean').reset_index()
    df_pivot.columns.name = None
    
    df_stats['year'] = df_stats['year'].astype(str)
    df_pivot['subject_id'] = df_pivot['subject_id'].str.replace('NDAR_', '', regex=False)
    
    df_merged = pd.merge(df_stats, df_pivot, on=['subject_id', 'year'], how='inner')
    df_baseline = df_merged[df_merged['year'] == 'baseline'].copy()
    
    req_cols = ['adhd', 'sex', 'iq', 'adhd_med_flag'] + [p for p in params_to_keep if p in df_baseline.columns]
    avail_cols = [c for c in req_cols if c in df_baseline.columns]
    df_clean = df_baseline.dropna(subset=avail_cols).copy()
    
    # Standardization (Z-score for continuous)
    continuous_vars = ['adhd', 'iq'] + [p for p in params_to_keep if p in df_clean.columns]
    for col in continuous_vars:
        if df_clean[col].std() > 0:
            df_clean[col] = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
            
    return df_clean

def fit_all_effect_sizes(df, params):
    rows = []
    base_terms = list(PREDICTORS_FORMULA.values())
    full_rhs = " + ".join(base_terms)
    
    for target_param in params:
        if target_param not in df.columns: continue
        dsub = df.dropna(subset=[target_param]).copy()
        if dsub.empty: continue
        
        try:
            # Full Model
            model_full = smf.ols(formula=f"{target_param} ~ {full_rhs}", data=dsub).fit()
            r2_full = model_full.rsquared
            
            row_data = {'param': target_param, 'param_label': PARAM_DISPLAY_NAMES.get(target_param, target_param)}
            
            # Loop over each predictor to calculate its specific Partial R2 and f2
            for pred_name, formula_term in PREDICTORS_FORMULA.items():
                # Extract Beta, SE, p
                extract_term = PREDICTORS_EXTRACT[pred_name]
                if extract_term in model_full.params.index:
                    beta = model_full.params[extract_term]
                    se = model_full.bse[extract_term]
                    p_val = model_full.pvalues[extract_term]
                else:
                    beta, se, p_val = np.nan, np.nan, np.nan
                
                # Fit Reduced Model (drop current predictor)
                reduced_terms = [t for t in base_terms if t != formula_term]
                reduced_rhs = " + ".join(reduced_terms)
                model_reduced = smf.ols(formula=f"{target_param} ~ {reduced_rhs}", data=dsub).fit()
                
                r2_reduced = model_reduced.rsquared
                
                # Effect Sizes
                partial_r2 = max(0.0, (r2_full - r2_reduced) / (1 - r2_reduced))
                cohens_f2 = partial_r2 / max(1 - partial_r2, 1e-10)
                
                row_data[f'{pred_name}_beta'] = beta
                row_data[f'{pred_name}_se'] = se
                row_data[f'{pred_name}_p'] = p_val
                row_data[f'{pred_name}_pr2'] = partial_r2
                row_data[f'{pred_name}_f2'] = cohens_f2
                
            rows.append(row_data)
        except Exception as e:
            print(f"Error on {target_param}: {e}")
            
    return pd.DataFrame(rows)

def build_latex_table(df_res, n_subjects):
    def p_to_stars(p):
        if pd.isna(p): return ""
        if p < 0.001: return r"\textbf{***}"
        if p < 0.01:  return r"\textbf{**}"
        if p < 0.05:  return r"\textbf{*}"
        return ""

    def fmt_cell(beta, se, p, f2):
        if pd.isna(beta): return "-"
        stars = p_to_stars(p)
        return f"{beta:.3f}{stars} ({se:.3f}) & {f2:.4f}"

    lines = [
        r"\begin{table}[H]",
        r"\begin{adjustwidth}{-1in}{-1in} % Widen table for multiple predictors",
        r"\centering",
        r"\caption{\textbf{Standardized coefficients and effect sizes across all predictors.}}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l | cc | cc | cc | cc}",
        r"\toprule",
        r" & \multicolumn{2}{c|}{\textbf{ADHD}} & \multicolumn{2}{c|}{\textbf{Sex (Male)}} & \multicolumn{2}{c|}{\textbf{IQ}} & \multicolumn{2}{c}{\textbf{Medication}} \\",
        r"\textbf{Param} & $\beta$ (SE) & $f^2$ & $\beta$ (SE) & $f^2$ & $\beta$ (SE) & $f^2$ & $\beta$ (SE) & $f^2$ \\",
        r"\midrule"
    ]

    for _, r in df_res.iterrows():
        row_str = f"{r['param_label']} & "
        row_str += f"{fmt_cell(r['ADHD_beta'], r['ADHD_se'], r['ADHD_p'], r['ADHD_f2'])} & "
        row_str += f"{fmt_cell(r['Sex_beta'], r['Sex_se'], r['Sex_p'], r['Sex_f2'])} & "
        row_str += f"{fmt_cell(r['IQ_beta'], r['IQ_se'], r['IQ_p'], r['IQ_f2'])} & "
        row_str += f"{fmt_cell(r['Medication_beta'], r['Medication_se'], r['Medication_p'], r['Medication_f2'])} \\\\"
        lines.append(row_str)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        rf"\begin{{flushleft}} \small Note: $N={n_subjects:,}$. Continuous variables were Z-score standardized. "
        r"Coefficients ($\beta$) with Wald SEs. Stars: \textbf{*} $p<.05$, \textbf{**} $p<.01$, \textbf{***} $p<.001$. "
        r"$f^2$ denotes Cohen's $f^2$. \end{flushleft}",
        r"\label{tab:all_effects}",
        r"\end{adjustwidth}",
        r"\end{table}"
    ])
    
    return "\n".join(lines)

# --- Main ---
def main():
    try:
        df = load_and_prep_data(PARAMS_CSV, STATS_CSV, PARAMS_OF_INTEREST)
        n_sub = len(df)
        if df.empty: return
        
        res_df = fit_all_effect_sizes(df, PARAMS_OF_INTEREST)
        
        # Format Console Print
        print("\n" + "="*110)
        print(f" ALL PREDICTORS EFFECT SIZES (N = {n_sub})")
        print("="*110)
        
        # Print block by block for readability
        for pred in ['ADHD', 'Sex', 'IQ', 'Medication']:
            print(f"\n--- Predictor: {pred.upper()} ---")
            print(f"{'Param':<15} {'Beta':>10} {'SE':>10} {'p-value':>10} {'Part_R2':>10} {'f2':>10}")
            for _, r in res_df.iterrows():
                print(f"{r['param']:<15} {r[f'{pred}_beta']:10.4f} {r[f'{pred}_se']:10.4f} "
                      f"{r[f'{pred}_p']:10.4f} {r[f'{pred}_pr2']:10.4f} {r[f'{pred}_f2']:10.4f}")
        
        print("\n" + "="*110)
        
        # Save LaTeX
        with open(OUTPUT_TEX, "w") as f:
            f.write(build_latex_table(res_df, n_sub))
        print(f"\nSaved comprehensive LaTeX table to: {OUTPUT_TEX}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
    
# ==============================================================================================================
#  ALL PREDICTORS EFFECT SIZES (N = 3567)
# ==============================================================================================================

# --- Predictor: ADHD ---
# Param                 Beta         SE    p-value    Part_R2         f2
# q_d                -0.0645     0.0182     0.0004     0.0035     0.0035
# q_s                 0.0223     0.0183     0.2229     0.0004     0.0004
# tau                 0.0285     0.0182     0.1179     0.0007     0.0007
# cost_stop_error    -0.0529     0.0181     0.0035     0.0024     0.0024
# cost_time          -0.0364     0.0182     0.0456     0.0011     0.0011

# --- Predictor: SEX ---
# Param                 Beta         SE    p-value    Part_R2         f2
# q_d                 0.0072     0.0336     0.8299     0.0000     0.0000
# q_s                -0.2090     0.0339     0.0000     0.0106     0.0107
# tau                -0.2465     0.0337     0.0000     0.0148     0.0151
# cost_stop_error     0.3233     0.0335     0.0000     0.0255     0.0261
# cost_time           0.2898     0.0336     0.0000     0.0204     0.0208

# --- Predictor: IQ ---
# Param                 Beta         SE    p-value    Part_R2         f2
# q_d                 0.1452     0.0166     0.0000     0.0210     0.0214
# q_s                -0.0151     0.0167     0.3679     0.0002     0.0002
# tau                -0.0914     0.0166     0.0000     0.0084     0.0085
# cost_stop_error     0.0803     0.0166     0.0000     0.0066     0.0066
# cost_time           0.0627     0.0166     0.0002     0.0040     0.0040

# --- Predictor: MEDICATION ---
# Param                 Beta         SE    p-value    Part_R2         f2
# q_d                 0.1964     0.0789     0.0129     0.0017     0.0017
# q_s                 0.0701     0.0796     0.3784     0.0002     0.0002
# tau                -0.0721     0.0791     0.3619     0.0002     0.0002
# cost_stop_error    -0.0069     0.0787     0.9306     0.0000     0.0000
# cost_time          -0.0432     0.0790     0.5845     0.0001     0.0001

# ==============================================================================================================


