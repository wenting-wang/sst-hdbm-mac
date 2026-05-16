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
OUTPUT_TEX = OUT_DIR / "tab_lm_params_5p_v1_interactions.tex"

PARAMS_OF_INTEREST = [
    'q_d',
    'q_s',
    'tau',
    'cost_stop_error',
    'cost_time'
]

PARAM_DISPLAY_NAMES = {
    'q_d': r"$\chi$ (Go Precision)",
    'q_s': r"$\delta$ (Stop Precision)",
    'cost_stop_error': r"$c_{\mathrm{se}}$ (Stop Error Cost)",
    'cost_time': r"$c_{\mathrm{t}}$ (Time Cost)",
    'tau': r"$\tau$ (Non-decision Time)",
}

# --- Expanded Predictors with Interactions ---
PREDICTORS_FORMULA = {
    'ADHD': 'adhd',
    'Sex': "C(sex, Treatment('Female'))",
    'IQ': 'iq',
    'Medication': 'adhd_med_flag',
    'ADHD_x_Sex': "adhd:C(sex, Treatment('Female'))",
    'ADHD_x_IQ': "adhd:iq",
    'ADHD_x_Med': "adhd:adhd_med_flag"
}

PREDICTORS_EXTRACT = {
    'ADHD': 'adhd',
    'Sex': "C(sex, Treatment('Female'))[T.Male]",
    'IQ': 'iq',
    'Medication': 'adhd_med_flag',
    'ADHD_x_Sex': "adhd:C(sex, Treatment('Female'))[T.Male]",
    'ADHD_x_IQ': "adhd:iq",
    'ADHD_x_Med': "adhd:adhd_med_flag"
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
            
            for pred_name, formula_term in PREDICTORS_FORMULA.items():
                extract_term = PREDICTORS_EXTRACT[pred_name]
                
                # Robust extraction: patsy sometimes swaps interaction terms alphabetically
                if extract_term not in model_full.params.index and ':' in extract_term:
                    parts = extract_term.split(':')
                    alt_term = f"{parts[1]}:{parts[0]}"
                    if alt_term in model_full.params.index:
                        extract_term = alt_term
                
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
        r"\caption{\textbf{Standardized coefficients and effect sizes (including ADHD interactions).}}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l | cc | cc | cc | cc | cc | cc | cc}",
        r"\toprule",
        r" & \multicolumn{2}{c|}{\textbf{ADHD}} & \multicolumn{2}{c|}{\textbf{Sex (Male)}} & \multicolumn{2}{c|}{\textbf{IQ}} & \multicolumn{2}{c|}{\textbf{Medication}} & \multicolumn{2}{c|}{\textbf{ADHD $\times$ Sex}} & \multicolumn{2}{c|}{\textbf{ADHD $\times$ IQ}} & \multicolumn{2}{c}{\textbf{ADHD $\times$ Med}} \\",
        r"\textbf{Param} & $\beta$ (SE) & $f^2$ & $\beta$ (SE) & $f^2$ & $\beta$ (SE) & $f^2$ & $\beta$ (SE) & $f^2$ & $\beta$ (SE) & $f^2$ & $\beta$ (SE) & $f^2$ & $\beta$ (SE) & $f^2$ \\",
        r"\midrule"
    ]

    for _, r in df_res.iterrows():
        row_str = f"{r['param_label']} & "
        row_str += f"{fmt_cell(r['ADHD_beta'], r['ADHD_se'], r['ADHD_p'], r['ADHD_f2'])} & "
        row_str += f"{fmt_cell(r['Sex_beta'], r['Sex_se'], r['Sex_p'], r['Sex_f2'])} & "
        row_str += f"{fmt_cell(r['IQ_beta'], r['IQ_se'], r['IQ_p'], r['IQ_f2'])} & "
        row_str += f"{fmt_cell(r['Medication_beta'], r['Medication_se'], r['Medication_p'], r['Medication_f2'])} & "
        row_str += f"{fmt_cell(r['ADHD_x_Sex_beta'], r['ADHD_x_Sex_se'], r['ADHD_x_Sex_p'], r['ADHD_x_Sex_f2'])} & "
        row_str += f"{fmt_cell(r['ADHD_x_IQ_beta'], r['ADHD_x_IQ_se'], r['ADHD_x_IQ_p'], r['ADHD_x_IQ_f2'])} & "
        row_str += f"{fmt_cell(r['ADHD_x_Med_beta'], r['ADHD_x_Med_se'], r['ADHD_x_Med_p'], r['ADHD_x_Med_f2'])} \\\\"
        lines.append(row_str)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        rf"\begin{{flushleft}} \small Note: $N={n_subjects:,}$. Continuous main effects were Z-score standardized. "
        r"Coefficients ($\beta$) with Wald SEs. Stars: \textbf{*} $p<.05$, \textbf{**} $p<.01$, \textbf{***} $p<.001$. "
        r"$f^2$ denotes Cohen's $f^2$. \end{flushleft}",
        r"\label{tab:all_effects_interactions}",
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
        print(f" ALL PREDICTORS & INTERACTIONS EFFECT SIZES (N = {n_sub})")
        print("="*110)
        
        # Print block by block for readability
        for pred in ['ADHD', 'Sex', 'IQ', 'Medication', 'ADHD_x_Sex', 'ADHD_x_IQ', 'ADHD_x_Med']:
            print(f"\n--- Predictor: {pred.replace('_', ' ').upper()} ---")
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
#  ALL PREDICTORS & INTERACTIONS EFFECT SIZES (N = 3567)
# ==============================================================================================================

# --- Predictor: ADHD ---
# Param                 Beta         SE    p-value    Part_R2         f2
# q_d                -0.0290     0.0272     0.2872     0.0000     0.0000
# q_s                 0.0269     0.0275     0.3284     0.0000     0.0000
# tau                 0.0307     0.0273     0.2608     0.0000     0.0000
# cost_stop_error    -0.0518     0.0272     0.0564     0.0000     0.0000
# cost_time          -0.0287     0.0273     0.2928     0.0000     0.0000

# --- Predictor: SEX ---
# Param                 Beta         SE    p-value    Part_R2         f2
# q_d                 0.0041     0.0336     0.9019     0.0000     0.0000
# q_s                -0.2096     0.0339     0.0000     0.0106     0.0107
# tau                -0.2459     0.0337     0.0000     0.0147     0.0150
# cost_stop_error     0.3238     0.0335     0.0000     0.0255     0.0262
# cost_time           0.2895     0.0337     0.0000     0.0203     0.0208

# --- Predictor: IQ ---
# Param                 Beta         SE    p-value    Part_R2         f2
# q_d                 0.1474     0.0167     0.0000     0.0215     0.0220
# q_s                -0.0163     0.0168     0.3330     0.0003     0.0003
# tau                -0.0923     0.0167     0.0000     0.0085     0.0086
# cost_stop_error     0.0828     0.0166     0.0000     0.0069     0.0070
# cost_time           0.0653     0.0167     0.0001     0.0043     0.0043

# --- Predictor: MEDICATION ---
# Param                 Beta         SE    p-value    Part_R2         f2
# q_d                 0.0974     0.1162     0.4020     0.0002     0.0002
# q_s                 0.1310     0.1172     0.2638     0.0004     0.0004
# tau                 0.1005     0.1164     0.3878     0.0002     0.0002
# cost_stop_error    -0.0868     0.1159     0.4540     0.0002     0.0002
# cost_time          -0.1448     0.1163     0.2133     0.0004     0.0004

# --- Predictor: ADHD X SEX ---
# Param                 Beta         SE    p-value    Part_R2         f2
# q_d                -0.0814     0.0359     0.0235     0.0014     0.0014
# q_s                 0.0028     0.0362     0.9389     0.0000     0.0000
# tau                 0.0219     0.0360     0.5437     0.0001     0.0001
# cost_stop_error    -0.0180     0.0358     0.6161     0.0001     0.0001
# cost_time          -0.0329     0.0360     0.3601     0.0002     0.0002

# --- Predictor: ADHD X IQ ---
# Param                 Beta         SE    p-value    Part_R2         f2
# q_d                -0.0246     0.0155     0.1110     0.0007     0.0007
# q_s                 0.0185     0.0156     0.2341     0.0004     0.0004
# tau                 0.0197     0.0155     0.2044     0.0005     0.0005
# cost_stop_error    -0.0346     0.0154     0.0247     0.0014     0.0014
# cost_time          -0.0351     0.0155     0.0234     0.0014     0.0014

# --- Predictor: ADHD X MED ---
# Param                 Beta         SE    p-value    Part_R2         f2
# q_d                 0.0892     0.0606     0.1409     0.0006     0.0006
# q_s                -0.0422     0.0611     0.4900     0.0001     0.0001
# tau                -0.1245     0.0607     0.0403     0.0012     0.0012
# cost_stop_error     0.0588     0.0604     0.3305     0.0003     0.0003
# cost_time           0.0777     0.0606     0.2000     0.0005     0.0005

# ==============================================================================================================

