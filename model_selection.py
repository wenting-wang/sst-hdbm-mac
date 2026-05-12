import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ==========================================
# Global Configuration
# ==========================================
CONFIG = {
    # 1. Extraction Paths
    'input_spec_folder': '/Users/w/sst-hdbm-mac/pomdp_v2/model_spec',
    'input_ppc_folder': '/Users/w/sst-hdbm-mac/pomdp_v2/model_ppc',
    'input_rec_folder': '/Users/w/sst-hdbm-mac/pomdp_v2/model_recovery',
    'input_cv_folder': '/Users/w/sst-hdbm-mac/pomdp_v2/model_cv',
    
    # 2. Intermediate CSV outputs
    'specs_csv': 'model_specs.csv',
    'ppc_csv': 'models_ppc.csv',
    'rec_csv': 'model_recovery.csv',
    'cv_csv': 'models_cv.csv',
    
    # 3. Final LaTeX output files
    'out_tex_specs': 'table1_specs.tex',
    'out_tex_ppc': 'table2_ppc.tex',
    'out_tex_rec': 'table3_recovery.tex',
    'out_tex_cv_ppc': 'table4_cv.tex'
}

# ==========================================
# PART 1: Data Extraction & Summarization
# ==========================================
def remove_outliers(x, y, lower_pct=1.0, upper_pct=99.0):
    mask_valid = np.isfinite(x) & np.isfinite(y)
    x = x[mask_valid]
    y = y[mask_valid]
    if len(y) < 2: return x, y
    y_low = np.percentile(y, lower_pct)
    y_high = np.percentile(y, upper_pct)
    mask_outliers = (y >= y_low) & (y <= y_high)
    return x[mask_outliers], y[mask_outliers]

def get_stats(x, y, is_log=False):
    if len(x) < 2: return np.nan, np.nan
    if is_log:
        mask = (x > 0) & (y > 0)
        x_log = np.log10(x[mask])
        y_log = np.log10(y[mask])
        if len(x_log) < 2: return np.nan, np.nan
        r, _ = pearsonr(x_log, y_log)
        rmse = np.sqrt(np.mean((x_log - y_log)**2))
    else:
        r, _ = pearsonr(x, y)
        rmse = np.sqrt(np.mean((x - y)**2))
    return r, rmse

def extract_recoveries():
    if not os.path.exists(CONFIG['input_rec_folder']): return
    known_log_params = ['cost_stop_error', 'cost_time', 'cost_go_error', 'cost_go_missing']
    csv_files = glob.glob(os.path.join(CONFIG['input_rec_folder'], '*.csv'))
    if not csv_files: return
        
    print(f"Extracting Recovery Stats from {len(csv_files)} files...")
    summary_data = []
    for file_path in csv_files:
        model_name = os.path.basename(file_path).replace('params_recovery_', '').replace('.csv', '')
        try:
            df = pd.read_csv(file_path)
            params = [col[3:] for col in df.columns if col.startswith('gt_')]
            model_summary = {'model_name': model_name}
            for param in params:
                if f"mu_{param}" in df.columns:
                    x, y = remove_outliers(df[f"gt_{param}"].to_numpy(float), df[f"mu_{param}"].to_numpy(float), 5.0, 95.0)
                    r_val, rmse_val = get_stats(x, y, is_log=(param in known_log_params))
                    model_summary[f"{param}_r"] = r_val
                    model_summary[f"{param}_rmse"] = rmse_val
            summary_data.append(model_summary)
        except Exception as e:
            print(f"  -> Error processing {file_path}: {e}")

    if summary_data:
        pd.DataFrame(summary_data).to_csv(CONFIG['rec_csv'], index=False)

def extract_model_specs():
    if not os.path.exists(CONFIG['input_spec_folder']): return
    py_files = glob.glob(os.path.join(CONFIG['input_spec_folder'], '*.py'))
    if not py_files: return

    tag_pattern = re.compile(r'MODEL_TAG\s*=\s*["\']([^"\']+)["\']')
    ranges_pattern = re.compile(r'PARAM_RANGES\s*=\s*(\{.*?\})', re.DOTALL)
    fixed_pattern = re.compile(r'FIXED_PARAMS\s*=\s*(\{.*?\})', re.DOTALL)

    extracted_data = []
    for file_path in py_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tag_match = tag_pattern.search(content)
            ranges_match = ranges_pattern.search(content)
            fixed_match = fixed_pattern.search(content)
            
            extracted_data.append({
                'file_name': os.path.basename(file_path),
                'MODEL_TAG': tag_match.group(1) if tag_match else None,
                'PARAM_RANGES': re.sub(r'\s+', ' ', ranges_match.group(1).strip()) if ranges_match else None,
                'FIXED_PARAMS': re.sub(r'\s+', ' ', fixed_match.group(1).strip()) if fixed_match else None
            })
        except Exception as e:
            print(f"  -> Error parsing {file_path}: {e}")

    if extracted_data:
        pd.DataFrame(extracted_data).to_csv(CONFIG['specs_csv'], index=False)

def summarize_ppc_metrics():
    if not os.path.exists(CONFIG['input_ppc_folder']): return
    target_columns = ['dis_perc_gs', 'dis_perc_ge', 'dis_perc_gm', 'dis_perc_ss', 'dis_ws_rt_gs', 'dis_ws_rt_ge', 'dis_ws_rt_se', 'dis_ks_rt_gs', 'dis_ks_rt_se', 'dis_ssd_mean', 'total_distance']
    csv_files = glob.glob(os.path.join(CONFIG['input_ppc_folder'], '*.csv'))
    if not csv_files: return

    summary_data = []
    for file_path in csv_files:
        model_name = os.path.basename(file_path).replace('ppc_model_summary_', '').replace('ppc_metrics_', '').replace('.csv', '')
        try:
            df = pd.read_csv(file_path)
            model_summary = {'model_name': model_name}
            for col in target_columns:
                model_summary[col] = df[col].mean() if col in df.columns else pd.NA
            summary_data.append(model_summary)
        except Exception as e:
            pass

    if summary_data:
        pd.DataFrame(summary_data).to_csv(CONFIG['ppc_csv'], index=False)

def summarize_cv_ppc_metrics():
    if not os.path.exists(CONFIG['input_cv_folder']): return
    target_columns = ['dis_perc_gs', 'dis_perc_ge', 'dis_perc_gm', 'dis_perc_ss', 'dis_ws_rt_gs', 'dis_ws_rt_ge', 'dis_ws_rt_se', 'dis_ks_rt_gs', 'dis_ks_rt_se', 'dis_ssd_mean', 'total_distance']
    
    search_pattern = os.path.join(CONFIG['input_cv_folder'], 'cv_summary_*.csv')
    csv_files = glob.glob(search_pattern)
    if not csv_files: return

    summary_data = []
    for file_path in csv_files:
        model_name = os.path.basename(file_path).replace('cv_summary_', '').replace('.csv', '')
        try:
            df = pd.read_csv(file_path)
            model_summary = {'model_name': model_name}
            for col in target_columns:
                model_summary[col] = df[col].mean() if col in df.columns else pd.NA
            summary_data.append(model_summary)
        except Exception as e:
            pass

    if summary_data:
        pd.DataFrame(summary_data).to_csv(CONFIG['cv_csv'], index=False)

# ==========================================
# PART 2: Formatting & LaTeX Generation
# ==========================================
def parse_dict_string(s):
    if pd.isna(s): return {}
    try: return eval(re.sub(r'#.*', '', str(s)).strip())
    except Exception: return {}

def escape_latex(text):
    return str(text).replace('_', r'\_')

def format_num(val):
    if pd.isna(val): return "---"
    try:
        f = float(val)
        if f == 0: return "0"
        if f.is_integer(): return str(int(f))
        return f"{f:.5f}".rstrip('0').rstrip('.')
    except Exception:
        return str(val)

def generate_latex_tables():
    print("\nStarting LaTeX formatting and file generation...")
    
    try:
        df_specs = pd.read_csv(CONFIG['specs_csv'])
        df_ppc = pd.read_csv(CONFIG['ppc_csv'])
        df_rec = pd.read_csv(CONFIG['rec_csv'])
    except Exception as e:
        print(f"Error loading intermediate CSV files. Details: {e}")
        return

    try:
        df_cv = pd.read_csv(CONFIG['cv_csv'])
    except Exception:
        df_cv = pd.DataFrame()

    if 'MODEL_TAG' in df_specs.columns: 
        df_specs.rename(columns={'MODEL_TAG': 'model_name'}, inplace=True)
    
    df = df_specs.merge(df_ppc, on='model_name', how='inner').merge(df_rec, on='model_name', how='inner')
    
    if df.empty:
        print("Warning: The merged dataframe is empty! Please check that 'MODEL_TAG' in your spec files exactly matches the suffixes of your CSV files.")
        return

    df['ranges_dict'] = df['PARAM_RANGES'].apply(parse_dict_string)
    df['fixed_dict'] = df['FIXED_PARAMS'].apply(parse_dict_string)
    
    df['p_count'] = df['model_name'].apply(lambda x: int(m.group(1)) if (m := re.search(r'(\d+)p', str(x))) else 99)
    df['v_num'] = df['model_name'].apply(lambda x: int(m.group(1)) if (m := re.search(r'v(\d+)', str(x))) else 99)
    df = df.sort_values(by=['p_count', 'v_num']).reset_index(drop=True)
    
    df['display_name'] = df.apply(lambda r: f"M{r['p_count']}.{r['v_num']}" if r['p_count']!=99 else r['model_name'], axis=1)
    
    # ---------------------------------------------------------
    # TABLE 1: Model Specs (Single Block)
    # ---------------------------------------------------------
    param_info = {
        'q_d_n': {'label': r'Go null ($\gonull$)', 'main_tuple': (0.0, 1.0)},
        'q_d': {'label': r'Go precision ($\godirr$)', 'main_tuple': (0.5, 1.0)},
        'q_s_n': {'label': r'Stop null ($\stnull$)', 'main_tuple': (0.0, 1.0)},
        'q_s': {'label': r'Stop precision ($\stdirr$)', 'main_tuple': (0.5, 1.0)},
        'cost_go_error': {'label': r'Go error cost ($c_{\text{ge}}$)', 'main_tuple': (1.0, 50.0)},
        'cost_go_missing': {'label': r'Go missing cost ($c_{\text{gm}}$)', 'main_tuple': (1.0, 50.0)},
        'cost_stop_error': {'label': r'Stop error cost ($c_{\text{se}}$)', 'main_tuple': (1.0, 50.0)},
        'cost_time': {'label': r'Time cost ($c_t$)', 'main_tuple': (0.01, 0.5)},
        'inv_temp': {'label': r'Inverse temperature ($\varphi$)', 'main_tuple': (10.0, 100.0)},
        'tau': {'label': r'Non-decision time ($\tau$)', 'main_tuple': (4, 16)},
        'rate_stop_trial': {'label': r'Stop prior ($\tilde{r}^{\nu}$)', 'main_tuple': None}
    }
    
    t1 = []
    t1.append(r"\begin{table}[H]")
    t1.append(r"    \centering")
    t1.append(r"    \scriptsize") 
    t1.append(r"    \setlength{\tabcolsep}{3pt}")
    t1.append(r"    \captionsetup{labelfont=bf}")
    t1.append(r"    \caption{\textbf{POMDP model specification and parameter priors}}")
    t1.append(r"    \label{tab:model_variants}")
    
    # Generate columns based on the total number of models in df
    col_format = "l l " + "c " * len(df)
    t1.append(r"    \makebox[\textwidth][c]{%") 
    t1.append(f"    \\begin{{tabular}}{{{col_format}}}")
    t1.append(r"        \toprule")
    
    model_headers = [f"\\textbf{{{escape_latex(row['display_name'])}}}" for _, row in df.iterrows()]
    t1.append(r"        \textbf{Parameter} & \textbf{Prior Range} & " + " & ".join(model_headers) + r" \\")
    t1.append(r"        \midrule")
    
    for p_key, info in param_info.items():
        main_tup = info['main_tuple']
        range_str = f"$[{format_num(main_tup[0])}, {format_num(main_tup[1])}]$" if main_tup else "Fixed"
        row_str = f"        {info['label']} & {range_str}"
        
        for _, row in df.iterrows():
            ranges = row['ranges_dict']
            fixed = row['fixed_dict']
            
            if p_key in ranges:
                val = ranges[p_key]
                if isinstance(val, tuple) and len(val) == 2 and main_tup is not None:
                    if abs(float(val[0]) - main_tup[0]) > 1e-5 or abs(float(val[1]) - main_tup[1]) > 1e-5:
                        row_str += f" & $[{format_num(val[0])}, {format_num(val[1])}]$"
                    else:
                        row_str += r" & \checkmark"
                else:
                    row_str += r" & \checkmark"
            elif p_key in fixed:
                val = fixed[p_key]
                if isinstance(val, float) and abs(val - 1.0/6.0) < 1e-4:
                    row_str += r" & $1/6$"
                else:
                    row_str += f" & {format_num(val)}"
            else:
                row_str += r" & ---"
        row_str += r" \\"
        t1.append(row_str)
        
    t1.append(r"        \bottomrule")
    t1.append(r"    \end{tabular}%")
    t1.append(r"    }")
    t1.append(r"\end{table}")
    
    with open(CONFIG['out_tex_specs'], 'w', encoding='utf-8') as f:
        f.write("\n".join(t1))

    # ---------------------------------------------------------
    # TABLE 2: PPC
    # ---------------------------------------------------------
    df_sorted_ppc = df.sort_values(by='total_distance', ascending=True).reset_index(drop=True)
    
    t2 = []
    t2.append(r"\begin{table}[H]")
    t2.append(r"    \centering")
    t2.append(r"    \scriptsize")
    t2.append(r"    \setlength{\tabcolsep}{3pt}")
    t2.append(r"    \captionsetup{labelfont=bf}")
    t2.append(r"    \caption{\textbf{In-sample empirical distances from posterior predictive checks}}")
    t2.append(r"    \label{tab:ppc_metrics}")
    t2.append(r"    \makebox[\textwidth][c]{%")
    t2.append(r"    \begin{tabular}{l cccc ccc cc c}")
    t2.append(r"        \toprule")
    t2.append(r"        & \multicolumn{4}{c}{\textbf{Choice Proportions}} & \multicolumn{3}{c}{\textbf{WS Dist. (RT)}} & \multicolumn{2}{c}{\textbf{KS Dist. (RT)}} & \\")
    t2.append(r"        \cmidrule(lr){2-5} \cmidrule(lr){6-8} \cmidrule(lr){9-10}")
    t2.append(r"        \textbf{Model} & \textbf{GS} & \textbf{GE} & \textbf{GM} & \textbf{SS} & \textbf{GS} & \textbf{GE} & \textbf{SE} & \textbf{GS} & \textbf{SE} & \textbf{Total Dist.} \\")
    t2.append(r"        \midrule")
    
    ppc_cols = ['dis_perc_gs', 'dis_perc_ge', 'dis_perc_gm', 'dis_perc_ss', 
                'dis_ws_rt_gs', 'dis_ws_rt_ge', 'dis_ws_rt_se', 
                'dis_ks_rt_gs', 'dis_ks_rt_se', 'total_distance']
                
    for _, row in df_sorted_ppc.iterrows():
        safe_name = escape_latex(row['display_name'])
        row_str = f"        {safe_name}"
        for col in ppc_cols:
            row_str += f" & {format_num(row[col])}"
        row_str += r" \\"
        t2.append(row_str)
        
    t2.append(r"        \bottomrule")
    t2.append(r"    \end{tabular}%")
    t2.append(r"    }") 
    t2.append(r"\end{table}")
    
    with open(CONFIG['out_tex_ppc'], 'w', encoding='utf-8') as f:
        f.write("\n".join(t2))

    # ---------------------------------------------------------
    # TABLE 3: Parameter Recovery
    # ---------------------------------------------------------
    t3 = []
    t3.append(r"\begin{table}[H]")
    t3.append(r"    \centering")
    t3.append(r"    \scriptsize")
    t3.append(r"    \setlength{\tabcolsep}{3pt}")
    t3.append(r"    \captionsetup{labelfont=bf}")
    t3.append(r"    \caption{\textbf{Parameter recovery correlations ($r$) across model variants}}")
    t3.append(r"    \label{tab:recovery_corr}")
    t3.append(r"    \makebox[\textwidth][c]{%")
    t3.append(r"    \begin{tabular}{l cccccccccc c}")
    t3.append(r"        \toprule")
    t3.append(r"        \textbf{Model} & $\godirr$ & $\stdirr$ & $\tau$ & $c_{\text{se}}$ & $c_t$ & $c_{\text{ge}}$ & $c_{\text{gm}}$ & $\gonull$ & $\stnull$ & $\varphi$ & \textbf{Valid} \\")
    t3.append(r"        \midrule")
    
    rec_cols = ['q_d_r', 'q_s_r', 'tau_r', 'cost_stop_error_r', 'cost_time_r', 
                'cost_go_error_r', 'cost_go_missing_r', 'q_d_n_r', 'q_s_n_r', 'inv_temp_r']
                
    for _, row in df.iterrows():
        safe_name = escape_latex(row['display_name'])
        row_str = f"        {safe_name}"
        
        is_valid = True
        has_params = False
        
        for col in rec_cols:
            val = row[col]
            if pd.isna(val):
                row_str += " & ---"
            else:
                row_str += f" & {format_num(val)}"
                has_params = True
                if float(val) <= 0.65:
                    is_valid = False
                    
        if is_valid and has_params:
            row_str += r" & \checkmark \\ "
        else:
            row_str += r" & \\"
            
        t3.append(row_str)
        
    t3.append(r"        \bottomrule")
    t3.append(r"    \end{tabular}%")
    t3.append(r"    }")
    t3.append(r"\end{table}")
    
    with open(CONFIG['out_tex_rec'], 'w', encoding='utf-8') as f:
        f.write("\n".join(t3))

    # ---------------------------------------------------------
    # TABLE 4: Out-of-Sample PPC
    # ---------------------------------------------------------
    if not df_cv.empty:
        df_cv['p_count'] = df_cv['model_name'].apply(lambda x: int(m.group(1)) if (m := re.search(r'(\d+)p', str(x))) else 99)
        df_cv['v_num'] = df_cv['model_name'].apply(lambda x: int(m.group(1)) if (m := re.search(r'v(\d+)', str(x))) else 99)
        df_cv['display_name'] = df_cv.apply(lambda r: f"M{r['p_count']}.{r['v_num']}" if r['p_count']!=99 else r['model_name'], axis=1)
        
        df_sorted_cv = df_cv.sort_values(by='total_distance', ascending=True).reset_index(drop=True)
        
        t4 = []
        t4.append(r"\begin{table}[H]")
        t4.append(r"    \centering")
        t4.append(r"    \scriptsize")
        t4.append(r"    \setlength{\tabcolsep}{3pt}")
        t4.append(r"    \captionsetup{labelfont=bf}")
        t4.append(r"    \caption{\textbf{Out-of-sample empirical distances from posterior predictive checks}}")
        t4.append(r"    \label{tab:cv_ppc_metrics}")
        t4.append(r"    \makebox[\textwidth][c]{%")
        t4.append(r"    \begin{tabular}{l cccc ccc cc c}")
        t4.append(r"        \toprule")
        t4.append(r"        & \multicolumn{4}{c}{\textbf{Choice Proportions}} & \multicolumn{3}{c}{\textbf{WS Dist. (RT)}} & \multicolumn{2}{c}{\textbf{KS Dist. (RT)}} & \\")
        t4.append(r"        \cmidrule(lr){2-5} \cmidrule(lr){6-8} \cmidrule(lr){9-10}")
        t4.append(r"        \textbf{Model} & \textbf{GS} & \textbf{GE} & \textbf{GM} & \textbf{SS} & \textbf{GS} & \textbf{GE} & \textbf{SE} & \textbf{GS} & \textbf{SE} & \textbf{Total Dist.} \\")
        t4.append(r"        \midrule")
        
        for _, row in df_sorted_cv.iterrows():
            safe_name = escape_latex(row['display_name'])
            row_str = f"        {safe_name}"
            for col in ppc_cols:
                row_str += f" & {format_num(row.get(col, pd.NA))}"
            row_str += r" \\"
            t4.append(row_str)
            
        t4.append(r"        \bottomrule")
        t4.append(r"    \end{tabular}%")
        t4.append(r"    }")
        t4.append(r"\end{table}")
        
        with open(CONFIG['out_tex_cv_ppc'], 'w', encoding='utf-8') as f:
            f.write("\n".join(t4))

    print("LaTeX files generated successfully:")
    print(f"  -> {CONFIG['out_tex_specs']}")
    print(f"  -> {CONFIG['out_tex_ppc']}")
    print(f"  -> {CONFIG['out_tex_rec']}")
    if not df_cv.empty:
        print(f"  -> {CONFIG['out_tex_cv_ppc']}")

# ==========================================
# Script Execution
# ==========================================
if __name__ == "__main__":
    extract_model_specs()
    summarize_ppc_metrics()
    summarize_cv_ppc_metrics()
    extract_recoveries()
    generate_latex_tables()