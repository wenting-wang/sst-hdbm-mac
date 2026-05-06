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
    # Replace these paths with your actual source directories
    'input_spec_folder': '/Users/w/sst-hdbm-mac/pomdp_v2/model_spec',
    'input_ppc_folder': '/Users/w/sst-hdbm-mac/pomdp_v2/model_outputs_filtered',
    'input_rec_folder': '/Users/w/sst-hdbm-mac/pomdp_v2/model_recovery',
    
    # Intermediate CSV outputs
    'specs_csv': 'model_specs.csv',
    'ppc_csv': 'models_ppc.csv',
    'rec_csv': 'model_recovery.csv',
    
    # Final LaTeX output files
    'out_tex_specs': 'table1_specs.tex',
    'out_tex_ppc': 'table2_ppc.tex',
    'out_tex_rec': 'table3_recovery.tex'
}

# ==========================================
# PART 1: Data Extraction & Filtering
# ==========================================
def remove_outliers(x, y, lower_pct=1.0, upper_pct=99.0):
    """Filters valid data and removes outliers based on y percentiles."""
    mask_valid = np.isfinite(x) & np.isfinite(y)
    x = x[mask_valid]
    y = y[mask_valid]
    if len(y) < 2:
        return x, y
    y_low = np.percentile(y, lower_pct)
    y_high = np.percentile(y, upper_pct)
    mask_outliers = (y >= y_low) & (y <= y_high)
    return x[mask_outliers], y[mask_outliers]

def get_stats(x, y, is_log=False):
    """Calculates Pearson r and RMSE, optionally in log scale."""
    if len(x) < 2:
        return np.nan, np.nan
    if is_log:
        mask = (x > 0) & (y > 0)
        x_log = np.log10(x[mask])
        y_log = np.log10(y[mask])
        if len(x_log) < 2:
            return np.nan, np.nan
        r, _ = pearsonr(x_log, y_log)
        rmse = np.sqrt(np.mean((x_log - y_log)**2))
    else:
        r, _ = pearsonr(x, y)
        rmse = np.sqrt(np.mean((x - y)**2))
    return r, rmse

def extract_recoveries():
    """Reads raw recovery CSVs, filters outliers, calculates r/rmse, saves to intermediate CSV."""
    if not os.path.exists(CONFIG['input_rec_folder']):
        print(f"Skipping recovery extraction: Folder {CONFIG['input_rec_folder']} not found.")
        return

    known_log_params = ['cost_stop_error', 'cost_time', 'cost_go_error', 'cost_go_missing']
    csv_files = glob.glob(os.path.join(CONFIG['input_rec_folder'], '*.csv'))
    
    if not csv_files:
        return
        
    print(f"Extracting Recovery Stats from {len(csv_files)} files...")
    summary_data = []
    
    for file_path in csv_files:
        model_name = os.path.basename(file_path).replace('params_recovery_', '').replace('.csv', '')
        try:
            df = pd.read_csv(file_path)
            gt_cols = [col for col in df.columns if col.startswith('gt_')]
            params = [col[3:] for col in gt_cols]
            model_summary = {'model_name': model_name}
            
            for param in params:
                col_true, col_rec = f"gt_{param}", f"mu_{param}"
                if col_rec in df.columns:
                    x_raw = df[col_true].to_numpy(float)
                    y_raw = df[col_rec].to_numpy(float)
                    x, y = remove_outliers(x_raw, y_raw, lower_pct=5.0, upper_pct=95.0)
                    is_log_param = param in known_log_params
                    r_val, rmse_val = get_stats(x, y, is_log=is_log_param)
                    model_summary[f"{param}_r"] = r_val
                    model_summary[f"{param}_rmse"] = rmse_val
                    
            summary_data.append(model_summary)
        except Exception as e:
            print(f"  -> Error processing {file_path}: {e}")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        cols = ['model_name'] + [c for c in summary_df.columns if c != 'model_name']
        summary_df[cols].to_csv(CONFIG['rec_csv'], index=False)
        print(f"Saved {CONFIG['rec_csv']}")

def extract_model_specs():
    """Extracts PARAM_RANGES and FIXED_PARAMS from raw python scripts."""
    if not os.path.exists(CONFIG['input_spec_folder']):
        print(f"Skipping spec extraction: Folder {CONFIG['input_spec_folder']} not found.")
        return

    tag_pattern = re.compile(r'MODEL_TAG\s*=\s*["\']([^"\']+)["\']')
    ranges_pattern = re.compile(r'PARAM_RANGES\s*=\s*(\{.*?\})', re.DOTALL)
    fixed_pattern = re.compile(r'FIXED_PARAMS\s*=\s*(\{.*?\})', re.DOTALL)

    py_files = glob.glob(os.path.join(CONFIG['input_spec_folder'], '*.py'))
    
    if not py_files:
        return

    print(f"Extracting Model Specs from {len(py_files)} files...")
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
        print(f"Saved {CONFIG['specs_csv']}")

def summarize_ppc_metrics():
    """Calculates mean values for specific PPC metrics across filtered outputs."""
    if not os.path.exists(CONFIG['input_ppc_folder']):
        print(f"Skipping PPC extraction: Folder {CONFIG['input_ppc_folder']} not found.")
        return

    target_columns = [
        'dis_perc_gs', 'dis_perc_ge', 'dis_perc_gm', 'dis_perc_ss',
        'dis_ws_rt_gs', 'dis_ws_rt_ge', 'dis_ws_rt_se', 
        'dis_ks_rt_gs', 'dis_ks_rt_se', 'dis_ssd_mean', 'total_distance'
    ]
    csv_files = glob.glob(os.path.join(CONFIG['input_ppc_folder'], '*.csv'))
    
    if not csv_files:
        return

    print(f"Extracting PPC Summary from {len(csv_files)} files...")
    summary_data = []

    for file_path in csv_files:
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            df = pd.read_csv(file_path)
            model_summary = {'model_name': model_name}
            for col in target_columns:
                model_summary[col] = df[col].mean() if col in df.columns else pd.NA
            summary_data.append(model_summary)
        except Exception as e:
            print(f"  -> Error processing {file_path}: {e}")

    if summary_data:
        pd.DataFrame(summary_data).to_csv(CONFIG['ppc_csv'], index=False)
        print(f"Saved {CONFIG['ppc_csv']}")

# ==========================================
# PART 2: Formatting & LaTeX Generation
# ==========================================
def parse_dict_string(s):
    if pd.isna(s): 
        return {}
    try:
        s = re.sub(r'#.*', '', str(s))
        return eval(s.strip())
    except Exception:
        return {}

def escape_latex(text):
    """Escapes LaTeX special characters like underscores."""
    return str(text).replace('_', r'\_')

def generate_latex_tables():
    print("\nStarting LaTeX formatting and file generation...")
    
    try:
        df_specs = pd.read_csv(CONFIG['specs_csv'])
        df_ppc = pd.read_csv(CONFIG['ppc_csv'])
        df_rec = pd.read_csv(CONFIG['rec_csv'])
    except Exception as e:
        print(f"Error loading intermediate CSV files. Please ensure extraction succeeded. Details: {e}")
        return

    # Normalize model names across tables for a clean join
    if 'MODEL_TAG' in df_specs.columns:
        df_specs.rename(columns={'MODEL_TAG': 'model_name'}, inplace=True)
    df_ppc['model_name'] = df_ppc['model_name'].str.replace('ppc_metrics_', '')
    
    # Inner Join to keep only fully matched models
    df = df_specs.merge(df_ppc, on='model_name', how='inner')
    df = df.merge(df_rec, on='model_name', how='inner')
    
    # Parse parameter structures
    df['ranges_dict'] = df['PARAM_RANGES'].apply(parse_dict_string)
    df['fixed_dict'] = df['FIXED_PARAMS'].apply(parse_dict_string)
    
    # Sorting Logic based on parameter count and version
    def get_p_count(name):
        match = re.search(r'(\d+)p', str(name))
        return int(match.group(1)) if match else 99

    def get_v_num(name):
        match = re.search(r'v(\d+)', str(name))
        return int(match.group(1)) if match else 99

    df['p_count'] = df['model_name'].apply(get_p_count)
    df['v_num'] = df['model_name'].apply(get_v_num)
    df = df.sort_values(by=['p_count', 'v_num']).reset_index(drop=True)
    
    print(f"Successfully aligned and sorted {len(df)} models.")

    # ---------------------------------------------------------
    # TABLE 1: Model Specs (Split by Parameter Count)
    # ---------------------------------------------------------
    param_info = {
        'q_d_n': {'label': r'Go null ($\gonull$)', 'range_str': r'$[0.0, 1.0]$', 'main_tuple': (0.0, 1.0)},
        'q_d': {'label': r'Go precision ($\godirr$)', 'range_str': r'$[0.5, 1.0]$', 'main_tuple': (0.5, 1.0)},
        'q_s_n': {'label': r'Stop null ($\stnull$)', 'range_str': r'$[0.0, 1.0]$', 'main_tuple': (0.0, 1.0)},
        'q_s': {'label': r'Stop precision ($\stdirr$)', 'range_str': r'$[0.5, 1.0]$', 'main_tuple': (0.5, 1.0)},
        'cost_go_error': {'label': r'Go error cost ($c_{\text{ge}}$)', 'range_str': r'$[1.0, 50.0]$', 'main_tuple': (1.0, 50.0)},
        'cost_go_missing': {'label': r'Go missing cost ($c_{\text{gm}}$)', 'range_str': r'$[1.0, 50.0]$', 'main_tuple': (1.0, 50.0)},
        'cost_stop_error': {'label': r'Stop error cost ($c_{\text{se}}$)', 'range_str': r'$[1.0, 50.0]$', 'main_tuple': (1.0, 50.0)},
        'cost_time': {'label': r'Time cost ($c_t$)', 'range_str': r'$[0.01, 0.5]$', 'main_tuple': (0.01, 0.5)},
        'inv_temp': {'label': r'Inverse temperature ($\varphi$)', 'range_str': r'$[10.0, 100.0]$', 'main_tuple': (10.0, 100.0)},
        'tau': {'label': r'Non-decision time ($\tau$)', 'range_str': r'$[4, 16]$ time steps', 'main_tuple': (4, 16)},
        'rate_stop_trial': {'label': r'Stop prior ($\tilde{r}^{\nu}$)', 'range_str': r'Fixed', 'main_tuple': None}
    }
    
    groups = [
        ("5-Parameter Models", df[df['p_count'] == 5]),
        ("6-Parameter Models", df[df['p_count'] == 6]),
        ("7- and 10-Parameter Models", df[df['p_count'] >= 7])
    ]
    
    t1 = []
    for title, sub_df in groups:
        if sub_df.empty:
            continue
            
        col_format = "l l " + "c " * len(sub_df)
        t1.append(r"\begin{table}[H]")
        t1.append(r"    \centering")
        t1.append(r"    \captionsetup{labelfont=bf}")
        t1.append(f"    \\caption{{\\textbf{{POMDP model parameters and descriptions ({title})}}}}")
        t1.append(r"    \resizebox{\textwidth}{!}{%")
        t1.append(f"    \\begin{{tabular}}{{{col_format}}}")
        t1.append(r"        \toprule")
        
        model_headers = [f"\\textbf{{\\texttt{{{escape_latex(row['model_name'])}}}}}" for _, row in sub_df.iterrows()]
        t1.append(r"        \textbf{Parameter} & \textbf{Prior Range} & " + " & ".join(model_headers) + r" \\")
        t1.append(r"        \midrule")
        
        for p_key, info in param_info.items():
            row_str = f"        {info['label']} & {info['range_str']}"
            main_tup = info['main_tuple']
            
            for _, row in sub_df.iterrows():
                ranges = row['ranges_dict']
                fixed = row['fixed_dict']
                
                if p_key in ranges:
                    val = ranges[p_key]
                    # Check if model specific range differs from the main prior range
                    if isinstance(val, tuple) and len(val) == 2 and main_tup is not None:
                        if abs(float(val[0]) - main_tup[0]) > 1e-5 or abs(float(val[1]) - main_tup[1]) > 1e-5:
                            row_str += f" & $[{val[0]}, {val[1]}]$"
                        else:
                            row_str += r" & \checkmark"
                    else:
                        row_str += r" & \checkmark"
                elif p_key in fixed:
                    val = fixed[p_key]
                    if isinstance(val, float) and abs(val - 1.0/6.0) < 1e-4:
                        row_str += r" & 1/6"
                    else:
                        row_str += f" & {val}"
                else:
                    row_str += r" & ---"
            row_str += r" \\"
            t1.append(row_str)
            
        t1.append(r"        \bottomrule")
        t1.append(r"    \end{tabular}%")
        t1.append(r"    }")
        t1.append(r"\end{table}")
        t1.append("") 
    
    with open(CONFIG['out_tex_specs'], 'w', encoding='utf-8') as f:
        f.write("\n".join(t1))

    # ---------------------------------------------------------
    # TABLE 2: PPC (Sorted by Total Distance)
    # ---------------------------------------------------------
    df_sorted_ppc = df.sort_values(by='total_distance', ascending=True).reset_index(drop=True)
    
    t2 = []
    t2.append(r"\begin{table}[H]")
    t2.append(r"    \centering")
    t2.append(r"    \captionsetup{labelfont=bf}")
    t2.append(r"    \caption{\textbf{Empirical distances from Posterior Predictive Checks (PPC).}}")
    t2.append(r"    \resizebox{\textwidth}{!}{%")
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
        safe_name = escape_latex(row['model_name'])
        row_str = f"        \\texttt{{{safe_name}}}"
        for col in ppc_cols:
            row_str += f" & {row[col]:.3f}"
        row_str += r" \\"
        t2.append(row_str)
        
    t2.append(r"        \bottomrule")
    t2.append(r"    \end{tabular}%")
    t2.append(r"    }")
    t2.append(r"\end{table}")
    
    with open(CONFIG['out_tex_ppc'], 'w', encoding='utf-8') as f:
        f.write("\n".join(t2))

    # ---------------------------------------------------------
    # TABLE 3: Parameter Recovery (With Valid Check)
    # ---------------------------------------------------------
    t3 = []
    t3.append(r"\begin{table}[H]")
    t3.append(r"    \centering")
    t3.append(r"    \captionsetup{labelfont=bf}")
    t3.append(r"    \caption{\textbf{Parameter recovery correlations ($r$) across model variants.}}")
    t3.append(r"    \resizebox{\textwidth}{!}{%")
    t3.append(r"    \begin{tabular}{l cccccccccc c}")
    t3.append(r"        \toprule")
    t3.append(r"        \textbf{Model} & $\godirr$ & $\stdirr$ & $\tau$ & $c_{\text{se}}$ & $c_t$ & $c_{\text{ge}}$ & $c_{\text{gm}}$ & $\gonull$ & $\stnull$ & $\varphi$ & \textbf{Valid} \\")
    t3.append(r"        \midrule")
    
    rec_cols = ['q_d_r', 'q_s_r', 'tau_r', 'cost_stop_error_r', 'cost_time_r', 
                'cost_go_error_r', 'cost_go_missing_r', 'q_d_n_r', 'q_s_n_r', 'inv_temp_r']
                
    for _, row in df.iterrows():
        safe_name = escape_latex(row['model_name'])
        row_str = f"        \\texttt{{{safe_name}}}"
        
        is_valid = True
        has_params = False
        
        for col in rec_cols:
            val = row[col]
            if pd.isna(val):
                row_str += " & ---"
            else:
                row_str += f" & {val:.3f}"
                has_params = True
                if float(val) <= 0.6:
                    is_valid = False
                    
        # Append checkmark ONLY if there are params and ALL are > 0.6
        if is_valid and has_params:
            row_str += r" & \checkmark \\"
        else:
            row_str += r" & \\"
            
        t3.append(row_str)
        
    t3.append(r"        \bottomrule")
    t3.append(r"    \end{tabular}%")
    t3.append(r"    }")
    t3.append(r"\end{table}")
    
    with open(CONFIG['out_tex_rec'], 'w', encoding='utf-8') as f:
        f.write("\n".join(t3))
        
    print("LaTeX files generated successfully:")
    print(f"  -> {CONFIG['out_tex_specs']}")
    print(f"  -> {CONFIG['out_tex_ppc']}")
    print(f"  -> {CONFIG['out_tex_rec']}")

# ==========================================
# Script Execution
# ==========================================
if __name__ == "__main__":
    # 1. Extract raw data to intermediate CSVs
    extract_model_specs()
    summarize_ppc_metrics()
    extract_recoveries()
    
    # 2. Filter, sort, and generate final LaTeX tables
    generate_latex_tables()