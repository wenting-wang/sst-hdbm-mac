import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ==========================================
# 核心统计函数 (参考 fig_params_recovery_adaptive)
# ==========================================
def remove_outliers(x, y, lower_pct=1.0, upper_pct=99.0):
    """过滤有效数据并移除 y (预测值) 的上下分位数离群点[cite: 2]"""
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
    """计算相关系数 r 和 RMSE，支持 log scale[cite: 2]"""
    if len(x) < 2:
        return np.nan, np.nan
        
    if is_log:
        # 在 log scale 下计算 r 和 RMSE[cite: 2]
        mask = (x > 0) & (y > 0)
        x_log = np.log10(x[mask])
        y_log = np.log10(y[mask])
        if len(x_log) < 2:
            return np.nan, np.nan
        r, _ = pearsonr(x_log, y_log)
        rmse = np.sqrt(np.mean((x_log - y_log)**2))
    else:
        # 常规线性 scale 计算[cite: 2]
        r, _ = pearsonr(x, y)
        rmse = np.sqrt(np.mean((x - y)**2))
        
    return r, rmse

# ==========================================
# 批量处理逻辑
# ==========================================
def calculate_all_recoveries():
    input_folder = '/Users/w/sst-hdbm-mac/pomdp_v2/model_recovery/'
    output_file = '/Users/w/sst-hdbm-mac/pomdp_v2/recovery_stats_summary.csv'
    
    # 启发式定义哪些参数在计算时需要转换成 log scale
    # 基于你之前代码的习惯，cost 系列参数通常会放在 log_params 里面
    known_log_params = ['cost_stop_error', 'cost_time', 'cost_go_error', 'cost_go_missing']
    
    search_pattern = os.path.join(input_folder, '*.csv')
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"在 {input_folder} 中没有找到任何 CSV 文件。")
        return
        
    print(f"找到 {len(csv_files)} 个 recovery 文件，开始计算统计指标...\n")
    
    summary_data = []
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        # 提取模型名称作为标识 (例如: params_recovery_5p_v1.csv -> 5p_v1)
        model_name = file_name.replace('params_recovery_', '').replace('.csv', '')
        print(f"正在处理模型: {model_name}")
        
        try:
            df = pd.read_csv(file_path)
            
            # 动态获取当前文件内的所有参数名称：找出所有以 'gt_' 开头的列[cite: 2]
            gt_cols = [col for col in df.columns if col.startswith('gt_')]
            params = [col[3:] for col in gt_cols]
            
            model_summary = {'model_name': model_name}
            
            for param in params:
                col_true = f"gt_{param}"
                col_rec = f"mu_{param}"
                
                # 确保真值列和预测值列都存在[cite: 2]
                if col_rec in df.columns:
                    x_raw = df[col_true].to_numpy(float)
                    y_raw = df[col_rec].to_numpy(float)
                    
                    # 1. 过滤异常值 (与原代码作图时的过滤范围 5.0% - 95.0% 保持一致)[cite: 2]
                    x, y = remove_outliers(x_raw, y_raw, lower_pct=5.0, upper_pct=95.0)
                    
                    # 2. 判断该参数是否在 log_params 列表中
                    is_log_param = param in known_log_params
                    
                    # 3. 获取 r 和 rmse[cite: 2]
                    r_val, rmse_val = get_stats(x, y, is_log=is_log_param)
                    
                    # 将结果存入字典，列名格式为：参数名_r 和 参数名_rmse
                    model_summary[f"{param}_r"] = r_val
                    model_summary[f"{param}_rmse"] = rmse_val
                    
            summary_data.append(model_summary)
            
        except Exception as e:
            print(f"  -> 处理 {file_name} 时发生错误: {e}")

    # 将所有模型结果汇总成一张 DataFrame
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # 排序：将 model_name 放在第一列
        cols = ['model_name'] + [c for c in summary_df.columns if c != 'model_name']
        summary_df = summary_df[cols]
        
        summary_df.to_csv(output_file, index=False)
        print(f"\n✅ 成功！大表已保存至: {output_file}")
        
        print("\n=== Recovery 汇总表预览 ===")
        print(summary_df.head().to_string())
    else:
        print("\n未能提取任何有效数据。")

if __name__ == '__main__':
    calculate_all_recoveries()