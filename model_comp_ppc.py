import pandas as pd
import os
import glob

def summarize_model_metrics():
    # 1. 定义文件夹路径和要计算的列
    input_folder = '/Users/w/sst-hdbm-mac/pomdp_v2/model_outputs_filtered/'
    # 汇总表的保存路径
    summary_output_file = '/Users/w/sst-hdbm-mac/pomdp_v2/models_summary_table.csv' 
    
    # 你指定需要计算平均值的列名列表
    target_columns = [
        'dis_perc_gs', 'dis_perc_ge', 'dis_perc_gm', 'dis_perc_ss',
        'dis_ws_rt_gs', 'dis_ws_rt_ge', 'dis_ws_rt_se', 
        'dis_ks_rt_gs', 'dis_ks_rt_se', 'dis_ssd_mean', 'total_distance'
    ]

    # 2. 获取所有过滤后的 csv 文件
    search_pattern = os.path.join(input_folder, '*.csv')
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"在 {input_folder} 中没有找到任何 CSV 文件，请检查路径。")
        return

    print(f"找到 {len(csv_files)} 个模型文件，正在计算平均值...\n")

    # 用于存放每个模型结果的列表
    summary_data = []

    # 3. 遍历每个文件进行计算
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        # 去掉 .csv 后缀作为模型名称
        model_name = os.path.splitext(file_name)[0] 
        
        try:
            df = pd.read_csv(file_path)
            
            # 创建一个字典来存储当前模型的数据，首先记录模型名字
            model_summary = {'model_name': model_name}
            
            # 检查指定的列是否存在，并计算平均值
            for col in target_columns:
                if col in df.columns:
                    # 计算该列的平均值，忽略 NaN 值
                    model_summary[col] = df[col].mean()
                else:
                    # 如果某个文件缺了某列，用 NaN 填补，并在控制台提示
                    model_summary[col] = pd.NA
                    print(f"  -> 警告: {file_name} 中缺失列 '{col}'")
            
            summary_data.append(model_summary)
            print(f"完成计算: {model_name}")
            
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")

    # 4. 将汇总数据转换为 DataFrame 大表
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # 打印大表的前几行预览
        print("\n=== 汇总大表预览 ===")
        print(summary_df.head().to_string())
        
        # 保存为新的 CSV 文件
        summary_df.to_csv(summary_output_file, index=False)
        print(f"\n✅ 成功！大表已保存至: {summary_output_file}")
    else:
        print("未能提取任何有效数据。")

if __name__ == "__main__":
    summarize_model_metrics()