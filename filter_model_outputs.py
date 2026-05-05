import pandas as pd
import os
import glob

def filter_csv_files():
    # 1. 定义文件路径
    reference_file = '/Users/w/sst-hdbm-mac/clinical_behavior.csv'
    target_folder = '/Users/w/sst-hdbm-mac/pomdp_v2/model_outputs_raw/'
    
    # 建议将过滤后的文件存入新文件夹，避免覆盖原始数据
    output_folder = '/Users/w/sst-hdbm-mac/pomdp_v2/model_outputs_filtered/'
    os.makedirs(output_folder, exist_ok=True)

    # 2. 读取参照文件，获取所有的 subject_id
    print(f"正在读取参照文件: {reference_file}")
    try:
        ref_df = pd.read_csv(reference_file)
        # 提取并去重 subject_id，存为 set 以提高匹配速度
        # 注意：如果你的列名大小写不同（如 'Subject_ID'），请在此处修改
        valid_subjects = set(ref_df['subject_id'].dropna().unique())
        print(f"成功获取了 {len(valid_subjects)} 个独立的 subject_id。\n")
    except FileNotFoundError:
        print(f"错误：找不到参照文件 {reference_file}，请检查路径。")
        return
    except KeyError:
        print(f"错误：参照文件中没有名为 'subject_id' 的列，请检查列名拼写。")
        return

    # 3. 遍历 target_folder 下的所有 csv 文件
    # 例如：/Users/w/sst-hdbm-mac/pomdp_v2/model_outputs_raw/ppc_metrics_5p_v1.csv
    search_pattern = os.path.join(target_folder, '*.csv')
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"在 {target_folder} 中没有找到任何 CSV 文件。")
        return

    print(f"找到 {len(csv_files)} 个待处理的 CSV 文件，开始过滤...\n")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"正在处理: {file_name}")
        
        try:
            # 读取目标文件
            target_df = pd.read_csv(file_path)
            
            # 检查文件是否包含 'subject_id' 列
            if 'subject_id' in target_df.columns:
                original_rows = len(target_df)
                
                # 核心过滤逻辑：只保留 subject_id 在 valid_subjects 里的行
                filtered_df = target_df[target_df['subject_id'].isin(valid_subjects)]
                filtered_rows = len(filtered_df)
                
                # 保存过滤后的文件
                output_path = os.path.join(output_folder, file_name)
                filtered_df.to_csv(output_path, index=False)
                
                print(f"  -> 完成！原始行数: {original_rows}, 过滤后行数: {filtered_rows}")
                print(f"  -> 已保存至: {output_path}")
            else:
                print(f"  -> 跳过！该文件不包含 'subject_id' 列。")
                
        except Exception as e:
            print(f"  -> 处理该文件时发生错误: {e}")
            
    print("\n所有文件处理完毕！")

if __name__ == "__main__":
    filter_csv_files()