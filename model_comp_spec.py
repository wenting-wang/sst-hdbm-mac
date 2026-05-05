import os
import glob
import re
import pandas as pd

def extract_model_specs():
    # 1. 定义文件夹路径和输出表格的路径
    input_folder = '/Users/w/sst-hdbm-mac/pomdp_v2/model_spec'
    output_file = '/Users/w/sst-hdbm-mac/pomdp_v2/model_specs_summary.csv'
    
    # 确保输入文件夹存在
    if not os.path.exists(input_folder):
        print(f"错误: 找不到文件夹 {input_folder}")
        return

    # 2. 定义用于提取的正则表达式
    # 匹配 MODEL_TAG = "xxx" 或 'xxx'
    tag_pattern = re.compile(r'MODEL_TAG\s*=\s*["\']([^"\']+)["\']')
    
    # 匹配 PARAM_RANGES = { ... }，使用 re.DOTALL 允许跨行匹配
    ranges_pattern = re.compile(r'PARAM_RANGES\s*=\s*(\{.*?\})', re.DOTALL)
    
    # 匹配 FIXED_PARAMS = { ... }
    fixed_pattern = re.compile(r'FIXED_PARAMS\s*=\s*(\{.*?\})', re.DOTALL)

    # 3. 寻找所有的 Python 文件
    search_pattern = os.path.join(input_folder, '*.py')
    py_files = glob.glob(search_pattern)
    
    if not py_files:
        print(f"在 {input_folder} 中没有找到任何 .py 文件。")
        return

    print(f"找到 {len(py_files)} 个 Python 配置文件，开始提取...\n")
    
    extracted_data = []

    # 4. 遍历并解析每个文件
    for file_path in py_files:
        file_name = os.path.basename(file_path)
        print(f"正在读取: {file_name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 提取信息
            tag_match = tag_pattern.search(content)
            ranges_match = ranges_pattern.search(content)
            fixed_match = fixed_pattern.search(content)
            
            # 格式化提取结果：如果匹配到了，去掉多余的换行和空格，压缩成单行
            model_tag = tag_match.group(1) if tag_match else None
            
            if ranges_match:
                param_ranges_str = re.sub(r'\s+', ' ', ranges_match.group(1).strip())
            else:
                param_ranges_str = None
                
            if fixed_match:
                fixed_params_str = re.sub(r'\s+', ' ', fixed_match.group(1).strip())
            else:
                fixed_params_str = None
                
            # 存入字典
            extracted_data.append({
                'file_name': file_name,
                'MODEL_TAG': model_tag,
                'PARAM_RANGES': param_ranges_str,
                'FIXED_PARAMS': fixed_params_str
            })
            
        except Exception as e:
            print(f"  -> 解析文件时出错: {e}")

    # 5. 生成大表并保存
    if extracted_data:
        df = pd.DataFrame(extracted_data)
        df.to_csv(output_file, index=False)
        print(f"\n✅ 提取完成！成功处理了 {len(extracted_data)} 个文件。")
        print(f"📊 汇总大表已保存至: {output_file}")
        
        # 打印前几行预览
        print("\n=== 表格预览 ===")
        print(df.head().to_string())
    else:
        print("未能提取任何有效信息。")

if __name__ == "__main__":
    extract_model_specs()