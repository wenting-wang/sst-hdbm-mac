import pandas as pd
import os

# ================= 配置区域 =================
# 文件列表 CSV (包含 filename 列)
LIST_FILE = '/Users/w/sst-abcd-adhd-mac/sst-hdbm-main/stats_beh_full.csv'
# 数据源文件夹
DATA_ROOT = '/Users/w/Desktop/data/sst_valid_base'
# ===============================================

def clean_files():
    print(">>> 开始执行 STEP 1: 数据清洗 (删除非 360 行的文件)...")
    
    try:
        # 读取列表
        df_list = pd.read_csv(LIST_FILE)
        if 'filename' in df_list.columns:
            file_list = df_list['filename'].tolist()
        else:
            file_list = df_list.iloc[:, 0].tolist()
    except Exception as e:
        print(f"列表文件读取失败: {e}")
        return

    deleted_count = 0
    kept_count = 0
    total_files = len(file_list)
    
    print(f"准备扫描 {total_files} 个文件...\n")

    for i, fname in enumerate(file_list):
        full_path = os.path.join(DATA_ROOT, fname)
        
        # 路径兼容处理 (检查子文件夹)
        if not os.path.exists(full_path):
             full_path = os.path.join(DATA_ROOT, 'SST/baseline_year_1_arm_1', fname)
             if not os.path.exists(full_path): 
                 # print(f"找不到文件 (跳过): {fname}")
                 continue
        
        try:
            # 这里的 nrows=361 是个小技巧：只读前361行就能判断长度是否超标或不足
            # 如果文件巨大，这样读会快很多；但对于SST这种小文件，全读也没事。
            df = pd.read_csv(full_path) 
            n_rows = len(df)
            
            if n_rows != 360:
                print(f"[删除] 长度异常 ({n_rows} lines): {fname}")
                os.remove(full_path) # <--- 执行删除
                deleted_count += 1
            else:
                kept_count += 1
                
        except Exception as e:
            print(f"[错误] 无法读取 {fname}: {e}")
            
        if (i + 1) % 1000 == 0:
            print(f"已扫描 {i + 1} / {total_files} ...")

    print("\n" + "="*50)
    print("清洗完成报告")
    print("="*50)
    print(f"扫描总数: {total_files}")
    print(f"保留文件: {kept_count} (均为标准的 360 行)")
    print(f"删除文件: {deleted_count} (长度不足或损坏)")
    print("="*50)
    print("现在可以放心地运行 Step 2 进行分析了。")

if __name__ == "__main__":
    clean_files()