import os
import glob
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 配置路径
BASE_DIR = '/Users/w/Desktop/data/sst_valid_base'
TARGET_COLUMN = 'sst_ssd_dur'

def main():
    # 找到所有的 zip 文件
    zip_pattern = os.path.join(BASE_DIR, '*.zip')
    zip_files = glob.glob(zip_pattern)
    
    if not zip_files:
        print(f"在 {BASE_DIR} 中没有找到任何 .zip 文件，请检查路径。")
        return

    print(f"共找到 {len(zip_files)} 个 zip 文件，开始处理...")
    
    all_ssd_durations = []
    error_files = []

    # 遍历所有 zip 文件（带进度条）
    for zf_path in tqdm(zip_files, desc="Processing ZIPs"):
        try:
            with zipfile.ZipFile(zf_path, 'r') as z:
                # 寻找压缩包内的 csv 文件
                # 你的文件路径类似 SST/baseline_year_1_arm_1/NDAR_INV..._sst.csv
                csv_filename = None
                for name in z.namelist():
                    if name.endswith('.csv'):
                        csv_filename = name
                        break
                
                if csv_filename:
                    # 直接在内存中打开并读取 CSV
                    with z.open(csv_filename) as f:
                        # 只读取我们需要的列，节省内存和时间
                        df = pd.read_csv(f, usecols=lambda c: c == TARGET_COLUMN)
                        
                        if TARGET_COLUMN in df.columns:
                            # 强制转换为数字，无法转换的变成 NaN，然后剔除 NaN
                            valid_durations = pd.to_numeric(df[TARGET_COLUMN], errors='coerce').dropna()
                            # 存入汇总列表
                            all_ssd_durations.extend(valid_durations.tolist())
                        else:
                            error_files.append(f"{os.path.basename(zf_path)}: 找不到列 {TARGET_COLUMN}")
                else:
                    error_files.append(f"{os.path.basename(zf_path)}: 压缩包内没有找到 csv 文件")
        except Exception as e:
            error_files.append(f"{os.path.basename(zf_path)}: 读取出错 - {str(e)}")

    # 打印错误/警告信息（如果有的话）
    if error_files:
        print(f"\n[注意] 有 {len(error_files)} 个文件出现问题 (前5个示例):")
        for err in error_files[:5]:
            print(f" - {err}")

    # 绘制直方图和打印统计表
    if all_ssd_durations:
        print(f"\n数据收集完毕，共提取到 {len(all_ssd_durations)} 个有效的 {TARGET_COLUMN} 数据点。")
        
        # 将列表转换为 pandas Series 
        ssd_series = pd.Series(all_ssd_durations)
        
        # === 1. 计算并打印 Mean, Median, Mode ===
        ssd_mean = ssd_series.mean()
        ssd_median = ssd_series.median()
        # mode 返回的是一个 Series (因为可能有多个众数)，我们取第一个
        ssd_mode = ssd_series.mode()[0] 
        
        print(f"\n=== SST SSD Duration 核心统计量 ===")
        print(f"Mean (平均值):   {ssd_mean:.2f}")
        print(f"Median (中位数): {ssd_median:.2f}")
        print(f"Mode (众数):     {ssd_mode:.2f}")
        print("===================================\n")
        
        # === 2. 计算并打印频数与比例的 Table ===
        stats_df = pd.DataFrame({
            'Count': ssd_series.value_counts(),
            'Proportion': ssd_series.value_counts(normalize=True)
        }).sort_index() 
        
        stats_df['Proportion (%)'] = stats_df['Proportion'].apply(lambda x: f"{x * 100:.2f}%")
        stats_df = stats_df.drop(columns=['Proportion'])
        
        print(f"=== {TARGET_COLUMN} 分布统计表 ===")
        print(stats_df.to_string())
        print("===================================\n")
        
        # === 3. 绘制修正后的柱状直方图 ===
        print("正在绘制图表...")
        plt.figure(figsize=(10, 6))
        
        # 使用 plt.bar 替代 plt.hist，直接根据离散的统计表绘图
        # width=40 意味着在间隔为 50 的 X 轴上，柱子宽度是 40，柱子之间会留出 10 的完美间隙
        plt.bar(stats_df.index, stats_df['Count'], width=40, color='skyblue', edgecolor='black', alpha=0.8)
        
        # 在图表上画三条竖线，直观标出均值、中位数和众数的位置
        plt.axvline(ssd_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {ssd_mean:.1f}')
        plt.axvline(ssd_median, color='green', linestyle='dotted', linewidth=2, label=f'Median: {ssd_median:.1f}')
        plt.axvline(ssd_mode, color='purple', linestyle='dashdot', linewidth=2, label=f'Mode: {ssd_mode:.1f}')
        
        # 设置 X 轴的刻度，让它正好对齐 0, 50, 100... 这些关键值
        plt.xticks(range(0, int(stats_df.index.max()) + 50, 50), rotation=45)
        
        plt.title(f'Distribution of {TARGET_COLUMN}', fontsize=14)
        plt.xlabel('SST SSD Duration (ms)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend() # 显示均值/中位数/众数的图例
        
        plt.tight_layout()
        plt.show()
    else:
        print("\n未能提取到任何有效数据，无法绘制图表。")

if __name__ == "__main__":
    main()