import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 定义文件路径
current_dir = os.getcwd()  
file_name = 'params_posteriors_10p_v1.csv'  
file_path = os.path.join(current_dir, file_name)

print(f"正在读取文件: {file_path}")

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"找不到文件 {file_name}，请确保它和 Python 脚本在同一个文件夹下！")
    exit()

# ==========================================
# 更新功能：计算并打印所有被试 mean 值的均值
# ==========================================
print("\n" + "="*45)
print("各个参数在所有被试中的平均值 (Mean of means)：")
print("="*45)

# 按照参数名 ('index') 分组，取出 'mean' 列，并计算均值
average_of_means = df.groupby('index')['mean'].mean()

# 打印结果，保留四位小数
for param, avg_val in average_of_means.items():
    print(f"{param:<16}: {avg_val:.4f}")
print("="*45 + "\n")

# ==========================================
# 剔除不需要画图的参数 'rate_stop_trial'
# ==========================================
df_plot = df[df['index'] != 'rate_stop_trial']

# 2. 设置绘图风格
sns.set_theme(style="whitegrid", palette="muted")

# 3. 获取剩余需要画图的参数名称
parameters = df_plot['index'].unique()

# 4. 动态计算子图的排版 (每行 3 个子图)
n_cols = 3
n_rows = (len(parameters) + n_cols - 1) // n_cols

# 创建画布和子图阵列
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

# 5. 遍历每个参数并绘制子图
for i, param in enumerate(parameters):
    ax = axes[i]
    param_data = df_plot[df_plot['index'] == param]
    
    # 画图：判断数据量是否足够画平滑曲线
    if len(param_data) > 1:
        sns.histplot(data=param_data, x='mean', kde=True, ax=ax, color='cornflowerblue', bins=10)
    else:
        sns.histplot(data=param_data, x='mean', kde=False, ax=ax, color='cornflowerblue', bins=10)
        
    ax.set_title(f'Parameter: {param}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Mean Estimate')
    ax.set_ylabel('Subject Count')

# 6. 隐藏多出来的空白子图
for j in range(len(parameters), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()

# 7. 保存图片，不显示
save_name = 'parameter_distributions_10p.png'
save_path = os.path.join(current_dir, save_name)

plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"图片已成功保存至: {save_path}")