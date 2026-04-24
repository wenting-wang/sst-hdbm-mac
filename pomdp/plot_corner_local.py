import pandas as pd
import corner
import matplotlib.pyplot as plt

# 1. 读取保存的 10000 个后验样本
df_samples = pd.read_csv("/Users/w/sst-hdbm-mac/pomdp/posterior_samples_NDAR_INV0WAF57A4.csv")

# 2. 获取参数列名
PARAM_ORDER = df_samples.columns.tolist()

# 3. 重新绘制 Corner 图，加入 range 参数剔除极端值
print("Plotting refined corner matrix...")
fig = corner.corner(
    df_samples,
    labels=PARAM_ORDER,
    quantiles=[0.05, 0.5, 0.95],
    show_titles=True,
    title_kwargs={"fontsize": 10},
    hist_kwargs={'density': True},
    color="royalblue",
    smooth=1.0,
    plot_datapoints=False,
    fill_contours=True,
    # 关键参数：0.98 表示画图时自动裁剪掉每个参数最高 1% 和最低 1% 的极端离群值
    range=[0.98] * len(PARAM_ORDER) 
)

# 4. 保存精修后的高清图
fig.savefig("/Users/w/sst-hdbm-mac/pomdp/corner_NDAR_INV0WAF57A4_refined.png", dpi=300, bbox_inches='tight')
print("Refined plot saved successfully!")