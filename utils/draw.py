import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

# 数据
categories = ['PASCAL VOC', 'COCO', 'iSAID']
attributes = ['SOTA mIoU', 'Background Area', 'Foreground Area']
values = np.array([
    [0.73, 0.70, 0.30],
    [0.45, 0.96, 0.04],
    [0.37, 0.97, 0.03]
])

# 设置字号
plt.rcParams['font.size'] = 11

# 绘图
bar_width = 0.2
bar_positions = np.arange(len(categories))

fig, ax = plt.subplots()

for i, attr in enumerate(attributes):
    bars = ax.bar(bar_positions + i * bar_width, values[:, i], bar_width, label=attr)

    # 在每个条形上方添加数字标注
    for bar, val in zip(bars, values[:, i]):
        color = 'red' if val == 0.37 else 'black'
        yval = bar.get_height()
        if val != 0.37:
            ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom', color=color)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.03, round(yval, 2), ha='center', va='bottom', color=color)

# 添加趋势线 - SOTA mIoU
trend_line_values = [np.mean(values[:, 0]) for _ in range(len(categories))]
ax.plot(bar_positions + 0.5, trend_line_values, marker='o', linestyle='-', color='gray', markersize=0)

# 添加趋势线 - Background Area
trend_line_values = [np.mean(values[:, 1]) for _ in range(len(categories))]
ax.plot(bar_positions + 1.5, trend_line_values, marker='o', linestyle='-', color='gray', markersize=0)

# 添加标签和标题
ax.set_xticks(bar_positions + bar_width)
ax.set_xticklabels(categories)
ax.set_xlabel('Datasets')
ax.set_ylabel('Values')
ax.set_title('WSSS Datasets Distribution and Comparison')

# 添加图例
ax.legend()

# 显示图形
plt.show()
plt.savefig('output_plot.pdf')




# import matplotlib.pyplot as plt
#
# # 数据
# labels = list(range(12))  # 0~12表示transformer blocks
# vit_b_data = [0.559, 0.5535, 0.5545, 0.58, 0.58, 0.593, 0.626, 0.676, 0.6808, 0.6638, 0.7017, 0.70943]
# ours_data = [0.6836, 0.726, 0.758, 0.782, 0.81, 0.82, 0.815, 0.801, 0.811, 0.847, 0.814, 0.905]
#
# # 画图
# plt.figure(figsize=(10, 6))
#
# plt.plot(labels, vit_b_data, marker='o', label='Ours', linestyle='-', color='b')
# plt.plot(labels, ours_data, marker='s', label='ViT-B', linestyle='--', color='r')
#
# # 在每个数据点上添加数字
# for i, txt in enumerate(vit_b_data):
#     plt.text(labels[i], vit_b_data[i], f'{vit_b_data[i]:.2f}', ha='right', va='bottom')
#
# for i, txt in enumerate(ours_data):
#     plt.text(labels[i], ours_data[i], f'{ours_data[i]:.2f}', ha='left', va='top')
#
# # 添加标题和标签
# # plt.title('Pair-wise Cosine Similarity vs Transformer Blocks')
# plt.xlabel('Transformer Blocks')
# plt.ylabel('Pair-wise Cosine Similarity')
# plt.legend()
#
# # 添加表格线
# plt.grid(True)
#
# # 使用Times New Roman字体
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.show()
# plt.savefig('output_plot.pdf')
# # 显示图形
#
