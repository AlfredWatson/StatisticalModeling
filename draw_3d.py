import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rules = pd.read_csv("fpgrowth/rules_30.csv")

selected_columns = rules.iloc[:, [5, 6, 7]]

columns = ['support', 'confidence', 'lift']

data_np = selected_columns.to_numpy()
# print(data_np)
supports = data_np[:, 0]
confidence = data_np[:, 1]
lift = data_np[:, 2]

# 使用scatter函数，c参数对应颜色映射的值（这里是z），cmap定义颜色映射方案
plt.figure(figsize=(16, 9), dpi=256)
sc = plt.scatter(supports, confidence, c=lift, cmap='viridis', s=100)  # 'viridis' 是一种常用的颜色映射

# 添加颜色条，用于解释颜色与数值的关系
plt.colorbar(sc, label=columns[2])

# 设置坐标轴标签
plt.xlabel(columns[0])
# plt.xlim([0.30, 0.65])
plt.ylabel(columns[1])
# plt.ylim([0.90, 1.70])
plt.title('Scatter Plot for rules')

# 显示图形
plt.show()
