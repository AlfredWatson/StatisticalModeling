import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

if plt.rcParams['font.family'] != 'SimHei':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于无衬线字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

if __name__ == '__main__':
    # 读取CSV文件
    rules_df = pd.read_csv('./fpgrowth/rules_30_sorted_by_confidence_2.csv', encoding='gbk')  # 请替换为你的CSV文件路径

    # 初始化一个无向图
    G = nx.Graph()

    # 添加节点：每个规则的前件和后件都是图中的节点
    for _, row in rules_df.iterrows():
        antecedents = row['antecedents']
        consequents = row['consequents']
        # 分割字符串假设它们是以某种分隔符（比如逗号）分隔的
        # antecedents_set = set(antecedents)
        # consequents_set = set(consequents)

        # 添加节点
        # for item in antecedents_set.union(consequents_set):
        #     G.add_node(item)
        G.add_node(antecedents)
        G.add_node(consequents)

        # 添加边，这里可以根据支持度、置信度等作为边的权重，或者简单连接前后件
        G.add_edge(antecedents, consequents, weight=row['confidence'])  # 使用置信度作为边的权重

    # 可视化网络图
    pos = nx.spring_layout(G)  # 使用spring布局

    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_size=100, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, edge_color=[G[u][v]['weight'] for u, v in G.edges()])
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='SimHei')  # 确保字体支持中文

    plt.title("Association Rule Network")
    plt.axis('off')
    plt.show()
