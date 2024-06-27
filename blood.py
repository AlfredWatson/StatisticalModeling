import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import kaggle_legacy

# 设置中文字体，这里以SimHei（黑体）为例，你需要确保该字体存在于你的系统中
if plt.rcParams['font.family'] != 'SimHei':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于无衬线字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def draw_zhu(y_list: list, x_list: list):
    # 假设我们有一组数据
    data = np.array(y_list)
    # 创建一个新的图形
    plt.figure(figsize=(16, 9), dpi=512)
    # 绘制柱状图
    bars = plt.bar(np.arange(len(data)), data)
    # 在每个柱子顶部添加数据标签
    for bar in bars:
        height = bar.get_height()
        # 获取每个柱子的中心点x坐标
        x = bar.get_x() + bar.get_width() / 2
        # 设置y坐标为柱子高度加上适当的偏移量以便显示清晰
        y = height + 0.01
        # 使用annotate方法在柱顶添加数据标签
        plt.annotate(f'{height:.2f}', xy=(x, y), ha='center', va='bottom', fontsize=10, rotation=45)

    # 添加坐标轴标签等细节
    plt.xlabel('labels')
    plt.ylabel('item frequency(relative)')
    plt.xticks(fontsize=10)
    plt.xticks(np.arange(len(y_list)), x_list)
    plt.xticks(rotation=45)
    plt.title('frequencies')

    # 显示图形
    plt.show()


def draw_line_by_y(x_list, y_list, y_label):
    plt.figure(figsize=(16, 9), dpi=512)
    plt.xlabel('antecedents')
    plt.ylabel(y_label)
    plt.xticks(fontsize=5)
    plt.xticks(np.arange(len(y_list)), x_list)
    plt.xticks(rotation=22.5)
    plt.title(y_label)
    plt.plot(x_list, y_list)
    for i in range(len(y_list)):
        plt.annotate(
            f'{y_list[i]:.2f}', xy=(x_list[i], y_list[i]),  # {x_list[i]}:
            ha='center', va='bottom', fontsize=6, rotation=60
        )
    plt.show()


def cal_statistic(y_list, y_label):
    # 使用matplotlib和seaborn绘制箱线图，seaborn提供了一些额外的美化功能
    sns.set(style="whitegrid")  # 设置背景样式
    f, ax = plt.subplots(figsize=(6, 10))  # 设置图形大小
    sns.boxplot(data=y_list, orient="v", showmeans=True, meanline=True, showfliers=True, saturation=0.5)  # 绘制箱线图
    ax.set_title('Detailed Boxplot of ' + y_label)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Value')

    # 如果需要，还可以添加上下四分位距（IQR）外的数据点的标签
    q1 = np.percentile(y_list, 25)
    q3 = np.percentile(y_list, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = [num for num in y_list if num < lower_bound or num > upper_bound]
    for outlier in outliers:
        plt.scatter(0, outlier, color='red', marker='o')  # 离群值用红色散点标记

    plt.show()


def frequent_itemsets_filter(file_dir: str, file_name: str, draw=False):
    frequent_itemsets = pd.read_csv(file_dir + '/' + file_name)
    # 筛选点对点频繁集到txt文件
    file_frequent = open(file_dir + '/' + file_name.strip('.csv') + '_filter.csv', 'w')
    file_frequent.write('itemsets,support\n')
    supports, itemsets = [], []
    for i in range(len(frequent_itemsets)):
        temp_itemsets = str(frequent_itemsets['itemsets'][i])
        temp_itemsets = temp_itemsets.strip('\"frozenset({')
        temp_itemsets = temp_itemsets.strip('})\"')
        temp_itemsets_list = temp_itemsets.split(', ')
        if len(temp_itemsets_list) == 1:
            support = float(frequent_itemsets['support'][i])
            supports.append(round(support, 2))
            itemsets.append(temp_itemsets_list[0].strip('\''))
            file_frequent.write(temp_itemsets_list[0] + ',' + str(support) + '\n')
    # print(supports, itemsets)
    file_frequent.close()
    if draw:
        draw_zhu(supports, itemsets)  # 画频率图
    return itemsets, supports


def calculate(dataframe, algorithm='fpgrowth', min_support=0.1, first=False):
    file_dir = algorithm
    file_name = 'frequent_itemsets_' + str(int(min_support * 100)) + '.csv'

    # step1: 找出频繁项集
    if algorithm == 'fpgrowth':  # 使用fpgrowth算法找出频繁项集
        frequent_itemsets = fpgrowth(dataframe, min_support=min_support, use_colnames=True)
    else:  # 使用apriori算法找出频繁项集
        frequent_itemsets = apriori(dataframe, min_support=min_support, use_colnames=True)
    # print(frequent_itemsets)

    # 将频繁项集按照support数值排序
    frequent_itemsets_sorted = frequent_itemsets.sort_values(by='support', ascending=False)
    # 保存频繁集到csv文件
    frequent_itemsets_sorted.to_csv(file_dir + '/' + file_name)

    # 过滤频繁集，仅保留supports为单项的数据
    # supports, itemsets = frequent_itemsets_filter(file_dir, file_name, True)

    # 读频繁集
    # frequent_itemsets_sorted = pd.read_csv(file_dir + '/' + file_name)

    # step2: 根据频繁项集生成关联规则
    rules = association_rules(frequent_itemsets_sorted, metric="confidence", min_threshold=1)
    # print(rules)
    # 保存规则集到csv文件
    file_name = 'rules_' + str(int(min_support * 100)) + '.csv'
    rules.to_csv(file_dir + '/' + file_name)

    # rules = pd.read_csv(file_dir + '/' + file_name)
    # rules_columns = rules.columns.tolist()
    # '''
    # [antecedents,consequents,
    # antecedent support,consequent support,
    # support,confidence,lift,leverage,conviction,zhangs_metric]
    # '''
    # for i in range(2, len(rules_columns)):
    #     x_list, y_list = [], []
    #     for j in range(0, len(rules)):
    #         x_up = ""
    #         for e in rules[rules_columns[0]][j]:
    #             x_up = x_up + e + " "
    #         x_down = ""
    #         for e in rules[rules_columns[1]][j]:
    #             x_down = x_down + e + " "
    #         x = x_up + "=>" + x_down  # antecedents => consequents
    #         x_list.append(x.strip())
    #
    #         y = rules[rules_columns[i]][j]
    #         y_list.append(y)
    #     # print(x_list, y_list)
    #     draw_line_by_y(x_list, y_list, rules_columns[i])
    #     cal_statistic(y_list, rules_columns[i])  # 计算y_list的统计量, 并画出箱式图
    #
    # # step3: 聚类
    # kaggle_legacy.kmeans(rules, rules_columns[2:])

    return True


def filter_rules(algorithm='fpgrowth', min_support=0.1):
    rules_data = pd.read_csv(algorithm + "/rules_" + str(int(min_support * 100)) + ".csv")
    df = pd.DataFrame(rules_data)
    columns = df.columns.tolist()
    # print(df.head())
    # 存放筛选后
    file_rules = open(algorithm + "/rules_" + str(int(min_support * 100)) + "_filter.csv", 'w')

    # 写入列
    line = ""
    for j in range(1, len(columns)):
        line = line + str(columns[j]) + ","
    file_rules.write(line + '\n')

    # 写入数据
    for i in range(len(df)):
        support = df['support'][i]
        confidence = df['confidence'][i]
        lift = df['lift'][i]
        if support > 0.2 and confidence > 0.3 and lift >= 1:
            # print(support, confidence, lift)
            line = ""
            for j in range(1, len(columns)):
                line = line + str(df[columns[j]][i]) + ","
            file_rules.write(line + '\n')


def ori_clustering():
    blood_data = pd.read_csv("blood_ori.csv")
    df_data = pd.DataFrame(blood_data)
    columns = df_data.columns.tolist()
    # kaggle_legacy.kmeans(df_data, columns)
    x = df_data.iloc[:, :].values

    kmeans_model = KMeans(n_clusters=3)
    kmeans_model.fit(x)

    cluster_labels = kmeans_model.fit_predict(x)

    # 初始化PCA，并指定降维到2维
    pca = PCA(n_components=2)

    # 转换数据
    X_pca = pca.fit_transform(x)

    # 结果合并，以便于根据聚类标签着色
    data_pca = np.hstack((X_pca, cluster_labels.reshape(-1, 1)))

    # 绘制散点图，根据聚类标签上色
    plt.figure(figsize=(10, 8))
    for i in range(3):
        # 选择特定聚类的数据点
        cluster_points = data_pca[data_pca[:, 2] == i][:, :2]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}', alpha=0.5)

    plt.title('PCA Visualization of Clusters in 2D')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()


def draw_clustering_tree(min_support=0.3, n=20):
    blood_data_temp = pd.read_excel("./blood_ori_zh.xlsx")
    x_labels = blood_data_temp.columns.tolist()

    file_dir = "fpgrowth"
    file_name = 'frequent_itemsets_' + str(int(min_support * 100)) + '.csv'
    itemsets, _ = frequent_itemsets_filter(file_dir, file_name, False)
    n = min(n, len(itemsets))

    mask = [0] * len(x_labels)
    x_labels_new = []
    for i in range(len(x_labels)):
        if x_labels[i] in itemsets[:n]:
            mask[i] = 1
            x_labels_new.append(x_labels[i])

    df_data = pd.read_excel("./blood_ori_zh_t.xlsx")
    features = df_data.columns.tolist()
    features = features[1:]

    for i in range(len(mask)):
        if mask[i] == 0:
            df_data.drop(i, inplace=True)

    y = df_data.iloc[:, :].values

    kmeans_kw = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 567,
    }

    inertia = []
    for k in range(1, 10):
        kmeans_model = KMeans(n_clusters=k, **kmeans_kw)
        kmeans_model.fit(y[0:19])
        inertia.append(kmeans_model.inertia_)

    """
    肘点：肘点方法是在 K 均值聚类中找到最佳“K”的图形表示。
    它的工作原理是找到 WCSS（簇内平方和），即簇中点与簇质心之间的平方距离之和。
    """
    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(range(1, 10), inertia, color='purple')
    for i in range(1, len(inertia) + 1):
        plt.annotate(f'{inertia[i - 1]:.2f}', xy=(i, inertia[i - 1]), ha='center', va='bottom', fontsize=6)
    plt.xticks(range(1, 10))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Iner")
    KL = KneeLocator(range(1, 10), inertia, curve="convex", direction="decreasing")
    print(KL.elbow)
    plt.axvline(x=KL.elbow, color='b', label='axvline - full height', linestyle="dashed")
    plt.show()

    """
    轮廓系数：轮廓系数或轮廓分数是用于计算聚类技术优度的指标。其值范围为 -1 到 1。
    1：表示集群彼此相距甚远且区分清楚。
    0：表示簇是无差别的，或者我们可以说簇之间的距离不显著。
    -1：表示集群的分配方式错误。
    """
    silhouette_coefficients = []
    for k in range(2, 10):
        kmeans_model = KMeans(n_clusters=k, **kmeans_kw)
        kmeans_model.fit(y[0:19])
        score = silhouette_score(y[0:19], kmeans_model.labels_)
        silhouette_coefficients.append(score)
    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(range(2, 10), silhouette_coefficients, color='purple')
    for i in range(2, len(silhouette_coefficients) + 2):
        plt.annotate(
            f'{silhouette_coefficients[i - 2]:.4f}', xy=(i, silhouette_coefficients[i - 2]),
            ha='center', va='bottom', fontsize=6
        )
    plt.xticks(range(2, 10))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()

    # # KMeans聚类
    # kmeans = KMeans(n_clusters=3)
    # kmeans_labels = kmeans.fit_predict(y)
    #
    # # 层次聚类
    # linked = linkage(y, 'ward')
    #
    # # 绘制树状图
    # plt.figure(figsize=(16, 9), dpi=512)
    # dendrogram(
    #     linked, orientation='top', distance_sort='descending', show_leaf_counts=True,
    #     labels=x_labels_new, leaf_rotation=90
    # )
    # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Distance')
    # plt.show()


def filter_rules_step2(algorithm='fpgrowth', min_support=0.1):
    blood_data = pd.read_csv("./" + algorithm + "/rules_" + str(int(min_support * 100)) + "_filter.csv", encoding='gbk')
    df_ori = pd.DataFrame(blood_data)
    columns = ['antecedent support', 'consequent support', 'leverage', 'conviction', 'zhangs_metric']
    df_temp = df_ori.drop(columns=columns)
    df_sorted = df_temp.sort_values(by='confidence', ascending=False)
    # print(df_sorted)
    df_sorted.to_csv(
        "./" + algorithm + "/rules_" + str(int(min_support * 100)) + "_sorted_by_" + 'confidence' + ".csv",
        index=False
    )


if __name__ == '__main__':
    df_data = pd.read_excel("./blood_ori_zh.xlsx")
    # df_data = pd.read_csv("./blood.csv")
    calculate(df_data, 'apriori', 0.01, True)

    # filter_rules(algorithm='fpgrowth', min_support=0.3)

    # ori_clustering()
    # draw_clustering_tree()
