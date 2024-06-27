import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


class color:
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def describe(df):
    variables, dtypes, count, unique, missing = [], [], [], [], []

    for item in df.columns:
        variables.append(item)
        dtypes.append(df[item].dtype)
        count.append(len(df[item]))
        unique.append(len(df[item].unique()))
        missing.append(df[item].isna().sum())

    output = pd.DataFrame({
        'variable': variables,
        'dtype': dtypes,
        'count': count,
        'unique': unique,
        'missing value': missing
    })

    return output


def plotly_visualizations(df):
    """
    使用Plotly散射，每个数据点都表示为一个标记点，其位置由x和y列给定。
    我们在5个主要变量上使用这个图（第5节）。
    """
    fig1 = px.scatter(
        df, x="Schizophrenia", y="Depressive", color="Anxiety",
        marginal_y="violin", marginal_x="box", trendline="ols", template="simple_white"
    )
    fig1.show()

    # fig2 = px.scatter(
    #     df, x="Depressive", y="Anxiety", color="Bipolar",
    #     marginal_y="violin", marginal_x="box", trendline="ols", template="simple_white"
    # )
    # fig2.show()
    #
    # fig3 = px.scatter(
    #     df, x="Anxiety", y="Bipolar", color="Eating",
    #     marginal_y="violin", marginal_x="box", trendline="ols", template="simple_white"
    # )
    # fig3.show()
    #
    # fig4 = px.scatter(
    #     df, x="Bipolar", y="Eating", color="Schizophrenia",
    #     marginal_y="violin", marginal_x="box", trendline="ols", template="simple_white"
    # )
    # fig4.show()
    #
    # """
    # A scatterplot matrix is a matrix associated to n numerical arrays (data variables), X1,X2,…,Xn, of the same length.
    # The cell (i,j) of such a matrix displays the scatter plot of the variable Xi versus Xj.
    # 散点图矩阵是与n个相同长度的数值阵列（数据变量）X1、X2、…、Xn相关联的矩阵。
    # 这种矩阵的单元（i，j）显示变量Xi对Xj的散点图。
    # """
    # fig = px.scatter_matrix(df, dimensions=["Schizophrenia", "Depressive", "Anxiety", "Bipolar"], color="Eating")
    # fig.show()


def seaborn_visualizations(df):
    """
    在描述性统计学中，箱图或箱图是一种通过四分位数以图形方式展示数字数据组的位置、分布和偏度的方法。
    除了方框图上的方框外，还可以有从方框延伸的线（称为须），指示上四分位数和下四分位数之外的可变性，因此，该图也称为方框图和方框图。
    """
    Numerical = ['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating']
    # i = 0
    # while i < 5:
    #     fig = plt.figure(figsize=[30, 3], dpi=200)
    #     plt.subplot(2, 2, 1)
    #     sns.boxplot(x=Numerical[i], data=df, boxprops=dict(facecolor="lightblue"))
    #     i += 1
    #     plt.show()
    """
    相关矩阵是探索性数据分析的重要工具。
    相关热图以一种视觉上吸引人的方式包含相同的信息。
    更重要的是：它们一目了然地显示了哪些变量是相关的，在多大程度上，在哪个方向上，并提醒我们潜在的多重共线性问题。
    """
    Corrmat = df[Numerical].corr()
    plt.figure(figsize=(10, 5), dpi=200)
    sns.heatmap(Corrmat, annot=True, fmt=".2f", linewidth=.5)
    plt.show()


def machine_learning_models(df):
    """
    正如我们之前提到的，我们将对这个数据集使用7种差分算法。只需查看每个算法的输出，就可以更好地了解集群的情况。
    """
    features = ['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating']
    # mini_batch_k_means(df, features)
    # agglomerative_clustering(df, features)
    # birch(df, features)
    # dbscan(df, features)
    kmeans(df, features)


def mini_batch_k_means(df, features):
    """
    Mini-BatchK-means算法的主要思想是使用固定大小的随机小批量数据，以便将它们存储在内存中。
    每次迭代都会从数据集中获得一个新的随机样本，并用于更新聚类，重复此过程直到收敛。
    每个小批量使用原型和数据的值的凸组合来更新聚类，应用随着迭代次数而减少的学习率。
    该学习速率与过程中分配给集群的数据数量相反。
    随着迭代次数的增加，新数据的影响减少，因此当在几个连续迭代中集群没有发生变化时，可以检测到收敛。
    """
    X_model_MiniB = df[features]
    X_model_MiniB = pd.DataFrame(X_model_MiniB)
    kmeans = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=6)
    X_model_MiniB["Cluster"] = kmeans.fit_predict(X_model_MiniB)
    X_model_MiniB["Cluster"] = X_model_MiniB["Cluster"].astype("int")
    print(X_model_MiniB.head())
    plt.figure(figsize=(10, 5), dpi=200)
    plt.style.use('seaborn-whitegrid')
    plt.rc("figure", autolayout=True)
    plt.rc("axes", labelweight='bold', labelsize='large', titleweight='bold', titlesize=14, titlepad=10)
    sns.relplot(x='Schizophrenia', y='Depressive', hue='Cluster', data=X_model_MiniB, height=6)
    sns.relplot(x='Schizophrenia', y='Anxiety', hue='Cluster', data=X_model_MiniB, height=6)
    sns.relplot(x='Schizophrenia', y='Bipolar', hue='Cluster', data=X_model_MiniB, height=6)
    sns.relplot(x='Schizophrenia', y='Eating', hue='Cluster', data=X_model_MiniB, height=6)
    plt.show()

    print(X_model_MiniB)

    print(X_model_MiniB["Cluster"].value_counts())

    df_Mini_C_0 = X_model_MiniB[X_model_MiniB["Cluster"] == 0]
    df_Mini_C_1 = X_model_MiniB[X_model_MiniB["Cluster"] == 1]
    df_Mini_C_2 = X_model_MiniB[X_model_MiniB["Cluster"] == 2]
    print(color.BOLD + color.BLUE + 'The Min and Max of Schizophrenia in Cluster = 0 : ' + color.END)
    print(df_Mini_C_0["Schizophrenia"].min(), "and", df_Mini_C_0["Schizophrenia"].max())
    print("\n")

    print(color.BOLD + color.BLUE + 'The Min and Max of Schizophrenia in Cluster = 1 : ' + color.END)
    print(df_Mini_C_1["Schizophrenia"].min(), "and", df_Mini_C_1["Schizophrenia"].max())
    print("\n")

    print(color.BOLD + color.BLUE + 'The Min and Max of Schizophrenia in Cluster = 2 : ' + color.END)
    print(df_Mini_C_2["Schizophrenia"].min(), "and", df_Mini_C_2["Schizophrenia"].max())
    print("\n")

    print(color.BOLD + color.BLUE + 'The Min and Max of Depressive in Cluster = 0 : ' + color.END)
    print(df_Mini_C_0["Depressive"].min(), "and", df_Mini_C_0["Depressive"].max())
    print("\n")

    print(color.BOLD + color.BLUE + 'The Min and Max of Depressive in Cluster = 1 : ' + color.END)
    print(df_Mini_C_1["Depressive"].min(), "and", df_Mini_C_1["Depressive"].max())
    print("\n")

    print(color.BOLD + color.BLUE + 'The Min and Max of Depressive in Cluster = 2 : ' + color.END)
    print(df_Mini_C_2["Depressive"].min(), "and", df_Mini_C_2["Depressive"].max())
    print("\n")

    print(color.BOLD + color.BLUE + 'The Min and Max of Anxiety in Cluster = 0 : ' + color.END)
    print(df_Mini_C_0["Anxiety"].min(), "and", df_Mini_C_0["Anxiety"].max())
    print("\n")

    print(color.BOLD + color.BLUE + 'The Min and Max of Anxiety in Cluster = 1 : ' + color.END)
    print(df_Mini_C_1["Anxiety"].min(), "and", df_Mini_C_1["Anxiety"].max())
    print("\n")

    print(color.BOLD + color.BLUE + 'The Min and Max of Anxiety in Cluster = 2 : ' + color.END)
    print(df_Mini_C_2["Anxiety"].min(), "and", df_Mini_C_2["Anxiety"].max())
    print("\n")

    print(color.BOLD + color.BLUE + 'The Min and Max of Bipolar in Cluster = 0 : ' + color.END)
    print(df_Mini_C_0["Bipolar"].min(), "and", df_Mini_C_0["Bipolar"].max())
    print("\n")

    print(color.BOLD + color.BLUE + 'The Min and Max of Bipolar in Cluster = 1 : ' + color.END)
    print(df_Mini_C_1["Bipolar"].min(), "and", df_Mini_C_1["Bipolar"].max())
    print("\n")

    print(color.BOLD + color.BLUE + 'The Min and Max of Bipolar in Cluster = 2 : ' + color.END)
    print(df_Mini_C_2["Bipolar"].min(), "and", df_Mini_C_2["Bipolar"].max())
    print("\n")

    print(color.BOLD + color.BLUE + 'The Min and Max of Eating in Cluster = 0 : ' + color.END)
    print(df_Mini_C_0["Eating"].min(), "and", df_Mini_C_0["Eating"].max())
    print("\n")

    print(color.BOLD + color.BLUE + 'The Min and Max of Eating in Cluster = 1 : ' + color.END)
    print(df_Mini_C_1["Eating"].min(), "and", df_Mini_C_1["Eating"].max())
    print("\n")

    print(color.BOLD + color.BLUE + 'The Min and Max of Eating in Cluster = 2 : ' + color.END)
    print(df_Mini_C_2["Eating"].min(), "and", df_Mini_C_2["Eating"].max())
    print("\n")


def agglomerative_clustering(df, features):
    """
    聚集聚类是一种层次聚类算法。
    这是一种无监督的机器学习技术，它将种群划分为几个集群，使同一集群中的数据点更相似，而不同集群中的信息点不同。
    """
    cluster_agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
    cluster_agg.fit(df[features])
    labels = cluster_agg.labels_
    print(labels)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Schizophrenia', y='Depressive').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Schizophrenia', y='Depressive', hue=labels).set_title('With clustering')
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Depressive', y='Anxiety').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Depressive', y='Anxiety', hue=labels).set_title('With clustering')
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Anxiety', y='Bipolar').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Anxiety', y='Bipolar', hue=labels).set_title('With clustering')
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Bipolar', y='Eating').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Bipolar', y='Eating', hue=labels).set_title('With clustering')
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Eating', y='Schizophrenia').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Eating', y='Schizophrenia', hue=labels).set_title('With clustering')
    plt.show()


def birch(df, features):
    """
    使用层次结构的平衡迭代约简和聚类（BIRCH）是一种聚类算法，
    它可以通过首先生成保留尽可能多信息的大型数据集的小型紧凑摘要来对大型数据集进行聚类。
    然后对这个较小的摘要进行聚类，而不是对较大的数据集进行聚类。
    BIRCH通常用于通过创建其他聚类算法现在可以使用的数据集摘要来补充其他聚类算法。
    然而，BIRCH有一个主要缺点——它只能处理度量属性。
    度量属性是其值可以在欧几里得空间中表示的任何属性，即不应存在分类属性。
    """
    cluster_birch = Birch(branching_factor=200, threshold=1).fit(df[features])
    print(cluster_birch)
    labels_b = cluster_birch.labels_
    print(labels_b)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Schizophrenia', y='Depressive').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Schizophrenia', y='Depressive', hue=labels_b).set_title('With clustering')
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Depressive', y='Anxiety').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Depressive', y='Anxiety', hue=labels_b).set_title('With clustering')
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Anxiety', y='Bipolar').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Anxiety', y='Bipolar', hue=labels_b).set_title('With clustering')
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Bipolar', y='Eating').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Bipolar', y='Eating', hue=labels_b).set_title('With clustering')
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Eating', y='Schizophrenia').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Eating', y='Schizophrenia', hue=labels_b).set_title('With clustering')
    plt.show()


def dbscan(df, features):
    """
    从根本上讲，所有聚类方法都使用相同的方法，即首先我们计算相似性，然后使用它将数据点聚类为组或批。
    在这里，我们将重点讨论基于密度的应用程序空间聚类与噪声（DBSCAN）聚类方法。
    DBSCAN算法基于“集群”和“噪声”的直观概念。关键思想是，对于簇的每个点，给定半径的邻域必须至少包含最小数量的点。
    """
    cluster_DB = DBSCAN(eps=0.55, min_samples=4).fit(df[features])
    labels_dB = cluster_DB.labels_
    print(set(cluster_DB.labels_))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Schizophrenia', y='Depressive').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Schizophrenia', y='Depressive', hue=labels_dB).set_title('With clustering')
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Depressive', y='Anxiety').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Depressive', y='Anxiety', hue=labels_dB).set_title('With clustering')
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Anxiety', y='Bipolar').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Anxiety', y='Bipolar', hue=labels_dB).set_title('With clustering')
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Bipolar', y='Eating').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Bipolar', y='Eating', hue=labels_dB).set_title('With clustering')
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Eating', y='Schizophrenia').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Eating', y='Schizophrenia', hue=labels_dB).set_title('With clustering')
    plt.show()


def kmeans(df_data, features):
    """
    K-Means 聚类是一种无监督机器学习算法，它将未标记的数据集分组到不同的聚类中。
    任务是将这些项目分类到组中。为了实现这一点，我们将使用 K-means 算法;
    一种无监督学习算法。算法名称中的“K”表示我们要将项目分类到的组/集群的数量。
    """
    # kmeans = KMeans(init="random", n_clusters=5, n_init=10, max_iter=300, random_state=42)
    # cluster_km = kmeans.fit(df[features])
    """
    .inertia_ 属性代表的是模型训练完成后的总平方误差（Sum of Squared Errors, SSE）。
    简单来说，它是所有样本点到其所属聚类中心的距离平方和。
    这个值越大，说明聚类效果越差，因为样本点距离所属中心越远；
    反之，值越小，则说明聚类效果越好，样本点更紧密地围绕各自的聚类中心分布。
    """
    # print(cluster_km.inertia_)
    # print(cluster_km.cluster_centers_)

    kmeans_kw = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 567,
    }

    inertia = []
    for k in range(1, 10):
        kmeans_model = KMeans(n_clusters=k, **kmeans_kw)
        kmeans_model.fit(df_data[features])
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
    plt.axvline(x=3, color='b', label='axvline - full height', linestyle="dashed")
    plt.show()

    KL = KneeLocator(range(1, 10), inertia, curve="convex", direction="decreasing")
    print(KL.elbow)
    """
    轮廓系数：轮廓系数或轮廓分数是用于计算聚类技术优度的指标。其值范围为 -1 到 1。
    1：表示集群彼此相距甚远且区分清楚。
    0：表示簇是无差别的，或者我们可以说簇之间的距离不显著。
    -1：表示集群的分配方式错误。
    """
    silhouette_coefficients = []
    for k in range(2, 10):
        kmeans_model = KMeans(n_clusters=k, **kmeans_kw)
        kmeans_model.fit(df_data[features])
        score = silhouette_score(df_data[features], kmeans_model.labels_)
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

    kmeans_model = KMeans(n_clusters=3)
    kmeans_model.fit(df_data[features])

    x = df_data.iloc[:, :].values

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

    # labels_Km = kmeans.labels_
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # sns.scatterplot(ax=axes[0], data=df, x=features[2], y=features[3]).set_title('Without clustering')
    # sns.scatterplot(ax=axes[1], data=df, x=features[2], y=features[3], hue=labels_Km).set_title('With clustering')
    # plt.show()

    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # sns.scatterplot(ax=axes[0], data=df, x='Schizophrenia', y='Depressive').set_title('Without clustering')
    # sns.scatterplot(ax=axes[1], data=df, x='Schizophrenia', y='Depressive', hue=labels_Km).set_title('With clustering')
    # plt.show()
    #
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # sns.scatterplot(ax=axes[0], data=df, x='Depressive', y='Anxiety').set_title('Without clustering')
    # sns.scatterplot(ax=axes[1], data=df, x='Depressive', y='Anxiety', hue=labels_Km).set_title('With clustering')
    # plt.show()
    #
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # sns.scatterplot(ax=axes[0], data=df, x='Anxiety', y='Bipolar').set_title('Without clustering')
    # sns.scatterplot(ax=axes[1], data=df, x='Anxiety', y='Bipolar', hue=labels_Km).set_title('With clustering')
    # plt.show()
    #
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # sns.scatterplot(ax=axes[0], data=df, x='Bipolar', y='Eating').set_title('Without clustering')
    # sns.scatterplot(ax=axes[1], data=df, x='Bipolar', y='Eating', hue=labels_Km).set_title('With clustering')
    # plt.show()
    #
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # sns.scatterplot(ax=axes[0], data=df, x='Eating', y='Schizophrenia').set_title('Without clustering')
    # sns.scatterplot(ax=axes[1], data=df, x='Eating', y='Schizophrenia', hue=labels_Km).set_title('With clustering')
    # plt.show()


if __name__ == '__main__':
    # Import dataset
    Data1 = pd.read_csv("input/1- mental-illnesses-prevalence.csv")
    df = pd.DataFrame(Data1)
    df = df.rename(
        columns={
            'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'Schizophrenia',
            'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'Depressive',
            'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'Anxiety',
            'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'Bipolar',
            'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'Eating'
        }
    )
    # print(color.BOLD + color.BLUE + color.UNDERLINE + '\"The describe table of df : Mental illness dataframe\"' + color.END)
    # print(describe(df))
    """
    在这个数据集中，我们只使用了5个变量。精神分裂症、抑郁、焦虑、双相情感障碍和饮食
            variable    dtype  count  unique  missing value
    0         Entity   object   6420     214              0
    1           Code   object   6420     206            270
    2           Year    int64   6420      30              0
    3  Schizophrenia  float64   6420    6406              0
    4     Depressive  float64   6420    6416              0
    5        Anxiety  float64   6420    6417              0
    6        Bipolar  float64   6420    6385              0
    7         Eating  float64   6420    6417              0
    """
    # plotly_visualizations(df)
    # seaborn_visualizations(df)
    machine_learning_models(df)
