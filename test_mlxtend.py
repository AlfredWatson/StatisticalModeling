import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def testv1():
    """
    在这个例子中，我们首先定义了一组交易数据，然后使用TransactionEncoder将其转换为适用于apriori函数的形式。
    apriori函数用于发现频繁项集，这里我们设定最小支持度为0.5（即项集出现的频率至少为50%）。
    之后，我们使用association_rules函数根据频繁项集生成关联规则，并设置了最小提升度阈值为1（仅显示提升度大于或等于1的规则）。
    最后，打印出生成的关联规则。
    关联规则通常表现为形如"A -> B"的形式，其中的支持度、置信度和提升度等指标可以反映规则的重要性。
    """
    # 假设我们有一个交易数据，格式如下：
    transactions = [
        ['Milk', 'Bread', 'Butter'],
        ['Bread', 'Butter'],
        ['Milk', 'Bread', 'Cheese'],
        ['Eggs'],
        ['Milk', 'Cheese', 'Bread'],
        ['Cheese', 'Bread']
    ]

    # 将交易数据转换为TransactionEncoder所需的格式
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    print(df)
    """
       Bread  Butter  Cheese   Eggs   Milk
    0   True    True   False  False   True
    1   True    True   False  False  False
    2   True   False    True  False   True
    3  False   False   False   True  False
    4   True   False    True  False   True
    5   True   False    True  False  False
    """

    # 使用Apriori算法找出频繁项集
    frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
    print(frequent_itemsets)
    """
        support         itemsets
    0  0.833333          (Bread)
    1  0.500000         (Cheese)
    2  0.500000           (Milk)
    3  0.500000  (Bread, Cheese)
    4  0.500000    (Bread, Milk)
    """

    # 根据频繁项集生成关联规则
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    print(rules)
    """
      antecedents consequents  ...  conviction  zhangs_metric
    0    (Cheese)     (Bread)  ...         inf       0.333333
    1     (Bread)    (Cheese)  ...        1.25       1.000000
    2      (Milk)     (Bread)  ...         inf       0.333333
    3     (Bread)      (Milk)  ...        1.25       1.000000
    
    [4 rows x 10 columns]
    """


if __name__ == '__main__':
    testv1()
