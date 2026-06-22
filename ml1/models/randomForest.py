import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def tree_config(model):
    depths = [tree.get_depth() for tree in model.estimators_]
    leaves = [tree.get_n_leaves() for tree in model.estimators_]
    nodes = [tree.tree_.node_count for tree in model.estimators_]
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(depths, bins=range(min(depths), max(depths) + 2), 
             edgecolor='black', align='left')
    plt.xlabel('Глубина дерева')
    plt.ylabel('Количество деревьев')
    plt.title('Распределение глубин деревьев')
    plt.xticks(range(min(depths), max(depths) + 1))

    plt.subplot(1, 3, 2)
    plt.hist(leaves, bins=20, edgecolor='black', color='green')
    plt.xlabel('Количество листьев')
    plt.ylabel('Количество деревьев')
    plt.title('Распределение листьев')

    plt.subplot(1, 3, 3)
    plt.hist(nodes, bins=20, edgecolor='black', color='orange')
    plt.xlabel('Количество узлов')
    plt.ylabel('Количество деревьев')
    plt.title('Распределение узлов')

    plt.tight_layout()
    plt.show()

    
def predict_randomforest(train_data, train_target, test_data):
    forest = RandomForestClassifier(max_features='log2', n_estimators=150, random_state=42, n_jobs=-1)
    model = forest.fit(train_data, train_target)
    prediction_target = model.predict(test_data)
    return prediction_target, model
