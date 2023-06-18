import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# 加载和处理数据集
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    X = []
    y = []
    for line in lines:
        values = line.strip().split()
        X.append(list(map(float, values[1:])))
        y.append(float(values[0]))

    X = np.array(X)
    y = np.array(y)

    return X, y


def forward_selection(X, y):
    num_features = X.shape[1]
    selected_features = []
    best_accuracy = 0.0

    for i in range(num_features):
        best_feature = None
        for feature in range(num_features):
            if feature not in selected_features:
                current_features = selected_features + [feature]
                accuracy = evaluate(X, y, current_features)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = feature
        if best_feature is not None:
            selected_features.append(best_feature)

    return selected_features, best_accuracy


def evaluate(X, y, feature_set):
    # 使用 K 近邻算法进行分类，进行交叉验证
    knn = KNeighborsClassifier()
    X_selected = X[:, feature_set]
    scores = cross_val_score(knn, X_selected, y, cv=10, scoring='accuracy')
    accuracy = scores.mean()
    return accuracy

def backward_elimination(X, y):
    num_features = X.shape[1]
    selected_features = list(range(num_features))
    best_accuracy = evaluate(X, y, selected_features)
    #for i in range (30,70):
    #    selected_features.remove(i);
    while len(selected_features) > 2:
        worst_feature = None
        for feature in selected_features:
            current_features = selected_features.copy()
            current_features.remove(feature)
            accuracy = evaluate(X, y, current_features)
            if accuracy is not None and accuracy >= best_accuracy:
                best_accuracy = accuracy
                worst_feature = feature
        if worst_feature is not None:
            selected_features.remove(worst_feature)
        else:
            break

    return selected_features, best_accuracy




# 主程序
def main():
    print("Welcome to Feature Selection Algorithm")

    # 加载和处理数据集
    file_name = input("Type in the name of the file to test: ")
    X, y = load_dataset(file_name)

    print("Running nearest neighbor with all features, using (leaving-one-out) evaluation, I get an accuracy of:")
    accuracy = evaluate(X, y, list(range(X.shape[1])))
    print(f"Accuracy: {accuracy}")

    print("Starting feature search:")
    #"""
    # 前向特征搜索
    print("(1) Forward Selection")
    selected_features, best_accuracy = forward_selection(X, y)
    print(f"Best feature set: {selected_features}")
    print(f"Best accuracy: {best_accuracy}")
    #"""

    # 后向消除
    print("(2) Backward Elimination")
    selected_features, best_accuracy = backward_elimination(X, y)
    print(f"Best feature set: {selected_features}")
    print(f"Best accuracy: {best_accuracy}")

if __name__ == "__main__":
    main()
