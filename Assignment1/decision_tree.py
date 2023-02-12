from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt

def dec_tree(X_train, y_train, X_test, y_test, problem_name):

    acc = []
    for i in range(1,40):
        clf = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = i)
        clf_fit = clf.fit(X_train, y_train)
        y_pred = clf_fit.predict(X_test)
        acc.append(metrics.accuracy_score(y_test, y_pred))
 
    plt.figure(figsize=(10,6))
    plt.plot(range(1,40),acc,color='blue',linestyle='dashed', marker='o',markerfacecolor='red',markersize=10)
    plt.title('Accuracy vs. Tree depth')
    plt.xlabel('L')
    plt.ylabel('Accuracy')
    plt.savefig(f'Decision_Tree_Accuracy_{problem_name}.png')
    L = acc.index(max(acc))
    est_dt = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = L)
    return max(acc), L, est_dt
