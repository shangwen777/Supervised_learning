from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

def knn(X_train, y_train, X_test, y_test, problem_name):
    acc = []
    for i in range(1,40):
        neigh = KNeighborsClassifier(n_neighbors = i)
        est_fit = neigh.fit(X_train, y_train)
        yhat = est_fit.predict(X_test)
        acc.append(metrics.accuracy_score(y_test, yhat))

    plt.figure(figsize=(10,6))
    plt.plot(range(1,40),acc,color='blue',linestyle='dashed', marker='o',markerfacecolor='red',markersize=10)
    plt.title('Accuracy vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.savefig(f'KNN_Accuracy_{problem_name}.png')
    K = acc.index(max(acc))
    est_knn = KNeighborsClassifier(n_neighbors = K)
 
    return max(acc), K, est_knn

