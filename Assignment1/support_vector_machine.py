from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def svm(X_train, y_train, X_test, y_test):
    clf = SVC(kernel='linear')
    clf_fit = clf.fit(X_train, y_train)
    y_pred = clf_fit.predict(X_test)

    return accuracy_score(y_test, y_pred), clf