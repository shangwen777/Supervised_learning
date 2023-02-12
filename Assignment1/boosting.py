from sklearn.metrics import classification_report
#from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score


def boost(X_train, y_train, X_test, y_test):
    gradient_booster = GradientBoostingClassifier(learning_rate=0.1)
    gb_fit = gradient_booster.fit(X_train, y_train)
    print(classification_report(y_test, gb_fit.predict(X_test)))


    return gb_fit.score(X_test, y_test), gradient_booster

def hist_boost(X_train, y_train, X_test, y_test):
    histogram_gradient_boosting = HistGradientBoostingClassifier()
    clf_fit = histogram_gradient_boosting.fit(X_train, y_train)
    y_pred = clf_fit.predict(X_test)
    accu = accuracy_score(y_test, y_pred)
    return accu, histogram_gradient_boosting
   



