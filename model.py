import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# Logistic Regression
def logisticRegression(X_train, Y_train, X_test, passengerId):
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    print(acc_log)
    results = pd.DataFrame({'PassengerId': passengerId,
                            'Survived': Y_pred})
    results.to_csv("predict.csv", index=False)


# Support Vector Machines
def svm(X_train, Y_train, X_test, passengerId):
    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
    print(acc_svc)
    results = pd.DataFrame({'PassengerId': passengerId,
                            'Survived': Y_pred})
    results.to_csv("predict.csv", index=False)


# KNN
def knn(X_train, Y_train, X_test, passengerId):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
    print(acc_knn)
    results = pd.DataFrame({'PassengerId': passengerId,
                            'Survived': Y_pred})
    results.to_csv("predict.csv", index=False)


# Gaussian Naive Bayes
def gaussianNaiveBayes(X_train, Y_train, X_test, passengerId):
    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
    print(acc_gaussian)
    results = pd.DataFrame({'PassengerId': passengerId,
                            'Survived': Y_pred})
    results.to_csv("predict.csv", index=False)


# Perceptron
def perceptron(X_train, Y_train, X_test, passengerId):
    perceptron = Perceptron()
    perceptron.fit(X_train, Y_train)
    Y_pred = perceptron.predict(X_test)
    acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
    print(acc_perceptron)
    results = pd.DataFrame({'PassengerId': passengerId,
                            'Survived': Y_pred})
    results.to_csv("predict.csv", index=False)


# Linear SVC
def linearSVC(X_train, Y_train, X_test, passengerId):
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, Y_train)
    Y_pred = linear_svc.predict(X_test)
    acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
    print(acc_linear_svc)
    results = pd.DataFrame({'PassengerId': passengerId,
                            'Survived': Y_pred})
    results.to_csv("predict.csv", index=False)


# Stochastic Gradient Descent
def sgd(X_train, Y_train, X_test, passengerId):
    sgd = SGDClassifier()
    sgd.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
    print(acc_sgd)
    results = pd.DataFrame({'PassengerId': passengerId,
                            'Survived': Y_pred})
    results.to_csv("predict.csv", index=False)


# Decision Tree
def decisionTree(X_train, Y_train, X_test, passengerId):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
    print(acc_decision_tree)
    results = pd.DataFrame({'PassengerId': passengerId,
                            'Survived': Y_pred})
    results.to_csv("predict.csv", index=False)


# Random Forest
def RF(X_train, Y_train, X_test, passengerId):
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print(acc_random_forest)
    results = pd.DataFrame({'PassengerId': passengerId,
                            'Survived': Y_pred})
    results.to_csv("predict.csv", index=False)

