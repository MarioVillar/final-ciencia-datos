# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 21:08:20 2023

@author: mario
"""

from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb

from train_utils import prepareDatasets, makePredictions

LINUX_GPU = False

if not LINUX_GPU:
    from sklearnex import patch_sklearn
    patch_sklearn()


def gridSrchSVC(train_dataset, train_target, test_dataset, test_passId):
    # Hiperparametros de SVC
    hyperparameters = {'kernel': ['rbf'], 'C': [0.1, 1], "gamma": [0.1, 1]}
    # hyperparameters = {'kernel': ['poly'], 'C': [10], "gamma": [1]}

    svc = svm.SVC()

    # Crear y entrenar en CV una svc
    clf = GridSearchCV(svc, hyperparameters, verbose=3,
                       scoring="accuracy", return_train_score=True)

    clf.fit(train_dataset, train_target)

    # Realizar predicciones en test
    makePredictions(clf, test_dataset, test_passId,
                    "../predicciones/svmPred.csv",
                    '../modelos/svmModel.pkl')
    return clf


def gridSrchKNNC(train_dataset, train_target, test_dataset, test_passId):
    # Hiperparametros a probar
    leaf_size = list(range(10, 50, 5))
    n_neighbors = list(range(5, 30, 5))
    p = [1, 2]

    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

    # Crear y entrenar en CV un clasificador Knn
    knn = KNeighborsClassifier()

    clf = GridSearchCV(knn, hyperparameters, verbose=3,
                       scoring="accuracy", return_train_score=True)

    clf.fit(train_dataset, train_target)

    # Realizar predicciones en test
    makePredictions(clf, test_dataset, test_passId,
                    "../predicciones/knnPred.csv",
                    '../modelos/knnModel.pkl')
    return clf


def gridSrchGradBoost(train_dataset, train_target, test_dataset, test_passId):
    # Hiperparametros a probar
    loss = ["log_loss"]  # "exponential"
    learning_rate = [0.005, 0.01, 0.1]
    n_estimators = list(range(50, 100, 10)) + list(range(100, 301, 100))
    max_depth = list(range(3, 10, 1))

    hyperparameters = dict(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                           max_depth=max_depth)

    # Crear y entrenar en CV un clasificador GradientBoostingClassifier
    grdbst = GradientBoostingClassifier()

    clf = GridSearchCV(grdbst, hyperparameters, verbose=3,
                       scoring="accuracy", return_train_score=True)

    clf.fit(train_dataset, train_target)

    # Realizar predicciones en test
    makePredictions(clf, test_dataset, test_passId,
                    "../predicciones/gradientBoostingLogLossPred.csv",
                    '../modelos/gradientBoostingLogLossModel.pkl')
    return clf


def gridSrchAdaBoost(train_dataset, train_target, test_dataset, test_passId):
    # Hiperparametros a probar
    learning_rate = [0.1, 0.01]
    n_estimators = [100, 200]

    hyperparameters = dict(learning_rate=learning_rate,
                           n_estimators=n_estimators)

    # Crear y entrenar en CV un clasificador AdaBoost
    adabst = AdaBoostClassifier()

    clf = GridSearchCV(adabst, hyperparameters, verbose=3,
                       scoring="accuracy", return_train_score=True)

    clf.fit(train_dataset, train_target)

    # Realizar predicciones en test
    makePredictions(clf, test_dataset, test_passId,
                    "../predicciones/adaBoostPred.csv",
                    '../modelos/adaBoostModel.pkl')
    return clf


def gridSrchLogReg(train_dataset, train_target, test_dataset, test_passId):
    # Hiperparametros a probar
    penalty = ["l1", "l2"]
    C = [0.01, 0.05, 0.1, 0.5]
    solver = ["liblinear", "newton-cholesky", "saga"]

    hyperparameters = dict(penalty=penalty, C=C, solver=solver)

    # Crear y entrenar en CV un clasificador Logisitic Regression
    logreg = LogisticRegression()

    clf = GridSearchCV(logreg, hyperparameters, verbose=3,
                       scoring="accuracy", return_train_score=True)

    clf.fit(train_dataset, train_target)

    # Realizar predicciones en test
    makePredictions(clf, test_dataset, test_passId,
                    "../predicciones/logRegPred.csv",
                    '../modelos/logRegModel.pkl')
    return clf


def gridSrchXBoost(train_dataset, train_target, test_dataset, test_passId):
    # Hiperparametros a probar
    learning_rate = [0.01, 0.5, 0.1]  # [0.06672065863100594]
    reg_lambda = [1, 10, 15]  # [3.0610042624477543]
    reg_alpha = [1, 10, 15]  # [4.581902571574289]
    colsample_bytree = [0.95, 1]  # [0.9241969052729379]
    subsample = [0.95, 1]  # [0.9527591724824661]
    n_estimators = list(range(100, 1100, 100))
    max_depth = [6, 8, 10]

    hyperparameters = dict(reg_lambda=reg_lambda, reg_alpha=reg_alpha, colsample_bytree=colsample_bytree,
                           subsample=subsample, learning_rate=learning_rate, n_estimators=n_estimators,
                           max_depth=max_depth)

    # Crear y entrenar en CV un clasificador XBoost
    if not LINUX_GPU:
        xbst = xgb.XGBClassifier(objective="binary:logistic", seed=0)
    else:
        xbst = xgb.XGBClassifier(objective="binary:logistic", seed=0,
                                 tree_method='gpu_hist', gpu_id=0)

    clf = GridSearchCV(xbst, hyperparameters, verbose=3,
                       scoring="accuracy", return_train_score=True, n_jobs=-1)

    clf.fit(train_dataset, train_target)

    # Realizar predicciones en test
    makePredictions(clf, test_dataset, test_passId,
                    "../predicciones/xBoostPred.csv", '../modelos/xBoostModel.pkl')
    return clf


def finalModelPred(clf, train_dataset, train_target, test_dataset, test_passId, output_pred_path):
    clf.fit(train_dataset, train_target)

    # Realizar predicciones en test
    makePredictions(clf, test_dataset, test_passId, output_pred_path)

    return clf


if __name__ == "__main__":
    train_dataset, train_target, test_dataset, test_passId = prepareDatasets()

    # clf = gridSrchSVC(train_dataset, train_target, test_dataset, test_passId)

    # clf = gridSrchKNNC(train_dataset, train_target, test_dataset, test_passId)

    # clf = gridSrchGradBoost(train_dataset, train_target, test_dataset, test_passId)

    # clf = gridSrchAdaBoost(train_dataset, train_target, test_dataset, test_passId)

    # clf = gridSrchLogReg(train_dataset, train_target, test_dataset, test_passId)

    # clf = gridSrchXBoost(train_dataset, train_target, test_dataset, test_passId)

    clf = xgb.XGBClassifier(objective="binary:logistic", seed=0, colsample_bytree=1, learning_rate=0.1,
                            max_depth=6, n_estimators=200, reg_alpha=10, reg_lambda=10, subsample=1)
