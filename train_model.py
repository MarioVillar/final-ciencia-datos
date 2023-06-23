# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 21:08:20 2023

@author: mario
"""

import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn import svm

from preprocess_dataset import load_train_preprocessed, preprocessDataset
from preprocess_attributes import load_test_data


TARGET_COL = "Transported"


def gridSrchSVC(train_dataset, test_dataset):
    train_target = train_dataset[TARGET_COL]
    train_dataset = train_dataset.drop([TARGET_COL], axis=1)

    test_passId = test_dataset["PassengerId"]

    # Preprocesar el dataset de test
    test_dataset = preprocessDataset(test_dataset, False, preprocess_data)

    # Hiperparametros de SVC
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}

    svc = svm.SVC()

    # Crear y entrenar en CV una svc
    clf = GridSearchCV(svc, parameters)

    clf.fit(train_dataset, train_target)

    # Realizar predicciones en test
    test_pred = clf.predict(test_dataset)

    # Guardar las predicciones en disco
    kagglePred(test_passId, test_pred, "svmPred.csv")

    return clf


def kagglePred(test_passId, test_pred, export_path):
    kaggle_pred = pd.DataFrame({'PassengerId': test_passId, 'Transported': test_pred})
    kaggle_pred.to_csv(export_path, index=False)


if __name__ == "__main__":
    train_dataset, preprocess_data = load_train_preprocessed()

    test_dataset = load_test_data()

    clf = gridSrchSVC(train_dataset, test_dataset)
