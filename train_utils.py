# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 18:25:47 2023

@author: mario
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance


from preprocess_attributes import load_test_data
from preprocess_dataset import load_train_preprocessed, preprocessDataset


TARGET_COL = "Transported"


def prepareDatasets(model_type=None):
    train_dataset, preprocess_data = load_train_preprocessed(model_type=model_type)

    test_dataset = load_test_data()

    train_target = train_dataset[TARGET_COL]
    train_dataset = train_dataset.drop([TARGET_COL], axis=1)

    test_passId = test_dataset["PassengerId"]

    # Preprocesar el dataset de test
    test_dataset = preprocessDataset(test_dataset, False, preprocess_data, model_type=model_type)

    return train_dataset, train_target, test_dataset, test_passId


def stratifyKFolds(train_dataset, train_target):
    skf = StratifiedKFold()
    return skf.split(train_dataset, train_target)


def transportedToBool(df):
    columna_transported = df['Transported']

    # Comprobar si la columna es booleana
    if columna_transported.dtype != bool and np.issubdtype(columna_transported.dtype, np.number):
        # Aproximar los valores a 0 o 1
        columna_transported = columna_transported.apply(lambda x: 0 if x < 0.5 else 1)

        # Castear la columna a booleana
        df['Transported'] = columna_transported.astype(bool)

    return df


def makePredictions(clf, test_dataset, test_passId, output_pred_path=None, output_model_path=None):
    if hasattr(clf, 'best_score_'):
        print("\nBest score in CV: ", clf.best_score_)

    # Realizar predicciones en test
    test_pred = clf.predict(test_dataset)

    # Guardar las predicciones en disco
    kaggle_pred = pd.DataFrame({'PassengerId': test_passId, 'Transported': test_pred})

    kaggle_pred = transportedToBool(kaggle_pred)

    if output_pred_path is not None:
        kaggle_pred.to_csv(output_pred_path, index=False)

    if output_model_path is not None:
        joblib.dump(clf, output_model_path)

    return kaggle_pred


def featureImportance(clf, train_dataset, train_target):
    # Obtén las importancias de permutación
    result = permutation_importance(clf, train_dataset, train_target, n_repeats=10, random_state=42)

    # Obtén las características importantes y sus puntajes
    importance_scores = result.importances_mean
    feature_names = clf.feature_names_in_  # Inserta aquí los nombres de las características utilizadas en el modelo

    # Ordena las características por su importancia
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_feature_names = [feature_names[idx] for idx in sorted_indices]
    sorted_scores = importance_scores[sorted_indices]

    # Imprime los resultados
    for feature_name, score in zip(sorted_feature_names, sorted_scores):
        print(f"{feature_name}: {score}")
