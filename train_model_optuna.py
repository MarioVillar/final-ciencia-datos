# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:33:27 2023

@author: mario
"""

import optuna
from sklearn.metrics import accuracy_score
import xgboost as xgb

import joblib

from train_utils import prepareDatasets, stratifyKFolds, makePredictions

from sklearnex import patch_sklearn
patch_sklearn()


def printBestOptuna(params, score):
    print("Best parameters:")
    for key, value in params.items():
        print(f"    {key}: {value}")
    print(f"Best score: {score}\n")


def optunaCallback(study, trial):
    # Cada cinco trials se guarda en disco
    if int(trial.number) % 20 == 0:
        joblib.dump(study, "../optuna_studies/" + study.study_name + ".pkl")

    print("\n", "-" * 40, " Finished current trial ", "-" * 40, "\n\n", sep="")


def optunaXBoost(trial):
    param = {
        'seed': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'lambda': trial.suggest_float('lambda', 0, 15.0),
        'alpha': trial.suggest_float('alpha', 0, 15.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 60, 1500),
        'max_depth': trial.suggest_categorical('max_depth', [4, 5, 6, 7, 8, 9, 10]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    accuracies = []

    for train_index, test_index in stratifyKFolds(train_dataset, train_target):
        # Obtener folds de entrenamiento y fold de validacion
        train_X, valid_X = train_dataset.iloc[train_index], train_dataset.iloc[test_index]
        train_y, valid_y = train_target.iloc[train_index], train_target.iloc[test_index]

        # Crear clasificador
        xbst = xgb.XGBClassifier(**param)

        # Entrenar el clasificador
        xbst.fit(train_X, train_y, eval_set=[(valid_X, valid_y)], verbose=0)

        # Evaluar el clasificador. Se realizan las pedicciones sobre el fold de validacion y se comparan con las targets
        accuracies.append(accuracy_score(valid_y, (xbst.predict(valid_X))))

    mean_acc = sum(accuracies) / len(accuracies)

    return mean_acc


def optunaOptimize(objective, n_trials, study_name, n_jobs=1, save_study=True):
    study = optuna.create_study(study_name=study_name, pruner=optuna.pruners.HyperbandPruner(), direction='maximize')

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[optunaCallback])

    print("\n", "-" * 102, "\n", "-" * 43, " Finished study ", "-" * 43, "\n\n", sep="")
    printBestOptuna(study.best_params, study.best_value)

    if save_study is not None:
        joblib.dump(study, "../optuna_studies/" + study_name + ".pkl")

    return study


if __name__ == "__main__":
    MODEL_NAME = "xbst"

    train_dataset, train_target, test_dataset, test_passId = prepareDatasets(model_type=MODEL_NAME)

    study = optunaOptimize(optunaXBoost, n_trials=1000, study_name=MODEL_NAME, n_jobs=6)

    study = joblib.load("../optuna_studies/xbst.pkl")

    best_params = study.best_params

    xbst = xgb.XGBClassifier(**best_params)

    # Entrenar el clasificador
    xbst.fit(train_dataset, train_target, verbose=0)

    pred = makePredictions(xbst, test_dataset, test_passId,
                           "../predicciones/xBoostOptunaPred.csv", '../modelos/xBoostOptunaModel.pkl')
