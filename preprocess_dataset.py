# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 17:00:55 2023

@author: mario
"""

from preprocess_attributes import preprocessAttributes, load_train_data, load_test_data

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding


# Entrenar y aplicar One Hot Encoding a una columna particular. Se guarda el encoder en
#   preprocess_data con la key "<col_name>Enc"
def fitTransformOneHotEnc(df, col_name, preprocess_data):
    enc = OneHotEncoder(handle_unknown="ignore")

    df_enc = df[col_name].to_frame()

    df_enc = df_enc.fillna("ValorDesconocido")

    enc.fit(df_enc)

    df = transformOneHotEnc(df, col_name, enc)

    preprocess_data[col_name + "Enc"] = enc

    return df, preprocess_data


# Aplicar One Hot Encoding a una columna particular
def transformOneHotEnc(df, col_name, enc):
    df_enc = df[col_name].to_frame()

    df_enc = df_enc.fillna("ValorDesconocido")

    array_enc = enc.transform(df_enc).toarray()

    df_enc = pd.DataFrame(array_enc, columns=enc.categories_[0])

    nonNan_cat = enc.categories_[0]
    nonNan_cat = np.delete(nonNan_cat, np.where(nonNan_cat == "ValorDesconocido"))

    df_enc.loc[df_enc["ValorDesconocido"] == 1, nonNan_cat] = np.nan

    df = pd.concat([df, df_enc], axis=1)

    df = df.drop(["ValorDesconocido"], axis=1)

    df = df.drop([col_name], axis=1)

    return df


# Hacer One Hot Encoding de todas las variables categoricas. En modo entrenamiento se hace un
#   fit de los encoders y se guardan en preprocess_data; en test se utilizand los encoders
#   de preprocess_data para realizar el One Hot Encoding
def oneHotEncoding(df, preprocess_data, train):
    if train:
        # One Hot Encoding de HomePlanet
        df, preprocess_data = fitTransformOneHotEnc(df, "HomePlanet", preprocess_data)

        # One Hot Encoding de Destination
        df, preprocess_data = fitTransformOneHotEnc(df, "Destination", preprocess_data)
    else:
        # One Hot Encoding de HomePlanet
        df = transformOneHotEnc(df, "HomePlanet", preprocess_data["HomePlanetEnc"])

        # One Hot Encoding de Destination
        df = transformOneHotEnc(df, "Destination", preprocess_data["DestinationEnc"])

    return df, preprocess_data


# Imputar valores perdidos mediante un KNN
def knnImputer(df, preprocess_data, train):
    if train:
        imputer = KNNImputer(n_neighbors=2, weights="uniform")

        df.loc[:, :] = imputer.fit_transform(df)

        preprocess_data["KnnImputer"] = imputer
    else:
        df.loc[:, :] = preprocess_data["KnnImputer"].transform(df)

    # Redondear a 0 o 1 los valores de las caracteristicas enteras
    int_charac = ["CryoSleep", "DeckNumber", "CabinNumber", "Stribor",
                  "NameLength", "NameInitial", "SurnameInitial",
                  "Earth", "Europa", "Mars",
                  "55 Cancri e", "PSO J318.5-22", "TRAPPIST-1e"]

    for col in int_charac:
        df[col] = df[col].round()

    return df, preprocess_data


# Detectar y eliminar los outliers del dataset
def removeOutliers(df, method="lof"):
    df_outlier_col = df[["Age", "TotalExpense"]]

    outlier_mask = None

    # LOF detection
    if method == "lof":
        clf = LocalOutlierFactor(n_neighbors=2, contamination=0.06)
        outlier_mask = clf.fit_predict(df_outlier_col)
    # Isolation Forest detection
    elif method == "isof":
        isof = IsolationForest(random_state=0, contamination=0.06)
        outlier_mask = isof.fit_predict(df_outlier_col)
    elif method == "elip":
        ellenv = EllipticEnvelope(random_state=0, contamination=0.06)
        outlier_mask = ellenv.fit_predict(df_outlier_col)

    if outlier_mask is not None:
        df = df[outlier_mask == 1]

    return df


# Escalar las caracteristicas numericas del dataset. Permite normalizarlas o estandarizarlas
def scaleAtributtes(df, preprocess_data, train, method="norm"):
    # Solo se estandarizan las columnas numericas
    scaleCols = ["PassengerId", "Age", "DeckNumber", "CabinNumber", "NameLength",
                 "NameInitial", "SurnameInitial", "TotalExpense"]

    if train:
        # Crear, entrenar y aplicar el Standarizado a los datos de entrenamiento
        if method == "std":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        df.loc[:, scaleCols] = scaler.fit_transform(df[scaleCols])

        # Guardar el scaler
        preprocess_data["Scaler"] = scaler
    else:
        # Aplicar el scaler (ya entrenado) a los datos de test
        df.loc[:, scaleCols] = preprocess_data["Scaler"].transform(df[scaleCols])

    return df, preprocess_data


# Preprocesar todo el df
def preprocessDataset(df, train, preprocess_data={}, model_type=None):
    if not train and ("HomePlanetEnc" not in preprocess_data or "DestinationEnc" not in preprocess_data
                      or "KnnImputer" not in preprocess_data):
        # or "Scaler" not in preprocess_data
        raise TypeError("En test se debe proporcionar HomePlanetEnc, DestinationEnc en preprocess_data")

    if train:
        df = shuffle_df(df)

    transported_col = None

    # Extraer la columna "Transported"
    if "Transported" in df.columns:
        transported_col = df.pop('Transported')

    # Aplicar el preprocesamiento de los diferentes atributos del dataset
    df = preprocessAttributes(df)

    # Realizar One Hot Encoding de las variables categoricas
    df, preprocess_data = oneHotEncoding(df, preprocess_data, train)

    # Imputar valores perdidos mediante KNN
    df, preprocess_data = knnImputer(df, preprocess_data, train)

    # En entrenamiento se eliminan outliers
    if train:
        df = removeOutliers(df)

    # Estandarizar los datos numericos
    df, preprocess_data = scaleAtributtes(df, preprocess_data, train, method="std")

    # Eliminar las caracteristicas que no tienen mucho sentido
    df = df.drop(["PassengerId", "NameLength", "NameInitial", "SurnameInitial"], axis=1)

    # Eliminar caracteristicas con alta correlacion con otras ('55 Cancri e' y 'Europa' respectivamente)
    df = df.drop(["TRAPPIST-1e", "DeckNumber"], axis=1)

    # Insertar la columna "Transported" al final del DataFrame
    if transported_col is not None:
        df.insert(len(df.columns), 'Transported', transported_col)

    if train:
        return df, preprocess_data
    else:
        return df


def shuffle_df(df):
    indices_aleatorios = np.random.permutation(df.index)
    df_reordenado = df.reindex(indices_aleatorios).reset_index(drop=True)
    return df_reordenado


def load_train_preprocessed(model_type=None):
    df_train = load_train_data()
    return preprocessDataset(df_train, train=True, model_type=model_type)


if __name__ == "__main__":
    df_train = load_train_data()

    df_train, preprocess_data = preprocessDataset(df_train, train=True)

    df_test = load_test_data()

    df_test = preprocessDataset(df_test, False, preprocess_data)
