# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 17:00:55 2023

@author: mario
"""

from preprocess_attributes import preprocessAttributes, load_train_data, reallocateTransported, zeroExpensesCryosleep

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer


# Entrenar y aplicar One Hot Encoding a una columna particular. Se guarda el encoder en
#   preprocess_data con la key "<col_name>Enc"
def fitTransformOneHotEnc(df, col_name, preprocess_data):
    enc = OneHotEncoder(handle_unknown='ignore')

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
def oneHotEncoding(df, preprocess_data, train=True):
    if train:
        # One Hot Encoding de HomePlanet
        df, preprocess_data = fitTransformOneHotEnc(df, "HomePlanet", preprocess_data)

        # One Hot Encoding de Destination
        df, preprocess_data = fitTransformOneHotEnc(df, "Destination", preprocess_data)
    else:
        # One Hot Encoding de HomePlanet
        df = fitTransformOneHotEnc(df, "HomePlanet", preprocess_data["HomePlanetEnc"])

        # One Hot Encoding de Destination
        df = fitTransformOneHotEnc(df, "Destination", preprocess_data["DestinationEnc"])

    return df, preprocess_data


# Imputar valores perdidos mediante un KNN
def knnImputer(df, preprocess_data={}, train=True):
    if train:
        # Quitar la columna transported
        df_train = df.drop(["Transported"], axis=1)

        imputer = KNNImputer(n_neighbors=2, weights="uniform")

        df_train = imputer.fit_transform(df_train)

        df[df.columns[:-1]] = df_train

        preprocess_data["KnnImputer"] = imputer
    else:
        df = preprocess_data["KnnImputer"].transform(df)

    # Redondear a 0 o 1 los valores de las caracteristicas enteras
    int_charac = ["CryoSleep", "DeckNumber", "CabinNumber", "Stribor",
                  "VIP", "NameLength", "NameInitial", "SurnameInitial",
                  "Earth", "Europa", "Mars",
                  "55 Cancri e", "PSO J318.5-22", "TRAPPIST-1e"]

    for col in int_charac:
        df[col] = df[col].round()

    return df, preprocess_data


# Preprocesar todo el df
def preprocessDataset(df, preprocess_data={}, train=True):
    if not train and ("HomePlanetEnc" not in preprocess_data or "DestinationEnc" not in preprocess_data):
        raise TypeError("En test se debe proporcionar HomePlanetEnc, DestinationEnc en preprocess_data")

    df = preprocessAttributes(df)

    # Realizar One Hot Encoding de las variables categoricas
    df, preprocess_data = oneHotEncoding(df, preprocess_data, train)

    # Reposicionar al final la columna transported (si existe, por lo que solo en train)
    df = reallocateTransported(df)

    # Imputar valores perdidos mediante KNN
    df, preprocess_data = knnImputer(df, preprocess_data, train)

    # Cuando el pasajero estuvo en cryosleep el gasto es cero
    df = zeroExpensesCryosleep(df)

    return df, preprocess_data


# def load_train_preprocessed():
#     df_train = load_train_data()
#     return preprocess(df_train)


if __name__ == "__main__":
    df_train = load_train_data()

    df_train, preprocess_data = preprocessDataset(df_train)

    # print(df_train)
