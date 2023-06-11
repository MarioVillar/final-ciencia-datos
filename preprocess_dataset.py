# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 17:00:55 2023

@author: mario
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def load_train_data():
    return pd.read_csv("../dataset/train.csv")


def load_test_data():
    return pd.read_csv("../dataset/test.csv")


# Preprocesar los valores de PassengerId
def preprocessId(df):
    # Función de transformación
    def convertir_formato(valor):
        gggg, pp = valor.split('_')
        return int(gggg) * 10 + int(pp)

    # Aplicar la función de transformación a la serie
    df["PassengerId"] = df["PassengerId"].apply(convertir_formato)

    return df


# Preprocesar los valores de CryoSleep
def preprocessCryoSleep(df):
    df["CryoSleep"] = df["CryoSleep"].replace({True: 1, False: 0})
    return df


# Entrenar y aplicar One Hot Encoding mediante un encoder a una columna particular
def fitTransformOneHotEnc(df, col_name, preprocess_data):
    enc = OneHotEncoder(handle_unknown='ignore')

    df_enc = df[col_name].to_frame()

    df_enc = df_enc.fillna("ValorDesconocido")

    enc.fit(df_enc)

    df = transformOneHotEnc(df, col_name, enc)

    df = df.drop([col_name], axis=1)

    preprocess_data[col_name + "Enc"] = enc

    return df, preprocess_data


# Aplicar One Hot Encoding mediante un encoder a una columna particular
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

    return df


# Hacer One Hot Encoding de todas las variables categoricas
def oneHotEncoding(df, preprocess_data):
    # One Hot Encoding de HomePlanet
    df, preprocess_data = fitTransformOneHotEnc(df, "HomePlanet", preprocess_data)

    # One Hot Encoding de Destination
    df, preprocess_data = fitTransformOneHotEnc(df, "Destination", preprocess_data)

    return df, preprocess_data


# Preprocesar todo el df
def preprocess(df):
    preprocess_data = {}

    # Preprocesar PassengerId
    df = preprocessId(df)

    # Preprocesar CryoSleep
    df = preprocessCryoSleep(df)

    # Realizar One Hot Encoding de las variables categoricas
    df, preprocess_data = oneHotEncoding(df, preprocess_data)

    return df, preprocess_data


# def load_train_preprocessed():
#     df_train = load_train_data()
#     return preprocess(df_train)


if __name__ == "__main__":
    df_train = load_train_data()

    df_train, preprocess_data = preprocess(df_train)

    # print(df_train)
