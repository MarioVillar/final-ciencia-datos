# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:55:28 2023

@author: mario
"""

import pandas as pd


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


# Preprocesar los valores de Cabin
def preprocessCabin(df):
    # Dividir en DeckNumber, CabinNumber y Stribor los valores de Cabin
    df[['DeckNumber', 'CabinNumber', 'Stribor']] = df['Cabin'].str.split('/', expand=True)

    # Cada cubierta se mapea a un numero diferente. La de arriba es la A y la de abajo es la T
    deck_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7}
    df["DeckNumber"] = df["DeckNumber"].map(deck_mapping)

    stribor_mapping = {"S": 1, "P": 0}
    df["Stribor"] = df["Stribor"].map(stribor_mapping)

    df["CabinNumber"] = df["CabinNumber"].astype(float)

    df = df.drop(["Cabin"], axis=1)

    return df


def preprocessVip(df):
    # df["VIP"] = df["VIP"].replace({True: 1, False: 0})
    df = df.drop(["VIP"], axis=1)
    return df


def preprocessName(df):
    alphabet_mapping = {letter: index for index, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}

    df["NameLength"] = df["Name"].str.len()

    df["NameInitial"] = df['Name'].str.split().str[0].str[0].str.upper().map(alphabet_mapping)

    df["SurnameInitial"] = df['Name'].str.split().str[1].str[0].str.upper().map(alphabet_mapping)

    df = df.drop(["Name"], axis=1)

    return df


def preprocessExpenses(df):
    df["TotalExpense"] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']

    df = df.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1)

    return df


# Convertir los NaN en cero cuando el pasajero estuvo en CryoSleep
def zeroExpensesCryosleep(df):
    exp_col_names = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    for col in exp_col_names:
        df.loc[df["CryoSleep"] == True, col] = 0
        df.loc[df["CryoSleep"].isna(), col] = 0

    return df


# Preprocesar todos los atributos de un DF
def preprocessAttributes(df):
    # Preprocesar PassengerId
    df = preprocessId(df)

    # Preprocesar CryoSleep
    df = preprocessCryoSleep(df)

    # Preprocesar Cabin
    df = preprocessCabin(df)

    # Preprocesar VIP
    df = preprocessVip(df)

    # Preprocesar Name
    df = preprocessName(df)

    df = zeroExpensesCryosleep(df)

    # Preprocesar caracteristicas numericas
    df = preprocessExpenses(df)

    return df
