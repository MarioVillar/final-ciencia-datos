# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 20:01:27 2023

@author: mario
"""

import numpy as np
from pandas.api.types import is_numeric_dtype

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from preprocess_dataset import load_train_data, preprocessDataset


def null_values_heatmap(df):
    heatmap_null = df_train.isnull().sum().to_frame()

    if "PassengerId" in heatmap_null:
        heatmap_null = heatmap_null.drop(index="PassengerId")

    if "Transported" in heatmap_null:
        heatmap_null = heatmap_null.drop(index="Transported")

    fig = px.imshow(heatmap_null.transpose(), text_auto=True,
                    title="Número de valores nulos de cada variable",
                    color_continuous_scale="teal")

    fig.update_yaxes(title='Número de valores nulos')

    fig.update_yaxes(tickvals=[])

    return fig


def histogram(df, col_name):
    # if not is_numeric_dtype(df[col_name]):
    #     df[col_name] = df[col_name].fillna("Missing value")
    # else:
    #     df[col_name] = df[col_name].fillna(0)

    fig = px.histogram(df, x=col_name,
                       # text_auto=True,
                       color="Transported",
                       # histnorm="percent",
                       title="Histograma del atributo " + col_name)

    fig.update_yaxes(title='Número de instancias')

    # fig.update_traces(texttemplate='%{y:.0f}')

    return fig


def histogramTransported(df, col_name):
    col_trans = df_train[df_train["Transported"] == True][col_name]
    col_no_trans = df_train[df_train["Transported"] == False][col_name]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=col_trans, name="Transported"))
    fig.add_trace(go.Histogram(x=col_no_trans, name="Not transported"))

    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)

    # Configurar diseño de la figura
    fig.update_layout(title='Histogramas del atributo ' + col_name)

    fig.update_yaxes(title='Número de instancias')

    return fig


def expensesBarChart(df, exp_col_names, cryoSleep_col_name):
    # Crear figura
    fig = go.Figure()

    # Crear histogramas para cada columna
    for col in exp_col_names:
        true_mean = df[col][df_train[cryoSleep_col_name] == 1].mean()
        false_mean = df[col][df_train[cryoSleep_col_name] == 0].mean()

        fig.add_trace(go.Bar(name=col, x=["False", "True"],
                             y=[false_mean, true_mean],
                             text=[false_mean, true_mean],
                             textposition='auto',
                             texttemplate='%{y:.2f}'
                             )
                      )

    # Ajustar la orientación del texto
    fig.update_traces(textangle=0)

    # Configurar diseño de la figura
    fig.update_layout(
        title='Total gastado de media en cada servicio según si el pasajero estuvo criogenizado o no',
        xaxis_title='CryoSleep',
        yaxis_title='Total gastado de media'
    )

    return fig


def expensesBoxPlots(df, exp_col_names):
    # Crear figura
    fig = go.Figure()

    # Crear box plots para cada columna
    for col in exp_col_names:
        # x_values = np.where(df['Transported'], col + ' (trans.)', col + ' (no trans.)')
        # fig.add_trace(go.Box(x=x_values, y=df[col], name=col))
        fig.add_trace(go.Box(y=df[col], name=col))

    # fig.update_layout(title='Diagramas de cajas de las variables numéricas.', boxmode='group')
    fig.update_layout(title='Diagramas de cajas de las variables numéricas.')

    return fig


def boxPlot(df, col_name):
    # Crear figura
    fig = go.Figure()

    fig.add_trace(go.Box(y=df[col_name], name=col_name, boxpoints='outliers',
                         marker=dict(color='rgb(8,81,156)', outliercolor='rgba(219, 64, 82, 0.6)',
                                     line=dict(outliercolor='rgba(219, 64, 82, 0.6)', outlierwidth=2)
                                     )
                         )
                  )

    fig.update_layout(title='Diagramas de cajas del atributo ' + col_name)

    return fig


def corrHeatmap(df):
    if "Transported" in df.columns:
        df = df.drop(["Transported"], axis=1)

    corrMatrix = df.corr()

    fig = px.imshow(corrMatrix, color_continuous_scale='RdBu_r')

    fig.update_layout(title='Matriz de correlación entre atributos del dataset.', title_x=0.5)

    return fig


def expensesHistogram(df):
    exp_col_names = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    for col in exp_col_names:
        umbral = 100
        df.loc[df[col] > umbral, col] = umbral

    # Crear figura con subplots
    fig = make_subplots(rows=3, cols=2)

    # Iterar sobre las columnas y agregar un histograma en cada subplot
    for i in range(len(exp_col_names)):
        fig.add_trace(go.Histogram(x=df[exp_col_names[i]]), row=(i // 2) + 1, col=(i % 2) + 1)
        fig.update_xaxes(title_text=exp_col_names[i], row=(i // 2) + 1, col=(i % 2) + 1)

    # Configurar diseño de la figura
    fig.update_layout(title='Histogramas de las características económicas', showlegend=False)

    return fig


def expensesScatters(df):
    exp_col_names = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Crear figura con subplots
    fig = make_subplots(rows=3, cols=2)

    # Iterar sobre las columnas y agregar un histograma en cada subplot
    for i in range(len(exp_col_names)):
        fig.add_trace(go.Scatter(x=df[exp_col_names[i]], y=df.index, mode='markers'), row=(i // 2) + 1, col=(i % 2) + 1)
        fig.update_xaxes(title_text=exp_col_names[i], row=(i // 2) + 1, col=(i % 2) + 1)

    # Configurar diseño de la figura
    fig.update_layout(title='Distribución de valores de las características económicas', showlegend=False)

    return fig


if __name__ == "__main__":
    import plotly.offline as poff

    df_train = load_train_data()

    fig = null_values_heatmap(df_train)
    poff.plot(fig, config=dict({'scrollZoom': True}))

    fig = histogram(df_train, "VIP")
    poff.plot(fig, config=dict({'scrollZoom': True}))

    fig = expensesBarChart(df_train, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], "CryoSleep")
    poff.plot(fig, config=dict({'scrollZoom': True}))

    fig = expensesBoxPlots(df_train, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])
    poff.plot(fig, config=dict({'scrollZoom': True}))

    fig = boxPlot(df_train, "Age")
    poff.plot(fig, config=dict({'scrollZoom': True}))

    fig = expensesScatters(df_train)
    poff.plot(fig, config=dict({'scrollZoom': True}))

    df_train, preprocess_data = preprocessDataset(df_train, train=True)

    fig = histogramTransported(df_train, "Age")
    poff.plot(fig, config=dict({'scrollZoom': True}))

    fig = corrHeatmap(df_train)
    poff.plot(fig, config=dict({'scrollZoom': True}))

    fig = expensesHistogram(df_train)
    poff.plot(fig, config=dict({'scrollZoom': True}))

    fig = histogram(df_train, "TotalExpense")
    poff.plot(fig, config=dict({'scrollZoom': True}))

    fig = boxPlot(df_train, "TotalExpense")
    poff.plot(fig, config=dict({'scrollZoom': True}))
