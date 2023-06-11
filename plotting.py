# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 20:01:27 2023

@author: mario
"""

import plotly.express as px
import plotly.graph_objects as go

from preprocess_dataset import load_train_data, preprocess


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
    df[col_name] = df[col_name].fillna("Missing value")

    fig = px.histogram(df, x=col_name, text_auto=True,
                       # color="Transported",
                       histnorm="percent",
                       title="Histograma del atributo " + col_name)

    fig.update_yaxes(title='Número de instancias')

    fig.update_traces(texttemplate='%{y:.0f}')

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


if __name__ == "__main__":
    import plotly.offline as poff

    df_train = load_train_data()

    # fig = null_values_heatmap(df_train)

    # fig = histogram(df_train, "Destination")

    fig = expensesHist(df_train, [
                       'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], "CryoSleep")

    # df_train = preprocess(df_train)

    poff.plot(fig, config=dict({'scrollZoom': True}))
