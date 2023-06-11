# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 20:01:27 2023

@author: mario
"""

import plotly.express as px

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


if __name__ == "__main__":
    import plotly.offline as poff

    df_train = load_train_data()

    # fig = null_values_heatmap(df_train)

    fig = histogram(df_train, "Destination")

    # df_train = preprocess(df_train)

    poff.plot(fig, config=dict({'scrollZoom': True}))
