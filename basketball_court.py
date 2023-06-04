# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 16:53:30 2023

@author: Mario Villar Sanz
"""


import plotly.offline as poff
import plotly.graph_objs as go


def full_court():
    # Create figure
    fig = go.Figure()

    # Court
    fig.add_trace(
        go.Scatter(x=[-50, 890, 890, -50, -50],
                   y=[-250, -250, 250, 250, -250],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Left basket
    fig.add_trace(
        go.Scatter(x=[-10, -10],
                   y=[-30, 30],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Right basket
    fig.add_trace(
        go.Scatter(x=[850, 850],
                   y=[-30, 30],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Left inside box
    fig.add_trace(
        go.Scatter(x=[-50, 140, 140, -50, -50],
                   y=[-80, -80, 80, 80, -80],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Left outside box
    fig.add_trace(
        go.Scatter(x=[-50, 140, 140, -50, -50],
                   y=[-60, -60, 60, 60, -60],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Right inside box
    fig.add_trace(
        go.Scatter(x=[700, 890, 890, 700, 700],
                   y=[-80, -80, 80, 80, -80],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Right outside box
    fig.add_trace(
        go.Scatter(x=[700, 890, 890, 700, 700],
                   y=[-60, -60, 60, 60, -60],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Left top 3-pointer straight line
    fig.add_trace(
        go.Scatter(x=[-50, 90],
                   y=[220, 220],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Left bottom 3-pointer straight line
    fig.add_trace(
        go.Scatter(x=[-50, 90],
                   y=[-220, -220],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Right top 3-pointer straight line
    fig.add_trace(
        go.Scatter(x=[750, 890],
                   y=[220, 220],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Right bottom 3-pointer straight line
    fig.add_trace(
        go.Scatter(x=[750, 890],
                   y=[-220, -220],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Half court line
    fig.add_trace(
        go.Scatter(x=[420, 420],
                   y=[-250, 250],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Left ring
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=11, y0=7.5, x1=-4, y1=-7.5,
                  line={"color": "rgba(0,0,0,1)", "width": 1},
                  )

    # Right ring
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=844, y0=7.5, x1=829, y1=-7.5,
                  line={"color": "rgba(0,0,0,1)", "width": 1},
                  )

    # Left free-throw circle
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=200, y0=60, x1=80, y1=-60,
                  line={"color": "rgba(0,0,0,1)", "width": 1},
                  )

    # Right free-throw circle
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=760, y0=60, x1=640, y1=-60,
                  line={"color": "rgba(0,0,0,1)", "width": 1},
                  )

    # Half court outside circle
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=480, y0=60, x1=360, y1=-60,
                  line={"color": "rgba(0,0,0,1)", "width": 1},
                  )

    # Half court inside circle
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=440, y0=20, x1=400, y1=-20,
                  line={"color": "rgba(0,0,0,1)", "width": 1},
                  )

    # Left 3-pointer half circle
    fig.add_shape(type="path",
                  path="M 90,220 Q 400,0 90,-220",
                  line={"color": "rgba(0,0,0,1)", "width": 1},
                  )

    # Right 3-pointer half circle
    fig.add_shape(type="path",
                  path="M 750,220 Q 440,0 750,-220",
                  line={"color": "rgba(0,0,0,1)", "width": 1},
                  )

    fig.update_xaxes(
        constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )

    fig.update_layout(plot_bgcolor='white')

    return fig


def half_court():
    # Create figure
    fig = go.Figure()

    # Court
    fig.add_trace(
        go.Scatter(y=[-50, 420, 420, -50, -50],
                   x=[-250, -250, 250, 250, -250],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Left basket
    fig.add_trace(
        go.Scatter(y=[-10, -10],
                   x=[-30, 30],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Left inside box
    fig.add_trace(
        go.Scatter(y=[-50, 140, 140, -50, -50],
                   x=[-80, -80, 80, 80, -80],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Left outside box
    fig.add_trace(
        go.Scatter(y=[-50, 140, 140, -50, -50],
                   x=[-60, -60, 60, 60, -60],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Left top 3-pointer straight line
    fig.add_trace(
        go.Scatter(y=[-50, 90],
                   x=[220, 220],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Left bottom 3-pointer straight line
    fig.add_trace(
        go.Scatter(y=[-50, 90],
                   x=[-220, -220],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Half court line
    fig.add_trace(
        go.Scatter(y=[420, 420],
                   x=[-250, 250],
                   line={"color": "rgba(0,0,0,1)", "width": 1},
                   mode='lines')
    )

    # Left ring
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  y0=11, x0=7.5, y1=-4, x1=-7.5,
                  line={"color": "rgba(0,0,0,1)", "width": 1},
                  )

    # Left free-throw circle
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  y0=200, x0=60, y1=80, x1=-60,
                  line={"color": "rgba(0,0,0,1)", "width": 1},
                  )

    # Half court outside circle
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  y0=480, x0=60, y1=360, x1=-60,
                  line={"color": "rgba(0,0,0,1)", "width": 1},
                  )

    # Half court inside circle
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  y0=440, x0=20, y1=400, x1=-20,
                  line={"color": "rgba(0,0,0,1)", "width": 1},
                  )

    # Left 3-pointer half circle
    fig.add_shape(type="path",
                  path="M 220,90 Q 0,400 -220,90",
                  line={"color": "rgba(0,0,0,1)", "width": 1},
                  )

    fig.update_xaxes(
        constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )

    fig.update_layout(plot_bgcolor='white')

    return fig


fig = half_court()
poff.plot(fig, config=dict({'scrollZoom': True}))
