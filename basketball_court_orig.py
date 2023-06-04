# Get this figure: fig = py.get_figure("https://plotly.com/~SamJCurtis/2/")
# Get this figure's data: data = py.get_figure("https://plotly.com/~SamJCurtis/2/").get_data()
# Add data to this figure: py.plot(Data([Scatter(x=[1, 2], y=[2, 3])]), filename ="basketcall_Court", fileopt="extend")
# Get y data of first trace: y1 = py.get_figure("https://plotly.com/~SamJCurtis/2/").get_data()[0]["y"]

# Get figure documentation: https://plotly.com/python/get-requests/
# Add data documentation: https://plotly.com/python/file-options/

# If you're using unicode in your file, you may need to specify the encoding.
# You can reproduce this figure in Python with the following code!

# Learn about API authentication here: https://plotly.com/python/getting-started
# Find your api_key here: https://plotly.com/settings/api

import plotly.offline as poff
import plotly.graph_objs as pgo

trace1 = {
    "uid": "3608439e-d007-11e8-bf2c-f2189834773b",
    "type": "scatter",
    "x": [47],
    "y": [25],
}

data = pgo.Data([trace1])

layout = {
    "shapes": [
        {
            "x0": 0,
            "x1": 94,
            "y0": 0,
            "y1": 50,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "rect",
        },
        {
            "x0": 4,
            "x1": 4,
            "y0": 22,
            "y1": 28,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "line",
        },
        {
            "x0": 90,
            "x1": 90,
            "y0": 22,
            "y1": 28,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "line",
        },
        {
            "x0": 0,
            "x1": 19,
            "y0": 17,
            "y1": 33,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "rect",
        },
        {
            "x0": 0,
            "x1": 19,
            "y0": 19,
            "y1": 31,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "rect",
        },
        {
            "x0": 75,
            "x1": 94,
            "y0": 17,
            "y1": 33,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "rect",
        },
        {
            "x0": 75,
            "x1": 94,
            "y0": 19,
            "y1": 31,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "rect",
        },
        {
            "x0": 0,
            "x1": 14,
            "y0": 47,
            "y1": 47,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "rect",
        },
        {
            "x0": 0,
            "x1": 14,
            "y0": 3,
            "y1": 3,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "rect",
        },
        {
            "x0": 80,
            "x1": 94,
            "y0": 47,
            "y1": 47,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "rect",
        },
        {
            "x0": 80,
            "x1": 94,
            "y0": 3,
            "y1": 3,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "rect",
        },
        {
            "x0": 47,
            "x1": 47,
            "y0": 0,
            "y1": 50,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "rect",
        },
        {
            "x0": 6.1,
            "x1": 4.6,
            "y0": 25.75,
            "y1": 24.25,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "circle",
        },
        {
            "x0": 89.4,
            "x1": 87.9,
            "y0": 25.75,
            "y1": 24.25,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "circle",
        },
        {
            "x0": 25,
            "x1": 13,
            "y0": 31,
            "y1": 19,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "circle",
        },
        {
            "x0": 81,
            "x1": 69,
            "y0": 31,
            "y1": 19,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "circle",
        },
        {
            "x0": 53,
            "x1": 41,
            "y0": 31,
            "y1": 19,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "circle",
        },
        {
            "x0": 49,
            "x1": 45,
            "y0": 27,
            "y1": 23,
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "type": "circle",
        },
        {
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "path": "M 14,47 Q 45,25 14,3",
            "type": "path",
        },
        {
            "line": {"color": "rgba(0,0,0,1)", "width": 1},
            "path": "M 80,47 Q 49,25 80,3",
            "type": "path",
        },
    ],
}

fig = pgo.Figure(data=data, layout=layout)

poff.plot(fig, config=dict({'scrollZoom': True}))
