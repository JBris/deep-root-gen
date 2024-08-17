import plotly.express as px
from dash import Input, Output, callback, dash_table, dcc, html, register_page
from plotly.graph_objects import Figure

register_page(__name__, name="Page 1", top_nav=True, path="/page1")


@callback(Output("graph", "figure"), Input("store", "data"))
def get_plot(data: dict) -> Figure:
    return px.scatter(
        data,
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="continent",
        log_x=True,
        size_max=60,
    )


def layout() -> html.Div:
    layout = html.Div(
        [
            dcc.Graph(id="graph"),
        ]
    )
    return layout
