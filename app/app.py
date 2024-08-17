#!/usr/bin/env python

######################################
# Imports
######################################

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, callback, dcc, html, page_container, page_registry
from dash_bootstrap_templates import load_figure_template
from hydra import compose, initialize
from omegaconf import OmegaConf

######################################
# Functions
######################################


def define_callbacks(app: Dash, df: pd.DataFrame) -> None:
    @callback(Output("graph-content", "figure"), Input("dropdown-selection", "value"))
    def update_graph(value: str) -> px.line:
        dff = df[df.country == value]
        return px.line(dff, x="year", y="pop")


def define_ui(app: Dash, df: pd.DataFrame) -> None:
    app_navbar = dbc.Nav(
        [
            dbc.NavLink(page["name"], href=page["path"], active="exact")
            for page in page_registry.values()
        ],
        vertical=False,
        pills=True,
    )

    app.layout = dcc.Loading(
        id="loading_page_content",
        children=[
            dcc.Store(id="store", storage_type="session", data=df.to_dict("records")),
            html.Div([app_navbar, page_container]),
        ],
        color="primary",
        fullscreen=True,
    )


######################################
# Main
######################################

load_figure_template("MINTY")
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.MINTY],
    use_pages=True,
    title="DeepRootGen",
)
server = app.server

df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv"
)

with initialize(version_base=None, config_path="conf", job_name="deep_root_gen"):
    cfg = compose(config_name="config")
    config = OmegaConf.to_container(cfg)

define_ui(app, df)
define_callbacks(app, df)

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port="8000")
