#!/usr/bin/env python

######################################
# Imports
######################################

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, callback, dcc, html
from dash_bootstrap_templates import load_figure_template
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

######################################
# Functions
######################################


def define_callbacks(app: Dash, df: pd.DataFrame) -> None:
    @callback(Output("graph-content", "figure"), Input("dropdown-selection", "value"))
    def update_graph(value: str) -> px.line:
        dff = df[df.country == value]
        return px.line(dff, x="year", y="pop")


def define_ui(app: Dash, df: pd.DataFrame) -> None:
    app.layout = [
        html.H1(children="Title of Dash App", style={"textAlign": "center"}),
        dcc.Dropdown(df.country.unique(), "Canada", id="dropdown-selection"),
        dcc.Graph(id="graph-content"),
    ]


######################################
# Main
######################################

load_figure_template("MINTY")
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
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
