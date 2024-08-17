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
from hydra.utils import instantiate

######################################
# Functions
######################################


def define_ui(app: Dash) -> None:
    nav_links = [
        dbc.NavLink(page["name"], href=page["path"], active="exact")
        for page in page_registry.values()
    ]

    app_navbar = dbc.Row(
        dbc.Nav(
            nav_links,
            vertical=False,
            pills=True,
            style={"padding-left": "1em", "padding-top": "0.5em"},
        )
    )

    app.layout = dcc.Loading(
        id="loading_page_content",
        children=[
            dcc.Store(id="store", storage_type="session", data=[{"name": "James"}]),
            html.Div(
                dbc.Row(
                    [
                        dbc.Card(
                            app_navbar,
                            style={"padding-bottom": "0.5em", "borderRadius": "0"},
                        ),
                        page_container,
                    ]
                ),
                id="app-contents",
            ),
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

with initialize(version_base=None, config_path="conf", job_name="deep_root_gen"):
    cfg = compose(config_name="config")
    config = instantiate(cfg)
app.settings = config

define_ui(app)

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port="8000")
