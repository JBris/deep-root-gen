#!/usr/bin/env python

######################################
# Imports
######################################

import os

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, page_container, page_registry
from dash_bootstrap_templates import load_figure_template
from hydra import compose, initialize
from hydra.utils import instantiate

######################################
# Environment
######################################

if os.environ.get("PREFECT_API_URL") is None:
    os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"

if os.environ.get("AWS_ACCESS_KEY_ID") is None:
    os.environ["AWS_ACCESS_KEY_ID"] = "user"

if os.environ.get("AWS_SECRET_ACCESS_KEY") is None:
    os.environ["AWS_SECRET_ACCESS_KEY"] = "password"

if os.environ.get("MLFLOW_S3_ENDPOINT_URL") is None:
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

if os.environ.get("ARANGO_DB") is None:
    os.environ["ARANGO_DB"] = "deeprootgen"

if os.environ.get("ARANGO_ROOT_USER") is None:
    os.environ["ARANGO_ROOT_USER"] = "root"

if os.environ.get("ARANGO_ROOT_PASSWORD") is None:
    os.environ["ARANGO_ROOT_PASSWORD"] = "password"

if os.environ.get("ARANGO_HOST_URL") is None:
    os.environ["ARANGO_HOST_URL"] = "http://localhost:8529 "

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

    app.layout = html.Div(
        [
            dcc.Store(id="store-simulation-run", storage_type="local", data=[]),
            dcc.Store(id="store-observed-data", storage_type="local", data=[]),
            dcc.Loading(
                id="loading_page_content",
                children=[
                    html.Div(
                        dbc.Row(
                            [
                                dbc.Card(
                                    app_navbar,
                                    style={
                                        "padding-bottom": "0.5em",
                                        "borderRadius": "0",
                                    },
                                ),
                                page_container,
                            ]
                        ),
                        id="app-contents",
                    ),
                ],
                color="primary",
                fullscreen=False,
                delay_show=500,
                delay_hide=500,
                type="circle",
            ),
        ],
        id="app-wrapper",
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
