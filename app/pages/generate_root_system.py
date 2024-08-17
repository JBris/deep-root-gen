#!/usr/bin/env python

######################################
# Imports
######################################

from pydoc import locate

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, get_app, html, register_page

######################################
# Functions
######################################


@callback(
    Output("table-content", "data"),
    Output("table-content", "page_size"),
    Input("store", "data"),
)
def get_table(data: dict) -> tuple:
    return data, 10


######################################
# Layout
######################################

register_page(__name__, name="Generate Root Data", top_nav=True, path="/")


@callback(
    Output("parameters-collapse", "is_open"),
    [Input("parameters-collapse-button", "n_clicks")],
    [State("parameters-collapse", "is_open")],
)
def toggle_parameters_collapse(n: int, is_open: bool) -> bool:
    if n:
        return not is_open
    return is_open


def layout() -> html.Div:
    app = get_app()
    form_model = app.settings["form"]

    components = []

    for component_spec in form_model.components["inputs"]:
        component_label = html.P(
            html.Span(
                dbc.Label(
                    component_spec.label,
                    html_for=component_spec.id,
                    id=f"{component_spec.id}-label",
                ),
                id=f"{component_spec.id}-tooltip-target",
                style={"cursor": "pointer"},
            ),
            style={"margin": "0"},
        )

        component_tooltip = dbc.Tooltip(
            component_spec.help,
            target=f"{component_spec.id}-tooltip-target",
            placement="right",
        )
        component = locate(component_spec.class_name)
        component_instance = component(id=component_spec.id, **component_spec.kwargs)  # type: ignore
        components.append(
            dbc.Row([component_label, component_tooltip, component_instance])
        )

    layout = html.Div(
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        dbc.Card(
                            [
                                html.H5(
                                    "Run Simulation",
                                    style={"margin-left": "1em", "margin-top": "0.2em"},
                                ),
                                dbc.Button(
                                    "Toggle Parameters",
                                    id="parameters-collapse-button",
                                    className="me-1",
                                    color="light",
                                    n_clicks=0,
                                ),
                                dbc.Collapse(
                                    dbc.Card(
                                        dbc.CardBody(
                                            dbc.Col(components),
                                        )
                                    ),
                                    id="parameters-collapse",
                                    is_open=True,
                                    dimension="height",
                                ),
                            ],
                        ),
                    ),
                    style={"width": "25%", "padding-right": "0"},
                ),
                html.Div(
                    dbc.Col(dbc.Card(components)),
                    style={"width": "75%", "padding-left": "0"},
                ),
            ]
        ),
        id="generate-root-system-page",
    )
    return layout
