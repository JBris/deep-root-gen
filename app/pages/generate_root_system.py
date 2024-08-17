#!/usr/bin/env python

######################################
# Imports
######################################

import os.path as osp

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, dcc, get_app, html, register_page

from deeprootgen.form import (
    build_collapsible,
    build_common_components,
    build_common_layout,
)

######################################
# Constants
######################################

PAGE_ID = "generate-root-system-page"

######################################
# Callbacks
######################################


@callback(
    Output(f"{PAGE_ID}-parameters-collapse", "is_open"),
    [Input(f"{PAGE_ID}-parameters-collapse-button", "n_clicks")],
    [State(f"{PAGE_ID}-parameters-collapse", "is_open")],
)
def toggle_parameters_collapse(n: int, is_open: bool) -> bool:
    """Toggle the collapsible for parameters.

    Args:
        n (int):
            The number of times that the button has been clicked.
        is_open (bool):
            Whether the collapsible is open.

    Returns:
        bool: The collapsible state.
    """
    if n:
        return not is_open
    return is_open


@callback(
    Output(f"{PAGE_ID}-data-collapse", "is_open"),
    [Input(f"{PAGE_ID}-data-collapse-button", "n_clicks")],
    [State(f"{PAGE_ID}-data-collapse", "is_open")],
)
def toggle_data_io_collapse(n: int, is_open: bool) -> bool:
    """Toggle the collapsible for data input/output.

    Args:
        n (int):
            The number of times that the button has been clicked.
        is_open (bool):
            Whether the collapsible is open.

    Returns:
        bool: The collapsible state.
    """
    if n:
        return not is_open
    return is_open


@callback(
    Output(f"{PAGE_ID}-external-links-collapse", "is_open"),
    [Input(f"{PAGE_ID}-external-links-collapse-button", "n_clicks")],
    [State(f"{PAGE_ID}-external-links-collapse", "is_open")],
)
def toggle_external_links_collapse(n: int, is_open: bool) -> bool:
    """Toggle the collapsible for external links.

    Args:
        n (int):
            The number of times that the button has been clicked.
        is_open (bool):
            Whether the collapsible is open.

    Returns:
        bool: The collapsible state.
    """
    if n:
        return not is_open
    return is_open


@callback(
    Output(f"{PAGE_ID}-download-content", "data"),
    [Input({"index": f"{PAGE_ID}-save-param-button", "type": ALL}, "n_clicks")],
    State({"type": f"{PAGE_ID}-parameters", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def save_param(n_clicks: int, param_inputs: list) -> None:
    """Save parameter form values.

    Args:
        n_clicks (int):
            The number of times that the button has been clicked.
        param_inputs (list):
            The parameter input data.
    """
    inputs = {}
    app = get_app()
    form_model = app.settings["form"]
    for i, input in enumerate(form_model.components["parameters"]["children"]):
        k = input["param"]
        inputs[k] = param_inputs[i]

    from datetime import datetime

    import yaml

    outfile = osp.join(
        "outputs", f"{datetime.today().strftime('%Y-%m-%d-%H-%M')}-{PAGE_ID}.yaml"
    )
    with open(outfile, "w") as f:
        yaml.dump(inputs, f, default_flow_style=False, sort_keys=False)
    return dcc.send_file(outfile)


@callback(
    Output({"type": f"{PAGE_ID}-parameters", "index": ALL}, "value"),
    Output(f"{PAGE_ID}-load-toast", "is_open"),
    Output(f"{PAGE_ID}-load-toast", "children"),
    Input({"index": f"{PAGE_ID}-upload-param-file-button", "type": ALL}, "contents"),
    State({"index": f"{PAGE_ID}-upload-param-file-button", "type": ALL}, "filename"),
    prevent_initial_call=True,
)
def update_output(list_of_contents: list, list_of_names: list) -> tuple:
    import base64

    import yaml

    _, content_string = list_of_contents[0].split(",")
    decoded = base64.b64decode(content_string)
    input_dict = yaml.safe_load(decoded.decode("utf-8"))
    inputs = list(input_dict.values())
    toast_message = f"Loading parameter specification from: {list_of_names[0]}"
    return inputs, True, toast_message


# @callback(
#     Output("generate-root-system-plot", "figure"),
#     Input({"type": f"{PAGE_ID}-parameters", "index": ALL}, "value"),
# )
# def plot_root_model(form_inputs: list) -> dcc.Graph:
#     """Run and plot the root model.

#     Args:
#         form_inputs (list):
#             The form input data.

#     Returns:
#         dcc.Graph: The visualised root model.
#     """
#     inputs = {}
#     app = get_app()
#     form_model = app.settings["form"]
#     for i, input in enumerate(form_model.components["parameters"]["children"]):
#         k = input["param"]
#         inputs[k] = form_inputs[i]

#     return dcc.Graph()


######################################
# Layout
######################################

register_page(__name__, name="Generate Root Data", top_nav=True, path="/")


def layout() -> html.Div:
    """Return the page layout.

    Returns:
        html.Div: The page layout.
    """
    app = get_app()
    form_model = app.settings["form"]

    parameter_components = build_common_components(
        form_model.components["parameters"]["children"], PAGE_ID, "parameters"
    )

    if form_model.components["parameters"]["collapsible"]:
        parameter_components = build_collapsible(
            parameter_components, PAGE_ID, "Parameters"
        )

    data_io_components = build_common_components(
        form_model.components["data"]["children"], PAGE_ID, "data"
    )

    if form_model.components["data"]["collapsible"]:
        data_io_components = build_collapsible(data_io_components, PAGE_ID, "Data")

    input_components = dbc.Col([parameter_components, data_io_components])
    output_components = dbc.Row(
        [
            dbc.Col(
                dcc.Graph(id="generate-root-system-plot"),
            )
        ]
    )

    layout = build_common_layout(
        "Run Simulation", PAGE_ID, input_components, output_components
    )

    return layout
