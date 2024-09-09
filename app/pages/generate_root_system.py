#!/usr/bin/env python

######################################
# Imports
######################################

import base64
import os.path as osp
from datetime import datetime

import dash_bootstrap_components as dbc
import pandas as pd
import yaml
from dash import ALL, Input, Output, State, callback, dcc, get_app, html, register_page

from deeprootgen.data_model import RootSimulationModel
from deeprootgen.form import (
    build_collapsible,
    build_common_components,
    build_common_layout,
)
from deeprootgen.model import RootSystemSimulation

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
    Output(f"{PAGE_ID}-simulation-collapse", "is_open"),
    [Input(f"{PAGE_ID}-simulation-collapse-button", "n_clicks")],
    [State(f"{PAGE_ID}-simulation-collapse", "is_open")],
)
def toggle_simulation_collapse(n: int, is_open: bool) -> bool:
    """Toggle the collapsible for simulation management.

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
    Output(f"{PAGE_ID}-download-params", "data"),
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
    _, content_string = list_of_contents[0].split(",")
    decoded = base64.b64decode(content_string)
    input_dict = yaml.safe_load(decoded.decode("utf-8"))
    inputs = list(input_dict.values())
    toast_message = f"Loading parameter specification from: {list_of_names[0]}"
    return inputs, True, toast_message


@callback(
    Output("generate-root-system-plot", "figure"),
    Output(f"{PAGE_ID}-download-content", "data"),
    Input({"index": f"{PAGE_ID}-run-sim-button", "type": ALL}, "n_clicks"),
    State({"type": f"{PAGE_ID}-parameters", "index": ALL}, "value"),
    State({"index": f"{PAGE_ID}-enable-soil-input", "type": ALL}, "on"),
    State({"index": f"{PAGE_ID}-download-sim-data-input", "type": ALL}, "on"),
    prevent_initial_call=True,
)
def run_root_model(
    n_clicks: list, form_values: list, enable_soils: list, download_data: list
) -> dcc.Graph:
    """Run and plot the root model.

    Args:
        n_clicks (list):
            Number of times the button has been clicked.
        form_values (list):
            The form input data.
        enable_soils (list):
            Enable visualisation of soil data.
        download_data (list):
            Whether to download the simulation results data.
    Returns:
        dcc.Graph: The visualised root model.
    """
    n_click: int = n_clicks[0]
    if n_click == 0:
        return None

    form_inputs = {}
    app = get_app()
    form_model = app.settings["form"]
    for i, input in enumerate(form_model.components["parameters"]["children"]):
        k = input["param"]
        form_inputs[k] = form_values[i]

    enable_soil: bool = enable_soils[0]
    form_inputs["enable_soil"] = enable_soil == True  # noqa: E712

    input_params = RootSimulationModel.parse_obj(form_inputs)
    simulation = RootSystemSimulation(
        simulation_tag=input_params.simulation_tag,
        random_seed=input_params.random_seed,
        visualise=True,
    )
    results = simulation.run(input_params)

    download_data = download_data[0]
    if download_data:
        from datetime import datetime

        now = datetime.today().strftime("%Y-%m-%d-%H-%M")
        outfile = osp.join("outputs", f"{now}-nodes.csv")
        df = pd.DataFrame(results.nodes)
        df.to_csv(outfile, index=False)
        download_file = dcc.send_file(outfile)
    else:
        download_file = None

    return results.figure, download_file


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

    k = "parameters"
    parameter_components = build_common_components(
        form_model.components[k]["children"], PAGE_ID, k
    )

    if form_model.components[k]["collapsible"]:
        parameter_components = build_collapsible(
            parameter_components, PAGE_ID, "Parameters"
        )

    k = "simulation"
    data_io_components = build_common_components(
        form_model.components[k]["children"], PAGE_ID, k
    )

    if form_model.components[k]["collapsible"]:
        data_io_components = build_collapsible(
            data_io_components, PAGE_ID, "Simulation"
        )

    input_components = dbc.Col([parameter_components, data_io_components])
    output_components = dbc.Row(
        [
            dbc.Col(
                dcc.Graph(id="generate-root-system-plot"),
            )
        ]
    )

    page_description = """
    Create synthetic root data by running the root system architecture simulation
    """
    layout = build_common_layout(
        "Run Simulation", PAGE_ID, input_components, output_components, page_description
    )

    return layout
