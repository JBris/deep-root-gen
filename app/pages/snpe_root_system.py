#!/usr/bin/env python

######################################
# Imports
######################################

import base64
import os
import os.path as osp

import dash_bootstrap_components as dbc
import pandas as pd
import yaml
from dash import (
    ALL,
    Input,
    Output,
    State,
    callback,
    dcc,
    get_app,
    html,
    no_update,
    register_page,
)
from prefect.deployments import run_deployment

from deeprootgen.form import (
    build_collapsible,
    build_common_components,
    build_common_layout,
    get_out_table_df,
)
from deeprootgen.io import s3_upload_file
from deeprootgen.pipeline import get_datetime_now, get_simulation_uuid

######################################
# Constants
######################################

TASK = "snpe"
PAGE_ID = "snpe-root-system-page"

######################################
# Callbacks
######################################


@callback(
    Output(f"{PAGE_ID}-sidebar-fade", "is_in"),
    Output(f"{PAGE_ID}-output-fade", "is_in"),
    Input(f"{PAGE_ID}-sidebar-fade", "is_in"),
)
def fade_in(is_in: bool) -> tuple:
    """Fade the page contents in.

    Args:
        is_in (bool):
            Whether the sidebar is in the page.

    Returns:
        tuple:
            A tuple for displaying the page contents.
    """
    if is_in:
        return True, True
    return True, True


@callback(
    Output({"index": f"{PAGE_ID}-simulation-runs-table", "type": ALL}, "data"),
    Input("store-simulation-run", "data"),
)
def update_table(runs: list | None) -> list | None:
    """Update the simulation run table.

    Args:
        runs (list | None):
            The list of simulation runs.

    Returns:
        list | None:
            The updated list of simulation runs.
    """
    if runs is None:
        return no_update
    return [runs]


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
def save_param(n_clicks: int | list[int], param_inputs: list) -> None:
    """Save parameter form values.

    Args:
        n_clicks (int | list[int]):
            The number of times that the button has been clicked.
        param_inputs (list):
            The parameter input data.
    """
    if n_clicks is None or len(n_clicks) == 0:  # type: ignore
        return no_update

    if n_clicks[0] is None or n_clicks[0] == 0:  # type: ignore
        return no_update

    inputs = {}
    app = get_app()
    form_model = app.settings["form"]
    for i, input in enumerate(form_model.components["parameters"]["children"]):
        k = input["param"]
        inputs[k] = param_inputs[i]

    file_name = f"{get_datetime_now()}-{PAGE_ID}.yaml"
    outfile = osp.join("outputs", file_name)
    with open(outfile, "w") as f:
        yaml.dump(inputs, f, default_flow_style=False, sort_keys=False)
    s3_upload_file(outfile, file_name)
    return dcc.send_file(outfile)


@callback(
    Output(f"{PAGE_ID}-download-results", "data"),
    [Input({"index": f"{PAGE_ID}-save-runs-button", "type": ALL}, "n_clicks")],
    State("store-simulation-run", "data"),
    prevent_initial_call=True,
)
def save_runs(n_clicks: int | list[int], simulation_runs: list) -> None:
    """Save simulation runs to file.

    Args:
        n_clicks (int | list[int]):
            The number of times that the button has been clicked.
        simulation_runs (list):
            A list of simulation run data.
    """
    if n_clicks is None or len(n_clicks) == 0:  # type: ignore
        return no_update

    if n_clicks[0] is None or n_clicks[0] == 0:  # type: ignore
        return no_update

    simulation_runs = simulation_runs[0]
    df = pd.DataFrame(simulation_runs)
    date_now = get_datetime_now()
    file_name = f"{date_now}-{PAGE_ID}-runs.csv"
    outfile = osp.join("outputs", file_name)
    df.to_csv(outfile, index=False)
    s3_upload_file(outfile, file_name)
    return dcc.send_file(outfile)


@callback(
    Output("store-simulation-run", "data", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "is_open", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "children", allow_duplicate=True),
    Input({"index": f"{PAGE_ID}-upload-runs-file-button", "type": ALL}, "contents"),
    State({"index": f"{PAGE_ID}-upload-runs-file-button", "type": ALL}, "filename"),
    prevent_initial_call=True,
)
def load_runs(list_of_contents: list, list_of_names: list) -> tuple:
    """Load simulation runs from file.

    Args:
        list_of_contents (list):
            The list of file contents.
        list_of_names (list):
            The list of file names.

    Returns:
        tuple:
            The updated form state.
    """
    if list_of_contents is None or len(list_of_contents) == 0:
        return no_update

    if list_of_contents[0] is None or list_of_contents[0] == 0:
        return no_update

    _, content_string = list_of_contents[0].split(",")
    decoded = base64.b64decode(content_string).decode("utf-8")
    from io import StringIO

    workflow_urls = pd.read_csv(StringIO(decoded)).to_dict("records")

    toast_message = f"Loading run history from: {list_of_names[0]}"
    return workflow_urls, True, toast_message


@callback(
    Output("store-simulation-run", "data", allow_duplicate=True),
    [Input({"index": f"{PAGE_ID}-clear-runs-button", "type": ALL}, "n_clicks")],
    prevent_initial_call=True,
)
def clear_runs(n_clicks: int | list[int]) -> list:
    """Clear the simulation runs table.

    Args:
        n_clicks (int | list[int]):
            The number of times that the button has been clicked.

    Returns:
        list:
            An empty list.
    """
    if n_clicks is None or len(n_clicks) == 0:  # type: ignore
        return no_update

    if n_clicks[0] is None or n_clicks[0] == 0:  # type: ignore
        return no_update

    return []


@callback(
    Output({"type": f"{PAGE_ID}-parameters", "index": ALL}, "value"),
    Output(f"{PAGE_ID}-load-toast", "is_open", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "children", allow_duplicate=True),
    Input({"index": f"{PAGE_ID}-upload-param-file-button", "type": ALL}, "contents"),
    State({"index": f"{PAGE_ID}-upload-param-file-button", "type": ALL}, "filename"),
    prevent_initial_call=True,
)
def load_params(list_of_contents: list, list_of_names: list) -> tuple:
    """Load the simulation parameters from file.

    Args:
        list_of_contents (list):
            The list of file contents.
        list_of_names (list):
            The list of file names.

    Returns:
        tuple:
            The updated form state.
    """
    if list_of_contents is None or len(list_of_contents) == 0:
        return no_update

    if list_of_contents[0] is None or list_of_contents[0] == 0:
        return no_update

    _, content_string = list_of_contents[0].split(",")
    decoded = base64.b64decode(content_string)
    input_dict = yaml.safe_load(decoded.decode("utf-8"))

    app = get_app()
    form_model = app.settings["form"]
    inputs = []
    for input in form_model.components["parameters"]["children"]:
        k = input["param"]
        inputs.append(input_dict[k])

    toast_message = f"Loading parameter specification from: {list_of_names[0]}"
    return inputs, True, toast_message


@callback(
    Output("store-simulation-run", "data", allow_duplicate=True),
    Output(f"{PAGE_ID}-results-toast", "is_open"),
    Output(f"{PAGE_ID}-results-toast", "children"),
    Input({"index": f"{PAGE_ID}-run-sim-button", "type": ALL}, "n_clicks"),
    State({"type": f"{PAGE_ID}-parameters", "index": ALL}, "value"),
    State({"index": f"{PAGE_ID}-enable-soil-input", "type": ALL}, "on"),
    State("store-simulation-run", "data"),
    prevent_initial_call=True,
)
def run_root_model(
    n_clicks: list, form_values: list, enable_soils: list, simulation_runs: list
) -> dcc.Graph:
    """Run and plot the root model.

    Args:
        n_clicks (list):
            Number of times the button has been clicked.
        form_values (list):
            The form input data.
        enable_soils (list):
            Enable visualisation of soil data.
        simulation_runs (list):
            A list of simulation run data.

    Returns:
        dcc.Graph: The visualised root model.
    """
    if n_clicks is None or len(n_clicks) == 0:
        return no_update

    if n_clicks[0] is None or n_clicks[0] == 0:
        return no_update

    form_inputs = {}
    app = get_app()
    form_model = app.settings["form"]
    for i, input in enumerate(form_model.components["parameters"]["children"]):
        k = input["param"]
        form_inputs[k] = form_values[i]

    enable_soil: bool = enable_soils[0]
    form_inputs["enable_soil"] = enable_soil == True  # noqa: E712

    simulation_uuid = get_simulation_uuid()
    flow_data = run_deployment(
        f"{TASK}/run_{TASK}_flow",
        parameters=dict(input_parameters=form_inputs, simulation_uuid=simulation_uuid),
        flow_run_name=f"run-{simulation_uuid}",
        timeout=0,
    )

    flow_run_id = str(flow_data.id)
    flow_name = flow_data.name
    simulation_tag = form_inputs["simulation_tag"]

    app_prefect_host = os.environ.get("APP_PREFECT_USER_HOST", "http://localhost:4200")
    prefect_flow_url = f"{app_prefect_host}/flow-runs/flow-run/{flow_run_id}"

    simulation_runs.append(
        {
            "workflow": f"<a href='{prefect_flow_url}' target='_blank'>{flow_name}</a>",
            "task": TASK,
            "date": get_datetime_now(),
            "tag": simulation_tag,
        }
    )

    toast_message = f"""
    Running simulation workflow: {flow_name}
    Simulation tag: {simulation_tag}
    """
    return simulation_runs, True, toast_message


######################################
# Layout
######################################

register_page(
    __name__, name="Sequential Neural Posterior Estimation", top_nav=True, path="/snpe"
)


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
    simulation_run_df = get_out_table_df()

    simulation_results_data = {"simulation-runs-table": simulation_run_df}

    k = "results"
    simulation_results_components = build_common_components(
        form_model.components[k]["children"],
        PAGE_ID,
        k,
        simulation_results_data,
        resize_component=False,
    )

    output_components = dbc.Row(
        dbc.Col(simulation_results_components, style={"margin-left": "0.5em"})
    )

    page_description = """
    Perform Bayesian parameter estimation for the root system architecture simulation against a target dataset using deep learning
    """
    layout = build_common_layout(
        "Run SNPE", PAGE_ID, input_components, output_components, page_description
    )

    return layout
