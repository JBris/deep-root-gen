#!/usr/bin/env python

######################################
# Imports
######################################

# mypy: ignore-errors

import os
import os.path as osp
from typing import Callable

import pandas as pd
import plotly.express as px
import requests
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

from deeprootgen.calibration import get_simulation_parameters
from deeprootgen.form import get_common_layout
from deeprootgen.io import load_data_from_file
from deeprootgen.pipeline import get_datetime_now, get_outdir

######################################
# Constants
######################################

TITLE = "Deployments"
TASK = "deployments"
PAGE_ID = f"{TASK}-root-system-page"
FORM_NAME = "deployments_form"
PROCEDURE = "Calibration"
DATA_COMPONENT_KEY = "summary_data"

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
    Output(f"{PAGE_ID}-deployments-collapse", "is_open"),
    [Input(f"{PAGE_ID}-deployments-collapse-button", "n_clicks")],
    [State(f"{PAGE_ID}-deployments-collapse", "is_open")],
)
def toggle_deployments_collapse(n: int, is_open: bool) -> bool:
    """Toggle the collapsible for deployments.

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
    Output(f"{PAGE_ID}-calibration-parameters-collapse", "is_open"),
    [Input(f"{PAGE_ID}-calibration-parameters-collapse-button", "n_clicks")],
    [State(f"{PAGE_ID}-calibration-parameters-collapse", "is_open")],
)
def toggle_calibration_parameters_collapse(n: int, is_open: bool) -> bool:
    """Toggle the collapsible for calibration inputs.

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
    Output(f"{PAGE_ID}-calibration-collapse", "is_open"),
    [Input(f"{PAGE_ID}-calibration-collapse-button", "n_clicks")],
    [State(f"{PAGE_ID}-calibration-collapse", "is_open")],
)
def toggle_calibration_collapse(n: int, is_open: bool) -> bool:
    """Toggle the collapsible for calibration inputs.

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
    Output(f"{PAGE_ID}-summary-data-collapse", "is_open"),
    [Input(f"{PAGE_ID}-summary-data-collapse-button", "n_clicks")],
    [State(f"{PAGE_ID}-summary-data-collapse", "is_open")],
)
def toggle_data_collapse(n: int, is_open: bool) -> bool:
    """Toggle the collapsible for statistics.

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
    Output(
        {"index": f"{PAGE_ID}-upload-obs-data-file-button", "type": ALL}, "children"
    ),
    Output({"index": f"{PAGE_ID}-clear-obs-data-file-button", "type": ALL}, "disabled"),
    Input("store-summary-data", "data"),
)
def update_summary_data(summary_data: dict) -> tuple:
    """Update the state of the summary data.

    Args:
        summary_data (dict | None):
            The summary data.

    Returns:
        tuple:
            The updated form state.
    """
    button_contents = "Load summary data"
    if summary_data is None:
        return [button_contents], [True]
    summary_label = summary_data.get("label")
    if summary_label is None:
        return [button_contents], [True]

    return [summary_label], [False]


@callback(
    Output("store-summary-data", "data", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "is_open", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "children", allow_duplicate=True),
    Input({"index": f"{PAGE_ID}-upload-obs-data-file-button", "type": ALL}, "contents"),
    State({"index": f"{PAGE_ID}-upload-obs-data-file-button", "type": ALL}, "filename"),
    prevent_initial_call=True,
)
def load_summary_data(list_of_contents: list, list_of_names: list) -> tuple:
    """Load statistics data from file.

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

    if list_of_contents[0] is None:
        return no_update

    loaded_data, _, toast_message = load_data_from_file(list_of_contents, list_of_names)
    summary_data = {"label": list_of_names[0], "values": loaded_data}
    return summary_data, True, toast_message


@callback(
    Output("store-summary-data", "data", allow_duplicate=True),
    Output(
        {"index": f"{PAGE_ID}-upload-obs-data-file-button", "type": ALL}, "contents"
    ),
    Output(f"{PAGE_ID}-load-toast", "is_open", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "children", allow_duplicate=True),
    Input({"index": f"{PAGE_ID}-clear-obs-data-file-button", "type": ALL}, "n_clicks"),
    State("store-summary-data", "data"),
    prevent_initial_call=True,
)
def clear_summary_data(n_clicks: int | list[int], summary_data: dict) -> tuple:
    """Clear summary data from the page.

    Args:
        n_clicks (int | list[int]):
            The number of form clicks.
        summary_data (dict):
            The exploratory data analysis data.

    Returns:
        tuple:
            The updated form state.
    """
    if n_clicks is None or len(n_clicks) == 0:  # type: ignore
        return no_update

    if n_clicks[0] is None or n_clicks[0] == 0:  # type: ignore
        return no_update

    summary_label = summary_data.get("label")
    if summary_data is None or summary_label is None:
        return no_update

    toast_message = f"Clearing: {summary_label}"
    return {}, [None], True, toast_message


def endpoint_predict(task: str, endpoint: str, json: dict) -> str:
    """Make a prediction to an endpoint.

    Args:
        task (str):
            The calibration task.
        endpoint (str):
            The endpoint url.
        json (dict):
            The JSON body.

    Returns:
        str:
            The response dataframe file path.
    """
    res = requests.post(url=f"{endpoint}/predict", json=json)
    json_data = res.json()
    print(json_data)
    df = pd.DataFrame(json_data)
    outdir = get_outdir()
    date_now = get_datetime_now()
    file_name = f"{date_now}-{task}-predict.csv"
    outfile = osp.join(outdir, file_name)
    df.to_csv(outfile, index=False)
    return outfile


@callback(
    Output(f"{PAGE_ID}-download-results", "data", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "is_open", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "children", allow_duplicate=True),
    Input({"index": f"{PAGE_ID}-call-optimiser-button", "type": ALL}, "n_clicks"),
    State({"index": f"{PAGE_ID}-n-trials-input", "type": ALL}, "value"),
    prevent_initial_call=True,
)
def call_optimisation(n_clicks: int | list[int], n_trials_list: list) -> Callable:
    """Call the optimisation endpoint.

    Args:
        n_clicks (int | list[int]):
            The number of form clicks.
        n_trials_list (list):
            The number of trials for optimisation.

    Returns:
        Callable:
            The form data.
    """
    if n_clicks is None or len(n_clicks) == 0:  # type: ignore
        return no_update

    if n_clicks[0] is None or n_clicks[0] == 0:  # type: ignore
        return no_update

    endpoint = os.environ.get(
        "DEPLOYMENT_OPTIMISATION_INTERNAL_LINK", "http://optimisation:3000"
    )

    n_trials = n_trials_list[0]
    json = {"n_trials": n_trials}
    task = "optimisation"
    outfile = endpoint_predict(task, endpoint, json)
    return dcc.send_file(outfile), True, f"Calling {task} calibrator"


@callback(
    Output(f"{PAGE_ID}-download-results", "data", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "is_open", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "children", allow_duplicate=True),
    Input({"index": f"{PAGE_ID}-call-sa-button", "type": ALL}, "n_clicks"),
    State({"index": f"{PAGE_ID}-parameter-names-dropdown", "type": ALL}, "value"),
    prevent_initial_call=True,
)
def call_sensitivity_analysis(
    n_clicks: int | list[int], parameter_names_list: list[str]
) -> Callable:
    """Call the sensitivity analysis endpoint.

    Args:
        n_clicks (int | list[int]):
            The number of form clicks.
        ts (list):
            The list of Sequential Monte Carlo time steps.

    Returns:
        Callable:
            The form data.
    """
    if n_clicks is None or len(n_clicks) == 0:  # type: ignore
        return no_update

    if n_clicks[0] is None or n_clicks[0] == 0:  # type: ignore
        return no_update

    endpoint = os.environ.get(
        "DEPLOYMENT_SENSITIVITY_ANALYSIS_INTERNAL_LINK",
        "http://sensitivity_analysis:3000",
    )

    parameter_names = parameter_names_list[0]
    json = {"names": parameter_names}
    task = "sensitivity_analysis"
    outfile = endpoint_predict(task, endpoint, json)
    return dcc.send_file(outfile), True, "Calling sensitivity analysis calibrator"


@callback(
    Output(f"{PAGE_ID}-download-results", "data", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "is_open", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "children", allow_duplicate=True),
    Input({"index": f"{PAGE_ID}-call-abc-button", "type": ALL}, "n_clicks"),
    State({"index": f"{PAGE_ID}-t-input", "type": ALL}, "value"),
    prevent_initial_call=True,
)
def call_abc(n_clicks: int | list[int], ts: list[int]) -> Callable:
    """Call the Approximate Bayesian Computation endpoint.

    Args:
        n_clicks (int | list[int]):
            The number of form clicks.
        ts (list):
            The list of Sequential Monte Carlo time steps.

    Returns:
        Callable:
            The form data.
    """
    if n_clicks is None or len(n_clicks) == 0:  # type: ignore
        return no_update

    if n_clicks[0] is None or n_clicks[0] == 0:  # type: ignore
        return no_update

    endpoint = os.environ.get("DEPLOYMENT_ABC_INTERNAL_LINK", "http://abc:3000")

    t = ts[0]
    json = {"t": [t]}
    task = "abc"
    outfile = endpoint_predict(task, endpoint, json)
    return dcc.send_file(outfile), True, f"Calling {task} calibrator"


@callback(
    Output(f"{PAGE_ID}-download-results", "data", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "is_open", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "children", allow_duplicate=True),
    Input({"index": f"{PAGE_ID}-call-snpe-button", "type": ALL}, "n_clicks"),
    State({"type": f"{PAGE_ID}-summary_statistics", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def call_snpe(n_clicks: int | list[int], statistics_inputs: list[float]) -> Callable:
    """Call the Sequential neural posterior estimation endpoint.

    Args:
        n_clicks (int | list[int]):
            The number of form clicks.
        statistics_inputs (list[float]):
            The list of summary statistic values.

    Returns:
        Callable:
            The form data.
    """
    if n_clicks is None or len(n_clicks) == 0:  # type: ignore
        return no_update

    if n_clicks[0] is None or n_clicks[0] == 0:  # type: ignore
        return no_update

    endpoint = os.environ.get("DEPLOYMENT_SNPE_INTERNAL_LINK", "http://snpe:3000")

    app = get_app()
    statistics_form = app.settings[FORM_NAME]
    summary_statistics = []
    for i, child in enumerate(
        statistics_form.components["summary_statistics"]["children"]
    ):
        statistic_name = child["param"]
        statistic_value = statistics_inputs[i]
        summary_statistics.append(
            {"statistic_name": statistic_name, "statistic_value": statistic_value}
        )

    json = {"summary_statistics": summary_statistics}
    task = "snpe"
    outfile = endpoint_predict(task, endpoint, json)
    return dcc.send_file(outfile), True, f"Calling {task} calibrator"


######################################
# Layout
######################################


register_page(__name__, name=TITLE, top_nav=True, path=f"/{TASK}")


def layout() -> html.Div:
    """Return the page layout.

    Returns:
        html.Div: The page layout.
    """
    page_description = """
    Interact with model calibration deployments and download data
    """

    layout = get_common_layout(
        title=TITLE,
        page_id=PAGE_ID,
        page_description=page_description,
        parameter_form_name=FORM_NAME,
        simulation_form_name=FORM_NAME,
        procedure=PROCEDURE,
        task=TASK,
        data_key=DATA_COMPONENT_KEY,
    )

    return layout
