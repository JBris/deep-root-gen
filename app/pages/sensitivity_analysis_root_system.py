#!/usr/bin/env python

######################################
# Imports
######################################

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

from deeprootgen.form import get_common_layout
from deeprootgen.io import load_data_from_file, s3_upload_file
from deeprootgen.pipeline import (
    dispatch_new_run,
    load_form_parameters,
    save_form_parameters,
    save_simulation_runs,
)

######################################
# Constants
######################################

TASK = "sensitivity_analysis"
PAGE_ID = f"{TASK.replace('_', '-')}-root-system-page"
FORM_NAME = "calibration_form"
PROCEDURE = "calibration"

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
    Output(f"{PAGE_ID}-{PROCEDURE}-collapse", "is_open"),
    [Input(f"{PAGE_ID}-{PROCEDURE}-collapse-button", "n_clicks")],
    [State(f"{PAGE_ID}-{PROCEDURE}-collapse", "is_open")],
)
def toggle_procedure_collapse(n: int, is_open: bool) -> bool:
    f"""Toggle the collapsible for {PROCEDURE} management.

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
    State({"type": f"{PAGE_ID}-{TASK}", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def save_param(
    n_clicks: int | list[int], param_inputs: list, calibration_inputs: list
) -> None:
    """Save parameter form values.

    Args:
        n_clicks (int | list[int]):
            The number of times that the button has been clicked.
        param_inputs (list):
            The parameter input data.
        calibration_inputs (list):
            The calibration parameter input data.
    """
    if n_clicks is None or len(n_clicks) == 0:  # type: ignore
        return no_update

    if n_clicks[0] is None or n_clicks[0] == 0:  # type: ignore
        return no_update

    outfile, file_name = save_form_parameters(
        PAGE_ID, FORM_NAME, param_inputs, TASK, calibration_inputs
    )
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

    if simulation_runs is None or len(simulation_runs) == 0:
        return no_update

    outfile, file_name = save_simulation_runs(simulation_runs)
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

    simulation_runs, toast_message = load_data_from_file(
        list_of_contents, list_of_names
    )
    return simulation_runs, True, toast_message


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
    Output({"type": f"{PAGE_ID}-{TASK}", "index": ALL}, "value"),
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

    inputs, toast_message = load_form_parameters(
        list_of_contents, list_of_names, FORM_NAME, TASK
    )

    parameter_inputs, calibration_inputs = inputs
    return parameter_inputs, calibration_inputs, True, toast_message


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
    form_model = app.settings[FORM_NAME]
    for i, input in enumerate(form_model.components["parameters"]["children"]):
        k = input["param"]
        form_inputs[k] = form_values[i]

    enable_soil: bool = enable_soils[0]
    form_inputs["enable_soil"] = enable_soil == True  # noqa: E712

    simulation_runs, toast_message = dispatch_new_run(
        TASK, form_inputs, simulation_runs
    )
    return simulation_runs, True, toast_message


@callback(
    Output(f"{PAGE_ID}-data-collapse", "is_open"),
    [Input(f"{PAGE_ID}-data-collapse-button", "n_clicks")],
    [State(f"{PAGE_ID}-data-collapse", "is_open")],
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
    Output(f"{PAGE_ID}-calibration-parameters-collapse", "is_open"),
    [Input(f"{PAGE_ID}-calibration-parameters-collapse-button", "n_clicks")],
    [State(f"{PAGE_ID}-calibration-parameters-collapse", "is_open")],
)
def toggle_calibration_parameters_collapse(n: int, is_open: bool) -> bool:
    """Toggle the collapsible for the calibrations parameters.

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


######################################
# Layout
######################################

register_page(__name__, name="Sensitivity Analysis", top_nav=True, path=f"/{TASK}")


def layout() -> html.Div:
    """Return the page layout.

    Returns:
        html.Div: The page layout.
    """
    title = "Run Sensitivity Analysis"
    page_description = """
    Perform a sensitivity analysis for the root system architecture simulation's parameters
    """

    layout = get_common_layout(
        title=title,
        page_id=PAGE_ID,
        page_description=page_description,
        parameter_form_name=FORM_NAME,
        procedure=PROCEDURE.title(),
        task=TASK,
    )
    return layout
