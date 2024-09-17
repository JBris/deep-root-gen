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
    html,
    no_update,
    register_page,
)

from deeprootgen.form import build_calibration_parameters, get_common_layout
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

TASK = "snpe"
PAGE_ID = f"{TASK}-root-system-page"
FORM_NAME = "calibration_form"
PROCEDURE = "calibration"
DATA_COMPONENT_KEY = "simulated_data"

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
    Output({"index": f"{PAGE_ID}-save-runs-button", "type": ALL}, "disabled"),
    Output({"index": f"{PAGE_ID}-clear-runs-button", "type": ALL}, "disabled"),
    Input("store-simulation-run", "data"),
)
def update_table(runs: list | None) -> tuple | None:
    """Update the simulation run table.

    Args:
        runs (list | None):
            The list of simulation runs.

    Returns:
        tuple | None:
            The updated form state.
    """
    if runs is None:
        return no_update
    if len(runs) == 0:
        return [runs], [True], [True]

    return [runs], [False], [False]


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

    simulation_runs, _, toast_message = load_data_from_file(
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
    Output(f"{PAGE_ID}-simulated-data-collapse", "is_open"),
    [Input(f"{PAGE_ID}-simulated-data-collapse-button", "n_clicks")],
    [State(f"{PAGE_ID}-simulated-data-collapse", "is_open")],
)
def toggle_data_collapse(n: int, is_open: bool) -> bool:
    """Toggle the collapsible for simulated data.

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


@callback(
    Output(
        {"index": f"{PAGE_ID}-upload-summary-data-file-button", "type": ALL}, "children"
    ),
    Output(
        {"index": f"{PAGE_ID}-upload-obs-data-file-button", "type": ALL}, "children"
    ),
    Output(
        {"index": f"{PAGE_ID}-upload-edge-data-file-button", "type": ALL}, "children"
    ),
    Output({"index": f"{PAGE_ID}-run-sim-button", "type": ALL}, "disabled"),
    Output({"index": f"{PAGE_ID}-clear-obs-data-file-button", "type": ALL}, "disabled"),
    Input("store-summary-data", "data"),
    Input("store-simulation-data", "data"),
    Input("store-edge-data", "data"),
    Input({"index": f"{PAGE_ID}-select-summary-stats-dropdown", "type": ALL}, "value"),
    Input({"index": f"{PAGE_ID}-use-summary-statistics-switch", "type": ALL}, "on"),
)
def update_uploaded_data_state(
    summary_data: dict | None,
    simulation_data: dict | None,
    edge_data: dict | None,
    summary_stats: list,
    use_summary_stats: list[bool],
) -> tuple:
    """Update the state of the uploaded data.

    Args:
        summary_data (dict | None):
            The summary statistics data.
        simulation_data (dict | None):
            The root simulation data.
        edge_data (dict | None):
            The graph edge data.
        summary_stats (list):
            The list of summary statistics.
        use_summary_stats (list):
            Whether to use summary statistics, rather than graph data.

    Returns:
        tuple:
            The updated form state.
    """

    def set_bttn_label(data: dict, data_type: str, k: str) -> str:
        if data is not None and data.get(k) is not None:
            bttn_label = data[k]
        else:
            bttn_label = f"Load {data_type} data"
        return bttn_label

    stats_bttn = set_bttn_label(summary_data, "statistics", "label")  # type: ignore[arg-type]
    sim_bttn = set_bttn_label(simulation_data, "statistics", "label")  # type: ignore[arg-type]
    edge_bttn = set_bttn_label(edge_data, "statistics", "label")  # type: ignore[arg-type]

    clear_disabled = True
    for data in [summary_data, simulation_data, edge_data]:
        if isinstance(data, list) and len(data) > 0:
            clear_disabled = False
            break
        if isinstance(data, dict):
            data_values = data.get("values", [])
            if len(data_values) > 0:
                clear_disabled = False
                break

    run_disabled = False
    use_summary_stat = use_summary_stats[0]
    if use_summary_stat:
        data_list = [summary_data]
    else:
        data_list = [simulation_data, edge_data]

    for data in data_list:
        if data is None:
            run_disabled = True
            break
        data_values = data.get("values", [])
        if len(data_values) == 0:
            run_disabled = True
            break
        if data_values[0] is None:
            run_disabled = True
            break

    stats_list = summary_stats[0]
    if use_summary_stat and not run_disabled:
        if len(stats_list) == 0:
            run_disabled = True
        elif stats_list[0] is None:
            run_disabled = True

    return [stats_bttn], [sim_bttn], [edge_bttn], [run_disabled], [clear_disabled]


@callback(
    Output("store-summary-data", "data", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "is_open", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "children", allow_duplicate=True),
    Input(
        {"index": f"{PAGE_ID}-upload-summary-data-file-button", "type": ALL}, "contents"
    ),
    State(
        {"index": f"{PAGE_ID}-upload-summary-data-file-button", "type": ALL}, "filename"
    ),
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
    Output("store-simulation-data", "data", allow_duplicate=True),
    Output("store-raw-simulation-data", "data", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "is_open", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "children", allow_duplicate=True),
    Input({"index": f"{PAGE_ID}-upload-obs-data-file-button", "type": ALL}, "contents"),
    State({"index": f"{PAGE_ID}-upload-obs-data-file-button", "type": ALL}, "filename"),
    prevent_initial_call=True,
)
def load_simulation_data(list_of_contents: list, list_of_names: list) -> tuple:
    """Load simulation data from file.

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

    loaded_data, content_string, toast_message = load_data_from_file(
        list_of_contents, list_of_names
    )
    summary_data = {"label": list_of_names[0], "values": loaded_data}
    return summary_data, {"values": content_string}, True, toast_message


@callback(
    Output("store-edge-data", "data", allow_duplicate=True),
    Output("store-raw-edge-data", "data", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "is_open", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "children", allow_duplicate=True),
    Input(
        {"index": f"{PAGE_ID}-upload-edge-data-file-button", "type": ALL}, "contents"
    ),
    State(
        {"index": f"{PAGE_ID}-upload-edge-data-file-button", "type": ALL}, "filename"
    ),
    prevent_initial_call=True,
)
def load_edge_data(list_of_contents: list, list_of_names: list) -> tuple:
    """Load edge data from file.

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

    loaded_data, content_string, toast_message = load_data_from_file(
        list_of_contents, list_of_names
    )
    summary_data = {"label": list_of_names[0], "values": loaded_data}
    return summary_data, {"values": content_string}, True, toast_message


@callback(
    Output("store-summary-data", "data", allow_duplicate=True),
    Output("store-simulation-data", "data", allow_duplicate=True),
    Output("store-edge-data", "data", allow_duplicate=True),
    Output("store-raw-simulation-data", "data", allow_duplicate=True),
    Output("store-raw-edge-data", "data", allow_duplicate=True),
    Output(
        {"index": f"{PAGE_ID}-upload-summary-data-file-button", "type": ALL}, "contents"
    ),
    Output(
        {"index": f"{PAGE_ID}-upload-obs-data-file-button", "type": ALL}, "contents"
    ),
    Output(
        {"index": f"{PAGE_ID}-upload-edge-data-file-button", "type": ALL}, "contents"
    ),
    Output(f"{PAGE_ID}-load-toast", "is_open", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "children", allow_duplicate=True),
    Input({"index": f"{PAGE_ID}-clear-obs-data-file-button", "type": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def clear_summary_data(n_clicks: int | list[int]) -> tuple:
    """Clear summary data from the page.

    Args:
        n_clicks (int | list[int]):
            The number of form clicks.

    Returns:
        tuple:
            The updated form state.
    """
    if n_clicks is None or len(n_clicks) == 0:  # type: ignore
        return no_update

    if n_clicks[0] is None or n_clicks[0] == 0:  # type: ignore
        return no_update

    toast_message = "Clearing all uploaded data"
    return {}, {}, {}, {}, {}, [None], [None], [None], True, toast_message


@callback(
    Output("store-simulation-run", "data", allow_duplicate=True),
    Output(f"{PAGE_ID}-results-toast", "is_open"),
    Output(f"{PAGE_ID}-results-toast", "children"),
    Input({"index": f"{PAGE_ID}-run-sim-button", "type": ALL}, "n_clicks"),
    State({"type": f"{PAGE_ID}-parameters", "index": ALL}, "value"),
    State({"type": f"{PAGE_ID}-{TASK}", "index": ALL}, "value"),
    State({"index": f"{PAGE_ID}-use-summary-statistics-switch", "type": ALL}, "on"),
    State({"index": f"{PAGE_ID}-stat-by-soil-layer-switch", "type": ALL}, "on"),
    State({"index": f"{PAGE_ID}-stat-by-soil-col-switch", "type": ALL}, "on"),
    State("store-simulation-run", "data"),
    State("store-summary-data", "data"),
    prevent_initial_call=True,
)
def run_root_model(
    n_clicks: list,
    parameter_values: list,
    calibration_values: list,
    use_summary_stats: list[bool],
    stats_by_layer: list[bool],
    stats_by_col: list[bool],
    simulation_runs: list,
    summary_data: dict,
) -> tuple:
    """Run and plot the root model.

    Args:
        n_clicks (list):
            Number of times the button has been clicked.
        parameter_values (list):
            The parameter form input data.
        calibration_values (list):
            The calibration parameter form input data.
        use_summary_stats (list):
            Whether to use summary statistics, rather than graph data.
        stats_by_layer (list):
            Whether to calculate statistics by soil layer.
        stats_by_col (list):
            Whether to calculate statistics by soil column.
        simulation_runs (list):
            A list of simulation run data.
        summary_data: (dict):
            The dictionary of observed summary statistic data.

    Returns:
        tuple:
            The updated form state.
    """
    if n_clicks is None or len(n_clicks) == 0:
        return no_update

    if n_clicks[0] is None or n_clicks[0] == 0:
        return no_update

    use_summary_stats[0]
    stat_by_layer = stats_by_layer[0]
    stat_by_col = stats_by_col[0]

    summary_statistics = summary_data.get("values", None)
    if summary_statistics is None:
        return no_update

    form_inputs = build_calibration_parameters(
        FORM_NAME,
        TASK,
        parameter_values,
        calibration_values,
        summary_statistics=summary_statistics,
        stat_by_layer=stat_by_layer,
        stat_by_col=stat_by_col,
    )
    if form_inputs is None:
        return no_update

    simulation_runs, toast_message = dispatch_new_run(
        TASK, form_inputs, simulation_runs
    )
    return simulation_runs, True, toast_message


######################################
# Layout
######################################

register_page(
    __name__,
    name="Sequential Neural Posterior Estimation",
    top_nav=True,
    path=f"/{TASK}",
)


def layout() -> html.Div:
    """Return the page layout.

    Returns:
        html.Div: The page layout.
    """
    title = "Run SNPE"
    page_description = """
    Perform Bayesian parameter estimation for the root system architecture simulation against a target dataset using deep learning
    """

    layout = get_common_layout(
        title=title,
        page_id=PAGE_ID,
        page_description=page_description,
        parameter_form_name=FORM_NAME,
        procedure=PROCEDURE.title(),
        task=TASK,
        data_key=DATA_COMPONENT_KEY,
    )
    return layout
