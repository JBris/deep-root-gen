#!/usr/bin/env python

######################################
# Imports
######################################

import pandas as pd
import plotly.express as px
from dash import (
    ALL,
    Input,
    Output,
    State,
    callback,
    get_app,
    html,
    no_update,
    register_page,
)

from deeprootgen.form import get_common_layout
from deeprootgen.io import load_data_from_file
from deeprootgen.statistics import get_summary_statistic_func

px.defaults.template = "ggplot2"

######################################
# Constants
######################################

TITLE = "Exploratory Data Analysis"
TASK = "eda"
PAGE_ID = f"{TASK}-root-system-page"
FORM_NAME = "eda_form"
PROCEDURE = "exploratory-data-analysis"

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
    Output("store-eda-data", "data", allow_duplicate=True),
    Output(
        {"index": f"{PAGE_ID}-upload-obs-data-file-button", "type": ALL},
        "children",
        allow_duplicate=True,
    ),
    Output(f"{PAGE_ID}-load-toast", "is_open", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "children", allow_duplicate=True),
    Input({"index": f"{PAGE_ID}-upload-obs-data-file-button", "type": ALL}, "contents"),
    State({"index": f"{PAGE_ID}-upload-obs-data-file-button", "type": ALL}, "filename"),
    prevent_initial_call=True,
)
def load_observation_data(list_of_contents: list, list_of_names: list) -> tuple:
    """Load observed data from file.

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

    loaded_data, toast_message = load_data_from_file(list_of_contents, list_of_names)

    return loaded_data, list_of_names, True, toast_message


@callback(
    Output("store-eda-data", "data", allow_duplicate=True),
    Output(
        {"index": f"{PAGE_ID}-upload-obs-data-file-button", "type": ALL},
        "children",
        allow_duplicate=True,
    ),
    Output(
        {"type": f"{PAGE_ID}-results", "index": ALL}, "figure", allow_duplicate=True
    ),
    Output(
        {"type": f"{PAGE_ID}-parameters", "index": ALL}, "value", allow_duplicate=True
    ),
    Output(
        {"index": f"{PAGE_ID}-upload-obs-data-file-button", "type": ALL}, "contents"
    ),
    Output(f"{PAGE_ID}-load-toast", "is_open", allow_duplicate=True),
    Output(f"{PAGE_ID}-load-toast", "children", allow_duplicate=True),
    [
        Input(
            {"index": f"{PAGE_ID}-clear-obs-data-file-button", "type": ALL}, "n_clicks"
        )
    ],
    State({"index": f"{PAGE_ID}-upload-obs-data-file-button", "type": ALL}, "filename"),
    prevent_initial_call=True,
)
def clear_observation_data(n_clicks: int | list[int], list_of_names: list) -> tuple:
    """Clear observation data from the page.

    Args:
        n_clicks (int | list[int]):
            The number of form clicks.
        list_of_names (list):
            The list of file names.

    Returns:
        tuple:
            The updated form state.
    """
    if n_clicks is None or len(n_clicks) == 0:  # type: ignore
        return no_update

    if n_clicks[0] is None or n_clicks[0] == 0:  # type: ignore
        return no_update

    if list_of_names is None or len(list_of_names) == 0:
        return no_update

    if list_of_names[0] is None:
        return no_update

    button_contents = "Load observed data"
    toast_message = f"Clearing: {list_of_names[0]}"

    app = get_app()
    form_model = app.settings[FORM_NAME]
    clear_plots: list[dict] = [{} for _ in form_model.components["results"]["children"]]

    clear_form = [None for _ in form_model.components["parameters"]["children"]]

    return [], [button_contents], clear_plots, clear_form, [None], True, toast_message


@callback(
    Output({"index": f"{PAGE_ID}-select-x-axis-dropdown", "type": ALL}, "options"),
    Output({"index": f"{PAGE_ID}-select-y-axis-dropdown", "type": ALL}, "options"),
    Output({"index": f"{PAGE_ID}-select-group-by-dropdown", "type": ALL}, "options"),
    Input("store-eda-data", "data"),
    prevent_initial_call=True,
)
def update_axes(eda_data: list | None) -> tuple | None:
    """Update the data axes.

    Args:
        eda_data (list | None):
            The list of observed data.

    Returns:
        tuple | None:
            The updated axes state.
    """
    if eda_data is None:
        return no_update

    df_columns = pd.DataFrame(eda_data).columns
    columns = []
    for df_column in df_columns:
        columns.append(
            {"label": df_column.replace("_", " ").title(), "value": df_column}
        )
    return [columns], [columns], [columns]


@callback(
    Output({"index": f"{PAGE_ID}-scatter-xy-plot", "type": ALL}, "figure"),
    Output({"index": f"{PAGE_ID}-heatmap-xy-plot", "type": ALL}, "figure"),
    Input({"index": f"{PAGE_ID}-select-x-axis-dropdown", "type": ALL}, "value"),
    Input({"index": f"{PAGE_ID}-select-y-axis-dropdown", "type": ALL}, "value"),
    Input({"index": f"{PAGE_ID}-select-group-by-dropdown", "type": ALL}, "value"),
    State("store-eda-data", "data"),
    prevent_initial_call=True,
)
def update_xy_plots(
    x_axes: list | None,
    y_axes: list | None,
    group_bys: list | None,
    eda_data: list | None,
) -> tuple | None:
    """Update the plot states.

    Args:
        x_axes (list | None):
            The x axis list of values.
        y_axes (list | None):
            The y axis list of values.
        group_bys (list | None):
            The column to group data by.
        eda_data (list | None):
            The list of observed data.

    Returns:
        tuple | None:
            The updated plot states.
    """
    if eda_data is None or len(eda_data) == 0:
        return no_update
    if x_axes is None or len(x_axes) == 0:
        return no_update
    if y_axes is None or len(y_axes) == 0:
        return no_update
    if group_bys is None or len(group_bys) == 0:
        return no_update

    if x_axes[0] is None or y_axes[0] is None:
        return [{}], [{}]

    x_axis = x_axes[0]
    y_axis = y_axes[0]
    group_by = group_bys[0]

    df = pd.DataFrame(eda_data).query("order > 0")
    if group_by is not None:
        df[group_by] = df[group_by].astype("category")

    scatter_plot = px.scatter(
        title=f"{y_axis.title()} against {x_axis.title()}",
        data_frame=df,
        x=x_axis,
        y=y_axis,
        color=group_by,
    ).update_layout(xaxis_title=x_axis.title(), yaxis_title=y_axis.title())

    heatmap = px.density_heatmap(
        title=f"{y_axis.title()} against {x_axis.title()}",
        data_frame=df,
        x=x_axis,
        y=y_axis,
    ).update_layout(xaxis_title=x_axis.title(), yaxis_title=y_axis.title())

    return [scatter_plot], [heatmap]


@callback(
    Output({"index": f"{PAGE_ID}-histogram-x-plot", "type": ALL}, "figure"),
    Output({"index": f"{PAGE_ID}-box-x-plot", "type": ALL}, "figure"),
    Input({"index": f"{PAGE_ID}-select-x-axis-dropdown", "type": ALL}, "value"),
    Input({"index": f"{PAGE_ID}-select-group-by-dropdown", "type": ALL}, "value"),
    State("store-eda-data", "data"),
    prevent_initial_call=True,
)
def update_x_plots(
    x_axes: list | None, group_bys: list | None, eda_data: list | None
) -> tuple | None:
    """Update the x axis plot states.

    Args:
        x_axes (list | None):
            The x axis list of values.
        group_bys (list | None):
            The column to group data by.
        eda_data (list | None):
            The list of observed data.

    Returns:
        tuple | None:
            The updated plot states.
    """
    if eda_data is None or len(eda_data) == 0:
        return no_update
    if x_axes is None or len(x_axes) == 0:
        return no_update
    if group_bys is None or len(group_bys) == 0:
        return no_update

    if x_axes[0] is None:
        return [{}], [{}]

    x_axis = x_axes[0]
    group_by = group_bys[0]

    df = pd.DataFrame(eda_data).query("order > 0")
    if group_by is not None:
        df[group_by] = df[group_by].astype("category")

    hist_x = px.histogram(
        title=x_axis.title(),
        data_frame=df,
        x=x_axis,
    ).update_layout(xaxis_title=x_axis.title(), yaxis_title="Count")

    box_x = px.box(
        title=x_axis.title(), data_frame=df, y=x_axis, x=group_by
    ).update_layout(
        yaxis_title=x_axis.title(),
    )

    return [hist_x], [box_x]


@callback(
    Output({"index": f"{PAGE_ID}-histogram-y-plot", "type": ALL}, "figure"),
    Output({"index": f"{PAGE_ID}-box-y-plot", "type": ALL}, "figure"),
    Input({"index": f"{PAGE_ID}-select-y-axis-dropdown", "type": ALL}, "value"),
    Input({"index": f"{PAGE_ID}-select-group-by-dropdown", "type": ALL}, "value"),
    State("store-eda-data", "data"),
    prevent_initial_call=True,
)
def update_y_plots(
    y_axes: list | None, group_bys: list | None, eda_data: list | None
) -> tuple | None:
    """Update the x axis plot states.

    Args:
        y_axes (list | None):
            The y axis list of values.
        group_bys (list | None):
            The column to group data by.
        eda_data (list | None):
            The list of observed data.

    Returns:
        tuple | None:
            The updated plot states.
    """
    if eda_data is None or len(eda_data) == 0:
        return no_update
    if y_axes is None or len(y_axes) == 0:
        return no_update
    if group_bys is None or len(group_bys) == 0:
        return no_update

    if y_axes[0] is None:
        return [{}], [{}]

    y_axis = y_axes[0]
    group_by = group_bys[0]

    df = pd.DataFrame(eda_data).query("order > 0")
    if group_by is not None:
        df[group_by] = df[group_by].astype("category")

    hist_y = px.histogram(
        title=y_axis.title(),
        data_frame=df,
        x=y_axis,
    ).update_layout(xaxis_title=y_axis.title(), yaxis_title="Count")

    box_y = px.box(
        title=y_axis.title(), data_frame=df, y=y_axis, x=group_by
    ).update_layout(
        yaxis_title=y_axis.title(),
    )

    return [hist_y], [box_y]


@callback(
    Output({"index": f"{PAGE_ID}-x-summary-statistic-plot", "type": ALL}, "figure"),
    Input(
        {"index": f"{PAGE_ID}-select-x-summary-stats-dropdown", "type": ALL}, "value"
    ),
    State("store-eda-data", "data"),
    prevent_initial_call=True,
)
def update_x_statistic_plot(
    summary_stats: list | None, eda_data: list | None
) -> list | None:
    """Update the x summary statistic plot state.

    Args:
        summary_stats (list | None):
            The list of summary statistics.
        eda_data (list | None):
            The list of observed data.

    Returns:
        list | None:
            The updated plot state.
    """
    if eda_data is None or len(eda_data) == 0:
        return no_update
    if summary_stats is None or len(summary_stats) == 0:
        return no_update
    if summary_stats[0] is None:
        return [{}]

    summary_stat = summary_stats[0]
    df = pd.DataFrame(eda_data).query("order > 0")
    if len(df) == 0:
        return no_update

    kwargs = dict(root_tissue_density=df.root_tissue_density.iloc[0])
    summary_statistic_func = get_summary_statistic_func(summary_stat)
    summary_statistic_instance = summary_statistic_func(**kwargs)
    summary_stat_plot = summary_statistic_instance.visualise(df)
    return [summary_stat_plot]


@callback(
    Output({"index": f"{PAGE_ID}-y-summary-statistic-plot", "type": ALL}, "figure"),
    Input(
        {"index": f"{PAGE_ID}-select-y-summary-stats-dropdown", "type": ALL},
        "value",
    ),
    State("store-eda-data", "data"),
    prevent_initial_call=True,
)
def update_y_statistic_plot(
    summary_stats: list | None, eda_data: list | None
) -> list | None:
    """Update the y summary statistic plot state.

    Args:
        summary_stats (list | None):
            The list of summary statistics.
        eda_data (list | None):
            The list of observed data.

    Returns:
        list | None:
            The updated plot state.
    """
    if eda_data is None or len(eda_data) == 0:
        return no_update
    if summary_stats is None or len(summary_stats) == 0:
        return no_update
    if summary_stats[0] is None:
        return [{}]

    summary_stat = summary_stats[0]
    df = pd.DataFrame(eda_data).query("order > 0")
    if len(df) == 0:
        return no_update

    kwargs = dict(root_tissue_density=df.root_tissue_density.iloc[0])
    summary_statistic_func = get_summary_statistic_func(summary_stat)
    summary_statistic_instance = summary_statistic_func(**kwargs)
    summary_stat_plot = summary_statistic_instance.visualise(df)
    return [summary_stat_plot]


@callback(
    Output({"index": f"{PAGE_ID}-scatter-statistics-plot", "type": ALL}, "figure"),
    Output({"index": f"{PAGE_ID}-heatmap-statistics-plot", "type": ALL}, "figure"),
    Input(
        {"index": f"{PAGE_ID}-select-x-summary-stats-dropdown", "type": ALL},
        "value",
    ),
    Input(
        {"index": f"{PAGE_ID}-select-y-summary-stats-dropdown", "type": ALL},
        "value",
    ),
    State("store-eda-data", "data"),
    prevent_initial_call=True,
)
def update_xy_statistic_plot(
    x_summary_stats: list | None, y_summary_stats: list | None, eda_data: list | None
) -> tuple | None:
    """Update the y summary statistic plot state.

    Args:
        x_summary_stats (list | None):
            The list of summary statistics for the x axis.
        y_summary_stats (list | None):
            The list of summary statistics for the x axis.
        eda_data (list | None):
            The list of observed data.

    Returns:
        tuple | None:
            The updated plot state.
    """
    if eda_data is None or len(eda_data) == 0:
        return no_update

    if x_summary_stats is None or len(x_summary_stats) == 0:
        return no_update
    if x_summary_stats[0] is None:
        return [{}], [{}]

    if y_summary_stats is None or len(y_summary_stats) == 0:
        return no_update
    if y_summary_stats[0] is None:
        return [{}], [{}]

    df = pd.DataFrame(eda_data).query("order > 0")
    if len(df) == 0:
        return no_update

    kwargs = dict(root_tissue_density=df.root_tissue_density.iloc[0])
    x_summary_stat = x_summary_stats[0]
    y_summary_stat = y_summary_stats[0]
    x_summary_statistic_func = get_summary_statistic_func(x_summary_stat)
    x_summary_statistic_instance = x_summary_statistic_func(**kwargs)
    y_summary_statistic_func = get_summary_statistic_func(y_summary_stat)
    y_summary_statistic_instance = y_summary_statistic_func(**kwargs)

    x_data = x_summary_statistic_instance.get_xy_comparison_data(df, 10)
    y_data = y_summary_statistic_instance.get_xy_comparison_data(df, 10)
    x_label = x_summary_stat.replace("_", " ").title()
    y_label = y_summary_stat.replace("_", " ").title()
    scatter_plot = px.scatter(
        x=x_data,
        y=y_data,
    ).update_layout(xaxis_title=x_label, yaxis_title=y_label)

    heatmap = px.density_heatmap(
        x=x_data,
        y=y_data,
    ).update_layout(xaxis_title=x_label, yaxis_title=y_label)

    return [scatter_plot], [heatmap]


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
    Optimise the parameters of the root system architecture simulation against a target dataset
    """

    layout = get_common_layout(
        title=TITLE,
        page_id=PAGE_ID,
        page_description=page_description,
        parameter_form_name=FORM_NAME,
        simulation_form_name=FORM_NAME,
        procedure=TITLE,
        task=TASK,
        left_sticky=True,
        right_sticky=False,
    )
    return layout
