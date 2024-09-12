"""Contains utilities for form components.

This module is composed of various utilities for form
components. These include common components and
components that are specific to a given page.

"""

from pydoc import locate

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, get_app, html


def build_common_components(
    component_specs: list,
    page_id: str,
    component_type: str,
    component_data: dict | None = None,
    resize_component: bool = True,
) -> list:
    """Build form components that are common across pages.

    Args:
        component_specs (list):
            The list of form component specifications.
        page_id (str):
            The page ID.
        component_type (str):
            The type of component. Used for grouping common components.
        component_data: (dict, optional):
            A dictionary of data to render within form component.
        resize_component: (bool, optional):
            Whether to resize the last component in the row.

    Returns:
        list: The common form components.
    """
    components = []

    row = []
    col_num = 0
    for component_spec in component_specs:
        component_label = html.P(
            html.Span(
                dbc.Label(
                    component_spec.label,
                    html_for=component_spec.id,
                    id=f"{page_id}-{component_spec.id}-label",
                ),
                id=f"{page_id}-{component_spec.id}-tooltip-target",
                style={
                    "cursor": "pointer",
                    "padding-left": "0.5em",
                    "padding-top": "0.5em",
                },
            ),
            style={"margin": "0"},
        )

        component_tooltip = dbc.Tooltip(
            component_spec.help,
            target=f"{page_id}-{component_spec.id}-tooltip-target",
            placement="right",
            delay={"show": "500"},
        )
        component = locate(component_spec.class_name)
        kwargs = component_spec.kwargs

        component_instance = component(
            id={
                "index": f"{page_id}-{component_spec.id}",
                "type": f"{page_id}-{component_type}",
            },
            **kwargs,
        )  # type: ignore
        component_instance.style = {"padding-left": "0.5em"}

        if hasattr(component_spec, "handler"):
            if component_spec.handler == "dropdown":
                options_func = locate(component_spec.options_func)
                summary_statistics = options_func()  # type: ignore
                component_instance.options = summary_statistics

            if component_spec.handler == "range_slider":
                component_instance.value = [
                    component_spec.min_value,
                    component_spec.max_value,
                ]
                component_instance.tooltip = {
                    "placement": "top",
                    "always_visible": True,
                }

            if component_spec.handler == "file_upload":
                component_instance.style = {
                    "width": "100%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "textAlign": "center",
                    "text-overflow": "ellipsis",
                    "overflow": "hidden",
                    "white-space": "nowrap",
                }

            if component_spec.handler == "data_table":
                component_instance.markdown_options = {"html": True}
                component_instance.persisted_props = ["columns.name", "data"]
                component_instance.filter_options = {
                    "placeholder_text": "Filter",
                    "case": "insensitive",
                }

                table_style = {"textAlign": "left"}
                component_instance.style_data = table_style
                component_instance.style_cell = table_style
                component_instance.style_header = table_style
                component_instance.style_filter = table_style
                component_instance.fixed_rows = {"headers": True}

                if (
                    component_data is not None
                    and component_data.get(component_spec.id) is not None
                ):
                    table_df = component_data[component_spec.id]
                    component_instance.data = table_df.to_dict("records")

                    component_instance.columns = [
                        {
                            "name": i.title(),
                            "id": i,
                            "selectable": True,
                            "presentation": "markdown",
                        }
                        for i in table_df.columns
                    ]

        row.append(dbc.Col([component_label, component_tooltip, component_instance]))

        col_num += 1
        if col_num >= 2:
            col_num = 0
            components.append(dbc.Row(row))
            row = []

    if resize_component:
        width = "52.5%"
    else:
        width = "100%"
    if len(row) == 1:
        components.append(dbc.Row(dbc.Col(row), style={"width": width}))

    return components


def build_collapsible(
    components: list,
    page_id: str,
    label: str,
) -> dbc.Row:
    """Build a collapsible form element.

    Args:
        components (list):
            A list of form components.
        page_id (str):
            The ID of the current page for grouping components.
        label (str):
            The label for the collapsible.

    Returns:
        dbc.Row: The collapsible.
    """
    ele_id = label.lower().replace(" ", "-")

    collapsible = dbc.Row(
        [
            dbc.Row(
                dbc.Button(
                    label,
                    id=f"{page_id}-{ele_id}-collapse-button",
                    className="me-1",
                    color="light",
                    n_clicks=0,
                )
            ),
            dbc.Row(
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(
                            dbc.Col(
                                components,
                            ),
                        ),
                    ),
                    id=f"{page_id}-{ele_id}-collapse",
                    is_open=True,
                    dimension="height",
                )
            ),
        ],
        id=f"{page_id}-{ele_id}-collapse-wrapper",
    )

    return collapsible


def build_common_layout(
    title: str,
    page_id: str,
    input_components: list,
    output_components: list,
    layout_description: str,
    left_sticky: bool = False,
    right_sticky: bool = True,
) -> html.Div:
    """Build a common form layout for interacting with the root model.

    Args:
        title (str):
            The page title.
        page_id (str):
            The page ID.
        input_components (list):
            The list of input form components.
        output_components (list):
            The list of modelling output components.
        layout_description (str):
            A description of the layout to add as page tooltip.
        left_sticky (bool, optional):
            Whether the left side of the page should be sticky. Defaults to False.
        right_sticky (bool, optional):
            Whether the right side of the page should be sticky. Defaults to True.

    Returns:
        html.Div:
            The common layout.
    """
    external_links = dbc.Nav(
        [
            dbc.NavItem(
                dbc.NavLink("MLflow", href="http://127.0.0.1:5000", target="_blank")
            ),
            dbc.NavItem(
                dbc.NavLink("Prefect", href="http://127.0.0.1:4200", target="_blank")
            ),
            dbc.NavItem(
                dbc.NavLink("MinIO", href="http://127.0.0.1:9001", target="_blank")
            ),
            dbc.NavItem(
                dbc.NavLink("Metabase", href="http://127.0.0.1:3000", target="_blank")
            ),
            dbc.NavItem(
                dbc.NavLink(
                    "Cloudbeaver", href="http://127.0.0.1:8978", target="_blank"
                )
            ),
            dbc.NavItem(
                dbc.NavLink("ArangoDB", href="http://127.0.0.1:8529", target="_blank")
            ),
        ],
        pills=True,
        justified=True,
        vertical="md",
    )
    external_links_collapsible = build_collapsible(
        external_links, page_id, "External Links"
    )

    sidebar_components = [
        html.H5(
            title,
            style={"margin-left": "1em", "margin-top": "0.2em", "text-align": "center"},
            id=f"{page_id}-title",
        ),
        dbc.Tooltip(
            layout_description,
            target=f"{page_id}-title",
            placement="right",
            delay={"show": "500"},
        ),
        input_components,
        dcc.Download(id=f"{page_id}-download-params"),
        dcc.Download(id=f"{page_id}-download-content"),
        dcc.Download(id=f"{page_id}-download-results"),
        dbc.Toast(
            "",
            id=f"{page_id}-load-toast",
            header="Data Notification",
            is_open=False,
            dismissable=True,
            icon="primary",
            duration=5000,
            style={"position": "fixed", "bottom": "0", "left": "0", "zIndex": "9999"},
        ),
        dbc.Toast(
            "",
            id=f"{page_id}-results-toast",
            header="Simulation Results",
            is_open=False,
            dismissable=True,
            icon="primary",
            duration=5000,
            style={"position": "fixed", "bottom": "0", "left": "0", "zIndex": "9999"},
        ),
        external_links_collapsible,
    ]

    sticky_style = {
        "position": "sticky",
        "top": "0",
        "max-height": "100vh",
    }
    left_style = {"width": "40%", "padding-right": "0"}
    right_style = {"width": "60%", "padding-left": "0", "text-align": "center"}

    if left_sticky:
        for k in sticky_style:
            left_style[k] = sticky_style[k]

    if right_sticky:
        for k in sticky_style:
            right_style[k] = sticky_style[k]

    layout = html.Div(
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        dbc.Fade(
                            id=f"{page_id}-sidebar-fade",
                            is_in=False,
                            appear=True,
                            timeout=750,
                            style={"transition": "opacity 750ms ease"},
                            children=dbc.Card(
                                sidebar_components,
                                style={"borderRadius": "0", "padding-top": "1em"},
                            ),
                        )
                    ),
                    style=left_style,
                ),
                html.Div(
                    dbc.Col(
                        dbc.Fade(
                            id=f"{page_id}-output-fade",
                            is_in=False,
                            appear=True,
                            timeout=1000,
                            style={"transition": "opacity 1000ms ease"},
                            children=dbc.Card(
                                output_components,
                                style={"borderRadius": "0", "padding-top": "1em"},
                            ),
                        )
                    ),
                    style=right_style,
                ),
            ],
            id=page_id,
        ),
    )
    return layout


def get_out_table_df() -> pd.DataFrame:
    """Get the default output table as a dataframe.

    Returns:
        pd.DataFrame:
            The output table as a dataframe.
    """
    out_df = pd.DataFrame([], columns=["workflow", "task", "date", "tag"])
    return out_df


def get_common_layout(
    title: str,
    page_id: str,
    page_description: str,
    parameter_form_name: str = "simulation_form",
    simulation_form_name: str = "simulation_form",
    procedure: str = "Simulation",
    task: str = "simulation",
    left_sticky: bool = False,
    right_sticky: bool = True,
) -> html.Div:
    """Get the common form layout for multiple dashboard pages.

    Args:
        title (str):
            The page title.
        page_id (str):
            The current page ID.
        page_description (str):
            A description of the page.
        parameter_form_name (str, optional):
            The name of the parameter form components specification. Defaults to "simulation_form".
        simulation_form_name (str, optional):
            The name of the simulation form components specification. Defaults to "simulation_form".
        procedure (str):
            The simulation procedure.
        task (str):
            The simulation task.
        left_sticky (bool, optional):
            Whether the left side of the page should be sticky. Defaults to False.
        right_sticky (bool, optional):
            Whether the right side of the page should be sticky. Defaults to True.

    Returns:
        html.Div:
            The common layout.
    """
    app = get_app()
    parameter_form = app.settings[parameter_form_name]
    simulation_form = app.settings[simulation_form_name]
    input_components = []

    k = "data"
    if parameter_form.components.get(k) is not None:
        calibration_components = build_common_components(
            parameter_form.components[k]["children"], page_id, k
        )

        if parameter_form.components[k].get("collapsible"):
            calibration_components = build_collapsible(
                calibration_components, page_id, k.title()
            )
            input_components.append(calibration_components)

    k = "parameters"
    parameter_components = build_common_components(
        parameter_form.components[k]["children"], page_id, k
    )

    if parameter_form.components[k].get("collapsible"):
        parameter_components = build_collapsible(
            parameter_components, page_id, k.title()
        )
    input_components.append(parameter_components)

    if procedure == "Calibration":
        k = task
        calibration_components = build_common_components(
            parameter_form.components[k]["children"], page_id, k
        )

        if parameter_form.components[k].get("collapsible"):
            calibration_components = build_collapsible(
                calibration_components, page_id, "Calibration Parameters"
            )
            input_components.append(calibration_components)

    k = "simulation"
    if simulation_form.components.get(k) is not None:
        data_io_components = build_common_components(
            simulation_form.components[k]["children"], page_id, k
        )

        if simulation_form.components[k]["collapsible"]:
            data_io_components = build_collapsible(
                data_io_components, page_id, procedure
            )
        input_components.append(data_io_components)

    input_components = dbc.Col(input_components)
    simulation_run_df = get_out_table_df()

    simulation_results_data = {"simulation-runs-table": simulation_run_df}

    k = "results"
    simulation_results_components = build_common_components(
        simulation_form.components[k]["children"],
        page_id,
        k,
        simulation_results_data,
        resize_component=False,
    )

    output_components = dbc.Row(
        dbc.Col(simulation_results_components, style={"margin-left": "0.5em"})
    )

    layout = build_common_layout(
        title,
        page_id,
        input_components,
        output_components,
        page_description,
        left_sticky,
        right_sticky,
    )
    return layout
