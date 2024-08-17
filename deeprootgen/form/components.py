"""Contains utilities for form components.

This module is composed of various utilities for form
components. These include common components and
components that are specific to a given page.

"""

from pydoc import locate

import dash_bootstrap_components as dbc
from dash import html


def build_common_components(
    component_specs: list, page_id: str, component_type: str
) -> list:
    """Build form components that are common across pages.

    Args:
        component_specs (list):
            The list of form component specifications.
        page_id (str):
            The page ID.
        component_type (str):
            The type of component. Used for grouping common components.

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
                style={"cursor": "pointer"},
            ),
            style={"margin": "0"},
        )

        component_tooltip = dbc.Tooltip(
            component_spec.help,
            target=f"{page_id}-{component_spec.id}-tooltip-target",
            placement="right",
        )
        component = locate(component_spec.class_name)
        component_instance = component(
            id={
                "index": f"{page_id}-{component_spec.id}",
                "type": f"{page_id}-{component_type}",
            },
            **component_spec.kwargs,
        )  # type: ignore

        row.append(dbc.Col([component_label, component_tooltip, component_instance]))

        col_num += 1
        if col_num >= 2:
            col_num = 0
            components.append(dbc.Row(row))
            row = []

    if len(row) == 1:
        components.append(dbc.Row(row))

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
    title: str, page_id: str, input_components: list, output_components: list
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
        ),
        input_components,
        external_links_collapsible,
    ]
    layout = html.Div(
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        dbc.Card(
                            sidebar_components,
                            style={"borderRadius": "0"},
                        ),
                    ),
                    style={"width": "40%", "padding-right": "0"},
                ),
                html.Div(
                    dbc.Col(dbc.Card(output_components, style={"borderRadius": "0"})),
                    style={"width": "60%", "padding-left": "0"},
                ),
            ],
            id=page_id,
        ),
    )
    return layout
