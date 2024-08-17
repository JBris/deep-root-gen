"""Contains utilities for form components.

This module is composed of various utilities for form
components. These include common components and
components that are specific to a given page.

"""

from pydoc import locate

import dash_bootstrap_components as dbc
from dash import html


def build_common_components(component_specs: list, page_id: str) -> list:
    """Build form components that are common across pages.

    Args:
        component_specs (list):
            The list of form component specifications.
        page_id (str):
            The page ID.

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
            id=f"{page_id}-{component_spec.id}", **component_spec.kwargs
        )  # type: ignore

        row.append(dbc.Col([component_label, component_tooltip, component_instance]))

        col_num += 1
        if col_num >= 2:
            col_num = 0
            components.append(dbc.Row(row))
            row = []

    return components


def build_common_layout(title: str, page_id: str, components: list) -> html.Div:
    """Build a common form layout for interacting with the root model.

    Args:
        title (str):
            The page title.
        page_id (str):
            The page ID.
        components (list):
            The list of form components.

    Returns:
        html.Div:
            The common layout.
    """
    parameter_components = [
        html.H5(
            title,
            style={"margin-left": "1em", "margin-top": "0.2em", "text-align": "center"},
        ),
        dbc.Button(
            "Toggle Parameters",
            id=f"{page_id}-parameters-collapse-button",
            className="me-1",
            color="light",
            n_clicks=0,
        ),
        dbc.Collapse(
            dbc.Card(
                dbc.CardBody(
                    dbc.Col(components),
                ),
            ),
            id=f"{page_id}-parameters-collapse",
            is_open=True,
            dimension="height",
        ),
    ]
    layout = html.Div(
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        dbc.Card(
                            parameter_components,
                            style={"borderRadius": "0"},
                        ),
                    ),
                    style={"width": "40%", "padding-right": "0"},
                ),
                html.Div(
                    dbc.Col(dbc.Card(components, style={"borderRadius": "0"})),
                    style={"width": "60%", "padding-left": "0"},
                ),
            ],
            id=page_id,
        ),
    )
    return layout
