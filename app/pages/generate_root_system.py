#!/usr/bin/env python

######################################
# Imports
######################################

from dash import Input, Output, State, callback, get_app, html, register_page

from deeprootgen.form import build_common_components, build_common_layout

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
    Output(f"{PAGE_ID}-data-io-collapse", "is_open"),
    [Input(f"{PAGE_ID}-data-io-collapse-button", "n_clicks")],
    [State(f"{PAGE_ID}-data-io-collapse", "is_open")],
)
def toggle_data_io_collapse(n: int, is_open: bool) -> bool:
    """Toggle the collapsible for data IO.

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

register_page(__name__, name="Generate Root Data", top_nav=True, path="/")


def layout() -> html.Div:
    """Return the page layout.

    Returns:
        html.Div: The page layout.
    """
    app = get_app()
    form_model = app.settings["form"]

    components = build_common_components(form_model.components["inputs"], PAGE_ID)

    layout = build_common_layout("Run Simulation", PAGE_ID, components)

    return layout
