from dash import Input, Output, callback, dash_table, html, register_page

register_page(__name__, name="Home", top_nav=True, path="/")


@callback(
    Output("table-content", "data"),
    Output("table-content", "page_size"),
    Input("store", "data"),
)
def get_table(data: dict) -> tuple:
    return data, 10


def layout() -> html.Div:
    layout = html.Div([dash_table.DataTable(id="table-content")])
    return layout
