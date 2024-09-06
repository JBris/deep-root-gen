import os
import uuid

from dash import Dash, Input, Output, State, callback, dcc, html
from prefect.deployments import run_deployment

os.environ["PREFECT_UI_URL"] = "http://127.0.0.1:4200/api"
os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

app = Dash(__name__)

app.layout = html.Div(
    [
        html.Div(dcc.Input(id="input-on-submit", type="text")),
        html.Button("Submit", id="submit-val", n_clicks=0),
        html.Div(
            id="container-button-basic", children="Enter a value and press submit"
        ),
    ]
)


@callback(
    Output("container-button-basic", "children"),
    Input("submit-val", "n_clicks"),
    prevent_initial_call=True,
)
def update_output(n_clicks: int) -> str:
    run_id = str(uuid.uuid4())
    run_deployment("Hello/Hello Flow", flow_run_name=f"run-{run_id}", timeout=0)

    run_deployment("Bye/Bye Flow", flow_run_name=f"run-{run_id}", timeout=0)

    return 'The input value was "{}" and the button has been clicked {} times'.format(
        run_id, n_clicks
    )


if __name__ == "__main__":
    app.run(debug=True)
