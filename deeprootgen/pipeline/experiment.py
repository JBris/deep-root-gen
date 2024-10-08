"""Contains utilities for performing experiment tracking.

This module defines utility functions for performing experiment
tracking with MLflow.

"""

import base64
import os
import os.path as osp
import uuid
from datetime import datetime

import mlflow
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from dash import get_app
from prefect import context, task
from prefect.deployments import run_deployment
from ydata_profiling import ProfileReport

from ..data_model import RootSimulationModel
from ..model import RootSystemSimulation
from ..statistics import get_summary_statistic_func, get_summary_statistics

OUT_DIR = osp.join("/app", "outputs")


def get_datetime_now() -> str:
    """Get the current datetime for now.

    Returns:
        str:
            The current datetime.
    """
    return datetime.today().strftime("%Y-%m-%d-%H-%M-%S")


def get_outdir() -> str:
    """Get the output directory.

    Returns:
        str:
            The output directory.
    """
    return OUT_DIR


def get_simulation_uuid() -> str:
    """Get a new simulation uuid.

    Returns:
        str:
            The simulation uuid.
    """
    simulation_uuid = str(uuid.uuid4())
    return simulation_uuid


@task
def begin_experiment(task: str, simulation_uuid: str, simulation_tag: str) -> None:
    """Begin the experiment session.

    Args:
        task (str):
            The name of the current task for the experiment.
        simulation_uuid (str):
            The simulation uuid.
        simulation_tag (str):
            The tag for the current root model simulation.
    """
    experiment_name = f"root_model_{task}"
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    app_url = os.environ.get("APP_USER_HOST")
    if app_url is None:
        app_url = "http://localhost:8000"

    app_prefect_host = os.environ.get("APP_PREFECT_USER_HOST")
    if app_prefect_host is None:
        app_prefect_host = "http://localhost:4200"

    flow_run_id = context.get_run_context().task_run.flow_run_id
    prefect_flow_url = f"{app_prefect_host}/flow-runs/flow-run/{flow_run_id}"

    run_description = f"""
# DeepRootGen URL

<{app_url}>

# Prefect flow URL

<{prefect_flow_url}>
    """
    mlflow.set_tag("mlflow.note.content", run_description)
    mlflow.set_tag("task", task)
    mlflow.set_tag("simulation_uuid", simulation_uuid)
    mlflow.set_tag("flow_run_id", flow_run_id)
    mlflow.set_tag("app_url", app_url)
    mlflow.set_tag("prefect_flow_url", prefect_flow_url)
    mlflow.set_tag("simulation_tag", simulation_tag)


@task
def log_config(
    config: dict,
    task: str,
) -> str:
    """Log the simulation configuration.

    Args:
        config (dict):
            The simulation configuration as a dictionary.
        task (str):
            The task name.

    Returns:
        str:
            The written configuration file.
    """
    for k, v in config.items():
        mlflow.log_param(k, v)

    outfile = osp.join(OUT_DIR, f"{get_datetime_now()}-{task}_config.yaml")
    with open(outfile, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    mlflow.log_artifact(outfile)
    return outfile


@task
def calculate_graph_metrics(G: nx.Graph, time_now: str, task: str) -> str:
    """Calculate graph metrics.

    Args:
        G (nx.Graph):
            The NetworkX graph.
        time_now (str):
            The current time.
        task (str):
            The current simulation task.

    Returns:
        str:
            The graph metric file.
    """
    metric_names = []
    metric_values = []
    for metric_func in [
        nx.diameter,
        nx.radius,
        nx.average_clustering,
        nx.node_connectivity,
        nx.degree_assortativity_coefficient,
        nx.degree_pearson_correlation_coefficient,
    ]:
        metric_name = metric_func.__name__
        metric_value = metric_func(G)

        mlflow.log_metric(metric_name, metric_value)
        metric_names.append(metric_name)
        metric_values.append(metric_value)

    metric_df = pd.DataFrame(
        {"metric_name": metric_names, "metric_value": metric_values}
    )
    outfile = osp.join(OUT_DIR, f"{time_now}-{task}_graph_metrics.csv")
    metric_df.to_csv(outfile, index=False)
    mlflow.log_artifact(outfile)
    return outfile


@task
def log_simulation(
    input_parameters: RootSimulationModel, simulation: RootSystemSimulation, task: str
) -> None:
    """Log details for the current simulation.

    Args:
        input_parameters (RootSimulationModel):
            The root simulation data model.
        simulation (RootSystemSimulation):
            The root system simulation instance.
        task (str):
            The task name.
    """
    node_df, edge_df = simulation.G.as_df()
    G = simulation.G.as_networkx()
    time_now = get_datetime_now()

    outfile = osp.join(OUT_DIR, f"{time_now}-{task}_nodes.csv")
    node_df.to_csv(outfile, index=False)
    mlflow.log_artifact(outfile)

    outfile = osp.join(OUT_DIR, f"{time_now}-{task}_edges.csv")
    edge_df.to_csv(outfile, index=False)
    mlflow.log_artifact(outfile)

    fig = simulation.init_fig(input_parameters)
    fig = simulation.plot_root_system(fig, node_df)
    outfile = osp.join(OUT_DIR, f"{time_now}-{task}_roots.html")
    fig.write_html(outfile)
    mlflow.log_artifact(outfile)

    fig = simulation.plot_hierarchical_graph(G)
    outfile = osp.join(OUT_DIR, f"{time_now}-{task}_hgraph.html")
    fig.write_html(outfile)
    mlflow.log_artifact(outfile)

    profile = ProfileReport(node_df, title="Root Model Report")
    outfile = osp.join(OUT_DIR, f"{time_now}-{task}_data_profile.html")
    profile.to_file(outfile)
    mlflow.log_artifact(outfile)

    kwargs = dict(root_tissue_density=input_parameters.root_tissue_density)
    statistic_names = []
    statistic_values = []
    summary_statistics = get_summary_statistics()
    for summary_statistic in summary_statistics:
        statistic_name = summary_statistic["value"]
        statistic_func = get_summary_statistic_func(statistic_name)
        statistic_instance = statistic_func(**kwargs)
        statistic_value = statistic_instance.calculate(node_df)

        # @TODO calculate by depth and horizontal distance
        if isinstance(statistic_value, tuple) or isinstance(statistic_value, tuple):
            statistic_name = f"average_{statistic_name}"
            statistic_value = np.array(statistic_value).flatten().mean()

        statistic_names.append(statistic_name)
        statistic_values.append(statistic_value)
        mlflow.log_metric(statistic_name, statistic_value)

    statistic_df = pd.DataFrame(
        {"statistic_name": statistic_names, "statistic_value": statistic_values}
    )
    outfile = osp.join(OUT_DIR, f"{time_now}-{task}_summary_statistics.csv")
    statistic_df.to_csv(outfile, index=False)
    mlflow.log_artifact(outfile)


def load_form_parameters(
    list_of_contents: list, list_of_names: list, form_name: str, task: str = ""
) -> tuple:
    """Load form parameters from file to a list.

    Args:
        list_of_contents (list):
            The uploaded list of contents.
        list_of_names (list):
            The list of file names.
        form_name (str):
            The current form name.
        task (str, optional):
            The current simulation task.

    Returns:
        tuple:
            The form inputs and toast message.
    """
    _, content_string = list_of_contents[0].split(",")
    decoded = base64.b64decode(content_string).decode("utf-8")
    input_dict = yaml.safe_load(decoded)

    app = get_app()
    form_model = app.settings[form_name]

    form_inputs = []
    for input in form_model.components["parameters"]["children"]:
        k = input["param"]
        form_inputs.append(input_dict[k])

    if task == "":
        inputs = form_inputs
    else:
        inputs = [form_inputs]
        calibration_inputs = []
        for input in form_model.components[task]["children"]:
            k = input["param"]
            calibration_inputs.append(input_dict[k])
        inputs.append(calibration_inputs)

    toast_message = f"Loading parameter specification from: {list_of_names[0]}"
    return inputs, toast_message


def save_form_parameters(
    page_id: str,
    form_name: str,
    param_inputs: list,
    task: str = "",
    calibration_inputs: list | None = None,
) -> tuple:
    """Write the current form parameters to file.

    Args:
        page_id (str):
            The current page ID.
        form_name (str):
            The name of the form component definitions.
        param_inputs (list):
            The list of parameter inputs.
        task (str, optional):
            The current simulation task. Defaults to ''.
        calibration_inputs (list | None, optional):
            The list of calibration parameter inputs. Defaults to None

    Returns:
        tuple:
            The output file and file name.
    """
    inputs = {}
    app = get_app()
    form_model = app.settings[form_name]
    for i, input in enumerate(form_model.components["parameters"]["children"]):
        k = input["param"]
        inputs[k] = param_inputs[i]

    if calibration_inputs is not None:
        for i, input in enumerate(form_model.components[task]["children"]):
            k = input["param"]
            inputs[k] = calibration_inputs[i]

    file_name = f"{get_datetime_now()}-{page_id}.yaml"
    outfile = osp.join(OUT_DIR, file_name)
    with open(outfile, "w") as f:
        yaml.dump(inputs, f, default_flow_style=False, sort_keys=False)

    return outfile, file_name


def save_simulation_runs(simulation_runs: list) -> tuple:
    """Save the current simulation runs to file.

    Args:
        simulation_runs (list):
            The list of simulation run data.

    Returns:
        tuple:
            The output file and file name.
    """
    df = pd.DataFrame(simulation_runs)
    date_now = get_datetime_now()
    file_name = f"{date_now}-root-simulation-runs.csv"
    outfile = osp.join("outputs", file_name)
    df.to_csv(outfile, index=False)
    return outfile, file_name


def dispatch_new_run(task: str, form_inputs: dict, simulation_runs: list) -> tuple:
    """Dispatch a new simulation run.

    Args:
        task (str):
            The name of the current task for the experiment.
        form_inputs (dict):
            The dictionary of form input data to pass as simulation parameters.
        simulation_runs (list):
            The list of simulation run data.

    Returns:
        tuple:
            The output file and file name.
    """
    simulation_uuid = get_simulation_uuid()
    flow_data = run_deployment(
        f"{task}/run_{task}_flow",
        parameters=dict(input_parameters=form_inputs, simulation_uuid=simulation_uuid),
        flow_run_name=simulation_uuid,
        timeout=0,
    )

    app_prefect_host = os.environ.get("APP_PREFECT_USER_HOST", "http://localhost:4200")
    flow_run_id = str(flow_data.id)
    prefect_flow_url = f"{app_prefect_host}/flow-runs/flow-run/{flow_run_id}"
    simulation_tag = form_inputs["simulation_tag"]

    simulation_runs.append(
        {
            "workflow": f"<a href='{prefect_flow_url}' target='_blank'>{simulation_uuid}</a>",
            "task": task,
            "date": get_datetime_now(),
            "tag": simulation_tag,
        }
    )

    toast_message = f"""
    Running simulation workflow: {simulation_uuid}
    Simulation tag: {simulation_tag}
    """

    return simulation_runs, toast_message
