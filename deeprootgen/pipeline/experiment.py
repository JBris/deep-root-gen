"""Contains utilities for performing experiment tracking.

This module defines utility functions for performing experiment
tracking with MLflow.

"""

import os
import os.path as osp
import uuid
from datetime import datetime

import mlflow
import networkx as nx
import pandas as pd
import yaml
from ydata_profiling import ProfileReport

from ..data_model import RootSimulationModel
from ..model import RootSystemSimulation

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


def begin_experiment(
    task: str, simulation_uuid: str, flow_run_id: str, simulation_tag: str
) -> None:
    """Begin the experiment session.

    Args:
        task (str):
            The name of the current task for the experiment.
        simulation_uuid (str):
            The simulation uuid.
        flow_run_id (str):
            The Prefect flow run ID.
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
