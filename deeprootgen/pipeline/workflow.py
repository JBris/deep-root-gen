"""Contains utilities for managing workflows.

This module defines utility functions for managing and orchestrating
workflows with Prefect.

"""

import os

import mlflow
from prefect import task
from prefect.artifacts import create_markdown_artifact


@task
def log_experiment_details(simulation_uuid: str) -> None:
    """Log the experiment details.

    Args:
        simulation_uuid (str):
            The simulation uuid.
    """
    run = mlflow.active_run()
    experiment_id = run.info.experiment_id
    run_id = run.info.run_id

    app_url = os.environ.get("APP_USER_HOST")
    if app_url is None:
        app_url = "http://localhost:8000"

    app_mlflow_host = os.environ.get("APP_MLFLOW_USER_HOST")
    if app_mlflow_host is None:
        app_mlflow_host = "http://localhost:5000"
    mlflow_experiment_url = (
        f"{app_mlflow_host}/#/experiments/{experiment_id}/runs/{run_id}"
    )

    flow_description = f"""
# DeepRootGen URL

Simulation UUID: {simulation_uuid}

<{app_url}>

# MLflow experiment URL

Run ID: {run_id}

<{mlflow_experiment_url}>
    """

    create_markdown_artifact(
        key="root-simulation-run",
        markdown=flow_description,
        description="Root simulation run.",
    )
