#!/usr/bin/env python

######################################
# Imports
######################################

import mlflow
from prefect import context, flow, task
from prefect.task_runners import ConcurrentTaskRunner

from deeprootgen.data_model import RootSimulationModel
from deeprootgen.model import RootSystemSimulation
from deeprootgen.pipeline import (
    begin_experiment,
    log_config,
    log_experiment_details,
    log_simulation,
)

######################################
# Main
######################################


@task
def run_sensitivity_analysis() -> None:
    print("hello")


@flow(
    name="sensitivity_analysis",
    description="Run a sensitivity analysis for the root model.",
    task_runner=ConcurrentTaskRunner(),
)
def run_sensitivity_analysis_flow(
    # input_params: RootSimulationModel
) -> None:
    run_sensitivity_analysis.submit()
