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
def run_snpe() -> None:
    print("hello")


@flow(
    name="snpe",
    description="Perform Bayesian parameter estimation for the root model using Sequential Neural Posterior Estimation.",
    task_runner=ConcurrentTaskRunner(),
)
def run_snpe_flow(
    # input_params: RootSimulationModel
) -> None:
    run_snpe.submit()
