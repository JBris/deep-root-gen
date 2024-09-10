#!/usr/bin/env python

######################################
# Imports
######################################

import mlflow
from prefect import context, flow, task
from prefect.task_runners import ConcurrentTaskRunner

from deeprootgen.data_model import RootSimulationModel
from deeprootgen.io import save_graph_to_db
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
def run_optimisation(
    input_parameters: RootSimulationModel, simulation_uuid: str
) -> None:
    """Running a optimisation procedure.

    Args:
        input_parameters (RootSimulationModel):
            The root simulation data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    print("hello")


@flow(
    name="optimisation",
    description="Run an optimisation procedure for the root model.",
    task_runner=ConcurrentTaskRunner(),
)
def run_optimisation_flow(
    input_parameters: RootSimulationModel, simulation_uuid: str
) -> None:
    """Flow for running a optimisation procedure.

    Args:
        input_parameters (RootSimulationModel):
            The root simulation data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    run_optimisation.submit(input_parameters, simulation_uuid)
