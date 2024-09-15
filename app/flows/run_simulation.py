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
def execute_simulation(input_parameters: RootSimulationModel) -> RootSystemSimulation:
    """Execute the root simulation.

    Args:
        input_parameters (RootSimulationModel):
            The root simulation data model.

    Returns:
        RootSystemSimulation:
            The root simulation.
    """
    simulation = RootSystemSimulation(
        simulation_tag=input_parameters.simulation_tag,  # type: ignore
        random_seed=input_parameters.random_seed,  # type: ignore
    )
    simulation.run(input_parameters)
    return simulation


@task
def run_simulation(input_parameters: RootSimulationModel, simulation_uuid: str) -> None:
    """Running a single root simulation.

    Args:
        input_parameters (RootSimulationModel):
            The root simulation data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    task = "simulation"
    flow_run_id = context.get_run_context().task_run.flow_run_id
    begin_experiment(
        task, simulation_uuid, flow_run_id, input_parameters.simulation_tag
    )
    log_experiment_details(simulation_uuid)
    simulation = execute_simulation(input_parameters)
    config = input_parameters.dict()

    log_config(config, task)
    log_simulation(input_parameters, simulation, task)
    save_graph_to_db(simulation, task, simulation_uuid)
    mlflow.end_run()


@flow(
    name="simulation",
    description="Run a single simulation for the root model.",
    task_runner=ConcurrentTaskRunner(),
)
def run_simulation_flow(
    input_parameters: RootSimulationModel, simulation_uuid: str
) -> None:
    """Flow for running a single root simulation.

    Args:
        input_parameters (RootSimulationModel):
            The root simulation data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    run_simulation.submit(input_parameters, simulation_uuid)