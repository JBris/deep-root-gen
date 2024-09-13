#!/usr/bin/env python

######################################
# Imports
######################################

import mlflow
from prefect import context, flow, task
from prefect.task_runners import ConcurrentTaskRunner

from deeprootgen.data_model import RootCalibrationModel
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
def run_sensitivity_analysis(
    input_parameters: RootCalibrationModel, simulation_uuid: str
) -> None:
    """Running a sensitivity analysis.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    print("hello")


@flow(
    name="sensitivity_analysis",
    description="Run a sensitivity analysis for the root model.",
    task_runner=ConcurrentTaskRunner(),
)
def run_sensitivity_analysis_flow(
    input_parameters: RootCalibrationModel, simulation_uuid: str
) -> None:
    """Flow for running a sensitivity analysis.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    run_sensitivity_analysis.submit(input_parameters, simulation_uuid)
