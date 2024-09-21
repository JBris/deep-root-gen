#!/usr/bin/env python

######################################
# Imports
######################################

# isort: off

# This is for compatibility with Prefect.

import multiprocessing

# isort: on


import gpytorch
import mlflow
import numpy as np
import pandas as pd
from joblib import dump as calibrator_dump
from matplotlib import pyplot as plt
from prefect import flow, task
from prefect.artifacts import create_table_artifact
from prefect.task_runners import SequentialTaskRunner
from scipy.stats import qmc

from deeprootgen.calibration import (
    AbcModel,
    calculate_summary_statistic_discrepancy,
    get_calibration_summary_stats,
    log_model,
    run_calibration_simulation,
)
from deeprootgen.data_model import RootCalibrationModel, SummaryStatisticsModel
from deeprootgen.io import save_graph_to_db
from deeprootgen.pipeline import (
    begin_experiment,
    get_datetime_now,
    get_outdir,
    log_config,
    log_experiment_details,
    log_simulation,
)
from deeprootgen.statistics import DistanceMetricBase

######################################
# Settings
######################################

multiprocessing.set_start_method("spawn", force=True)

######################################
# Constants
######################################

TASK = "surrogate"

######################################
# Main
######################################

# class SurrogateSampler:
#     def sample() -> pd.DataFrame:


@task
def run_surrogate(input_parameters: RootCalibrationModel, simulation_uuid: str) -> None:
    """Running a surrogate model on the root model.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    begin_experiment(TASK, simulation_uuid, input_parameters.simulation_tag)
    log_experiment_details(simulation_uuid)

    config = input_parameters.dict()
    log_config(config, TASK)
    mlflow.end_run()


@flow(
    name="surrogate",
    description="Train a surrogate model for the root model.",
    task_runner=SequentialTaskRunner(),
)
def run_surrogate_flow(
    input_parameters: RootCalibrationModel, simulation_uuid: str
) -> None:
    """Flow for training a surrogate model on the root model.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    run_surrogate.submit(input_parameters, simulation_uuid)
