#!/usr/bin/env python

######################################
# Imports
######################################

# isort: off

# This is for compatibility with Prefect.

import multiprocessing

# isort: on


import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import pyabc
from joblib import dump as calibrator_dump
from matplotlib import pyplot as plt
from prefect import flow, task
from prefect.artifacts import create_table_artifact
from prefect.task_runners import SequentialTaskRunner

from deeprootgen.calibration import (
    SensitivityAnalysisModel,
    calculate_summary_statistic_discrepancy,
    get_calibration_summary_stats,
    log_model,
)
from deeprootgen.data_model import RootCalibrationModel, SummaryStatisticsModel
from deeprootgen.pipeline import (
    begin_experiment,
    get_datetime_now,
    get_outdir,
    log_config,
    log_experiment_details,
)
from deeprootgen.statistics import DistanceMetricBase

######################################
# Settings
######################################

multiprocessing.set_start_method("spawn", force=True)

######################################
# Constants
######################################

TASK = "abc"

######################################
# Main
######################################


@task
def prepare_task(input_parameters: RootCalibrationModel) -> tuple:
    """Prepare the Bayesian parameter estimation procedure.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.

    Raises:
        ValueError:
            Error thrown when the summary statistic list is empty.

    Returns:
        tuple:
            The Bayesian model specification.
    """
    distance, statistics_list = get_calibration_summary_stats(input_parameters)
    observed_values = [statistic.statistic_value for statistic in statistics_list]

    return distance, statistics_list, observed_values


@task
def run_abc(input_parameters: RootCalibrationModel, simulation_uuid: str) -> None:
    """Running Approximate Bayesian Computation.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    begin_experiment(TASK, simulation_uuid, input_parameters.simulation_tag)
    log_experiment_details(simulation_uuid)

    distance, statistics_list, observed_values = prepare_task(input_parameters)

    def distance_func(x: dict, _: dict) -> float:
        return x["discrepancy"]

    parameter_intervals = input_parameters.parameter_intervals.dict()
    # calibration_parameters = input_parameters.calibration_parameters
    names = []
    data_types = {}
    dist_kwargs = {}
    for name, v in parameter_intervals.items():
        names.append(name)

        lower_bound = v["lower_bound"]
        upper_bound = v["upper_bound"]

        data_type = v["data_type"]
        data_types[name] = data_type

        dist_kwargs[name] = pyabc.RV("uniform", lower_bound, upper_bound - lower_bound)

    prior = pyabc.Distribution(**dist_kwargs)

    def simulator_func(theta: dict) -> dict:
        parameter_specs = theta.copy()
        for name in names:
            data_type = data_types[name]
            if data_type == "discrete":
                parameter_specs[name] = int(parameter_specs[name])

        discrepancy = calculate_summary_statistic_discrepancy(
            parameter_specs, input_parameters, statistics_list, distance
        )

        return {"discrepancy": discrepancy}

    abc = pyabc.ABCSMC(
        simulator_func,
        prior,
        distance_func,
        population_size=2,
        sampler=pyabc.SingleCoreSampler(check_max_eval=True),
    )

    abc.new("sqlite://")

    # history = abc.run(minimum_epsilon=0.1, max_nr_populations=2)

    config = input_parameters.dict()
    log_config(config, TASK)
    mlflow.end_run()


@flow(
    name="abc",
    description="Perform Bayesian parameter estimation for the root model using Approximate Bayesian Computation.",
    task_runner=SequentialTaskRunner(),
)
def run_abc_flow(input_parameters: RootCalibrationModel, simulation_uuid: str) -> None:
    """Flow for running Approximate Bayesian Computation.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    run_abc.submit(input_parameters, simulation_uuid)
