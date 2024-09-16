#!/usr/bin/env python

######################################
# Imports
######################################


import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import pymc as pm
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
# Constants
######################################

TASK = "abc"

######################################
# Main
######################################


def distance_func(e: float, observed: np.ndarray, simulated: np.ndarray) -> float:
    print(simulated)
    return simulated.item()


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

    # distance, statistics_list = get_calibration_summary_stats(input_parameters)
    # [statistic.statistic_value for statistic in statistics_list]

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

    # names = []
    # priors = []
    # with pm.Model() as model:
    #     parameter_intervals = input_parameters.parameter_intervals.dict()
    #     for name, v in parameter_intervals.items():
    #         names.append(name)

    #         lower_bound = v["lower_bound"]
    #         upper_bound = v["upper_bound"]
    #         data_type = v["data_type"]

    #         if data_type == "discrete":
    #             prior = pm.DiscreteUniform(name, lower_bound, upper_bound)
    #         else:
    #             prior = pm.Uniform(name, lower_bound, upper_bound)
    #         priors.append(prior)

    #     params = tuple(priors)

    #     def simulator_func(
    #         _, *parameters
    #     ):
    #         parameters = parameters[:-1]

    #         parameter_specs = {}
    #         for i, name in enumerate(names):
    #             parameter_specs[name] = parameters[i].item()

    #         discrepancy = calculate_summary_statistic_discrepancy(
    #             parameter_specs, input_parameters, statistics_list, distance
    #         )

    #         print(parameter_specs)
    #         print(discrepancy)
    #         return np.array([discrepancy])

    #     calibration_parameters = input_parameters.calibration_parameters
    #     pm.Simulator(
    #         "root_simulator",
    #         simulator_func,
    #         params = params,
    #         distance = distance_func,
    #         sum_stat = "identity",
    #         epsilon = calibration_parameters["epsilon"],
    #         observed = observed_values,
    #     )

    #     time_now = get_datetime_now()
    #     outdir = get_outdir()
    #     pgm = pm.model_to_graphviz(model = model)
    #     outfile =  f"{time_now}-{TASK}_model_graph"
    #     pgm.render(format = "png", directory = outdir, filename = outfile)
    #     outfile = osp.join(outdir, f"{outfile}.png")
    #     mlflow.log_artifact(outfile)

    # trace = pm.sample_smc(
    #     # draws = calibration_parameters["draws"],
    #     model = model,
    #     draws = 3,
    #     chains = 1,
    #     cores = 1,
    #     # chains = calibration_parameters["chains"],
    #     # cores = calibration_parameters["cores"],
    #     # compute_convergence_checks = False,
    #     # return_inferencedata = False,
    #     # random_seed = input_parameters.random_seed,
    #     # progressbar = False
    # )

    # del trace
    # del model
    # print(trace)
