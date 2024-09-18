#!/usr/bin/env python

######################################
# Imports
######################################

# isort: off

# This is for compatibility with Prefect.

import multiprocessing

# isort: on

import os.path as osp
from datetime import timedelta

import mlflow
import numpy as np
import pandas as pd
import pyabc
from joblib import dump as calibrator_dump
from matplotlib import pyplot as plt
from prefect import flow, task
from prefect.artifacts import create_table_artifact
from prefect.task_runners import SequentialTaskRunner

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

TASK = "abc"

######################################
# Main
######################################


def distance_func(simulated: dict, _: dict) -> float:
    """The function to calculate the distance between observed and simulated data.

    Args:
        simulated (dict):
            The simulated data.
        _ (dict):
            The observed data.

    Returns:
        float:
            The distance between observed and simulated data.
    """
    return simulated["discrepancy"]


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

    parameter_intervals = input_parameters.parameter_intervals.dict()
    dist_kwargs = {}
    transition_mapping = {}

    for name, v in parameter_intervals.items():
        lower_bound = v["lower_bound"]
        upper_bound = v["upper_bound"]
        data_type = v["data_type"]

        if data_type == "discrete":
            lower_bound = np.floor(lower_bound).astype("int")
            upper_bound = np.floor(upper_bound).astype("int")

            discrete_domain = np.arange(lower_bound, upper_bound + 1)
            dist_kwargs[name] = pyabc.RV(
                "rv_discrete",
                values=(
                    discrete_domain,
                    np.repeat(1 / len(discrete_domain), len(discrete_domain)),
                ),
            )
            transition_mapping[name] = pyabc.DiscreteJumpTransition(
                domain=discrete_domain, p_stay=0.7
            )
        else:
            dist_kwargs[name] = pyabc.RV(
                "uniform", lower_bound, upper_bound - lower_bound
            )
            transition_mapping[name] = pyabc.MultivariateNormalTransition(scaling=1)

    prior = pyabc.Distribution(**dist_kwargs)
    transitions = pyabc.AggregatedTransition(mapping=transition_mapping)
    distances, statistics_list = get_calibration_summary_stats(input_parameters)
    return distances, statistics_list, prior, transitions


def perform_task(
    input_parameters: RootCalibrationModel,
    prior: pyabc.Distribution,
    transitions: pyabc.AggregatedTransition,
    statistics_list: list[SummaryStatisticsModel],
    distances: list[DistanceMetricBase],
) -> tuple:
    """Perform the Bayesian parameter estimation procedure.

    Args:
        parameter_intervals (RootCalibrationIntervals):
            The simulation parameter intervals.
        prior (pyabc.Distribution):
            The model prior specification.
        transitions (pyabc.AggregatedTransition):
            The parameter transition kernels.
        statistics_list (list[SummaryStatisticsModel]):
            The list of summary statistics.
        distances (list[DistanceMetricBase]):
            The distance metric object.

    Returns:
        tuple:
            The fitted model and sampling history.
    """

    def simulator_func(theta: dict) -> dict:
        parameter_specs = theta.copy()
        discrepancy = calculate_summary_statistic_discrepancy(
            parameter_specs, input_parameters, statistics_list, distances
        )

        return {"discrepancy": discrepancy}

    calibration_parameters = input_parameters.calibration_parameters
    adaptive_pop_size = pyabc.AdaptivePopulationSize(
        calibration_parameters["start_nr_particles"],
        min_population_size=2,
        n_bootstrap=5,
    )

    abc = pyabc.ABCSMC(
        simulator_func,
        prior,
        distance_func,
        population_size=adaptive_pop_size,
        transitions=transitions,
        eps=pyabc.MedianEpsilon(),
        sampler=pyabc.SingleCoreSampler(check_max_eval=True),
    )

    abc.new("sqlite://")

    history = abc.run(
        minimum_epsilon=calibration_parameters["minimum_epsilon"],
        max_nr_populations=calibration_parameters["max_nr_populations"],
        min_acceptance_rate=0.0,
        max_total_nr_simulations=calibration_parameters["max_total_nr_simulations"],
        max_walltime=timedelta(minutes=calibration_parameters["max_walltime"]),  # type: ignore[arg-type]
    )

    return abc, history


def log_task(
    history: pyabc.History,
    input_parameters: RootCalibrationModel,
    simulation_uuid: str,
) -> tuple:
    """Log the Bayesian model results.

    Args:
        abc (pyabc.ABCSMC):
            The Bayesian model.
        history (pyabc.History):
            The Sequential Monte Carlo sampling history.
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        simulation_uuid (str):
            The simulation uuid.

    Returns:
        tuple:
            The simulation and its parameters.
    """
    time_now = get_datetime_now()
    outdir = get_outdir()

    distribution_dfs = []
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        df["t"] = t
        df["w"] = w
        distribution_dfs.append(df)

    distribution_dfs = pd.concat(distribution_dfs)
    outfile = osp.join(outdir, f"{time_now}-{TASK}_parameters.csv")
    distribution_dfs.to_csv(outfile, index=False)
    mlflow.log_artifact(outfile)

    create_table_artifact(
        key="parameter-estimates",
        table=distribution_dfs.sort_values("t", ascending=False)
        .head(10)
        .to_dict(orient="records"),
        description="# Parameter estimates.",
    )

    populations_df = history.get_all_populations()
    outfile = osp.join(outdir, f"{time_now}-{TASK}_populations.csv")
    populations_df.to_csv(outfile, index=False)
    mlflow.log_artifact(outfile)

    population_particles_df = history.get_nr_particles_per_population()
    outfile = osp.join(outdir, f"{time_now}-{TASK}_nr_particles_per_population.csv")
    population_particles_df.to_csv(outfile, index=False)
    mlflow.log_artifact(outfile)

    distances_df = []
    for t in range(history.max_t + 1):
        df = history.get_weighted_distances(t=t)
        df["t"] = t
        distances_df.append(df)
    distances_df = pd.concat(distances_df)

    create_table_artifact(
        key="simulation-distances",
        table=distances_df.sort_values("t", ascending=False)
        .head(10)
        .to_dict(orient="records"),
        description="# Simulation distances.",
    )

    outfile = osp.join(outdir, f"{time_now}-{TASK}_distances.csv")
    distances_df.to_csv(outfile, index=False)
    mlflow.log_artifact(outfile)

    for plot_func in [
        pyabc.visualization.plot_sample_numbers,
        pyabc.visualization.plot_total_sample_numbers,
        pyabc.visualization.plot_sample_numbers_trajectory,
        pyabc.visualization.plot_epsilons,
        pyabc.visualization.plot_effective_sample_sizes,
        pyabc.visualization.plot_walltime,
        pyabc.visualization.plot_total_walltime,
        pyabc.visualization.plot_contour_matrix,
        pyabc.visualization.plot_acceptance_rates_trajectory,
        pyabc.visualization.plot_kde_matrix_highlevel,
    ]:
        outfile = osp.join(outdir, f"{plot_func.__name__}.png")
        plot_func(history)
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()
        mlflow.log_artifact(outfile)

    parameter_specs = {}
    best_parameters = distribution_dfs.sort_values("t", ascending=False).head(1)
    for parameter_name in distribution_dfs.drop(columns=["t", "w"]).columns:
        best_parameter = best_parameters[parameter_name]
        parameter_specs[parameter_name] = best_parameter
        mlflow.log_metric(parameter_name, best_parameter)

    simulation, simulation_parameters = run_calibration_simulation(
        parameter_specs, input_parameters
    )

    outfile = osp.join(outdir, f"{time_now}-{TASK}_calibrator.pkl")
    calibrator_dump(distribution_dfs, outfile)
    calibration_model = AbcModel()

    artifacts = {"calibrator": outfile}
    signature_x = pd.DataFrame({"t": [1, 2, 3]})
    signature_y = distribution_dfs.head(5)

    log_model(
        TASK,
        input_parameters,
        calibration_model,
        artifacts,
        simulation_uuid,
        signature_x,
        signature_y,
    )

    return simulation, simulation_parameters


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

    distances, statistics_list, prior, transitions = prepare_task(input_parameters)

    # @BUG: Issue with SQLite + Prefect multithreading here
    # perform_task cannot be decorated with @task
    @task
    def dispatching_perform_task() -> bool:
        return True

    dispatching_perform_task()

    _, history = perform_task(
        input_parameters, prior, transitions, statistics_list, distances
    )

    # @BUG: Issue with SQLite + Prefect multithreading here
    # log_task cannot be decorated with @task
    @task
    def dispatching_log_task() -> bool:
        return True

    dispatching_log_task()

    simulation, simulation_parameters = log_task(
        history, input_parameters, simulation_uuid
    )

    config = input_parameters.dict()
    log_config(config, TASK)
    log_simulation(simulation_parameters, simulation, TASK)
    save_graph_to_db(simulation, TASK, simulation_uuid)
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
