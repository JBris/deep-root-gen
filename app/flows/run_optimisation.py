#!/usr/bin/env python

######################################
# Imports
######################################

import os.path as osp

import mlflow
import optuna
import pandas as pd
from joblib import dump as calibrator_dump
from optuna.samplers import TPESampler
from prefect import context, flow, task
from prefect.artifacts import create_table_artifact
from prefect.task_runners import ConcurrentTaskRunner

from deeprootgen.calibration import (
    OptimisationModel,
    calculate_summary_statistic_discrepency,
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
# Constants
######################################

TASK = "optimisation"

######################################
# Main
######################################


@task
def prepare_task(input_parameters: RootCalibrationModel) -> tuple:
    """Prepare the optimisation procedure.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.

    Raises:
        ValueError:
            Error thrown when the summary statistic list is empty.

    Returns:
        tuple:
            The optimisation study, data, and cost function.
    """
    distance, statistics_list = get_calibration_summary_stats(input_parameters)
    calibration_parameters = input_parameters.calibration_parameters
    sampler = TPESampler(
        n_startup_trials=calibration_parameters["n_startup_trials"],
        n_ei_candidates=calibration_parameters["n_ei_candidates"],
        consider_prior=True,
        prior_weight=1.0,
        consider_magic_clip=True,
        consider_endpoints=False,
    )
    study = optuna.create_study(
        sampler=sampler, study_name="root_model", direction="minimize"
    )

    return study, statistics_list, distance


@task
def perform_task(
    input_parameters: RootCalibrationModel,
    study: optuna.study.Study,
    statistics_list: list[SummaryStatisticsModel],
    distance: DistanceMetricBase,
) -> None:
    """Perform the optimisation procedure.

    Args:
        parameter_intervals (RootCalibrationIntervals):
            The simulation parameter intervals.
        study (optuna.study.Study):
            The optimisation study.
        statistics_list (list[SummaryStatisticsModel]):
            The list of summary statistics.
        distance (DistanceMetricBase):
            The distance metric object.
    """
    calibration_parameters = input_parameters.calibration_parameters
    parameter_intervals = input_parameters.parameter_intervals.dict()

    study.optimize(
        lambda trial: objective(
            trial, parameter_intervals, statistics_list, distance, input_parameters
        ),
        n_trials=calibration_parameters["n_trials"],
        n_jobs=calibration_parameters["n_jobs"],
        gc_after_trial=False,
        show_progress_bar=False,
    )


def objective(
    trial: optuna.trial.Trial,
    parameter_intervals: dict,
    statistics_list: list[SummaryStatisticsModel],
    distance: DistanceMetricBase,
    input_parameters: RootCalibrationModel,
) -> float:
    """The optimisation objective.

    Args:
        trial (optuna.trial.Trial):
            The optimisation trial object.
        parameter_intervals (dict):
            The simulation parameter intervals.
        statistics_list (list[SummaryStatisticsModel]):
            The list of summary statistics.
        distance (DistanceMetricBase):
            The distance metric object.
        input_parameters (RootCalibrationModel):
            The root calibration data model.

    Returns:
        float:
            The optimisation cost.
    """
    parameter_specs = {}
    for parameter, value in parameter_intervals.items():
        lower_bound = value["lower_bound"]
        upper_bound = value["upper_bound"]
        data_type = value["data_type"]

        if data_type == "continuous":
            parameter_specs[parameter] = trial.suggest_float(
                parameter, lower_bound, upper_bound
            )
        else:
            parameter_specs[parameter] = trial.suggest_int(
                parameter, lower_bound, upper_bound
            )

    discrepency = calculate_summary_statistic_discrepency(
        parameter_specs, input_parameters, statistics_list, distance
    )
    return discrepency


@task
def log_task(
    study: optuna.study.Study,
    input_parameters: RootCalibrationModel,
    simulation_uuid: str,
) -> tuple:
    """Log the optimisation results.

    Args:
        study (optuna.study.Study):
            The optimisation study.
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

    trials_df: pd.DataFrame = study.trials_dataframe().sort_values(
        "value", ascending=True
    )
    outfile = osp.join(outdir, f"{time_now}-{TASK}_trials.csv")
    trials_df.to_csv(outfile, index=False)
    mlflow.log_artifact(outfile)

    create_table_artifact(
        key="opt-params",
        table=trials_df.drop(
            columns=["datetime_start", "datetime_complete", "duration"]
        )
        .head(10)
        .to_dict(orient="records"),
        description="# Optimised parameters.",
    )

    for plot_func in [
        optuna.visualization.plot_edf,
        optuna.visualization.plot_optimization_history,
        optuna.visualization.plot_parallel_coordinate,
        optuna.visualization.plot_param_importances,
        optuna.visualization.plot_slice,
    ]:
        img_file = osp.join(outdir, f"{plot_func.__name__}.png")
        plot_func(study).write_image(img_file)
        mlflow.log_artifact(img_file)

    parameter_specs = {}
    best_parameters = trials_df.head(1)
    params_prefix = "params_"
    for parameter_name in trials_df.columns:
        if not parameter_name.startswith(params_prefix):
            continue
        parameter_name = parameter_name.replace(params_prefix, "")
        best_parameter = best_parameters[f"{params_prefix}{parameter_name}"]
        parameter_specs[parameter_name] = best_parameter
        mlflow.log_metric(parameter_name, best_parameter)

    simulation, simulation_parameters = run_calibration_simulation(
        parameter_specs, input_parameters
    )

    outfile = osp.join(outdir, f"{time_now}-{TASK}_calibrator.pkl")
    calibrator_dump(study, outfile)
    calibration_model = OptimisationModel()

    artifacts = {"calibrator": outfile}
    signature_x = pd.DataFrame({"n_trials": [1, 2, 3]})
    signature_y = trials_df.head(5).drop(
        columns=["datetime_start", "datetime_complete", "duration"]
    )

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
def run_optimisation(
    input_parameters: RootCalibrationModel, simulation_uuid: str
) -> None:
    """Running a optimisation procedure.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    flow_run_id = context.get_run_context().task_run.flow_run_id
    begin_experiment(
        TASK, simulation_uuid, flow_run_id, input_parameters.simulation_tag
    )
    log_experiment_details(simulation_uuid)

    study, statistics_list, distance = prepare_task(input_parameters)
    perform_task(input_parameters, study, statistics_list, distance)
    simulation, simulation_parameters = log_task(
        study, input_parameters, simulation_uuid
    )

    config = input_parameters.dict()
    log_config(config, TASK)
    log_simulation(simulation_parameters, simulation, TASK)
    save_graph_to_db(simulation, TASK, simulation_uuid)
    mlflow.end_run()


@flow(
    name="optimisation",
    description="Run an optimisation procedure for the root model.",
    task_runner=ConcurrentTaskRunner(),
)
def run_optimisation_flow(
    input_parameters: RootCalibrationModel, simulation_uuid: str
) -> None:
    """Flow for running an optimisation procedure.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    run_optimisation.submit(input_parameters, simulation_uuid)
