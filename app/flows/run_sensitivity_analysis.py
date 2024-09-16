#!/usr/bin/env python

######################################
# Imports
######################################

# isort: off

# This is for compatibility with Prefect.
import multiprocessing

# isort: on

import os.path as osp

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
from joblib import dump as calibrator_dump
from matplotlib import pyplot as plt
from prefect import flow, task
from prefect.artifacts import create_table_artifact
from prefect.task_runners import ConcurrentTaskRunner
from SALib import ProblemSpec

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

TASK = "sensitivity_analysis"

######################################
# Main
######################################


@task
def prepare_task(input_parameters: RootCalibrationModel) -> tuple:
    """Prepare the sensitivity analysis procedure.

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
    names = []
    bounds = []
    dists = []
    data_types = []

    parameter_intervals = input_parameters.parameter_intervals.dict()
    distance_metric = input_parameters.statistics_comparison.distance_metric.replace(  # type: ignore[union-attr]
        "_", " "
    ).title()

    for k, v in parameter_intervals.items():
        names.append(k)

        lower_bound = v["lower_bound"]
        upper_bound = v["upper_bound"]
        bounds.append([lower_bound, upper_bound])

        data_type = v["data_type"]
        data_types.append(data_type)

        dists.append("unif")

    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
        "dists": dists,
        "groups": None,
        "outputs": [distance_metric],
    }

    sp = ProblemSpec(problem)
    distance, statistics_list = get_calibration_summary_stats(input_parameters)

    return sp, names, data_types, distance, statistics_list


@task
def perform_task(
    sp: ProblemSpec,
    input_parameters: RootCalibrationModel,
    statistics_list: list[SummaryStatisticsModel],
    distance: DistanceMetricBase,
    names: list[str],
    data_types: list[str],
) -> tuple:
    """Perform the sensitivity analysis procedure.

    Args:
        sp (ProblemSpec):
            The sensitivity analysis problem specification.
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        statistics_list (list[SummaryStatisticsModel]):
            The list of summary statistics.
        distance (DistanceMetricBase):
            The distance metric object.
        names (list[str]):
            The list of parameter names.
        data_types (list[str]):
            The list of parameter data types.

    Returns:
        tuple:
            The sensitivity analysis problem specification and samples.
    """
    sample_list: list[dict] = []

    def simulator_func(
        X: np.ndarray,
        statistics_list: list[SummaryStatisticsModel],
        distance: DistanceMetricBase,
        names: list,
        data_types: list,
        sample_list: list,
        distance_metric: str,
    ) -> np.ndarray:
        import numpy as np

        discrepancies = []
        for theta in X:
            parameter_specs = {}
            for i, parameter_value in enumerate(theta):
                data_type = data_types[i]
                if data_type == "discrete":
                    parameter_value = int(parameter_value)

                k = names[i]
                parameter_specs[k] = parameter_value

            discrepancy = calculate_summary_statistic_discrepancy(
                parameter_specs, input_parameters, statistics_list, distance
            )

            discrepancies.append(discrepancy)
            parameter_specs[distance_metric] = discrepancy
            sample_list.append(parameter_specs)

        discrepancies = np.array(discrepancies)
        return discrepancies

    calibration_parameters = input_parameters.calibration_parameters
    distance_metric = (
        input_parameters.statistics_comparison.distance_metric  # type: ignore[union-attr]
    )
    (
        sp.sample_sobol(
            calibration_parameters["n_samples"],
            scramble=True,
            calc_second_order=True,
            seed=input_parameters.random_seed,
        )
        .evaluate(
            simulator_func,
            statistics_list,
            distance,
            names,
            data_types,
            sample_list,
            distance_metric,
        )
        .analyze_sobol()
    )

    return sp, sample_list


@task
def log_task(
    sp: ProblemSpec,
    sample_list: list[dict],
    names: list[str],
    input_parameters: RootCalibrationModel,
    simulation_uuid: str,
) -> None:
    """Log the sensitivity analysis results.

    Args:
        sp (ProblemSpec):
            The sensitivity analysis problem specification.
        sample_list (list[dict]):
            The list of summary statistics.
        names (list[str]):
            The list of parameter names.
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    time_now = get_datetime_now()
    outdir = get_outdir()

    si_dfs = sp.to_df()
    for i, si_label in enumerate(["total_si", "first_si", "second_si"]):
        outfile = osp.join(outdir, f"{time_now}-{TASK}_{si_label}.csv")
        si_df = si_dfs[i]

        if si_label != "second_si":
            si_df["name"] = names
            cols = list(si_df.columns)
            cols = [cols[-1]] + cols[:-1]
            si_df = si_df[cols]

        si_df.to_csv(outfile, index=False)
        mlflow.log_artifact(outfile)

    sample_df = pd.DataFrame(sample_list)
    outfile = osp.join(outdir, f"{time_now}-{TASK}_sobol_samples.csv")
    sample_df.to_csv(outfile, index=False)
    mlflow.log_artifact(outfile)

    total_si_df = si_dfs[0]
    fig = px.bar(
        total_si_df,
        x="name",
        y="ST",
        title="Total order Sobol indices",
        error_y="ST_conf",
    ).update_layout(xaxis_title="Parameter", yaxis_title="Sensitivity Index")
    outfile = osp.join(outdir, f"{time_now}-{TASK}_total_order_indices.png")
    fig.write_image(outfile, width=1200, height=1200)
    mlflow.log_artifact(outfile)

    st_confs = total_si_df.ST_conf.values
    for i, sensitivity_index in enumerate(total_si_df.ST.values):
        name = names[i]
        st_conf = st_confs[i]
        mlflow.log_metric(name, sensitivity_index)
        mlflow.log_metric(f"{name}_conf", st_conf)

    create_table_artifact(
        key="sensitivity-analysis-indices",
        table=total_si_df.to_dict(orient="records"),
        description="# Sobol sensitivity indices (total order).",
    )

    outfile = osp.join(outdir, f"{time_now}-{TASK}_sobol_heatmap.png")
    sp.heatmap()
    plt.tight_layout()
    plt.savefig(outfile)
    mlflow.log_artifact(outfile)

    sp.total_si_df = total_si_df
    outfile = osp.join(outdir, f"{time_now}-{TASK}_calibrator.pkl")
    calibrator_dump(sp, outfile)
    calibration_model = SensitivityAnalysisModel()

    artifacts = {"calibrator": outfile}
    signature_x = pd.DataFrame({"name": names})
    signature_y = total_si_df

    log_model(
        TASK,
        input_parameters,
        calibration_model,
        artifacts,
        simulation_uuid,
        signature_x,
        signature_y,
    )


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
    begin_experiment(TASK, simulation_uuid, input_parameters.simulation_tag)
    log_experiment_details(simulation_uuid)

    sp, names, data_types, distance, statistics_list = prepare_task(input_parameters)
    sp, sample_list = perform_task(
        sp, input_parameters, statistics_list, distance, names, data_types
    )
    log_task(sp, sample_list, names, input_parameters, simulation_uuid)

    config = input_parameters.dict()
    log_config(config, TASK)
    mlflow.end_run()


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
