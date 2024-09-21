"""Contains utilities for calibrating models from summary statistics.

This module defines utility functions for calibrating simulation models from summary statistics.
"""

import numpy as np
import pandas as pd

from ..data_model import (
    RootCalibrationModel,
    RootSimulationModel,
    SummaryStatisticsModel,
)
from ..model import RootSystemSimulation
from ..statistics import (
    DistanceMetricBase,
    get_distance_metric_func,
    get_summary_statistic_func,
)


def get_calibration_summary_stats(input_parameters: RootCalibrationModel) -> tuple:
    """Extract summary statistics needed for model calibration.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
    Raises:
        ValueError:
            Error thrown when the summary statistic list is empty.

    Returns:
        tuple:
            The calibration distance metric and summary statistics.
    """
    statistics_comparison = input_parameters.statistics_comparison

    distance_metrics = []
    if isinstance(statistics_comparison.distance_metrics, list):  # type: ignore[union-attr]
        for distance_metric in statistics_comparison.distance_metrics:  # type: ignore[union-attr]
            distance_func = get_distance_metric_func(distance_metric)
            distance = distance_func()
            distance_metrics.append(distance)

    statistics_list: list[SummaryStatisticsModel] = []
    if input_parameters.summary_statistics is None:
        return distance_metrics, statistics_list

    input_statistics = []
    for summary_statistic in input_parameters.summary_statistics:
        statistics_list.append(summary_statistic.dict())
        input_statistics.append(summary_statistic.statistic_name)

    summary_statistics = set(  # noqa: F841
        statistics_comparison.summary_statistics  # type: ignore
    ).intersection(input_statistics)

    statistics_records = (
        pd.DataFrame(statistics_list)
        .query("statistic_name in @summary_statistics")
        .to_dict("records")
    )
    if len(statistics_records) == 0:
        raise ValueError("Summary statistics list cannot be empty.")

    statistics_list = [
        SummaryStatisticsModel.parse_obj(statistic) for statistic in statistics_records
    ]

    return distance_metrics, statistics_list


def run_calibration_simulation(
    parameter_specs: dict,
    input_parameters: RootCalibrationModel,
) -> tuple:
    """Run a simulation for model calibration purposes.

    Args:
        parameter_specs (dict):
            The simulation parameter specification.
        input_parameters (RootCalibrationModel):
            The root calibration data model.

    Returns:
        tuple:
            The simulation and its parameters.
    """
    parameter_specs["random_seed"] = input_parameters.random_seed
    simulation_parameters = RootSimulationModel.parse_obj(parameter_specs)
    simulation = RootSystemSimulation(
        simulation_tag=input_parameters.simulation_tag,  # type: ignore
        random_seed=input_parameters.random_seed,  # type: ignore
    )

    simulation.run(simulation_parameters)
    return simulation, simulation_parameters


def calculate_summary_statistics(
    parameter_specs: dict,
    input_parameters: RootCalibrationModel,
    statistics_list: list[SummaryStatisticsModel],
) -> tuple:
    """Calculate summary statistics for observed and simulated data.

    Args:
        parameter_specs (dict):
            The simulation parameter specification.
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        statistics_list (list[SummaryStatisticsModel]):
            The list of summary statistics.

    Returns:
        tuple:
            The simulated and observed data.
    """
    simulation, simulation_parameters = run_calibration_simulation(
        parameter_specs, input_parameters
    )

    node_df, _ = simulation.G.as_df()
    observed_values = []
    simulated_values = []
    kwargs = dict(root_tissue_density=simulation_parameters.root_tissue_density)

    for statistic in statistics_list:
        statistic_name = statistic.statistic_name
        statistic_func = get_summary_statistic_func(statistic_name)
        statistic_instance = statistic_func(**kwargs)
        statistic_value = statistic_instance.calculate(node_df)

        # @TODO calculate by depth and horizontal distance
        if isinstance(statistic_value, tuple) or isinstance(statistic_value, list):
            continue

        observed_values.append(statistic.statistic_value)
        simulated_values.append(statistic_value)

    observed = np.array(observed_values).flatten()
    simulated = np.array(simulated_values).flatten()

    return simulated, observed


def calculate_summary_statistic_discrepancy(
    parameter_specs: dict,
    input_parameters: RootCalibrationModel,
    statistics_list: list[SummaryStatisticsModel],
    distances: list[DistanceMetricBase],
) -> float:
    """Calculate the discrepancy between simulated and observed data.

    Args:
        parameter_specs (dict):
            The simulation parameter specification.
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        statistics_list (list[SummaryStatisticsModel]):
            The list of summary statistics.
        distances (list[DistanceMetricBase]):
            The distance metric object.

    Returns:
        float:
            The discrepancy between simulated and observed data.
    """
    simulated, observed = calculate_summary_statistics(
        parameter_specs, input_parameters, statistics_list
    )

    discrepancy_list = []
    for distance in distances:
        discrepancy = distance.calculate(observed, simulated)
        discrepancy_list.append(discrepancy)

    discrepancies: float = np.array(discrepancy_list).flatten().sum()
    return discrepancies
