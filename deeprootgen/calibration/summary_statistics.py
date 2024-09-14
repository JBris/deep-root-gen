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
    distance_func = get_distance_metric_func(statistics_comparison.distance_metric)  # type: ignore
    distance = distance_func()

    statistics_list = []
    for summary_statistic in input_parameters.summary_statistics:  # type: ignore
        statistics_list.append(summary_statistic.dict())

    statistics_records = (
        pd.DataFrame(statistics_list)
        .query("statistic_name in @statistics_comparison.summary_statistics")
        .to_dict("records")
    )
    if len(statistics_records) == 0:
        raise ValueError("Summary statistics list cannot be empty.")

    statistics_list = [
        SummaryStatisticsModel.parse_obj(statistic) for statistic in statistics_records
    ]

    return distance, statistics_list


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


def calculate_summary_statistic_discrepency(
    parameter_specs: dict,
    input_parameters: RootCalibrationModel,
    statistics_list: list[SummaryStatisticsModel],
    distance: DistanceMetricBase,
) -> float:
    """Calculate the discrepency between simulated and observed data.

    Args:
        parameter_specs (dict):
            The simulation parameter specification.
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        statistics_list (list[SummaryStatisticsModel]):
            The list of summary statistics.
        distance (DistanceMetricBase):
            The distance metric object.

    Returns:
        float:
            The discrepency between simulated and observed data.
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

        observed_values.append(statistic.statistic_value)
        simulated_values.append(statistic_value)

    observed = np.array(observed_values)
    simulated = np.array(simulated_values)
    discrepency = distance.calculate(observed, simulated)
    return discrepency
