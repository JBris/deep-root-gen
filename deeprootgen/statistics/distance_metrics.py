"""Contains utilities for calculating distance values.

This module defines utility functions for distance metrics that can be used to compare simulated and observational data.
"""

from pydoc import locate
from typing import Callable


def get_distance_metric_func(distance_metric: str) -> Callable:
    """Get the distance metric function by name.

    Args:
        distance_metric (str):
            The distance metric name.

    Returns:
        Callable:
            The distance metric function.
    """
    distance_metric = distance_metric.replace("_", " ").title().replace(" ", "")
    module = "deeprootgen.statistics.distance_metrics"
    func: Callable = locate(f"{module}.{distance_metric}")  # type: ignore
    return func


def get_distance_metrics() -> list[dict]:
    """Get a list of available distance metrics and labels.

    Returns:
        list[dict]:
            A list of available distance metrics and labels.
    """
    distance_metrics: list[str] = [
        "l2_norm",
        "l1_norm",
        "mean_squared_error",
        "mean_absolute_error",
        "root_mean_squared_error",
        "mean_pinball_loss",
        "mean_absolute_percentage_error",
        "median_absolute_error",
    ]

    distance_metric_list = []
    for distance_metric in distance_metrics:
        label = distance_metric.replace("_", " ").title()
        distance_metric_list.append({"value": distance_metric, "label": label})

    return distance_metric_list
