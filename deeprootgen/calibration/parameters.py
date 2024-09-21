"""Contains utilities for working with simulation parameters and their distributions.

This module defines various utlities for working with distributions of simulation parameters.
"""

import uuid

import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
from dash import get_app
from scipy.stats import qmc

from ..data_model import RootCalibrationModel


class PriorCollection:
    """A wrapper around a collection of priors."""

    def __init__(self, priors: list[dist.Distribution]) -> None:
        """PriorCollection constructor.

        Args:
            priors (list[dist.Distribution]):
                The list of prior distributions.
        """
        self.priors = priors

    def sample(self, batch_shape: tuple = ()) -> torch.Tensor:
        """Sample from the priors.

        Args:
            batch_shape (tuple, optional):
                The batch shape of the sampled priors. Defaults to ().

        Returns:
            torch.Tensor:
                The sampled priors.
        """
        prior_sample = []
        for prior in self.priors:
            prior_sample.append(prior.sample(batch_shape).squeeze())
        return torch.stack(prior_sample).T


def get_simulation_parameters(
    parameter_form_name: str = "calibration_form",
) -> list[dict]:
    """Get a list of available simulation parameters and labels.

    Args:
        parameter_form_name (str, optional):
            The name of the parameter form components specification. Defaults to "simulation_form".

    Returns:
        list[dict]:
            A list of available simulation parameters and labels.
    """
    app = get_app()
    parameter_form = app.settings[parameter_form_name]

    parameter_list = []
    for child in parameter_form.components["parameters"]["children"]:
        if child["class_name"] != "dash.dcc.RangeSlider":
            continue

        label = child["label"]
        parameter = child["param"]
        parameter_list.append({"value": parameter, "label": label})

    return parameter_list


def lhc_sample(input_parameters: RootCalibrationModel) -> pd.DataFrame:
    """Sample from the parameter space.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.

    Returns:
        pd.DataFrame:
            The sample dataframe.
    """
    statistics_comparison = input_parameters.statistics_comparison
    if statistics_comparison.exclude_parameters is None:  # type: ignore[union-attr]
        exclude_parameters = set()
    else:
        exclude_parameters = set(statistics_comparison.exclude_parameters)  # type: ignore[union-attr]
    if statistics_comparison.include_parameters is None:  # type: ignore[union-attr]
        include_parameters = set()
    else:
        include_parameters = set(statistics_comparison.include_parameters)  # type: ignore[union-attr]
    include_parameters = include_parameters.difference(exclude_parameters)

    names = []
    lower_bounds = []
    upper_bounds = []
    parameter_intervals = input_parameters.parameter_intervals.dict()
    for parameter, value in parameter_intervals.items():
        if parameter in exclude_parameters and len(exclude_parameters) > 0:
            continue
        if parameter not in include_parameters and len(include_parameters) > 0:
            continue

        names.append(parameter)
        lower_bounds.append(value["lower_bound"])
        upper_bounds.append(value["upper_bound"])

    d = len(names)
    sampler = qmc.LatinHypercube(d=d, seed=input_parameters.random_seed)
    calibration_parameters = input_parameters.calibration_parameters

    n_simulations = calibration_parameters["n_simulations"]
    n_validation_simulations = calibration_parameters["n_validation_simulations"]
    n_holdout_simulations = calibration_parameters["n_holdout_simulations"]
    total_samples = n_simulations + n_validation_simulations + n_holdout_simulations  # type: ignore[operator]

    sample = sampler.random(n=total_samples)
    sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)
    sample_df = pd.DataFrame(sample_scaled, columns=names).sample(frac=1)
    sim_ids = [str(uuid.uuid4()) for _ in range(total_samples)]  # type: ignore[arg-type]
    sample_df["sim_ids"] = sim_ids

    training_data_split = []
    for i, n_sims in enumerate(
        [n_simulations, n_validation_simulations, n_holdout_simulations]
    ):
        training_data_split.extend(np.repeat(i, n_sims))

    sample_df["training_data_split"] = training_data_split

    names = set(names)  # type: ignore[assignment]
    for parameter, value in parameter_intervals.items():
        if parameter not in names:
            lower_bound = value["lower_bound"]
            upper_bound = value["upper_bound"]
            midpoint = np.median([lower_bound, upper_bound])
            sample_df[parameter] = midpoint

        data_type = value["data_type"]
        if data_type == "discrete":
            sample_df[parameter] = sample_df[parameter].astype("int")

    return sample_df
