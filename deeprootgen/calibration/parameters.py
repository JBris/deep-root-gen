"""Contains utilities for working with simulation parameters and their distributions.

This module defines various utlities for working with distributions of simulation parameters.
"""

import torch
import torch.distributions as dist
from dash import get_app


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
