"""Contains utilities for working with simulation parameters and their distributions.

This module defines various utlities for working with distributions of simulation parameters.
"""

import torch
import torch.distributions as dist


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
