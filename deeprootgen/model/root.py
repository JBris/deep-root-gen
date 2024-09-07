"""Contains the underlying root system simulation model.

This module defines the root system architecture simulation model for constructing 3D root systems.

"""

from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..data_model import RootSimulationModel
from .soil import Soil


class RootSystemSimulation:
    """The root system architecture simulation model."""

    def __init__(self) -> None:
        self.nodes: List[dict] = []
        self.edges: List[dict] = []
        self.soil: Soil = Soil()

    def run(self, input_parameters: RootSimulationModel) -> dict:
        soil_df = self.soil.create_soil_grid(
            input_parameters.soil_layer_height,
            input_parameters.soil_n_layers,
            input_parameters.soil_layer_width,
            input_parameters.soil_n_cols,
        )

        fig = self.soil.create_soil_fig(soil_df)

        return {"figure": fig}
