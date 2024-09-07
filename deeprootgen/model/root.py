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
        run_results = {}

        # Initialise figure (optionally with soil)
        if input_parameters.enable_soil:
            soil_df = self.soil.create_soil_grid(
                input_parameters.soil_layer_height,
                input_parameters.soil_n_layers,
                input_parameters.soil_layer_width,
                input_parameters.soil_n_cols,
            )

            run_results["soil"] = soil_df
            fig = self.soil.create_soil_fig(soil_df)
        else:
            run_results["soil"] = None
            fig = go.Figure()

        fig.update_layout(
            scene=dict(
                xaxis=dict(title="x"), yaxis=dict(title="z"), zaxis=dict(title="y")
            )
        )
        run_results["figure"] = fig
        return run_results
