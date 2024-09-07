"""Contains the soil component of the root system simulation.

This module defines the soil component of the
root system architecture simulation model for constructing 3D root systems.

"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class Soil:
    """The soil surrounding the root system."""

    def create_soil_grid(
        self,
        soil_layer_height: float,
        soil_n_layers: int,
        soil_layer_width: float,
        soil_n_cols: int,
    ) -> pd.DataFrame:
        """Create a soil voxel grid.

        Args:
            soil_layer_height (float):
                The height per vertical soil layer.
            soil_n_layers (int):
                The number of vertical soil layers.
            soil_layer_width (float):
                The width per horizontal soil layers/columns.
            soil_n_cols (int): _description_
                The number of horizontal soil layers/columns.

        Returns:
            pd.DataFrame:
                The soil voxel grid dataframe.
        """
        voxel_height = soil_layer_height
        n_layers = soil_n_layers + 1
        voxel_width = soil_layer_width
        n_cols = soil_n_cols + 1

        # Create unit hypercube
        x = np.linspace(0, 1, n_cols)
        y = np.linspace(0, 1, n_layers)
        z = np.linspace(0, 1, 2)
        M = np.meshgrid(x, y, z)
        grid = [vector.flatten() for vector in M]

        # Rescale and translate x,z about the origin
        xv, yv, zv = grid
        xv *= voxel_width * (n_cols - 1)
        xv -= voxel_width / 2 * (n_cols - 1)
        yv *= -voxel_height * (n_layers - 1)
        zv *= voxel_width
        zv -= voxel_width / 2

        soil_df = pd.DataFrame({"x": xv, "y": yv, "z": zv})
        return soil_df

    def create_soil_fig(self, soil_df: pd.DataFrame) -> go.Figure:
        """Create a figure from a soil grid.

        Args:
            soil_df (pd.DataFrame):
                The soil voxel grid dataframe.

        Returns:
            go.Figure:
                The soil grid figure.
        """
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=soil_df["x"],
                    y=soil_df["z"],
                    z=soil_df["y"],
                    mode="markers",
                    marker=dict(
                        size=4, color="brown", colorscale="brwnyl", opacity=0.5
                    ),
                )
            ]
        )

        return fig
