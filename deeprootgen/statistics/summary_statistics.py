"""Contains utilities for producing summary statistics.

This module defines utility functions for producing summary statistics that can be used to compare simulated and observational data.
"""

# mypy: ignore-errors

from abc import ABC, abstractmethod
from pydoc import locate
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

px.defaults.template = "ggplot2"


class SummaryStatisticBase(ABC):
    """The summary statistic abstract class."""

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> float | np.ndarray:
        """Calculate the summary statistic.

        Args:
            df (pd.DataFrame):
                The root dataframe.

        Raises:
            NotImplementedError:
                Error raised for the unimplemented abstract method.
        """
        raise NotImplementedError("calculate() method not implemented.")

    @abstractmethod
    def get_xy_comparison_data(
        self, df: pd.DataFrame, n_elements: int = 10
    ) -> np.ndarray:
        """Get summary statistic data for comparing against another summary statistic.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.
            n_elements (int, optional):
                The number of elements for the comparison. Defaults to 10.

        Returns:
            np.ndarray:
                The comparison data.
        """
        raise NotImplementedError("get_xy_comparison_data() method not implemented.")

    def get_number_of_roots(self, df: pd.DataFrame) -> int:
        """Get the number of roots in the root system.

        Args:
            df (pd.DataFrame):
                The root dataframe.

        Returns:
            int:
                The number of roots.
        """
        df = df.query("order > 0")
        n = len(df.organ_id.unique())
        return n

    def calculate_statistic_per_layer(
        self, df: pd.DataFrame, layer_decrement: int = 10
    ) -> tuple:
        """Calculate a summary statistic per soil layer.

        Args:
            df (pd.DataFrame):
                The root dataframe.
            layer_decrement (int, optional):
                The depth to decrement each soil layer. Defaults to 10.

        Returns:
            tuple:
                The list of summary statistics and soil layers.
        """
        soil_layers = range(0, df.z.min().astype("int"), -layer_decrement)
        soil_layers = np.array(soil_layers)

        statistics_per_layer: list[float] = []
        for soil_layer in soil_layers:
            layer_df = df.query(
                f"z > {soil_layer - layer_decrement} & z < {soil_layer}"
            )
            statistic = self.calculate(layer_df)
            statistics_per_layer.append(statistic)
        return statistics_per_layer, soil_layers


class DepthDistribution(SummaryStatisticBase):
    """The DepthDistribution summary statistic."""

    def calculate(self, df: pd.DataFrame, bins: int = 10) -> tuple:
        """Get the cumulative root distribution by soil depth.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.
            bins (int, optional):
                The number of bins for a histogram. Defaults to 10.

        Returns:
            tuple:
                The cumulative root distribution summary statistic.
        """
        depth = abs(df.z)
        count, bins_count = np.histogram(depth, bins=bins)
        bins_count = np.insert(bins_count, 0, 0)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        cdf = np.insert(cdf, 0, 0)
        bins_count = bins_count[1:]
        return cdf, bins_count

    def visualise(self, df: pd.DataFrame, bins: int = 10) -> go.Figure:
        """Visualise the cumulative root distribution by soil depth.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.
            bins (int, optional):
                The number of bins for a histogram. Defaults to 10.

        Returns:
            go.Figure:
                The visualisation of the cumulative root distribution summary statistic.
        """
        cdf, bins_count = self.calculate(df, bins)
        return px.scatter(
            title="Cumulative root distribution by soil depth",
            x=cdf,
            y=bins_count,
        ).update_layout(
            xaxis_title="Cumulative root fraction", yaxis_title="Soil depth (cm)"
        )

    def get_xy_comparison_data(
        self, df: pd.DataFrame, n_elements: int = 10
    ) -> np.ndarray:
        """Get summary statistic data for comparing against another summary statistic.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.
            n_elements (int, optional):
                The number of elements for the comparison. Defaults to 10.

        Returns:
            np.ndarray:
                The comparison data.
        """
        _, bins_count = self.calculate(df, n_elements)
        bins_count = bins_count[:n_elements]
        return bins_count


class RadialDistribution(SummaryStatisticBase):
    """The RadialDistribution summary statistic."""

    def calculate(self, df: pd.DataFrame, bins: int = 10) -> tuple:
        """Get the cumulative root distribution by horizontal distance.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.
            bins (int, optional):
                The number of bins for a histogram. Defaults to 10.

        Returns:
            tuple:
                The cumulative root distribution summary statistic.
        """
        horizontal = abs(df.melt(value_vars=["x", "y"]).value)
        count, bins_count = np.histogram(horizontal, bins=bins)
        bins_count = np.insert(bins_count, 0, 0)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        cdf = np.insert(cdf, 0, 0)
        bins_count = bins_count[1:]
        return cdf, bins_count

    def visualise(self, df: pd.DataFrame, bins: int = 10) -> go.Figure:
        """Visualise the cumulative root distribution by horizontal distance.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.
            bins (int, optional):
                The number of bins for a histogram. Defaults to 10.

        Returns:
            go.Figure:
                The visualisation of the cumulative root distribution summary statistic.
        """
        cdf, bins_count = self.calculate(df, bins)
        return px.scatter(
            title="Cumulative root distribution by horizontal distance",
            x=cdf,
            y=bins_count,
        ).update_layout(
            xaxis_title="Cumulative root fraction",
            yaxis_title="Horizontal root distance (cm)",
        )

    def get_xy_comparison_data(
        self, df: pd.DataFrame, n_elements: int = 10
    ) -> np.ndarray:
        """Get summary statistic data for comparing against another summary statistic.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.
            n_elements (int, optional):
                The number of elements for the comparison. Defaults to 10.

        Returns:
            np.ndarray:
                The comparison data.
        """
        _, bins_count = self.calculate(df, n_elements)
        bins_count = bins_count[:n_elements]
        return bins_count


class TotalVolume(SummaryStatisticBase):
    """The TotalVolume summary statistic."""

    def calculate(self, df: pd.DataFrame) -> float:
        """Caculate the total root volume.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            float:
                The total root volume.
        """
        radius = df.diameter / 2
        height = df.length
        volume = np.pi * radius**2 * height
        return np.sum(volume)

    def visualise(self, df: pd.DataFrame) -> go.Figure:
        """Visualise the total root volume.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            go.Figure:
                The visualisation of the total root volume.
        """
        volumes, soil_layers = self.calculate_statistic_per_layer(df)

        return px.scatter(
            title="Total root volume per soil layer",
            x=volumes,
            y=abs(soil_layers),
        ).update_layout(
            xaxis_title="Total root volume (cm^3)", yaxis_title="Soil depth (cm)"
        )

    def get_xy_comparison_data(
        self, df: pd.DataFrame, n_elements: int = 10
    ) -> np.ndarray:
        """Get summary statistic data for comparing against another summary statistic.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.
            n_elements (int, optional):
                The number of elements for the comparison. Defaults to 10.

        Returns:
            np.ndarray:
                The comparison data.
        """
        comparison_data, _ = self.calculate_statistic_per_layer(df)
        comparison_data = comparison_data[:n_elements]
        return comparison_data


class AverageVolume(SummaryStatisticBase):
    """The AverageVolume summary statistic."""

    def calculate(self, df: pd.DataFrame) -> float:
        """Caculate the average root volume.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            float:
                The average root volume.
        """
        radius = df.diameter / 2
        height = df.length
        volume = np.pi * radius**2 * height
        total_volume = np.sum(volume)
        n = self.get_number_of_roots(df)
        average = total_volume / n
        return average

    def visualise(self, df: pd.DataFrame) -> go.Figure:
        """Visualise the average root volume.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            go.Figure:
                The visualisation of the average root volume.
        """
        volumes, soil_layers = self.calculate_statistic_per_layer(df)

        return px.scatter(
            title="Average root volume per soil layer",
            x=volumes,
            y=abs(soil_layers),
        ).update_layout(
            xaxis_title="Average root volume (cm^3)", yaxis_title="Soil depth (cm)"
        )

    def get_xy_comparison_data(
        self, df: pd.DataFrame, n_elements: int = 10
    ) -> np.ndarray:
        """Get summary statistic data for comparing against another summary statistic.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.
            n_elements (int, optional):
                The number of elements for the comparison. Defaults to 10.

        Returns:
            np.ndarray:
                The comparison data.
        """
        comparison_data, _ = self.calculate_statistic_per_layer(df)
        comparison_data = comparison_data[:n_elements]
        return comparison_data


class TotalLength(SummaryStatisticBase):
    """The TotalLength summary statistic."""

    def calculate(self, df: pd.DataFrame) -> float:
        """Caculate the total root length.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            float:
                The total root length.
        """
        return df.length.sum()

    def visualise(self, df: pd.DataFrame) -> go.Figure:
        """Visualise the total root length.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            go.Figure:
                The visualisation of the total root length.
        """
        lengths, soil_layers = self.calculate_statistic_per_layer(df)

        return px.scatter(
            title="Total root length per soil layer",
            x=lengths,
            y=abs(soil_layers),
        ).update_layout(
            xaxis_title="Total root length (cm)", yaxis_title="Soil depth (cm)"
        )

    def get_xy_comparison_data(
        self, df: pd.DataFrame, n_elements: int = 10
    ) -> np.ndarray:
        """Get summary statistic data for comparing against another summary statistic.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.
            n_elements (int, optional):
                The number of elements for the comparison. Defaults to 10.

        Returns:
            np.ndarray:
                The comparison data.
        """
        comparison_data, _ = self.calculate_statistic_per_layer(df)
        comparison_data = comparison_data[:n_elements]
        return comparison_data


class AverageLength(SummaryStatisticBase):
    """The AverageLength summary statistic."""

    def calculate(self, df: pd.DataFrame) -> float:
        """Caculate the average root length.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            float:
                The average root length.
        """
        total_length = df.length.sum()
        n = self.get_number_of_roots(df)
        average = total_length / n
        return average

    def visualise(self, df: pd.DataFrame) -> go.Figure:
        """Visualise the average root length.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            go.Figure:
                The visualisation of the average root length.
        """
        lengths, soil_layers = self.calculate_statistic_per_layer(df)

        return px.scatter(
            title="Average root length per soil layer",
            x=lengths,
            y=abs(soil_layers),
        ).update_layout(
            xaxis_title="Average root length (cm)", yaxis_title="Soil depth (cm)"
        )

    def get_xy_comparison_data(
        self, df: pd.DataFrame, n_elements: int = 10
    ) -> np.ndarray:
        """Get summary statistic data for comparing against another summary statistic.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.
            n_elements (int, optional):
                The number of elements for the comparison. Defaults to 10.

        Returns:
            np.ndarray:
                The comparison data.
        """
        comparison_data, _ = self.calculate_statistic_per_layer(df)
        comparison_data = comparison_data[:n_elements]
        return comparison_data


class TotalDiameter(SummaryStatisticBase):
    """The TotalDiameter summary statistic."""

    def calculate(self, df: pd.DataFrame) -> float:
        """Caculate the total root diameter.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            float:
                The total root diameter.
        """
        return df.diameter.sum()

    def visualise(self, df: pd.DataFrame) -> go.Figure:
        """Visualise the total root diameter.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            go.Figure:
                The visualisation of the total root diameter.
        """
        diameters, soil_layers = self.calculate_statistic_per_layer(df)

        return px.scatter(
            title="Total root diameter per soil layer",
            x=diameters,
            y=abs(soil_layers),
        ).update_layout(
            xaxis_title="Total root diameter (cm)", yaxis_title="Soil depth (cm)"
        )

    def get_xy_comparison_data(
        self, df: pd.DataFrame, n_elements: int = 10
    ) -> np.ndarray:
        """Get summary statistic data for comparing against another summary statistic.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.
            n_elements (int, optional):
                The number of elements for the comparison. Defaults to 10.

        Returns:
            np.ndarray:
                The comparison data.
        """
        comparison_data, _ = self.calculate_statistic_per_layer(df)
        comparison_data = comparison_data[:n_elements]
        return comparison_data


class AverageDiameter(SummaryStatisticBase):
    """The AverageDiameter summary statistic."""

    def calculate(self, df: pd.DataFrame) -> float:
        """Caculate the average root diameter.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            float:
                The average root diameter.
        """
        total_length = df.diameter.sum()
        n = self.get_number_of_roots(df)
        average = total_length / n
        return average

    def visualise(self, df: pd.DataFrame) -> go.Figure:
        """Visualise the average root diameter.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            go.Figure:
                The visualisation of the average root diameter.
        """
        diameters, soil_layers = self.calculate_statistic_per_layer(df)

        return px.scatter(
            title="Average root diameter by soil layer",
            x=diameters,
            y=abs(soil_layers),
        ).update_layout(
            xaxis_title="Average root diameter (cm)", yaxis_title="Soil depth (cm)"
        )

    def get_xy_comparison_data(
        self, df: pd.DataFrame, n_elements: int = 10
    ) -> np.ndarray:
        """Get summary statistic data for comparing against another summary statistic.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.
            n_elements (int, optional):
                The number of elements for the comparison. Defaults to 10.

        Returns:
            np.ndarray:
                The comparison data.
        """
        comparison_data, _ = self.calculate_statistic_per_layer(df)
        comparison_data = comparison_data[:n_elements]
        return comparison_data


class ConvexHullArea(SummaryStatisticBase):
    """The ConvexHullArea statistic."""

    def calculate(self, df: pd.DataFrame) -> float:
        """Caculate the area of the convex hull.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            float:
                The area of the convex hull.
        """
        coordinates = df[["x", "y", "z"]].values
        hull = ConvexHull(points=coordinates)
        return hull.area

    def visualise(self, df: pd.DataFrame) -> go.Figure:
        """Visualise the area of the convex hull.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            go.Figure:
                The visualisation of the area of the convex hull.
        """
        areas, soil_layers = self.calculate_statistic_per_layer(df)

        return px.scatter(
            title="Convex hull area by soil layer",
            x=areas,
            y=abs(soil_layers),
        ).update_layout(
            xaxis_title="Convex hull area (cm^2)", yaxis_title="Soil depth (cm)"
        )

    def get_xy_comparison_data(
        self, df: pd.DataFrame, n_elements: int = 10
    ) -> np.ndarray:
        """Get summary statistic data for comparing against another summary statistic.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.
            n_elements (int, optional):
                The number of elements for the comparison. Defaults to 10.

        Returns:
            np.ndarray:
                The comparison data.
        """
        comparison_data, _ = self.calculate_statistic_per_layer(df)
        comparison_data = comparison_data[:n_elements]
        return comparison_data


class ConvexHullVolume(SummaryStatisticBase):
    """The ConvexHullVolume statistic."""

    def calculate(self, df: pd.DataFrame) -> float:
        """Caculate the volume of the convex hull.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            float:
                The volume of the convex hull.
        """
        coordinates = df[["x", "y", "z"]].values
        hull = ConvexHull(points=coordinates)
        return hull.volume

    def visualise(self, df: pd.DataFrame) -> go.Figure:
        """Visualise the volume of the convex hull.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.

        Returns:
            go.Figure:
                The visualisation of the volume of the convex hull.
        """
        areas, soil_layers = self.calculate_statistic_per_layer(df)

        return px.scatter(
            title="Convex hull volume by soil layer",
            x=areas,
            y=abs(soil_layers),
        ).update_layout(
            xaxis_title="Convex hull volume (cm^3)", yaxis_title="Soil depth (cm)"
        )

    def get_xy_comparison_data(
        self, df: pd.DataFrame, n_elements: int = 10
    ) -> np.ndarray:
        """Get summary statistic data for comparing against another summary statistic.

        Args:
            df (pd.DataFrame):
                The dataframe of root data.
            n_elements (int, optional):
                The number of elements for the comparison. Defaults to 10.

        Returns:
            np.ndarray:
                The comparison data.
        """
        comparison_data, _ = self.calculate_statistic_per_layer(df)
        comparison_data = comparison_data[:n_elements]
        return comparison_data


def get_summary_statistic_func(summary_statistic: str) -> Callable:
    """Get the summary statistic function by name.

    Args:
        summary_statistic (str):
            The summary statistic name.

    Returns:
        Callable:
            The summary statistic function.
    """
    summary_statistic = summary_statistic.replace("_", " ").title().replace(" ", "")
    module = "deeprootgen.statistics.summary_statistics"
    return locate(f"{module}.{summary_statistic}")


def get_summary_statistics() -> list[dict]:
    """Get a list of available summary statistics and labels.

    Returns:
        list[dict]:
            A list of available summary statistics and labels.
    """
    summary_statistics: list[str] = [
        "depth_distribution",
        "radial_distribution",
        "total_volume",
        "average_volume",
        "total_length",
        "average_length",
        "total_diameter",
        "average_diameter",
        "convex_hull_area",
        "convex_hull_volume",
    ]

    summary_statistic_list = []
    for summary_statistic in summary_statistics:
        label = summary_statistic.replace("_", " ").title()
        summary_statistic_list.append({"value": summary_statistic, "label": label})

    return summary_statistic_list
