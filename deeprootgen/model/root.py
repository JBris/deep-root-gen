"""Contains the underlying root system simulation model.

This module defines the root system architecture simulation model for constructing 3D root systems.

"""

# mypy: ignore-errors

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.random import default_rng

from ..data_model import (
    RootNodeModel,
    RootSimulationModel,
    RootSimulationResultModel,
    RootType,
    RootTypeModel,
)
from ..spatial import get_transform_matrix, make_homogenous
from .hgraph import RootNode, RootSystemGraph
from .soil import Soil


class RootOrgan:
    """A single root organ within the root system."""

    def __init__(
        self,
        parent_node: RootNode,
        input_parameters: RootSimulationModel,
        root_type: RootTypeModel,
        simulation_tag: str,
        rng: np.random.Generator,
    ) -> None:
        """RootOrgan constructor.

        Args:
            parent_node (RootNode):
                The parent root node.
            input_parameters (RootSimulationModel):
                The root simulation data model.
            root_type (RootTypeModel):
                The root type data.
            simulation_tag (str, optional):
                A tag to group together multiple simulations.
            rng (np.random.Generator):
                The random number generator.

        Returns:
            RootOrgan:
                A root organ within the root system.
        """
        self.parent_node = parent_node
        self.segments: List[RootNode] = []
        self.child_organs: List["RootOrgan"] = []
        self.input_parameters = input_parameters
        self.root_type = root_type

        # Diameter
        self.base_diameter = input_parameters.base_diameter
        self.fine_root_threshold = input_parameters.fine_root_threshold

        # Length
        self.proot_length_interval = np.array(
            [input_parameters.min_primary_length, input_parameters.max_primary_length]
        )
        self.sroot_length_interval = np.array(
            [input_parameters.min_sec_root_length, input_parameters.max_sec_root_length]
        )
        self.length_reduction = input_parameters.length_reduction

        self.reset_transform()
        self.simulation_tag = simulation_tag
        self.rng = rng
        self.invalid_root = False

    def init_diameters(self, segments_per_root: int, apex_diameter: int) -> np.ndarray:
        """Initialise root diameters for the root organ.

        Args:
            segments_per_root (int):
                The number of segments for a single root organ.
            apex_diameter (int):
                The diameter of the root apex.
        """
        diameter_reduction = 1 - self.input_parameters.diameter_reduction
        base_diameter = self.base_diameter * diameter_reduction ** (
            self.parent_node.node_data.order
        )
        if base_diameter > apex_diameter:
            ub, lb = base_diameter, apex_diameter
        else:
            lb, ub = base_diameter, apex_diameter

        diameters = self.rng.uniform(lb, ub, segments_per_root)
        diameters = np.sort(diameters)[::-1]
        return diameters

    def init_lengths(self, segments_per_root: int) -> np.ndarray:
        """Initialise root lengths for the root organ.

        Args:
            segments_per_root (int):
                The number of segments for a single root organ.
            length_range (tuple[float]):
                The minimum and maximum value for the segment lengths.
        """
        order = self.parent_node.node_data.order + 1
        if order > 1:
            reduction_factor = self.length_reduction ** (
                self.parent_node.node_data.order
            )
            min_length, max_length = self.sroot_length_interval * reduction_factor
        else:
            min_length, max_length = self.proot_length_interval

        if max_length > min_length:
            ub, lb = max_length, min_length
        else:
            lb, ub = max_length, min_length

        root_length = self.rng.uniform(lb, ub)
        # Rescale segment samples
        segment_samples = self.rng.uniform(0, 1, segments_per_root)
        segment_samples = np.sort(segment_samples)[::-1]
        segment_samples /= segment_samples.sum(axis=0)
        lengths = segment_samples * root_length
        return lengths

    def add_child_node(
        self,
        parent_node: RootNode,
        diameters: np.ndarray,
        lengths: np.ndarray,
        coordinates: np.ndarray,
        root_type: RootTypeModel,
        i: int,
        new_organ: bool = False,
    ) -> RootNode:
        """Add a new child node to the root organ.

        Args:
            parent_node (RootNode):
                The parent node of the root organ.
            diameters (np.ndarray):
                The array of segment diameters.
            lengths (np.ndarray):
                The array of segment lengths.
            coordinates (np.ndarray):
                The 3D coordinates.
            root_type (RootTypeModel):
                The root type data model.
            i (int):
                The current array index.
            new_organ (bool, optional):
                Whether the node belongs to a new root organ. Defaults to False.

        Returns:
            RootNode:
                The child node.
        """
        diameter = diameters[i]
        length = lengths[i]
        x, y, z = coordinates[i]

        node_data = RootNodeModel(
            x=x,
            y=y,
            z=z,
            diameter=diameter,
            length=length,
            mode="marker",
            root_type=root_type.root_type,
            order_type=root_type.order_type,
            position_type=root_type.position_type,
            simulation_tag=self.simulation_tag,
        )

        child_node = parent_node.add_child_node(node_data, new_organ=new_organ)
        child_node.organ = self
        self.segments.append(child_node)
        return child_node

    def init_segment_coordinates(
        self, segments_per_root: int, lengths: np.ndarray
    ) -> np.ndarray:
        """Initialise the coordinates of the root segments.

        Args:
            segments_per_root (int):
                The number of segments for a single root organ.
            lengths (np.ndarray):
                The lengths of each root segment.

        Returns:
            np.ndarray:
                The 3D root segment coordinates.
        """
        coordinates = [np.repeat(0, 3)]
        root_vary = self.input_parameters.root_vary
        y_rotations = self.rng.uniform(-root_vary, root_vary, segments_per_root)
        z_rotations = self.rng.uniform(-root_vary, root_vary, segments_per_root)
        # noise = self.rng.uniform(1e-4, 1e-3, segments_per_root)

        for i in range(lengths.shape[0]):
            segment_length = lengths[i]
            coord = np.array([np.repeat(segment_length, 3)])
            homogenous_coordinates = make_homogenous(coord)
            y_rotate = y_rotations[i]
            z_rotate = z_rotations[i]
            current_coord = coordinates[i]
            transformation_matrix = get_transform_matrix(
                pitch=y_rotate, yaw=z_rotate, translation=current_coord
            )
            transformed_coordinates = (
                transformation_matrix[:-1] @ homogenous_coordinates
            )
            coord = transformed_coordinates.T[0]
            coordinates.append(coord)

        coordinates = np.array(coordinates)
        coordinates[:, 2] *= -1
        return coordinates

    def construct_root(
        self,
        segments_per_root: int,
        apex_diameter: int,
    ) -> List[RootNode]:
        """Construct all root segments for the root organ.

        Args:
            segments_per_root (int):
                The number of segments for a single root organ.
            apex_diameter (int):
                The diameter of the root apex.

        Returns:
            List[RootNode]:
                The root segments for the root organ.
        """
        diameters = self.init_diameters(segments_per_root, apex_diameter)
        lengths = self.init_lengths(segments_per_root)
        coordinates = self.init_segment_coordinates(segments_per_root, lengths)

        self.base_node = self.add_child_node(
            self.parent_node,
            diameters=diameters,
            lengths=lengths,
            coordinates=coordinates,
            root_type=self.root_type,
            i=0,
            new_organ=True,
        )
        child_node = self.base_node

        for i in range(1, segments_per_root):
            child_node = self.add_child_node(
                child_node,
                diameters=diameters,
                lengths=lengths,
                coordinates=coordinates,
                root_type=self.root_type,
                i=i,
                new_organ=False,
            )

        return self.segments

    def add_child_organ(
        self, floor_threshold: float = 0.4, ceiling_threshold: float = 0.9
    ) -> "RootOrgan":
        floor = int(len(self.segments) * floor_threshold)
        ceiling = int(len(self.segments) * ceiling_threshold)
        if floor > ceiling:
            floor, ceiling = ceiling, floor
        if floor <= 0:
            floor = 1

        indx = self.rng.integers(floor, ceiling)
        parent_node = self.segments[indx]

        child_organ = RootOrgan(
            parent_node,
            input_parameters=self.input_parameters,
            root_type=self.root_type,
            simulation_tag=self.simulation_tag,
            rng=self.rng,
        )
        self.child_organs.append(child_organ)
        return child_organ

    def construct_root_from_parent(
        self,
        segments_per_root: int,
        apex_diameter: int,
    ) -> List[RootNode]:
        """Construct root segments for the root organ, inheriting plant properties from the parent organ.

        Args:
            segments_per_root (int):
                The number of segments for a single root organ.
            apex_diameter (int):
                The diameter of the root apex.

        Returns:
            List[RootNode]:
                The root segments for the root organ.
        """
        diameters = self.init_diameters(segments_per_root, apex_diameter)
        lengths = self.init_lengths(segments_per_root)
        coordinates = self.init_segment_coordinates(segments_per_root, lengths)

        parent_data = self.parent_node.node_data
        parent_order = parent_data.order
        diameters += parent_data.diameter * 0.1**parent_order
        lengths += parent_data.length * 0.1**parent_order

        avg_diameter = diameters.mean(axis=0)
        if avg_diameter > self.fine_root_threshold:
            root_type = RootType.STRUCTURAL.value
        else:
            root_type = RootType.FINE.value

        root_type = RootTypeModel(
            root_type=root_type,
            order_type=RootType.SECONDARY.value,
            position_type=self.root_type.position_type,
        )

        self.base_node = self.add_child_node(
            self.parent_node,
            diameters=diameters,
            lengths=lengths,
            coordinates=coordinates,
            root_type=root_type,
            i=0,
            new_organ=True,
        )
        child_node = self.base_node

        for i in range(1, segments_per_root):
            child_node = self.add_child_node(
                child_node,
                diameters=diameters,
                lengths=lengths,
                coordinates=coordinates,
                root_type=root_type,
                i=i,
                new_organ=False,
            )

        return self.segments

    def reset_transform(self) -> np.ndarray:
        """Reset the transformation matrix.

        Returns:
            np.ndarray:
                The reset transformation matrix.
        """
        self.transform_matrix = np.eye(4)
        return self.transform_matrix

    def update_transform(
        self,
        roll: float = 0,
        pitch: float = 0,
        yaw: float = 0,
        translation: List[float] = [0, 0, 0],
        reflect: List[float] = [1, 1, 1, 1],
        scale: List[float] = [1, 1, 1, 1],
    ) -> np.ndarray:
        """Update the transformation matrix.

        Args:
            roll (float, optional):
                The roll transform in degrees. Defaults to 0.
            pitch (float, optional):
                The pitch transform in degrees. Defaults to 0.
            yaw (float, optional):
                The yaw transform in degrees. Defaults to 0.
            translation (List[float], optional):
                The translation transform in degrees. Defaults to [0, 0, 0].
            reflect (List[float], optional):
                The reflect transform in degrees. Defaults to [1, 1, 1, 1].
            scale (List[float], optional):
                The scale transform in degrees. Defaults to [1, 1, 1, 1].

        Returns:
            np.ndarray:
                The updated transformation matrix.
        """
        transformation_matrix = get_transform_matrix(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            translation=translation,
            reflect=reflect,
            scale=scale,
        )
        self.transform_matrix = self.transform_matrix @ transformation_matrix
        return self.transform_matrix

    def cascading_update_transform(
        self,
        roll: float = 0,
        pitch: float = 0,
        yaw: float = 0,
        translation: List[float] = [0, 0, 0],
        reflect: List[float] = [1, 1, 1, 1],
        scale: List[float] = [1, 1, 1, 1],
    ) -> None:
        """Update the transformation matrix for the organ and child organs.

        Args:
            roll (float, optional):
                The roll transform in degrees. Defaults to 0.
            pitch (float, optional):
                The pitch transform in degrees. Defaults to 0.
            yaw (float, optional):
                The yaw transform in degrees. Defaults to 0.
            translation (List[float], optional):
                The translation transform in degrees. Defaults to [0, 0, 0].
            reflect (List[float], optional):
                The reflect transform in degrees. Defaults to [1, 1, 1, 1].
            scale (List[float], optional):
                The scale transform in degrees. Defaults to [1, 1, 1, 1].
        """
        self.update_transform(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            translation=translation,
            reflect=reflect,
            scale=scale,
        )

        for child_organ in self.child_organs:
            child_organ.cascading_update_transform(
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                translation=translation,
                reflect=reflect,
                scale=scale,
            )

    def get_coordinates(self) -> np.ndarray:
        """Get the coordinates of the root segments.

        Returns:
            np.ndarray:
                The coordinates of the root segments
        """
        coordinates = []
        for segment in self.segments:
            node_data = segment.node_data
            coordinate = [node_data.x, node_data.y, node_data.z]
            coordinates.append(coordinate)

        coordinates = np.array(coordinates)
        return coordinates

    def transform(self) -> np.ndarray:
        """Apply the transformation matrix to the root system coordinates.

        Returns:
            np.ndarray:
                The transformation matrix.
        """

        coordinates = self.get_coordinates()
        ones_matrix = np.ones((len(coordinates), 1))
        homogenous_coordinates = np.hstack((coordinates, ones_matrix)).T
        transformed_coordinates = self.transform_matrix[:-1] @ homogenous_coordinates
        coordinates = transformed_coordinates.T

        for i, segment in enumerate(self.segments):
            node_data = segment.node_data
            node_data.x, node_data.y, node_data.z = coordinates[i]

        return self.reset_transform()

    def cascading_transform(self) -> None:
        """Apply the transformation matrix for the organ and child organs."""
        self.transform()
        for child_organ in self.child_organs:
            child_organ.cascading_transform()

    def get_parent_origin(self) -> np.ndarray:
        """Get the origin of the parent node.

        Returns:
            np.ndarray:
                The origin.
        """
        node_data = self.parent_node.node_data
        x, y, z = node_data.x, node_data.y, node_data.z

        origin = np.array([x, y, z])
        return origin

    def get_local_origin(self) -> np.ndarray:
        """Get the origin of the current root.

        Returns:
            np.ndarray:
                The local origin.
        """
        node_data = self.segments[0].node_data
        origin = np.array([node_data.x, node_data.y, node_data.z])
        return origin

    def get_apex_coordinates(self) -> np.ndarray:
        """Get the apex coordinates of the current root.

        Returns:
            np.ndarray:
                The apex coordinates.
        """
        node_data = self.segments[-1].node_data
        apex = np.array([node_data.x, node_data.y, node_data.z])
        return apex

    def cascading_to_world_origin(self) -> None:
        """
        Translate all child nodes to the world origin.
        """
        local_origin = -self.get_local_origin()
        self.cascading_update_transform(translation=local_origin)

    def set_invalid_root(self) -> None:
        """Specify that the root is invalid."""
        self.invalid_root = True
        for segment in self.segments:
            segment.node_data.invalid_root = True

    def cascading_set_invalid_root(self) -> None:
        """Specify that the root and its children are invalid."""
        self.set_invalid_root()
        for child in self.child_organs:
            child.cascading_set_invalid_root()

    def validate(
        self, no_root_zone: float, pitch: int = 90, max_attempts: int = 50
    ) -> None:
        """Validate the plausibility of the root organ.

        Args:
            no_root_zone (float):
                The minimum depth threshold for root growth.
            pitch (int, optional):
                Pitch in degrees to rotate roots. Defaults to 90.
            max_attempts (int, optional):
                Maximum number of validation attempts. Defaults to 50.
        """
        if self.invalid_root:
            return

        def __transform(**kwargs):
            """Translate to world origin. Apply transform. Translate back to local origin."""
            local_origin = self.get_local_origin()
            self.cascading_to_world_origin()
            self.cascading_transform()
            self.cascading_update_transform(**kwargs)
            self.cascading_transform()
            self.cascading_update_transform(translation=local_origin)
            self.cascading_transform()

        coin_flip = self.rng.binomial(1, 0.5)
        if coin_flip == 1:
            pitch *= -1

        # No upwards growing roots
        # Gravitropism
        iter_count = 0
        current_order = self.segments[0].node_data.order
        if current_order > 1:
            while self.get_apex_coordinates()[2] > self.get_local_origin()[2]:
                if iter_count > max_attempts:
                    return self.cascading_set_invalid_root()
                __transform(pitch=pitch)
                iter_count += 1

        # Coordinates above no root zone
        iter_count = 0
        coordinates = self.get_coordinates()
        while np.any(coordinates[:, 2] > no_root_zone):
            if iter_count > max_attempts:
                return self.cascading_set_invalid_root()
            __transform(pitch=pitch)
            coordinates = self.get_coordinates()
            iter_count += 1

        # Remove detached roots
        if current_order > 1:
            local_origin = np.around(self.get_local_origin())
            parent_coordinates = np.around(self.get_parent_origin())
            if np.any(np.not_equal(local_origin, parent_coordinates)):
                return self.cascading_set_invalid_root()


class RootSystemSimulation:
    """The root system architecture simulation model."""

    def __init__(
        self,
        simulation_tag: str = "default",
        random_seed: int = None,
        visualise: bool = False,
    ) -> None:
        """RootSystemSimulation constructor.

        Args:
            simulation_tag (str, optional):
                A tag to group together multiple simulations. Defaults to 'default'.
            random_seed (int, optional):
                The seed for the random number generator. Defaults to None.
            visualise (bool, optional):
                Whether to visualise the results. Defaults to False

        Returns:
            RootSystemSimulation:
                The RootSystemSimulation instance.
        """
        self.soil: Soil = Soil()
        self.G: RootSystemGraph = RootSystemGraph()
        self.organs: Dict[int, List[RootOrgan]] = {}
        self.simulation_tag = simulation_tag
        self.visualise = visualise
        self.rng = default_rng(random_seed)

    def get_yaw(self, number_of_roots: int) -> tuple:
        """Get the yaw for rotating the root organs.

        Args:
            number_of_roots (int):
                The number of roots.

        Returns:
            tuple:
                The yaw and base yaw.
        """
        yaw_base = 360 / number_of_roots
        return yaw_base, yaw_base * 0.05, yaw_base

    def plot_root_system(self, fig: go.Figure, node_df: pd.DataFrame) -> go.Figure:
        """Create a visualisation of the root system.

        Args:
            fig (int):
                The base plotly figure.
            node_df (pd.DataFrame):
                The root node dataframe.

        Returns:
            go.Figure:
                The visualisation of the root system.
        """
        node_df = node_df.query("invalid_root == False")

        fig.add_trace(
            go.Scatter3d(
                name="root",
                x=node_df["x"],
                y=node_df["y"],
                z=node_df["z"],
                mode="markers",
                line=dict(color="green", colorscale="brwnyl", width=10),
                marker=dict(size=7, color="green", colorscale="brwnyl", opacity=1),
                customdata=np.stack(
                    (
                        node_df.organ_id,
                        node_df.order,
                        node_df.segment_rank,
                        node_df.diameter,
                        node_df.length,
                        node_df.root_type,
                        node_df.order_type,
                        node_df.position_type,
                        node_df.simulation_tag,
                    ),
                    axis=-1,
                ),
                hovertemplate="""
                x: %{x}<br>
                y: %{y}<br>
                z: %{z}<br>
                Organ ID: %{customdata[0]}<br>
                Order: %{customdata[1]}<br>
                Segment rank: %{customdata[2]}<br>
                Diameter: %{customdata[3]}<br>
                Length: %{customdata[4]}<br>
                Root type: %{customdata[5]}<br>
                Order type: %{customdata[6]}<br>
                Position type: %{customdata[7]}<br>
                Simulation tag: %{customdata[8]}<br>""",
            )
        )
        fig.update_traces(connectgaps=False)
        return fig

    def init_fig(self, input_parameters: RootSimulationModel) -> go.Figure | None:
        """Initialise the root system figure.

        Args:
            input_parameters (RootSimulationModel):
                The root simulation data model.

        Returns:
            go.Figure | None:
                The root system visualisation.
        """
        fig = None
        if self.visualise:
            # Initialise figure (optionally with soil)
            if input_parameters.enable_soil:
                soil_df = self.soil.create_soil_grid(
                    input_parameters.soil_layer_height,
                    input_parameters.soil_n_layers,
                    input_parameters.soil_layer_width,
                    input_parameters.soil_n_cols,
                )

                fig = self.soil.create_soil_fig(soil_df)
            else:
                fig = go.Figure()

            fig.update_layout(
                scene=dict(
                    xaxis=dict(title="x"), yaxis=dict(title="y"), zaxis=dict(title="z")
                )
            )

        return fig

    def init_organs(
        self, input_parameters: RootSimulationModel
    ) -> Dict[str, List[RootOrgan]]:
        """Initialise the root organs for the simulation.

        Args:
            input_parameters (RootSimulationModel):
                The root simulation data model.

        Returns:
            Dict[str, List[RootOrgan]]:
                The initialised root organs.
        """
        for order in range(1, input_parameters.max_order + 1):
            self.organs[order] = []

        order = 1
        segments_per_root = input_parameters.segments_per_root
        apex_diameter = input_parameters.apex_diameter

        root_type = RootTypeModel(
            root_type=RootType.STRUCTURAL.value,
            order_type=RootType.PRIMARY.value,
            position_type=RootType.OUTER.value,
        )
        for _ in range(input_parameters.outer_root_num):
            organ = RootOrgan(
                self.G.base_node,
                input_parameters=input_parameters,
                root_type=root_type,
                simulation_tag=self.simulation_tag,
                rng=self.rng,
            )
            organ.construct_root(segments_per_root, apex_diameter)
            self.organs[order].append(organ)

        root_type = RootTypeModel(
            root_type=RootType.STRUCTURAL.value,
            order_type=RootType.PRIMARY.value,
            position_type=RootType.INNER.value,
        )
        for _ in range(input_parameters.inner_root_num):
            organ = RootOrgan(
                self.G.base_node,
                input_parameters=input_parameters,
                root_type=root_type,
                simulation_tag=self.simulation_tag,
                rng=self.rng,
            )
            organ.construct_root(segments_per_root, apex_diameter)
            self.organs[order].append(organ)

        min_sec_root_num = input_parameters.min_sec_root_num
        max_sec_root_num = input_parameters.max_sec_root_num
        if min_sec_root_num > max_sec_root_num:
            min_sec_root_num, max_sec_root_num = max_sec_root_num, min_sec_root_num

        for order in range(2, input_parameters.max_order + 1):
            prev_order = order - 1
            growth_sec_root = (1 - input_parameters.growth_sec_root) ** -(order - 2)

            for parent_organ in self.organs[prev_order]:
                n_secondary_roots = self.rng.integers(
                    min_sec_root_num, max_sec_root_num
                )
                n_secondary_roots = int(n_secondary_roots * growth_sec_root)
                for _ in range(n_secondary_roots):
                    child_organ = parent_organ.add_child_organ(
                        floor_threshold=input_parameters.floor_threshold,
                        ceiling_threshold=input_parameters.ceiling_threshold,
                    )
                    self.organs[order].append(child_organ)
                    child_organ.construct_root_from_parent(
                        segments_per_root,
                        apex_diameter,
                    )

        return self.organs

    def position_secondary_roots(self, input_parameters: RootSimulationModel) -> None:
        """Position secondary roots about the origin.

        Args:
            input_parameters (RootSimulationModel):
                The root simulation data model.
        """
        for order in range(2, input_parameters.max_order + 1):
            for secondary_root in self.organs[order]:
                yaw = self.rng.uniform(-30, 240)
                pitch, roll = self.rng.uniform(60, 110, 2)
                secondary_root.update_transform(yaw=yaw)
                secondary_root.update_transform(pitch=pitch, roll=-roll)
                secondary_root.transform()
                secondary_root.update_transform(pitch=-45, roll=35)
                secondary_root.transform()

        for order in range(input_parameters.max_order, 1, -1):
            for secondary_root in self.organs[order]:
                parent_origin = secondary_root.get_parent_origin()
                secondary_root.cascading_update_transform(translation=parent_origin)
                secondary_root.cascading_transform()

    def position_primary_roots(self, input_parameters: RootSimulationModel) -> None:
        """Position primary roots about the origin.

        Args:
            input_parameters (RootSimulationModel):
                The root simulation data model.
        """
        position_type = RootType.OUTER.value
        yaw_base, yaw_noise_base, yaw = self.get_yaw(input_parameters.outer_root_num)
        for primary_root in self.organs[1]:
            if primary_root.root_type.position_type != position_type:
                continue
            pitch = self.rng.uniform(-20, -15)
            primary_root.cascading_update_transform(pitch=pitch, yaw=yaw)
            primary_root.cascading_transform()
            yaw += yaw_base + self.rng.uniform(-yaw_noise_base, yaw_noise_base)

        position_type = RootType.INNER.value
        yaw_base, yaw_noise_base, yaw = self.get_yaw(input_parameters.inner_root_num)
        for primary_root in self.organs[1]:
            if primary_root.root_type.position_type != position_type:
                continue
            pitch = self.rng.uniform(0, 45)
            primary_root.cascading_update_transform(pitch=pitch, yaw=yaw)
            primary_root.cascading_transform()
            yaw += yaw_base + self.rng.uniform(-yaw_noise_base, yaw_noise_base)

    def validate(
        self,
        input_parameters: RootSimulationModel,
        pitch: int = 45,
    ) -> None:
        """Validate the plausibility of the root system.

        Args:
            input_parameters (RootSimulationModel):
                The root simulation data model.
            pitch (int, optional):
                Pitch in degrees to rotate roots. Defaults to 90.
        """
        for order in range(1, input_parameters.max_order + 1):
            for organ in self.organs[order]:
                organ.validate(
                    input_parameters.no_root_zone,
                    pitch,
                    input_parameters.max_val_attempts,
                )

    def run(self, input_parameters: RootSimulationModel) -> None:
        """Run a root system architecture simulation.

        Args:
            input_parameters (RootSimulationModel):
                The root simulation data model.

        Returns:
            dict:
                The simulation results.
        """
        self.init_organs(input_parameters)
        self.position_secondary_roots(input_parameters)
        self.position_primary_roots(input_parameters)
        self.validate(input_parameters, pitch=60)
