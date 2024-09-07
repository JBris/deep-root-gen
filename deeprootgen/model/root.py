"""Contains the underlying root system simulation model.

This module defines the root system architecture simulation model for constructing 3D root systems.

"""

# mypy: ignore-errors

from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.random import default_rng
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from ..data_model import RootEdgeModel, RootNodeModel, RootSimulationModel
from .soil import Soil


class RootNode:
    """A node within the hierarchical graph representation of the root system."""

    def __init__(self, G: "RootSystemGraph", node_data: RootNodeModel) -> None:
        """RootNode constructor.

        Args:
            G: (RootSystemGraph):
                The hierarchical graph representation of the root system.
            node_data (RootNodeModel):
                The root node data model.

        Returns:
            RootNode:
                The RootNode instance.
        """
        self.G = G
        self.node_data = node_data

    def add_child_node(
        self, child_data: RootNodeModel, new_organ: bool = False
    ) -> "RootNode":
        """Add a child node to the hierarchical graph.

        Args:
            child_data (RootNodeModel):
                The node data for the child node.
            new_organ (bool):
                Whether the new child node belongs to a new plant organ.

        Returns:
            RootNode:
                The child node.
        """
        if new_organ:
            organ_id = self.G.increment_organ_id()
            segment_rank = 0
            order = self.node_data.order + 1
        else:
            organ_id = self.node_data.organ_id
            segment_rank = self.node_data.segment_rank + 1
            order = self.node_data.order

        child_data.parent_id = self.node_data.node_id
        child_data.organ_id = organ_id
        child_data.plant_id = self.node_data.plant_id
        child_data.segment_rank = segment_rank
        child_data.order = order

        child_node = self.G.add_node(child_data)
        edge_data = RootEdgeModel(
            parent_id=self.node_data.node_id, child_id=child_node.node_data.node_id
        )
        self.G.add_edge(edge_data)

        return child_node

    def as_dict(self) -> dict:
        """Return the graph node as a dictionary.

        Returns:
            dict:
                The node as a dictionary.
        """
        return self.node_data.dict()


class RootEdge:
    """An edge within the hierarchical graph representation of the root system."""

    def __init__(self, G: "RootSystemGraph", edge_data: RootEdgeModel) -> None:
        """RootEdge constructor.

        Args:
            G: (RootSystemGraph):
                The hierarchical graph representation of the root system.
            edge_data (RootEdgeModel):
                The root edge data model.

        Returns:
            RootEdge:
                The RootEdge instance.
        """
        self.G = G
        self.edge_data = edge_data

    def as_dict(self) -> dict:
        """Return the graph edge as a dictionary.

        Returns:
            dict:
                The edge as a dictionary.
        """
        return self.edge_data.dict()


class RootSystemGraph:
    """The hierarchical graph representation of the root system."""

    def __init__(self) -> None:
        """RootSystemGraph constructor.

        Returns:
            RootSystemSimulation:
                The RootSystemGraph instance.
        """
        self.nodes: List[RootNode] = []
        self.edges: List[RootEdge] = []
        self.node_id = 0
        self.organ_id = 0

        # Base organ node
        node_data = RootNodeModel()
        node_data.organ_id = self.increment_organ_id()
        self.base_node = self.add_node(node_data)

    def add_node(self, node_data: RootNodeModel) -> RootNode:
        """Construct a new RootNode.

        Args:
            node_data (RootNodeModel):
                The root node data model.

        Returns:
            RootNode:
                The new RootNode.
        """
        node_data.node_id = self.increment_node_id()
        node = RootNode(self, node_data)

        self.nodes.append(node)
        return node

    def add_edge(self, edge_data: RootEdgeModel) -> RootEdge:
        """Construct a new RootEdge.

        Args:
            node_data (RootEdgeModel):
                The root edge data model.

        Returns:
            RootEdge:
                The new RootEdge.
        """
        edge = RootEdge(self, edge_data)
        self.edges.append(edge)
        return edge

    def increment_node_id(self) -> int:
        """Increment the node ID.

        Returns:
            int:
                The node ID prior to incrementation.
        """
        node_id = self.node_id
        self.node_id += 1
        return node_id

    def increment_organ_id(self) -> int:
        """Increment the organ ID.

        Returns:
            int:
                The organ ID prior to incrementation.
        """
        organ_id = self.organ_id
        self.organ_id += 1
        return organ_id

    def as_dict(self) -> tuple:
        """Return the graph as a tuple of node and edge lists.

        Returns:
            tuple:
                The graph as a tuple of node and edge lists.
        """
        nodes = []
        for n in self.nodes:
            nodes.append(n.as_dict())

        edges = []
        for e in self.edges:
            edges.append(e.as_dict())

        return nodes, edges

    def as_df(self) -> tuple:
        """Return the graph as a tuple of node and edge dataframes.

        Returns:
            tuple:
                The graph as a tuple of node and edge dataframes.
        """
        nodes, edges = self.as_dict()
        node_df = pd.DataFrame(nodes)
        edge_df = pd.DataFrame(edges)
        return node_df, edge_df

    def as_networkx(self) -> nx.Graph:
        """Return the graph as a NetworkX graph.

        Returns:
            tuple:
                The graph in NetworkX format.
        """
        node_df, edge_df = self.as_df()

        G = nx.from_pandas_edgelist(
            edge_df, "parent_id", "child_id", create_using=nx.Graph()
        )

        node_features = node_df.set_index("node_id", drop=False).T.squeeze().to_dict()
        nx.set_node_attributes(G, node_features, "x")
        return G

    def as_torch(self) -> Data:
        """Return the graph as a PyTorch Geometric graph dataset.

        Returns:
            tuple:
                The graph as a PyTorch Geometric graph dataset.
        """
        G = self.as_networkx()
        torch_G = from_networkx(G).double()
        return torch_G


class RootOrgan:
    """A single root organ within the root system."""

    def __init__(
        self,
        parent_node: RootNode,
        input_parameters: RootSimulationModel,
        rng: np.random.Generator,
    ) -> None:
        """RootOrgan constructor.

        Args:
            parent_node (RootNode):
                The parent root node.
            input_parameters (RootSimulationModel):
                The root simulation data model.
            rng (np.random.Generator):
                The random number generator.

        Returns:
            RootOrgan:
                A root organ within the root system.
        """
        self.parent_node = parent_node
        self.segments: List[RootNode] = []
        self.child_roots: List["RootOrgan"] = []

        # Diameter
        self.base_diameter = input_parameters.base_diameter * input_parameters.max_order
        self.diameter_reduction = input_parameters.diameter_reduction

        # Length
        self.proot_length_interval = np.array(
            [input_parameters.min_primary_length, input_parameters.max_primary_length]
        )
        self.sroot_length_interval = np.array(
            [input_parameters.min_sec_root_length, input_parameters.max_sec_root_length]
        )
        self.length_reduction = input_parameters.length_reduction

        self.rng = rng

    def init_diameters(self, segments_per_root: int, apex_diameter: int) -> np.ndarray:
        """Initialise root diameters for the root organ.

        Args:
            segments_per_root (int):
                The number of segments for a single root organ.
            apex_diameter (int):
                The diameter of the root apex.
        """
        base_diameter = self.base_diameter * self.diameter_reduction ** (
            self.parent_node.node_data.order
        )
        diameters = self.rng.uniform(apex_diameter, base_diameter, segments_per_root)
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

        lengths = self.rng.uniform(min_length, max_length, segments_per_root)
        lengths = np.sort(lengths)[::-1]
        return lengths

    def add_child_node(
        self,
        parent_node: RootNode,
        diameters: np.ndarray,
        lengths: np.ndarray,
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
        node_data = RootNodeModel(diameter=diameter, length=length)

        child_node = parent_node.add_child_node(node_data, new_organ=new_organ)
        self.segments.append(child_node)
        return child_node

    def construct_root(
        self, segments_per_root: int, apex_diameter: int
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

        child_node = self.add_child_node(
            self.parent_node, diameters=diameters, lengths=lengths, i=0, new_organ=True
        )

        for i in range(1, segments_per_root):
            child_node = self.add_child_node(
                child_node, diameters=diameters, lengths=lengths, i=i, new_organ=False
            )

        return self.segments


class RootSystemSimulation:
    """The root system architecture simulation model."""

    def __init__(self, random_seed: int = None) -> None:
        """RootSystemSimulation constructor.

        Args:
            random_seed (int, optional):
                The seed for the random number generator. Defaults to None.

        Returns:
            RootSystemSimulation:
                The RootSystemSimulation instance.
        """
        self.soil: Soil = Soil()
        self.G: RootSystemGraph = RootSystemGraph()
        self.organs: List["RootOrgan"] = []
        self.rng = default_rng(random_seed)

    def run(self, input_parameters: RootSimulationModel) -> dict:
        """Run a root system architecture simulation.

        Args:
            input_parameters (RootSimulationModel):
                The root simulation data model.

        Returns:
            dict:
                The simulation results.
        """
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

        segments_per_root = input_parameters.segments_per_root
        apex_diameter = input_parameters.apex_diameter

        organ = RootOrgan(
            self.G.base_node, input_parameters=input_parameters, rng=self.rng
        )
        organ.construct_root(segments_per_root, apex_diameter)
        print(self.G.as_dict())
        run_results["figure"] = fig
        return run_results
