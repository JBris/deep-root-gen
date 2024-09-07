"""Contains the underlying root system simulation model.

This module defines the root system architecture simulation model for constructing 3D root systems.

"""

from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
        self.children: List[RootNode] = []

    def add_child_node(self, child_data: RootNodeModel) -> "RootNode":
        """Add a child node to the hierarchical graph.

        Args:
            child_data (RootNodeModel):
                The node data for the child node.

        Returns:
            RootNode:
                The child node.
        """
        child_data.parent_id = self.node_data.node_id
        child_node = self.G.add_node(child_data)
        self.children.append(child_node)

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

        # Base organ node
        node_data = RootNodeModel()
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

        node_features = node_df.set_index("node_id").T.squeeze().to_dict()
        nx.set_node_attributes(G, node_features, "x")
        return G

    def as_torch(self) -> Data:
        """Return the graph as a PyTorch Geometric graph dataset.

        Returns:
            tuple:
                The graph as a PyTorch Geometric graph dataset.
        """
        G = self.as_networkx()
        torch_G = from_networkx(G)
        return torch_G


class RootSystemSimulation:
    """The root system architecture simulation model."""

    def __init__(self) -> None:
        """RootSystemSimulation constructor.

        Returns:
            RootSystemSimulation:
                The RootSystemSimulation instance.
        """
        self.soil: Soil = Soil()
        self.G: RootSystemGraph = RootSystemGraph()

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

        node_data = RootNodeModel()
        self.G.base_node.add_child_node(node_data)

        run_results["figure"] = fig
        return run_results
