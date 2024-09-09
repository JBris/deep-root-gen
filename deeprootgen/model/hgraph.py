"""Contains the hierarchical graph representation of the root system.

This module defines the hierarchical graph representation of the root system.
This includes integration with NetworkX and PyTorch Geometric.

"""

# mypy: ignore-errors

from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from ..data_model import RootEdgeModel, RootNodeModel
from ..spatial import get_transform_matrix, make_homogenous


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
            segment_rank = 1
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
