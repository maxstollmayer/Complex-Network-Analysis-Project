"""
Defines Layout class for storing a graph and its layout,
previewing and writing the node and edge tables to files.
"""

from dataclasses import dataclass
from typing import Any, Collection, Dict, Optional, Tuple, Union
import networkx as nx
import numpy as np
import pandas as pd
from plotly import graph_objects as go


# defaults
NODE_COLUMNS = ["x", "y", "z", "r", "g", "b", "a", "name"]
EDGE_COLUMNS = ["i", "j", "r", "g", "b", "a"]
DEFAULT_2D_Z = 0
DEFAULT_NODE_COLOR = (31, 119, 180, 100)
DEFAULT_EDGE_COLOR = (0, 0, 0, 100)
DEFAULT_NAME_PREFIX = "node_"
DEFAULT_NODE_FILE = "nodes.csv"
DEFAULT_EDGE_FILE = "edges.csv"

# type aliases
Number = Union[int, float]
Graph = nx.Graph
Positions = Dict[Any, Collection[Number]]
Color = Tuple[Number, Number, Number, Number]
Colors = Collection[Color]
Names = Collection[str]


def normalize_colors(colors: Colors) -> Colors:
    """Returns normalized RGBA colors in range [0, 1]."""
    normalized = np.array(colors)
    normalized[:, -1] = np.where(normalized[:, -1] >= 100, 100, normalized[:, -1])
    normalized = normalized / np.array([255, 255, 255, 100])
    return normalized


@dataclass
class Layout:
    """
    Class for storing a graph and its layout, previewing and writing tables to files.
    """

    graph: Graph
    positions: Positions
    node_names: Optional[Names] = None
    node_colors: Optional[Colors] = None
    edge_colors: Optional[Colors] = None

    def __post_init__(self) -> None:
        """Data validation and assignments of default colors and names where necessary."""
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()

        if len(self.positions) != n_nodes:
            raise ValueError("Given positions are not compatible with graph.")

        if np.array(list(self.positions.values())).shape[1] not in [2, 3]:
            raise ValueError("Only 2D and 3D layouts are supported.")

        if self.node_names is None or len(self.node_names) != n_nodes:
            self.node_names = [
                DEFAULT_NAME_PREFIX + str(node) for node in self.positions.keys()
            ]

        if self.node_colors is None or len(self.node_colors) != n_nodes:
            self.node_colors = self.node_colors = np.repeat(
                [DEFAULT_NODE_COLOR], repeats=n_nodes, axis=0
            )

        if self.edge_colors is None or len(self.edge_colors) != n_edges:
            self.edge_colors = np.repeat(
                [DEFAULT_EDGE_COLOR], repeats=self.graph.number_of_edges(), axis=0
            )

    def node_table(self) -> pd.DataFrame:
        """
        Returns table with normalized node coordinates, colors and names.
        """

        # normalizes coordinates
        coords = np.array(list(self.positions.values()))
        max_values = coords.max(axis=0)
        min_values = coords.min(axis=0)
        coords = (coords - min_values) / (max_values - min_values)

        dim = coords.shape[1]
        if dim == 2:
            data = [
                [*xy, DEFAULT_2D_Z, *rgba, name]
                for xy, rgba, name in zip(coords, self.node_colors, self.node_names)
            ]
        else:
            data = [
                [*xyz, *rgba, name]
                for xyz, rgba, name in zip(coords, self.node_colors, self.node_names)
            ]

        return pd.DataFrame.from_records(data, columns=NODE_COLUMNS)

    def edge_table(self) -> pd.DataFrame:
        """Returns table with edge data and colors."""
        data = [[*ij, *rgba] for ij, rgba in zip(self.graph.edges, self.edge_colors)]
        return pd.DataFrame(data, columns=EDGE_COLUMNS)

    def preview(self, renderer: Optional[str] = "notebook_connected") -> None:
        """
        Displays a 3D plotly figure of the given network layout.
        The following are the most relevant renderers.
        None: plotly chooses (can fail in notebook)
        "browser": opens new tab in default web browser
        "notebook": adds notebook renderer to file
        "notebook_connected": calls online notebook renderer
        """
        normalized_node_colors = normalize_colors(self.node_colors)
        normalized_edge_colors = normalize_colors(self.edge_colors)
        nodes = self.node_table()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=nodes["x"],
                y=nodes["y"],
                z=nodes["z"],
                mode="markers",
                marker=dict(color=normalized_node_colors),
                text=nodes["name"],
                name="",
                showlegend=False,
            )
        )
        for u, v in self.graph.edges:
            df = nodes.loc[[u, v], :]
            fig.add_trace(
                go.Scatter3d(
                    x=df["x"],
                    y=df["y"],
                    z=df["z"],
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                    line=dict(color=normalized_edge_colors),
                )
            )
        fig.show(renderer=renderer)

    def write(
        self,
        node_file_path: str = DEFAULT_NODE_FILE,
        edge_file_path: str = DEFAULT_EDGE_FILE,
    ) -> None:
        """Writes node and edge tables to separate csv files."""
        self.node_table().to_csv(path_or_buf=node_file_path, index=False, header=False)
        self.edge_table().to_csv(path_or_buf=edge_file_path, index=False, header=False)


# only for testing purposes
def main() -> None:
    """
    Generates exemplary graph and layout, displays a preview and prints node and edge tables.
    """
    G: Graph = nx.erdos_renyi_graph(10, 0.5)
    pos = nx.spring_layout(G, dim=3)
    layout = Layout(G, pos)
    print("Node data:")
    print(layout.node_table())
    print("\nEdge data:")
    print(layout.edge_table())
    layout.preview(renderer=None)


if __name__ == "__main__":
    main()
