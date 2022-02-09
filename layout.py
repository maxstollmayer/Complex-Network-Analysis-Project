"""
Defines Layout class for storing a graph and its layout,
previewing and writing the node and edge tables to files.
"""

from typing import Any, Collection, Dict, Optional, Tuple, Union
import networkx as nx
import numpy as np
import pandas as pd
from plotly.graph_objects import Figure, Scatter3d


# defaults
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


class Layout:
    """
    Class for storing a graph and its layout, previewing and writing tables to files.
    """

    def __init__(
        self,
        G: Graph,
        pos: Positions,
        node_names: Optional[Names] = None,
        node_colors: Optional[Colors] = None,
        edge_colors: Optional[Colors] = None,
    ) -> None:
        """Initializes and validates given data and sets some internal variables."""

        # relabels nodes for combatability with VR
        self.G: Graph = nx.convert_node_labels_to_integers(G)
        self.pos: Positions = {idx: value for idx, (_, value) in enumerate(pos.items())}

        # internal variables
        self._n_nodes = self.G.number_of_nodes()
        self._n_edges = self.G.number_of_edges()
        self._dim = len(pos[0])

        # data validation
        if len(self.pos) != self._n_nodes:
            raise ValueError("Given positions do not match the number of nodes.")
        if self._dim not in [2, 3]:
            raise ValueError(
                f"Only 2D and 3D layouts are supported. Not dimension {self._dim}."
            )

        # calls setters on the optional arguments
        self.node_names = node_names
        self.node_colors = node_colors
        self.edge_colors = edge_colors

    @property
    def node_names(self) -> Names:
        """Getter for node name collection."""
        return self._node_names

    @node_names.setter
    def node_names(self, names: Optional[Names]) -> None:
        """
        Setter for node name collection.
        Sets default values if no names are given or they do not match the number of nodes.
        """
        if names is None:
            self._node_names = [DEFAULT_NAME_PREFIX + str(node) for node in self.G]
            return
        if len(names) != self._n_nodes:
            print("Node names do not match the number of nodes. Uses defaults instead.")
            self._node_names = [DEFAULT_NAME_PREFIX + str(node) for node in self.G]
            return
        self._node_names = names

    @property
    def node_colors(self) -> Colors:
        """Getter for node name collection."""
        return self._node_colors

    @node_colors.setter
    def node_colors(self, colors: Optional[Colors]) -> None:
        """
        Setter for node color collection.
        Sets default values if no colors are given or they do not match the number of nodes.
        """
        if colors is None:
            self._node_colors = np.repeat(
                [DEFAULT_NODE_COLOR], repeats=self._n_nodes, axis=0
            )
            return
        if len(colors) != self._n_nodes:
            print(
                "Node colors do not match the number of nodes. Uses defaults instead."
            )
            self._node_colors = np.repeat(
                [DEFAULT_NODE_COLOR], repeats=self._n_nodes, axis=0
            )
            return
        self._node_colors = colors

    @property
    def edge_colors(self) -> Colors:
        """Getter for edge name collection."""
        return self._edge_colors

    @edge_colors.setter
    def edge_colors(self, colors: Optional[Colors]) -> None:
        """
        Setter for edge color collection.
        Sets default values if no colors are given or they do not match the number of edges.
        """
        if colors is None:
            self._edge_colors = np.repeat(
                [DEFAULT_EDGE_COLOR], repeats=self._n_edges, axis=0
            )
            return
        if len(colors) != self._n_edges:
            print(
                "Edge colors do not match the number of edges. Uses defaults instead."
            )
            self._edge_colors = np.repeat(
                [DEFAULT_EDGE_COLOR], repeats=self._n_edges, axis=0
            )
            return
        self._edge_colors = colors

    @property
    def node_table(self) -> pd.DataFrame:
        """Returns table with node data for VR software."""

        # normalizes coordinates to [0, 1]
        coords = np.array(list(self.pos.values()))
        max_values = coords.max(axis=0)
        min_values = coords.min(axis=0)
        coords = (coords - min_values) / (max_values - min_values)

        if self._dim == 2:
            data = [
                [*xy, DEFAULT_2D_Z, *rgba, name]
                for xy, rgba, name in zip(coords, self.node_colors, self.node_names)
            ]
        else:  # dim == 3
            data = [
                [*xyz, *rgba, name]
                for xyz, rgba, name in zip(coords, self.node_colors, self.node_names)
            ]
        return pd.DataFrame(data, columns=["x", "y", "z", "r", "g", "b", "a", "name"])

    @property
    def edge_table(self) -> pd.DataFrame:
        """Returns table with edge data for VR software."""
        data = [[*ij, *rgba] for ij, rgba in zip(self.G.edges, self.edge_colors)]
        return pd.DataFrame(data, columns=["i", "j", "r", "g", "b", "a"])

    def preview(self, renderer: Optional[str] = "notebook_connected") -> None:
        """
        Displays a 3D plotly figure of the given network layout.
        
        The following are the most relevant renderers.
        None: plotly chooses (can fail in notebook)
        "browser": opens new tab in default web browser
        "notebook": adds notebook renderer to file
        "notebook_connected": calls online notebook renderer
        """

        # normalizes colors and alpha to [0, 1]
        node_colors_normalized = normalize_colors(self.node_colors)
        edge_colors_normalized = normalize_colors(self.edge_colors)

        fig = Figure()
        fig.add_trace(
            Scatter3d(
                x=self.node_table["x"],
                y=self.node_table["y"],
                z=self.node_table["z"],
                mode="markers",
                marker=dict(color=node_colors_normalized),
                text=self.node_table["name"],
                name="",
                showlegend=False,
            )
        )
        for i, j in self.G.edges:
            line = self.node_table.loc[[i, j], :]
            fig.add_trace(
                Scatter3d(
                    x=line["x"],
                    y=line["y"],
                    z=line["z"],
                    mode="lines",
                    line=dict(color=edge_colors_normalized),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.show(renderer=renderer)

    def write(
        self,
        node_file_path: str = DEFAULT_NODE_FILE,
        edge_file_path: str = DEFAULT_EDGE_FILE,
    ) -> None:
        """Writes node and edge tables to separate csv files."""
        self.node_table.to_csv(path_or_buf=node_file_path, index=False, header=False)
        self.edge_table.to_csv(path_or_buf=edge_file_path, index=False, header=False)


# only for testing purposes
def main() -> None:
    """
    Generates exemplary graph and layout, displays a preview and prints node and edge tables.
    """
    G: Graph = nx.erdos_renyi_graph(10, 0.5)
    pos = nx.spring_layout(G, dim=3)
    layout = Layout(G, pos)
    print("Node data:")
    print(layout.node_table)
    print("\nEdge data:")
    print(layout.edge_table)
    layout.preview(renderer=None)


if __name__ == "__main__":
    main()
