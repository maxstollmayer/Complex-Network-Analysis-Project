"""
Defines Layout class for storing a graph and its layout,
previewing and writing the node and edge tables to files.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Union
import networkx as nx
import numpy as np
import pandas as pd


# defaults
NODE_COLUMNS = ["x", "y", "z", "r", "g", "b", "a", "name"]
EDGE_COLUMNS = ["i", "j", "r", "g", "b", "a"]
DEFAULT_2D_Z = 0
DEFAULT_NODE_COLOR = [31, 119, 180, 100]
DEFAULT_EDGE_COLOR = [0, 0, 0, 100]
DEFAULT_NAME_PREFIX = "node_"
DEFAULT_NODE_FILE = "nodes.csv"
DEFAULT_EDGE_FILE = "edges.csv"

# type aliases
Graph = nx.Graph
Positions = Dict[Any, Iterable[Union[int, float]]]
Color = Iterable[int]
Colors = Iterable[Color]
Names = Iterable[str]


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

    def node_table(self) -> pd.DataFrame:
        """
        Returns table with normalized node coordinates, colors and names.
        """

        # sets default node names and colors if not given
        n_nodes = self.graph.number_of_nodes()
        if self.node_names is None:
            self.node_names = [DEFAULT_NAME_PREFIX + str(idx) for idx in range(n_nodes)]
        if self.node_colors is None:
            self.node_colors = np.repeat([DEFAULT_NODE_COLOR], repeats=n_nodes, axis=0)

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
        elif dim == 3:
            data = [
                [*xyz, *rgba, name]
                for xyz, rgba, name in zip(coords, self.node_colors, self.node_names)
            ]
        else:
            raise ValueError(
                f"Only 2D and 3D layouts are supported. Not dimension {dim}."
            )

        return pd.DataFrame.from_records(data, columns=NODE_COLUMNS)

    def edge_table(self) -> pd.DataFrame:
        """Returns table with edge data and colors."""

        # sets default edge colors if not given
        if self.edge_colors is None:
            self.edge_colors = np.repeat(
                [DEFAULT_EDGE_COLOR], repeats=self.graph.number_of_edges(), axis=0
            )

        data = [[*ij, *rgba] for ij, rgba in zip(self.graph.edges, self.edge_colors)]
        return pd.DataFrame(data, columns=EDGE_COLUMNS)

    def preview(self) -> None:
        """"""
        pass

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
    Generates exemplary graph and layout, displays a preview, prints and writes node and edge tables.
    """
    G: Graph = nx.erdos_renyi_graph(10, 0.25)
    pos = nx.spring_layout(G, dim=3)
    layout = Layout(G, pos)
    print("Node data:")
    print(layout.node_table())
    print("\nEdge data:")
    print(layout.edge_table())
    layout.preview()
    layout.write()


if __name__ == "__main__":
    main()
