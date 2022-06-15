"""
Defines Layout class for storing a graph and its layout,
previewing and writing the node and edge tables to files.
"""

from typing import Optional

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.aliases import Graph, Positions, Colors, Names


# defaults
DEFAULT_NODE_COLOR = (31, 119, 180, 100)
DEFAULT_EDGE_COLOR = (0, 0, 0, 100)
NODE_NAME_PREFIX = "node_"
NODE_FILE_SUFFIX = "nodes.csv"
EDGE_FILE_SUFFIX = "edges.csv"


def normalize_colors(colors: Colors) -> Colors:
    """Returns normalized RGBA colors in range [0, 1]."""
    normalized = np.array(colors)
    normalized[:, -1] = np.where(normalized[:, -1] >= 100, 100, normalized[:, -1])
    normalized = normalized / np.array([255, 255, 255, 100])
    return normalized


class Layout:
    """
    Class for storing a graph and its layout, previewing and writing tables to files for the VR app.
    """

    def __init__(
        self,
        G: Graph,
        pos: Optional[Positions] = None,
        node_names: Optional[Names] = None,
        node_colors: Optional[Colors] = None,
        edge_colors: Optional[Colors] = None,
    ) -> None:
        """Initializes class and sets some internal variables."""

        # copies graph
        self.G: Graph = G.copy()

        # internal variables
        self._n_nodes = self.G.number_of_nodes()
        self._n_edges = self.G.number_of_edges()
        self._relabeled_edges = nx.convert_node_labels_to_integers(self.G).edges

        # calls setters on optional arguments
        self.pos = pos
        self.node_names = node_names
        self.node_colors = node_colors
        self.edge_colors = edge_colors

    @property
    def pos(self) -> Positions:
        """Getter for node positions."""
        return self._pos

    @pos.setter
    def pos(self, pos: Optional[Positions]) -> None:
        """
        Setter for node positions. Sets spring layout positions of none given.
        Raises ValueError if they are not 3D or do not match number of nodes.
        """
        if pos is None:
            self._pos = nx.spring_layout(self.G, dim=3)
            return
        if len(pos) != self._n_nodes:
            raise ValueError("Positions do not match the number of nodes.")
        if len(pos[0]) != 3:
            raise ValueError(f"Only 3D layouts are supported.")
        self._pos = pos

    @property
    def node_names(self) -> Names:
        """Getter for node name collection."""
        return self._node_names

    @node_names.setter
    def node_names(self, names: Optional[Names]) -> None:
        """
        Setter for node name collection. Sets default values if none given.
        Raises ValueError if they do not match the number of nodes.
        """
        if names is None:
            self._node_names = self.G.nodes
            return
        if len(names) != self._n_nodes:
            raise ValueError("Node names do not match the number of nodes.")
        self._node_names = names

    @property
    def node_colors(self) -> Colors:
        """Getter for node name collection."""
        return self._node_colors

    @node_colors.setter
    def node_colors(self, colors: Optional[Colors]) -> None:
        """
        Setter for node color collection. Sets default color if none given.
        Raises ValueError if they do not match the number of nodes.
        """
        if colors is None:
            self._node_colors = np.repeat(
                [DEFAULT_NODE_COLOR], repeats=self._n_nodes, axis=0
            )
            return
        if len(colors) != self._n_nodes:
            raise ValueError("Node colors do not match the number of nodes.")
        self._node_colors = colors

    @property
    def edge_colors(self) -> Colors:
        """Getter for edge name collection."""
        return self._edge_colors

    @edge_colors.setter
    def edge_colors(self, colors: Optional[Colors]) -> None:
        """
        Setter for edge color collection. Sets default values if none given.
        Raises ValueError if they do not match the number of edges.
        """
        if colors is None:
            self._edge_colors = np.repeat(
                [DEFAULT_EDGE_COLOR], repeats=self._n_edges, axis=0
            )
            return
        if len(colors) != self._n_edges:
            raise ValueError("Edge colors do not match the number of edges.")
        self._edge_colors = colors

    @property
    def node_table(self) -> pd.DataFrame:
        """Returns table with node data for VR software."""

        # normalizes coordinates to [0, 1]
        coords = np.array(list(self.pos.values()))
        max_value = coords.max()
        min_value = coords.min()
        coords = (coords - min_value) / (max_value - min_value)

        data = [
            [*xyz, *rgba, name]
            for xyz, rgba, name in zip(coords, self.node_colors, self.node_names)
        ]
        return pd.DataFrame(data, columns=["x", "y", "z", "r", "g", "b", "a", "name"])

    @property
    def edge_table(self) -> pd.DataFrame:
        """Returns table with edge data for VR software."""
        data = [
            [*ij, *rgba] for ij, rgba in zip(self._relabeled_edges, self.edge_colors)
        ]
        return pd.DataFrame(data, columns=["i", "j", "r", "g", "b", "a"])

    def preview(self, renderer: Optional[str] = "notebook_connected") -> None:
        """
        Displays a 3D figure of the given network layout.

        The following are the most relevant renderers.
        None: plotly with automatic renderer
        "matplotlib": static 3D matplotlib plot
        "browser": opens new tab in default web browser
        "notebook": adds notebook renderer to file
        "notebook_connected": calls online notebook renderer
        """

        # normalizes colors and alpha to [0, 1]
        node_colors_normalized = normalize_colors(self.node_colors)
        edge_colors_normalized = normalize_colors(self.edge_colors)

        # static matplotlib
        if renderer == "matplotlib":
            plt.style.use("default")
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            ax.scatter(
                self.node_table["x"],
                self.node_table["y"],
                self.node_table["z"],
                c=node_colors_normalized,
            )

            for k, (i, j) in enumerate(self._relabeled_edges):
                line = self.node_table.loc[[i, j], :]
                ax.plot(
                    line["x"], line["y"], line["z"], color=edge_colors_normalized[k]
                )

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.axes.set_xlim3d(left=0, right=1)
            ax.axes.set_ylim3d(bottom=0, top=1)
            ax.axes.set_zlim3d(bottom=0, top=1)

            fig.tight_layout()
            plt.show()
            return

        # interactive plotly
        fig = go.Figure(
            layout=go.Layout(
                scene=dict(
                    aspectmode="cube",
                    xaxis=dict(
                        range=[0, 1],
                    ),
                    yaxis=dict(
                        range=[0, 1],
                    ),
                    zaxis=dict(
                        range=[0, 1],
                    ),
                ),
            )
        )
        fig.add_trace(
            go.Scatter3d(
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
        for k, (i, j) in enumerate(self._relabeled_edges):
            line = self.node_table.loc[[i, j], :]
            color = f"rgba({self.edge_colors[k][0]}, {self.edge_colors[k][1]}, {self.edge_colors[k][2]}, {edge_colors_normalized[k][-1]})"
            fig.add_trace(
                go.Scatter3d(
                    x=line["x"],
                    y=line["y"],
                    z=line["z"],
                    mode="lines",
                    line=dict(color=color),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.show(renderer=renderer)
        return

    def write(
        self,
        file_path: Optional[str] = None,
    ) -> None:
        """Writes node and edge tables to separate csv files."""
        if file_path is None:
            node_file = NODE_FILE_SUFFIX
            edge_file = EDGE_FILE_SUFFIX
        else:
            node_file = file_path + "_" + NODE_FILE_SUFFIX
            edge_file = file_path + "_" + EDGE_FILE_SUFFIX
        self.node_table.to_csv(node_file, index=False, header=False)
        self.edge_table.to_csv(edge_file, index=False, header=False)


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
