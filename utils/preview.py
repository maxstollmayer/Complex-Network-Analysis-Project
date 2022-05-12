"""
Defines function for graph previewing in 2D and 3D.
"""

from typing import Optional

from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import plotly.graph_objects as go

from aliases import Graph, Positions


def preview(
    G: Graph,
    pos: Positions,
    dim: int = 2,
    renderer: Optional[str] = None,
) -> None:
    "Previews graph layouts in 2D and 3D."
    if dim == 2:
        fig, ax = plt.subplots()
        nx.draw(G, pos, ax=ax, node_size=50)
        ax.set_aspect("equal")
        plt.show()
        return

    if dim == 3:
        P = np.array(list(pos.values()))
        fig = go.Figure(layout=go.Layout(scene=dict(aspectmode="data")))
        fig.add_trace(
            go.Scatter3d(
                x=P[:, 0],
                y=P[:, 1],
                z=P[:, 2],
                mode="markers",
                marker=dict(color=0),
                name="",
                text=list(pos.keys()),
                showlegend=False,
            )
        )
        for i, j in G.edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[pos[i][0], pos[j][0]],
                    y=[pos[i][1], pos[j][1]],
                    z=[pos[i][2], pos[j][2]],
                    mode="lines",
                    line=dict(color="black"),
                    opacity=0.5,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.show(renderer=renderer)
        return

    raise ValueError(
        "Plotting for humans is only supported in 2D and 3D. Sorry for your inconvenience, higher-dimensional being."
    )
