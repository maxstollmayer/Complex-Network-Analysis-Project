"""
Defines functions for spherical layout and to create spheres around nodes.
"""


from typing import Tuple, Optional
import networkx as nx
import numpy as np

from aliases import Graph, Position, Positions
from preview import preview


def fibonacci_sphere(n_samples: int) -> np.ndarray:
    "Distributes points radially on a 3D unit sphere according to golden ratio."
    phi = (3 - np.sqrt(5)) * np.pi  # golden ratio in radians
    theta = phi * np.arange(n_samples)

    points = np.zeros((n_samples, 3))
    points[:, 2] = np.linspace(1 / n_samples - 1, 1 - 1 / n_samples, n_samples)
    radii = np.sqrt(1 - points[:, 2] ** 2)
    points[:, 0] = radii * np.sin(theta)
    points[:, 1] = radii * np.cos(theta)

    return points


def spherical_layout(
    G: Graph, scale: float = 1, center: Optional[Position] = None, dim: int = 2
) -> Positions:
    "Position nodes on a sphere in given dimension."
    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)
    if len(center) != dim:
        raise ValueError("Length of center coordinates does not match dimension.")

    N = len(G)
    if N <= 0:
        return {}
    elif N == 1:
        return {nx.utils.arbitrary_element(G): center}
    else:
        if dim == 2:
            return nx.circular_layout(G, scale, center)
        if dim == 3:
            pos = scale * fibonacci_sphere(N) + center
            return dict(zip(G, pos))
        raise NotImplementedError("Only dimensions 2 and 3 are currently supported.")


def fibonacci_sphere_graph(n_nodes: int) -> Graph:
    "Returns graph of 3D sphere with edges that connect nodes that are close in Fibonacci layout."
    G = nx.empty_graph(n_nodes)
    pos = spherical_layout(G, dim=3)

    # rough estimate of distance between neighboring nodes on fibonacci sphere
    radius = 1.5 * 4 * 2 * np.pi / np.sqrt(3) / n_nodes

    # connect close nodes
    nodes = list(G.nodes)
    for node in G:
        nodes.remove(node)
        for other in nodes:
            dist = np.sum((pos[node] - pos[other]) ** 2)
            if dist <= radius:
                G.add_edge(node, other)
    return G


def spherical_graph(n_nodes: int, dim: int = 2) -> Graph:
    "Returns wireframe graph of 2D or 3D sphere."
    if dim == 2:
        return nx.cycle_graph(n_nodes)
    if dim == 3:
        return fibonacci_sphere_graph(n_nodes)
    raise NotImplementedError(
        "Higher-dimensional spherical graphs are not yet supported."
    )


def spheres(
    G: Graph,
    pos: Positions,
    radius: float,
    samples: int = 10,
    dim: int = 2,
) -> Tuple[Graph, Positions]:
    "Creates 2D or 3D spheres of given radius around each node in given layout."
    G = G.copy()
    pos = pos.copy()
    for node in G:
        # create sphere around node
        S = spherical_graph(samples, dim=dim)
        nx.relabel_nodes(S, lambda x: f"{node}_s{x}", copy=False)

        # add sphere to graph and layout
        G = nx.compose(G, S)
        S_pos = spherical_layout(S, scale=radius, center=pos[node], dim=dim)
        pos.update(S_pos)
    return G, pos


def main() -> None:
    "Previews a graph in 2D and 3D with spheres."
    G = nx.empty_graph(1)
    for dim in [2, 3]:
        pos = nx.spring_layout(G, dim=dim)
        G2, pos2 = spheres(G, pos, radius=0.1, samples=3**dim, dim=dim)
        preview(G2, pos2, dim=dim)


if __name__ == "__main__":
    main()
