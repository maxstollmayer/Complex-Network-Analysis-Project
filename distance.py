from typing import Union, Collection, Callable, Any, Dict
from itertools import combinations
import numpy as np


# type aliases
Number = Union[int, float]
Position = Collection[Number]
Positions = Dict[Any, Position]
Metric = Callable[[Position, Position], float]


def euclidean_distance(pos1: Position, pos2: Position) -> float:
    return np.sqrt(np.sum((np.array(pos1) - np.array(pos2)) ** 2))


def get_distance(positions: Positions, metric: Metric = euclidean_distance) -> Dict:
    edge_weights = dict()

    nodes = list(positions.keys())
    for node1 in positions.keys():
        nodes.remove(node1)
        for node2 in nodes:
            if not node1 == node2:
                pos1 = positions[node1]
                pos2 = positions[node2]
                edge_weights[(node1, node2)] = metric(pos1, pos2)

    return edge_weights


if __name__ == "__main__":
    import networkx as nx

    G = nx.erdos_renyi_graph(10, 0.2)
    pos = nx.spring_layout(G)
    print(get_distance(pos))
