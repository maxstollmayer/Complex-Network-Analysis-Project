"""
Defines type aliases.
"""

from typing import Union, Collection, Tuple, Dict, Any
import networkx as nx


Number = Union[int, float]
Position = Collection[Number]
Positions = Dict[Any, Position]
Color = Tuple[Number, Number, Number, Number]
Colors = Collection[Color]
Names = Collection[str]
Graph = nx.Graph
