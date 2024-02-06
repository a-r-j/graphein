"""ML utility Functions for working with graphs."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT

# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import Any, List

import networkx as nx


def add_labels_to_graph(
    graphs: List[nx.Graph], labels: List[Any], name: str
) -> List[nx.Graph]:
    """Adds labels to a graph.

    :param graphs: A list of graphs to add labels to.
    :type graphs: List[nx.Graph]
    :param labels: A list of labels to add to the graphs.
    :type labels: List[Any]
    :param name: The name to add the label under.
    :type name: str
    :return: A list of graphs with labels added.
    :rtype: List[nx.Graph]
    """
    for i, g in enumerate(graphs):
        g.graph[name] = labels[i]
    return graphs
