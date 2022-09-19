"""ML utility Functions for working with graphs."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT

# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from typing import Any, List

import networkx as nx
from loguru import logger as log

from graphein.utils.utils import import_message

try:
    from torch_geometric.data import Data
except ImportError:
    message = import_message(
        "graphein.ml.utils",
        "torch_geometric",
        pip_install=True,
        conda_channel="pyg",
    )
    log.warning(message)


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


def combine_pyg_data(interactor_a: Data, interactor_b: Data) -> Data:
    """
    Combines two pytorch geometric Data objects into a single object. This is
    useful in paired input problems, e.g. protein-protein interaction.

    Attributes stored in each interactor are given prefixes in the returned
    object. E.g.: ``interactor_a.x`` becomes ``output.a_x` (similarly we will
    also have: ``output.b_x``.

    :param interactor_a: The first interactor.
    :type interactor_a: Data
    :param interactor_b: The second interactor.
    :type interactor_b: Data
    """
    a = interactor_a.to_dict()
    b = interactor_b.to_dict()
    a = {f"a_{k}": v for k, v in a.items()}
    b = {f"b_{k}": v for k, v in b.items()}
    return Data().from_dict({**a, **b})
