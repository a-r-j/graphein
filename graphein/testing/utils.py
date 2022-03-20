"""Testing utilities for the Graphein library."""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import logging as log
from typing import Any, Callable, Dict

import networkx as nx
import numpy as np


def graphs_isomorphic(g: nx.Graph, h: nx.Graph) -> bool:
    """Checks for structural isomorphism between two graphs: ``g`` and ``h``.

    :param g: The first graph.
    :type g: :class:`networkx.Graph`
    :param h: The second graph.
    :type h: :class:`networkx.Graph`
    :return: ``True`` if the graphs are isomorphic, ``False`` otherwise.
    :rtype: bool
    """
    return nx.is_isomorphic(g, h)


def assert_graphs_isomorphic(g: nx.Graph, h: nx.Graph):
    """Checks for structural isomorphism between two graphs: ``g`` and ``h``.

    :param g: The first graph.
    :param h: The second graph.
    :raises AssertionError: If the graphs are not isomorphic.
    """
    assert graphs_isomorphic(g, h), "Graphs are not isomorphic."


def nodes_equal(g: nx.Graph, h: nx.Graph):
    """Checks whether two graphs have the same nodes.

    :param g: The first graph.
    :type g: :class:`networkx.Graph`
    :param h: The second graph.
    :type h: :class:`networkx.Graph`
    :raises AssertionError: If the graphs do not contain the same nodes
    """
    for n in g.nodes():
        assert n in h.nodes(), f"Node {n} (graph g) not in graph h"
    for n in h.nodes():
        assert n in g.nodes(), f"Node {n} (graph h) not in graph g"


def node_data_equal(g: nx.Graph, h: nx.Graph):
    """Checks whether two graphs have the same node features.

    :param g: The first graph.
    :type g: :class:`networkx.Graph`
    :param h: The second graph.
    :type h: :class:`networkx.Graph`
    :raises AssertionError: If the graphs do not contain the same nodes
    """
    for n in g.nodes():
        assert dictionaries_equal(g.nodes[n], h.nodes[n]), \
            f"Node {n} (graph g) features do not match graph h"
    for n in h.nodes():
        assert dictionaries_equal(g.nodes[n], h.nodes[n]), \
            f"Node {n} (graph h) features do not match graph g"


def dictionaries_equal(dic1: Dict[str, Any], dic2: Dict[str, Any]) -> bool:
    """Checks if two dictionaries are equal.

    :param dic1: _description_
    :type dic1: Dict[str, Any]
    :param dic2: _description_
    :type dic2: Dict[str, Any]
    :return: _description_
    :rtype: bool
    """
    for key, value in dic1.items():
        key1 = key
        value1 = value
    for key, value in dic2.items():
        key2 = key
        value2 = value
    if np.array_equal(value1, value2) == False or key1 != key2:
        log.info(
            f"Graphs differ at key {key1} with value {value1} and {key2} with value {value2}")
        return False
    else:
        return True


def graphs_equal(
    g: nx.Graph,
    h: nx.Graph,
    node_match_func: Callable = dictionaries_equal,
    edge_match_func: Callable = dictionaries_equal
) -> bool:

    return nx.is_isomorphic(g, h, node_match_func, edge_match_func)


def assert_graphs_equal(
    g: nx.Graph,
    h: nx.Graph,
    node_match_func: Callable = dictionaries_equal,
    edge_match_func: Callable = dictionaries_equal,
):
    """Asserts whether two graphs are equal (structural isomorphism and edge and node features match)

    :param g: The first graph.
    :param h: The second graph.
    :param node_match_func: Matching function for node features. Takes two node dictionaries and returns True if they are equal.
    :param edge_match_func: Matching function for edge features. A function that takes two edge dictionaries and returns True if they are equal.
    :raises AssertionError: If the graphs are not equal.
    """
    assert graphs_equal(g, h, node_match_func,
                        edge_match_func), "Graphs are not isomorphic."
