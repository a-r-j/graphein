"""Testing utilities for the Graphein library."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import Any, Callable, Dict

import networkx as nx
import numpy as np
from loguru import logger as log

__all__ = [
    "compare_exact",
    "compare_approximate",
    "graphs_isomorphic",
    "nodes_equal",
    "edges_equal",
    "edge_data_equal",
]


def compare_exact(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    """Return whether two dicts of arrays are exactly equal.

    :param first: The first dictionary.
    :type first: Dict[str, Any]
    :param second: The second dictionary.
    :type second: Dict[str, Any]
    :return: ``True`` if the dictionaries are exactly equal, ``False``
        otherwise.
    :rtype: bool
    """
    if first.keys() != second.keys():
        return False
    return all(np.array_equal(first[key], second[key]) for key in first)


def compare_approximate(first, second):
    """Return whether two dicts of arrays are approximates equal.

    :param first: The first dictionary.
    :type first: Dict[str, Any]
    :param second: The second dictionary.
    :type second: Dict[str, Any]
    :return: ``True`` if the dictionaries are approx equal, ``False`` otherwise.
    :rtype: bool
    """
    if first.keys() != second.keys():
        return False
    return all(np.allclose(first[key], second[key]) for key in first)


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


def nodes_equal(g: nx.Graph, h: nx.Graph) -> bool:
    """Checks whether two graphs have the same nodes.

    :param g: The first graph.
    :type g: :class:`networkx.Graph`
    :param h: The second graph.
    :type h: :class:`networkx.Graph`
    :raises AssertionError: If the graphs do not contain the same nodes
    """
    for n in g.nodes():
        if n not in h.nodes():
            log.info(f"Node {n} (graph g) not in graph h")
            return False
    for n in h.nodes():
        if n not in g.nodes():
            log.info(f"Node {n} (graph h) not in graph g")
            return False
    return True


def edges_equal(g: nx.Graph, h: nx.Graph) -> bool:
    """Checks whether two graphs have the same edges.

    :param g: The first graph.
    :type g: :class:`networkx.Graph`
    :param h: The second graph.
    :type h: :class:`networkx.Graph`
    :raises AssertionError: If the graphs do not contain the same nodes
    """
    for u, v in g.edges():
        if (u, v) not in h.edges():
            log.info(f"Edge {u}-{v} (graph g) not in graph h")
            return False
    for u, v in h.edges():
        if (u, v) not in g.edges():
            log.info(f"Edge {u}-{v} (graph h) not in graph g")
            return False
    return True


def edge_data_equal(
    g: nx.Graph, h: nx.Graph, comparison_func: Callable = compare_exact
) -> bool:
    """Checks whether two graphs have the same edge features.

    :param g: The first graph.
    :type g: :class:`networkx.Graph`
    :param h: The second graph.
    :type h: :class:`networkx.Graph`
    :param comparison_func: Matching function for edge features.
        Takes two edge feature dictionaries and returns ``True`` if they are
        equal. Defaults to :func:`compare_exact`
    :type node_match_func: Callable
    :returns: ``True`` if the graphs have the same node features, ``False``
        otherwise.
    :rtype: bool
    """
    if not edges_equal(g, h):
        log.info("Edge lists do not match")
        return False
    for u, v in g.edges():
        if not compare_exact(g.edges[u, v], h.edges[u, v]):
            log.info(f"Edge {u}-{v} (graph g) features do not match graph h")
            return False
    for u, v in h.edges():
        if not compare_exact(g.edges[u, v], h.edges[u, v]):
            log.info(f"Edge {u}-{v} (graph h) features do not match graph g")
            return False
    return True


def node_data_equal(
    g: nx.Graph, h: nx.Graph, comparison_func: Callable = compare_exact
) -> bool:
    """Checks whether two graphs have the same node features.

    :param g: The first graph.
    :type g: :class:`networkx.Graph`
    :param h: The second graph.
    :type h: :class:`networkx.Graph`
    :param comparison_func: Matching function for node features.
        Takes two node dictionaries and returns True if they are equal.
        Defaults to :func:`compare_exact`
    :type comparison_func: Callable
    :returns: ``True`` if the graphs have the same node features, ``False``
        otherwise.
    :rtype: bool
    """
    if not nodes_equal(g, h):
        return False
    for n in g.nodes():
        if not compare_exact(g.nodes[n], h.nodes[n]):
            log.info(f"Node {n} (graph g) features do not match graph h")
            return False
    for n in h.nodes():
        if not compare_exact(g.nodes[n], h.nodes[n]):
            log.info(f"Node {n} (graph h) features do not match graph g")
            return False
    return True


def graphs_equal(
    g: nx.Graph,
    h: nx.Graph,
    node_match_func: Callable = compare_exact,
    edge_match_func: Callable = compare_exact,
) -> bool:
    """Asserts whether two graphs are equal
    (structural isomorphism and edge and node features match).

    :param g: The first graph.
    :type g: :class:`networkx.Graph`
    :param h: The second graph.
    :type h: :class:`networkx.Graph`
    :param node_match_func: Matching function for node features.
        Takes two node dictionaries and returns True if they are equal.
        Defaults to :func:`compare_exact`
    :type node_match_func: Callable
    :param edge_match_func: Matching function for edge features.
        A function that takes two edge dictionaries and returns ``True``
        if they are equal. Defaults to :func:`compare_exact`
    :type edge_match_func: Callable
    :return: ``True`` if the graphs are equal, ``False`` otherwise.
    :rtype: bool
    """
    return nx.is_isomorphic(g, h, node_match_func, edge_match_func)
