"""Functions for computing biochemical edges of graphs."""

# Graphein
# Author: Eric Ma, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import itertools
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger as log
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph


def compute_distmat(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise euclidean distances between every atom.

    Design choice: passed in a DataFrame to enable easier testing on
    dummy data.

    :param coords: pd.Dataframe containing molecule structure. Must contain
        columns ``["x_coord", "y_coord", "z_coord"]``.
    :type coords: pd.DataFrame
    :return: np.ndarray of euclidean distance matrix.
    :rtype: np.ndarray
    """
    return pairwise_distances(coords, metric="euclidean")


def get_interacting_atoms(angstroms: float, distmat: np.ndarray) -> np.ndarray:
    """Find the atoms that are within a particular radius of one another.

    :param angstroms: Radius in angstroms.
    :type angstroms: float
    :param distmat: Distance matrix.
    :type distmat: np.ndarray
    :returns: Array of interacting atoms
    :rtype: np.ndarray
    """
    return np.where(distmat <= angstroms)


def add_distance_threshold(G: nx.Graph, threshold: float = 5.0):
    """
    Adds edges to any nodes within a given distance of each other.

    :param G: molecule structure graph to add distance edges to
    :type G: nx.Graph
    :param threshold: Distance in angstroms, below which two nodes are
        connected.
    :type threshold: float
    :return: Graph with distance-based edges added
    """

    dist_mat = compute_distmat(G.graph["coords"])
    interacting_nodes = get_interacting_atoms(threshold, distmat=dist_mat)
    outgoing = [list(G.nodes())[i] for i in interacting_nodes[0]]
    incoming = [list(G.nodes())[i] for i in interacting_nodes[1]]
    interacting_nodes = list(zip(outgoing, incoming))

    log.info(
        f"Found: {len(interacting_nodes)} distance edges for radius {threshold}"
    )
    for n1, n2 in interacting_nodes:
        if G.has_edge(n1, n2):
            G.edges[n1, n2]["kind"].add("distance_threshold")
        else:
            G.add_edge(n1, n2, kind={"distance_threshold"})


def add_fully_connected_edges(
    G: nx.Graph,
):
    """
    Adds fully connected edges to nodes.

    :param G: Molecule structure graph to add distance edges to.
    :type G: nx.Graph
    """
    length = len(G.graph["coords"])

    for n1, n2 in itertools.product(G.nodes(), G.nodes()):
        if G.has_edge(n1, n2):
            G.edges[n1, n2]["kind"].add("fully_connected")
        else:
            G.add_edge(n1, n2, kind={"fully_connected"})


def add_k_nn_edges(
    G: nx.Graph,
    k: int = 1,
    mode: str = "connectivity",
    metric: str = "minkowski",
    p: int = 2,
    include_self: Union[bool, str] = False,
):
    """
    Adds edges to nodes based on K nearest neighbours.

    :param G: Molecule structure graph to add distance edges to.
    :type G: nx.Graph
    :param k: Number of neighbors for each sample.
    :type k: int
    :param mode: Type of returned matrix: ``"connectivity"`` will return the
        connectivity matrix with ones and zeros, and ``"distance"`` will return
        the distances between neighbors according to the given metric.
    :type mode: str
    :param metric: The distance metric used to calculate the k-Neighbors for
        each sample point. The DistanceMetric class gives a list of available
        metrics. The default distance is ``"euclidean"`` (``"minkowski"``
        metric with the ``p`` param equal to ``2``).
    :type metric: str
    :param p: Power parameter for the Minkowski metric. When ``p = 1``, this is
        equivalent to using ``manhattan_distance`` (l1), and
        ``euclidean_distance`` (l2) for ``p = 2``. For arbitrary ``p``,
        ``minkowski_distance`` (l_p) is used. Default is ``2`` (euclidean).
    :type p: int
    :param include_self: Whether or not to mark each sample as the first nearest
        neighbor to itself. If ``"auto"``, then ``True`` is used for
        ``mode="connectivity"`` and ``False`` for ``mode="distance"``. Default
        is ``False``.
    :type include_self: Union[bool, str]
    :return: Graph with knn-based edges added.
    :rtype: nx.Graph
    """
    dist_mat = compute_distmat(G.graph["coords"])

    nn = kneighbors_graph(
        X=dist_mat,
        n_neighbors=k,
        mode=mode,
        metric=metric,
        p=p,
        include_self=include_self,
    )

    # Create iterable of node indices
    outgoing = np.repeat(np.array(range(len(G.graph["coords"]))), k)
    outgoing = [list(G.nodes())[i] for i in outgoing]
    incoming = [list(G.nodes())[i] for i in nn.indices]
    interacting_nodes = list(zip(outgoing, incoming))
    log.info(f"Found: {len(interacting_nodes)} KNN edges")
    for n1, n2 in interacting_nodes:
        if G.has_edge(n1, n2):
            G.edges[n1, n2]["kind"].add(f"k_nn_{k}")
        else:
            G.add_edge(n1, n2, kind={f"k_nn_{k}"})
