"""Functions for computing biochemical edges of graphs."""
# Graphein
# Author: Eric Ma, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph


log = logging.getLogger(__name__)


def compute_distmat(coords: np.array) -> np.array:
    """
    Compute pairwise euclidean distances between every atom.

    Design choice: passed in a DataFrame to enable easier testing on
    dummy data.

    :param coords: pd.Dataframe containing molecule structure. Must contain columns ["x_coord", "y_coord", "z_coord"]
    :type coords: pd.DataFrame
    :return: pd.Dataframe of euclidean distance matrix
    :rtype: pd.DataFrame
    """
    eucl_dists = pairwise_distances(
        coords, metric="euclidean"
    )

    return eucl_dists

def get_interacting_atoms(angstroms: float, distmat: pd.DataFrame):
    """Find the atoms that are within a particular radius of one another."""
    return np.where(distmat <= angstroms)

def add_distance_threshold(
    G: nx.Graph, long_interaction_threshold: int = 5.0, threshold: float = 5.0
):
    """
    Adds edges to any nodes within a given distance of each other. Long interaction threshold is used
    to specify minimum separation in sequence to add an edge between networkx nodes within the distance threshold

    :param G: molecule structure graph to add distance edges to
    :type G: nx.Graph
    :param long_interaction_threshold: minimum distance in sequence for two nodes to be connected
    :type long_interaction_threshold: int
    :param threshold: Distance in angstroms, below which two nodes are connected
    :type threshold: float
    :return: Graph with distance-based edges added
    """
    
    dist_mat = compute_distmat(G.graph["coords"])
    interacting_nodes = get_interacting_atoms(threshold, distmat=dist_mat)
    interacting_nodes = list(zip(interacting_nodes[0], interacting_nodes[1]))

    log.info(f"Found: {len(interacting_nodes)} distance edges")
    count = 0
    for a1, a2 in interacting_nodes:
        n1 = a1
        n2 = a2
        count += 1
        if G.has_edge(n1, n2):
            G.edges[n1, n2]["kind"].add("distance_threshold")
        else:
            G.add_edge(n1, n2, kind={"distance_threshold"})
    log.info(
        f"Added {count} distance edges. ({len(list(interacting_nodes)) - count} removed by LIN)"
    )

def add_fully_connected_edges(
    G: nx.Graph,
):
    """
    Adds fully connected edges to nodes 

    :param G: molecule structure graph to add distance edges to
    :type G: nx.Graph
    :return: Graph with knn-based edges added
    :rtype: nx.Graph
    """
    length = len(G.graph["coords"])
    
    for n1 in range(length):
        for n2 in range(length):
            if G.has_edge(n1, n2):
                G.edges[n1, n2]["kind"].add("fully_connected")
            else:
                G.add_edge(n1, n2, kind={"fully_connected"})

def add_k_nn_edges(
    G: nx.Graph,
    long_interaction_threshold: int = 5,
    k: int = 1,
    mode: str = "connectivity",
    metric: str = "minkowski",
    p: int = 2,
    include_self: Union[bool, str] = False,
):
    """
    Adds edges to nodes based on K nearest neighbours. Long interaction threshold is used
    to specify minimum separation in sequence to add an edge between networkx nodes within the distance threshold

    :param G: molecule structure graph to add distance edges to
    :type G: nx.Graph
    :param long_interaction_threshold: minimum distance in sequence for two nodes to be connected
    :type long_interaction_threshold: int
    :param k: Number of neighbors for each sample.
    :type k: int
    :param mode: Type of returned matrix: ``"connectivity"`` will return the connectivity matrix with ones and zeros,
        and ``"distance"`` will return the distances between neighbors according to the given metric.
    :type mode: str
    :param metric: The distance metric used to calculate the k-Neighbors for each sample point.
        The DistanceMetric class gives a list of available metrics.
        The default distance is ``"euclidean"`` (``"minkowski"`` metric with the ``p`` param equal to ``2``).
    :type metric: str
    :param p: Power parameter for the Minkowski metric. When ``p = 1``, this is equivalent to using ``manhattan_distance`` (l1),
        and ``euclidean_distance`` (l2) for ``p = 2``. For arbitrary ``p``, ``minkowski_distance`` (l_p) is used. Default is ``2`` (euclidean).
    :type p: int
    :param include_self: Whether or not to mark each sample as the first nearest neighbor to itself.
        If ``"auto"``, then ``True`` is used for ``mode="connectivity"`` and ``False`` for ``mode="distance"``. Default is ``False``.
    :type include_self: Union[bool, str]
    :return: Graph with knn-based edges added
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
    incoming = nn.indices
    interacting_nodes = list(zip(outgoing, incoming))
    log.info(f"Found: {len(interacting_nodes)} KNN edges")
    for a1, a2 in interacting_nodes:
        n1 = a1
        n2 = a2
        if G.has_edge(n1, n2):
            G.edges[n1, n2]["kind"].add("k_nn")
        else:
            G.add_edge(n1, n2, kind={"k_nn"})
            


