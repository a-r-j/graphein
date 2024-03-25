"""Functions for graph-level featurization of the sequence of a protein. This submodule is focussed on physicochemical
proporties of the sequence."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from functools import partial
from typing import List, Optional

import networkx as nx
import numpy as np
from Bio import SeqUtils
from multipledispatch import dispatch

from graphein.protein.features.sequence.utils import (
    aggregate_feature_over_chains,
    compute_feature_over_chains,
)


@dispatch(str, str)
def molecular_weight(protein: str, seq_type: str = "protein"):
    func = partial(SeqUtils.molecular_weight, seq_type=seq_type)

    return func(protein)


@dispatch(nx.Graph, seq_type=str)
def molecular_weight(protein, seq_type: str = "protein"):
    func = partial(SeqUtils.molecular_weight, seq_type=seq_type)

    return compute_feature_over_chains(
        protein, func, feature_name="molecular_weight"
    )


def get_sinusoid_encoding_table(
    n_position: int, d_hid: int, padding_idx: Optional[int] = None
) -> np.ndarray:
    """
    Numpy-based implementation of sinusoid position encoding used in Transformer models.

    Based on implementation by @foowaa (https://gist.github.com/foowaa/5b20aebd1dff19ee024b6c72e14347bb)

    :param n_position: Number of positions to encode (length of graph) (``N``).
    :param d_hid: dimension of embedding vector (``M``).
    :param padding_idx: Set 0 dimension. Defaults to ``None``.
    :return: Sinusoid table. (``NxM``).
    :rtype: np.ndarray
    """

    def calc_angle(position: int, hid_idx: int) -> float:
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_pos_angle_vec(position: int) -> List[float]:
        return [calc_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_pos_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return sinusoid_table


def add_positional_encoding(
    G: nx.Graph,
    d_hid: int,
    padding_idx: Optional[int] = None,
    add_to_nodes: bool = True,
) -> nx.Graph:
    """Adds transformer positional encoding (based on sequence) to graph.

    Accessed via: ``g.graph["positional_encoding"]`` if added to the graph
    (``add_to_nodes=False``) or ``d["positional_encoding"] for _, d in g.nodes(data=True)``
    if added to the nodes (``add_to_nodes=True``).

    Nodes are numbered as they occur in list(g.nodes()). By default,
    this corresponds to stacked N->C sequences (ie. in multichain graphs:
    ``SeqA N->C, SeqB N->C, SeqC N->C``).

    :param G: Graph to add positional encoding to.
    :type G: nx.Graph
    :param d_hid: Dimensionality of positional encoding.
    :type d_hid: int
    :param padding_idx: Set 0 dimension. Defaults to ``None``.
    :type padding_idx: Optional[int], optional
    :param add_to_nodes: Whether to add the positional encoding to the graph as a graph-level feature
        (``Nxd_hid`` matrix, where ``N`` is the length (number of nodes) of the graph), or as sliced arrays
        (size ``d_hid``) added to the nodes, defaults to ``True``.
    :type add_to_nodes: bool
    :return: Graph with positional encoding added as either a graph feature or node features.
    :rtype: nx.Graph
    """
    n = len(G)
    sinusoid_table = get_sinusoid_encoding_table(n, d_hid, padding_idx)
    if not add_to_nodes:
        G.graph["positional_encoding"] = sinusoid_table
    else:
        for i, (n, d) in enumerate(G.nodes(data=True)):
            d["positional_encoding"] = sinusoid_table[i, :]

    return G
