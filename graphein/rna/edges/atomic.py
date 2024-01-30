"""Functions for computing atomic structure of proteins."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import Any, Dict

import networkx as nx
import numpy as np
from loguru import logger as log
from scipy.spatial import distance_matrix

import graphein.protein.edges.atomic as protein
from graphein.rna.features.atom import add_atomic_radii

# TODO dealing with metals
# TODO There are other check and balances that can be implemented from here:
# https://www.daylight.com/meetings/mug01/Sayle/m4xbondage.html


def add_atomic_edges(G: nx.Graph, tolerance: float = 0.56) -> nx.Graph:
    """
    Computes covalent edges based on atomic distances. Covalent radii are
    assigned to each atom based on its bond state. The distance matrix is then
    thresholded to entries less than this distance plus some tolerance to
    create an adjacency matrix. This adjacency matrix is then parsed into an
    edge list and covalent edges added.

    Bond states and covalent radii are retrieved from:

        Structures of the Molecular Components in DNA and RNA with Bond Lengths
        Interpreted as Sums of Atomic Covalent Radii
        *Raji Heyrovska*

    :param G: Atomic graph (nodes correspond to atoms) to populate with atomic
        bonds as edges
    :type G: nx.Graph
    :param tolerance: Tolerance for atomic distance. Default is ``0.56``
        Angstroms. Commonly used values are: ``0.4, 0.45, 0.56``
    :type tolerance: float
    :return: Atomic graph with edges between bonded atoms added
    :rtype: nx.Graph
    """
    dist_mat = distance_matrix(G.graph["coords"], G.graph["coords"])

    # Add atomic radii if not present
    for n, d in G.nodes(data=True):
        if "atomic_radius" not in d.keys():
            add_atomic_radii(n, d)

    radii = np.array(list(nx.get_node_attributes(G, "atomic_radius").values()))

    # Create a covalent 'distance' matrix by adding the radius arrays with
    # its transpose
    covalent_radius_distance_matrix = np.add(
        radii.reshape(-1, 1),
        radii.reshape(1, -1),
    )
    # Add the tolerance
    covalent_radius_distance_matrix = (
        covalent_radius_distance_matrix + tolerance
    )

    # Threshold Distance Matrix to entries where the euclidean distance is
    # less than the covalent radius plus tolerance and larger than 0.4
    dist_mat[dist_mat < 0.4] = np.nan
    dist_mat[dist_mat > covalent_radius_distance_matrix] = np.nan
    # Store atomic adjacency matrix in graph
    G.graph["atomic_adj_mat"] = np.nan_to_num(dist_mat)

    # Get node IDs from non NaN entries in the thresholded distance matrix
    # and add the edge to the graph
    inds = zip(*np.where(~np.isnan(dist_mat)))
    for i in inds:
        length = dist_mat[i[0]][i[1]]
        node_1 = G.graph["pdb_df"]["node_id"][i[0]]
        node_2 = G.graph["pdb_df"]["node_id"][i[1]]
        chain_1 = G.graph["pdb_df"]["chain_id"][i[0]]
        chain_2 = G.graph["pdb_df"]["chain_id"][i[1]]

        # Check nodes are in graph
        if not (G.has_node(node_1) and G.has_node(node_2)):
            continue

        # Check atoms are in the same chain
        if not (chain_1 and chain_2):
            continue

        if G.has_edge(node_1, node_2):
            G.edges[node_1, node_2]["kind"].add("covalent")
            G.edges[node_1, node_2]["bond_length"] = length
        else:
            G.add_edge(node_1, node_2, kind={"covalent"}, bond_length=length)

    # TODO checking degree against MAX_NEIGHBOURS

    return G


def add_ring_status(G: nx.Graph) -> nx.Graph:
    """
    Identifies rings in the atomic RNA graph. Assigns the edge attribute
    ``"RING"`` to edges in the ring. We do not distinguish between aromatic and
    non-aromatic rings. Functions by identifying all cycles in the graph.

    :param G: Atom-level RNA structure graph to add ring edge types to
    :type G: nx.Graph
    :return: Atom-level RNA structure graph with added ``"RING"`` edge attribute
    :rtype: nx.Graph
    """
    return protein.add_ring_status(G)


def add_bond_order(G: nx.Graph) -> nx.Graph:
    """
    Assign bond orders to the covalent bond edges between atoms on the basis of
    bond length. Values are taken from:

        Automatic Assignment of Chemical Connectivity to Organic Molecules in
        the Cambridge Structural Database.
        *Jon C. Baber and Edward E. Hodgkin*

    :param G: Atomic-level RNA graph with covalent edges.
    :type G: nx.Graph
    :return: Atomic-level RNA graph with covalent edges annotated with putative
        bond order.
    :rtype: nx.Graph
    """
    return protein.add_bond_order(G)
