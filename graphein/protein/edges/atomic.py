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
import pandas as pd
from loguru import logger as log

from graphein.protein.edges.distance import compute_distmat
from graphein.protein.resi_atoms import (
    BOND_LENGTHS,
    BOND_ORDERS,
    COVALENT_RADII,
    DEFAULT_BOND_STATE,
    RESIDUE_ATOM_BOND_STATE,
)

# TODO dealing with metals
# TODO There are other check and balances that can be implemented from here:
# https://www.daylight.com/meetings/mug01/Sayle/m4xbondage.html


def assign_bond_states_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a ``PandasPDB`` atom DataFrame and assigns bond states to each atom
    based on:

        *Atomic Structures of all the Twenty Essential Amino Acids and a*
        *Tripeptide, with Bond Lengths as Sums of Atomic Covalent Radii*
        Heyrovska, 2008

    First, maps atoms to their standard bond states
    (:const:`~graphein.protein.resi_atoms.DEFAULT_BOND_STATE`). Second, maps
    non-standard bonds states
    (:const:`~graphein.protein.resi_atoms.RESIDUE_ATOM_BOND_STATE`). Fills
    ``NaNs`` with standard bond states.

    :param df: Pandas PDB DataFrame.
    :type df: pd.DataFrame
    :return: DataFrame with added ``atom_bond_state`` column.
    :rtype: pd.DataFrame
    """

    # Map atoms to their standard bond states
    naive_bond_states = pd.Series(df["atom_name"].map(DEFAULT_BOND_STATE))

    # Create series of bond states for the non-standard states
    ss = (
        pd.DataFrame(RESIDUE_ATOM_BOND_STATE)
        .unstack()
        .rename_axis(("residue_name", "atom_name"))
        .rename("atom_bond_state")
    )

    # Map non-standard states to the DataFrame based on the residue and atom
    # name
    df = df.join(ss, on=["residue_name", "atom_name"])

    # Fill the NaNs with the standard states
    df = df.fillna(value={"atom_bond_state": naive_bond_states})

    return df


def assign_covalent_radii_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns covalent radius
    (:const:`~graphein.protein.resi_atoms.COVALENT_RADII`) to each atom based
    on its bond state. Adds a ``covalent_radius`` column. Using values from:

        *Atomic Structures of all the Twenty Essential Amino Acids and a*
        *Tripeptide, with Bond Lengths as Sums of Atomic Covalent Radii*
        Heyrovska, 2008

    :param df: Pandas PDB DataFrame with a ``bond_states_column``.
    :type df: pd.DataFrame
    :return: Pandas PDB DataFrame with added ``covalent_radius`` column.
    :rtype: pd.DataFrame
    """
    # Assign covalent radius to each atom
    df["covalent_radius"] = df["atom_bond_state"].map(COVALENT_RADII)

    return df


def add_atomic_edges(G: nx.Graph, tolerance: float = 0.56) -> nx.Graph:
    """
    Computes covalent edges based on atomic distances. Covalent radii are
    assigned to each atom based on its bond assign_bond_states_to_dataframe.
    The distance matrix is then thresholded to entries less than this distance
    plus some tolerance to create an adjacency matrix. This adjacency matrix is
    then parsed into an edge list and covalent edges added

    :param G: Atomic graph (nodes correspond to atoms) to populate with atomic
        bonds as edges
    :type G: nx.Graph
    :param tolerance: Tolerance for atomic distance. Default is ``0.56``
        Angstroms. Commonly used values are: ``0.4, 0.45, 0.56``
    :type tolerance: float
    :return: Atomic graph with edges between bonded atoms added
    :rtype: nx.Graph
    """
    dist_mat = compute_distmat(G.graph["pdb_df"])

    # We assign bond states to the dataframe, and then map these to covalent
    # radii
    G.graph["pdb_df"] = assign_bond_states_to_dataframe(G.graph["pdb_df"])
    G.graph["pdb_df"] = assign_covalent_radii_to_dataframe(G.graph["pdb_df"])

    # Create a covalent 'distance' matrix by adding the radius arrays with its
    # transpose
    covalent_radius_distance_matrix = np.add(
        np.array(G.graph["pdb_df"]["covalent_radius"]).reshape(-1, 1),
        np.array(G.graph["pdb_df"]["covalent_radius"]).reshape(1, -1),
    )

    # Add the tolerance
    covalent_radius_distance_matrix = (
        covalent_radius_distance_matrix + tolerance
    )

    # Threshold Distance Matrix to entries where the eucl distance is less than
    # the covalent radius plus tolerance and larger than 0.4
    dist_mat = dist_mat[dist_mat > 0.4]
    t_distmat = dist_mat[dist_mat < covalent_radius_distance_matrix]

    # Store atomic adjacency matrix in graph
    G.graph["atomic_adj_mat"] = np.nan_to_num(t_distmat)

    # Get node IDs from non NaN entries in the thresholded distance matrix and
    # add the edge to the graph
    inds = zip(*np.where(~np.isnan(t_distmat)))
    for i in inds:
        length = t_distmat[i[0]][i[1]]
        node_1 = G.graph["pdb_df"]["node_id"][i[0]]
        node_2 = G.graph["pdb_df"]["node_id"][i[1]]
        chain_1 = G.graph["pdb_df"]["chain_id"][i[0]]
        chain_2 = G.graph["pdb_df"]["chain_id"][i[1]]

        # Check nodes are in graph
        if not (G.has_node(node_1) and G.has_node(node_2)):
            continue

        # Check atoms are in the same chain
        if chain_1 != chain_2:
            continue

        if G.has_edge(node_1, node_2):
            G.edges[node_1, node_2]["kind"].add("covalent")
            G.edges[node_1, node_2]["bond_length"] = length
        else:
            G.add_edge(node_1, node_2, kind={"covalent"}, bond_length=length)

    # Todo checking degree against MAX_NEIGHBOURS

    return G


def add_ring_status(G: nx.Graph) -> nx.Graph:
    """
    Identifies rings in the atomic graph. Assigns the edge attribute ``"RING"``
    to edges in the ring. We do not distinguish between aromatic and
    non-aromatic rings. Functions by identifying all cycles in the graph.

    :param G: Atom-level protein structure graph to add ring edge types to.
    :type G: nx.Graph
    :return: Atom-level protein structure graph with added ``"RING"`` edge
        attribute.
    :rtype: nx.Graph
    """
    cycles = nx.cycle_basis(
        G
    )  # Produces a list of lists containing nodes in each cycle
    # Iterate over cycles, check for an edge between the nodes
    # if there is one, add a "RING" attribute
    for cycle in cycles:
        [
            G.edges[x, y]["kind"].add("RING")
            for i, x in enumerate(cycle)
            for j, y in enumerate(cycle)
            if G.has_edge(x, y)
            if i != j
        ]

    return G


def add_bond_order(G: nx.Graph) -> nx.Graph:
    """
    Assign bond orders to the covalent bond edges between atoms on the basis of
    bond length. Values are taken from:

        *Automatic Assignment of Chemical Connectivity to Organic Molecules in*
        *the Cambridge Structural Database.*
        Jon C. Baber and Edward E. Hodgkin*

    :param G: Atomic-level protein graph with covalent edges.
    :type G: nx.Graph
    :return: Atomic-level protein graph with covalent edges annotated with
        putative bond order.
    :rtype: mx.Graph
    """
    for u, v, a in G.edges(data=True):
        atom_a = G.nodes[u]["element_symbol"]
        atom_b = G.nodes[v]["element_symbol"]

        # Assign bonds with hydrogens to 1
        if atom_a == "H" or atom_b == "H":
            G.edges[u, v]["kind"].add("SINGLE")
        # If not, we need to identify the bond type from the bond length
        else:
            query = f"{atom_a}-{atom_b}"
            # We need this try block as the dictionary keys may be X-Y, whereas
            # the query we construct may be Y-X
            try:
                identify_bond_type_from_mapping(G, u, v, a, query)
            except KeyError:
                query = f"{atom_b}-{atom_a}"
                try:
                    identify_bond_type_from_mapping(G, u, v, a, query)
                except KeyError:
                    log.debug(
                        f"Could not identify bond type for {query}. Adding a \
                            single bond."
                    )
                    G.edges[u, v]["kind"].add("SINGLE")

    return G


def identify_bond_type_from_mapping(
    G: nx.Graph, u: str, v: str, a: Dict[str, Any], query: str
):
    """
    Compares the bond length between two atoms in the graph, and the relevant
    experimental value by performing a lookup against the watershed values in:

        *Automatic Assignment of Chemical Connectivity to Organic Molecules in*
        *the Cambridge Structural Database.*
        Jon C. Baber and Edward E. Hodgkin*

    Bond orders are assigned in the order ``triple`` < ``double`` < ``single``
    (e.g. if a bond is shorter than the triple bond watershed (``w_dt``) then
    it is assigned as a triple bond. Similarly, if a bond is longer than this
    but shorter than the double bond watershed (``w_sd``), it is assigned double
    bond status.

    :param G: ``nx.Graph`` of atom-protein structure with atomic edges added.
    :type G: nx.Graph
    :param u: Node 1 in edge.
    :type u: str
    :param v: Node 2 in edge.
    :type v: str
    :param a: edge data
    :type a: Dict[str, Any]
    :param query: ``"ELEMENTX-ELEMENTY"`` to perform lookup with
        (E.g. ``"C-O"``,``"N-N"``)
    :type query: str
    :return: Graph with atomic edge bond order assigned
    :rtype: nx.Graph
    """
    # Perform lookup of allowable bond orders for the given atom pair
    allowable_order = BOND_ORDERS[query]
    # If max double, compare the length to the double watershed distance, w_sd,
    # else assign single
    if len(allowable_order) == 2:
        if a["bond_length"] < BOND_LENGTHS[query]["w_sd"]:
            G.edges[u, v]["kind"].add("DOUBLE")
        else:
            G.edges[u, v]["kind"].add("SINGLE")
    else:
        # If max triple, compare the length to the triple watershed distance,
        # w_dt, then double, else assign single
        if a["bond_length"] < BOND_LENGTHS[query]["w_dt"]:
            G.edges[u, v]["kind"].add("TRIPLE")
        elif a["bond_length"] < BOND_LENGTHS[query]["w_sd"]:
            G.edges[u, v]["kind"].add("DOUBLE")
        else:
            G.edges[u, v]["kind"].add("SINGLE")
    return G


# The codeblock below was used in an initial pass at solving the bond order
# assignment problem based on hybridisation state.
# We instead use a simpler method of construction based on bond lengths,
# but I am loathe to remove this code as it may prove useful later
"""
def cosinus(x0, x1, x2):
    e0 = x0 - x1
    e1 = x2 - x1
    e0 = e0 / np.linalg.norm(e0)
    e1 = e1 / np.linalg.norm(e1)
    cosinus = np.dot(e0, e1)
    angle = np.arccos(cosinus)

    return 180 - np.degrees(angle)


def dihedral(x0, x1, x2, x3):
    b0 = -1.0 * (x1 - x0)
    b1 = x2 - x1
    b2 = x3 - x2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    y = np.dot(b0xb1_x_b1xb2, b1) * (1.0 / np.linalg.norm(b1))
    x = np.dot(b0xb1, b1xb2)

    grad = 180 - np.degrees(np.arctan2(y, x))
    return grad


def assign_bond_orders(G: nx.Graph) -> nx.Graph:

    bond_angles: Dict[str, float] = {}
    for n, d in G.nodes(data=True):
        neighbours = list(G.neighbors(n))

        if len(neighbours) == 1:
            G.edges[n, neighbours[0]]["kind"].add("SINGLE")
            bond_angles[n] = 0.0
        elif len(neighbours) == 2:
            cos_angle = cosinus(
                G.nodes[n]["coords"],
                G.nodes[neighbours[0]]["coords"],
                G.nodes[neighbours[1]]["coords"],
            )
            bond_angles[n] = cos_angle
        elif len(neighbours) == 3:
            dihed = dihedral(
                G.nodes[n]["coords"],
                G.nodes[neighbours[0]]["coords"],
                G.nodes[neighbours[1]]["coords"],
                G.nodes[neighbours[2]]["coords"],
            )
            bond_angles[n] = dihed

    print(bond_angles)

    # Assign Bond angles to dataframe
    G.graph["pdb_df"]["bond_angles"] = G.graph["pdb_df"]["node_id"].map(
        bond_angles
    )
    print(G.graph["pdb_df"].to_string())

    # Assign Hybridisation state from Bond Angles
    hybridisation_state = {
        n: "sp"
        if d > 155
        else "sp2"
        if d > 115
        else "sp3"
        if d <= 115
        else "UNK"
        for n, d in bond_angles.items()
    }
    G.graph["pdb_df"]["bond_angles"] = G.graph["pdb_df"]["node_id"].map(
        hybridisation_state
    )

    return G
"""
