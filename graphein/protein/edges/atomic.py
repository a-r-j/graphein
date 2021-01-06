"""Functions for computing atomic structre of proteins"""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import networkx as nx
import numpy as np
import pandas as pd

from graphein.protein.edges.distance import compute_distmat
from graphein.protein.resi_atoms import (
    COVALENT_RADII,
    DEFAULT_BOND_STATE,
    RESIDUE_ATOM_BOND_STATE,
)

# TODO BOND ORDER - currently we make no distinction between single, double and aromatic bonds
# Todo dealing with metals
# Todo There are other check and balances that can be implemented from here: https://www.daylight.com/meetings/mug01/Sayle/m4xbondage.html
# Todo detect aromaticity


def assign_bond_states_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a PandasPDB atom dataframe and assigns bond states to each atom based on:
    Atomic Structures of all the Twenty Essential Amino Acids and a Tripeptide, with Bond Lengths as Sums of Atomic Covalent Radii
    Heyrovska, 2008
    :param df: Pandas PDB dataframe
    :return: Dataframe with added atom_bond_state column
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

    # Map non-standard states to the dataframe based on the residue and atom name
    df = df.join(ss, on=["residue_name", "atom_name"])

    # Fill the NaNs with the standard states
    df = df.fillna(value={"atom_bond_state": naive_bond_states})

    return df


def assign_covalent_radii_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns covalent radius to each atom based on its bond state. Using Values from :
    Atomic Structures of all the Twenty Essential Amino Acids and a Tripeptide, with Bond Lengths as Sums of Atomic Covalent Radii
    Heyrovska, 2008
    :param df: Pandas PDB dataframe with a bond_states_column
    :return: Pandas PDB dataframe with added covalent_radius column
    """
    # Assign covalent radius to each atom
    df["covalent_radius"] = df["atom_bond_state"].map(COVALENT_RADII)

    return df


def add_atomic_edges(G: nx.Graph) -> nx.Graph:
    """
    This computes covalent edges based on atomic distances. Covalent radii add_atomic_edges assigned to each atom based on its bond assign_bond_states_to_dataframe
    The distance matrix is then thresholded to entries less than this distance plus some tolerance to create and adjacency matrix.
    This adjacency matrix is then parsed into an edge list and covalent edges added
    :param G:
    :return:
    """
    TOLERANCE = 0.56  # 0.4 0.45, 0.56 This is the distance tolerance
    dist_mat = compute_distmat(G.graph["pdb_df"])

    # We assign bond states to the dataframe, and then map these to covalent radii
    G.graph["pdb_df"] = assign_bond_states_to_dataframe(G.graph["pdb_df"])
    G.graph["pdb_df"] = assign_covalent_radii_to_dataframe(G.graph["pdb_df"])

    # Create a covalent 'distance' matrix by adding the radius arrays with its transpose
    covalent_radius_distance_matrix = np.add(
        np.array(G.graph["pdb_df"]["covalent_radius"]).reshape(-1, 1),
        np.array(G.graph["pdb_df"]["covalent_radius"]).reshape(1, -1),
    )

    # Add the tolerance
    covalent_radius_distance_matrix = (
        covalent_radius_distance_matrix + TOLERANCE
    )

    # Threshold Distance Matrix to entries where the eucl distance is less than the covalent radius plus tolerance and larger than 0.4
    dist_mat = dist_mat[dist_mat > 0.4]
    t_distmat = dist_mat[dist_mat < covalent_radius_distance_matrix]

    # Store atomic adjacency matrix in graph
    G.graph["atomic_adj_mat"] = np.nan_to_num(t_distmat)

    # Get node IDs from non NaN entries in the thresholded distance matrix and add the edge to the graph
    inds = zip(*np.where(~np.isnan(t_distmat)))
    for i in inds:
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
        else:
            G.add_edge(node_1, node_2, kind={"covalent"})

    # Todo checking degree against MAX_NEIGHBOURS

    return G
