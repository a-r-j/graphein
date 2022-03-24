"""Provides geometry-based featurisation functions."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import logging

import networkx as nx
import numpy as np
import pandas as pd

from graphein.protein.utils import filter_dataframe


def add_sidechain_vector(
    g: nx.Graph, scale: bool = True, reverse: bool = False
):
    """Adds vector from node to average position of sidechain atoms.

    We compute the mean of the sidechain atoms for each node. For this we use the ``rgroup_df`` dataframe.
    If the graph does not contain the ``rgroup_df`` dataframe, we compute it from the ``raw_pdb_df``.
    If scale, we scale the vector to the unit vector. If reverse is True,
    we reverse the vector (``sidechain - node``). If reverse is false (default) we compute (``node - sidechain``).

    :param g: Graph to add vector to.
    :type g: nx.Graph
    :param scale: Scale vector to unit vector. Defaults to ``True``.
    :type scale: bool
    :param reverse: Reverse vector. Defaults to ``False``.
    :type reverse: bool
    """
    # Get or compute R-Group DF
    if "rgroup_df" not in g.graph.keys():
        g.graph["rgroup_df"] = compute_rgroup_dataframe(g.graph["raw_pdb_df"])

    sc_centroid = g.graph["rgroup_df"].groupby("node_id").mean()

    # Iterate over nodes and compute vector
    for n, d in g.nodes(data=True):
        if d["residue_name"] == "GLY":
            # If GLY, set vector to 0
            vec = np.array([0, 0, 0])
        else:
            if reverse:
                vec = d["coords"] - np.array(
                    sc_centroid.loc[n][["x_coord", "y_coord", "z_coord"]]
                )
            else:
                vec = (
                    np.array(
                        sc_centroid.loc[n][["x_coord", "y_coord", "z_coord"]]
                    )
                    - d["coords"]
                )

            if scale:
                vec = vec / np.linalg.norm(vec)

        d["sidechain_vector"] = vec


def add_beta_carbon_vector(
    g: nx.Graph, scale: bool = True, reverse: bool = False
):
    """Adds vector from node (typically alpha carbon) to position of beta carbon.

    Glycine does not have a beta carbon, so we set it to ``np.array([0, 0, 0])``.
    We extract the position of the beta carbon from the unprocessed atomic PDB dataframe.
    For this we use the ``raw_pdb_df`` dataframe.
    If scale, we scale the vector to the unit vector. If reverse is True,
    we reverse the vector (``C beta - node``). If reverse is false (default) we compute (``node - C beta``).

    :param g: Graph to add vector to.
    :type g: nx.Graph
    :param scale: Scale vector to unit vector. Defaults to ``True``.
    :type scale: bool
    :param reverse: Reverse vector. Defaults to ``False``.
    :type reverse: bool
    """

    c_beta_coords = filter_dataframe(
        g.graph["raw_pdb_df"], "atom_name", ["CB"], boolean=True
    )
    c_beta_coords.index = c_beta_coords["node_id"]

    # Iterate over nodes and compute vector
    for n, d in g.nodes(data=True):
        if d["residue_name"] == "GLY":
            vec = np.array([0, 0, 0])
        else:
            if reverse:
                vec = d["coords"] - np.array(
                    c_beta_coords.loc[n][["x_coord", "y_coord", "z_coord"]]
                )
            else:
                vec = (
                    np.array(
                        c_beta_coords.loc[n][["x_coord", "y_coord", "z_coord"]]
                    )
                    - d["coords"]
                )

            if scale:
                vec = vec / np.linalg.norm(vec)
        d["c_beta_vector"] = vec


def add_sequence_neighbour_vector(
    g: nx.Graph, scale: bool = True, reverse: bool = False, n_to_c: bool = True
):
    """Computes vector from node to adjacent node in sequence.
    Typically used with ``CA`` (alpha carbon) graphs.

    If ``n_to_c`` is ``True`` (default), we compute the vectors from the N terminus to the C terminus (canonical direction).
    If ``reverse`` is ``False`` (default), we compute ``Node_i - Node_{i+1}``.
    If ``reverse is ``True``, we compute ``Node_{i+1} - Node_i``.
    :param g: Graph to add vector to.
    :type g: nx.Graph
    :param scale: Scale vector to unit vector. Defaults to ``True``.
    :type scale: bool
    :param reverse: Reverse vector. Defaults to ``False``.
    :type reverse: bool
    :param n_to_c: Compute vector from N to C or C to N. Defaults to ``True``.
    :type n_to_c: bool
    """
    suffix = "n_to_c" if n_to_c else "c_to_n"
    # Iterate over every chain
    for chain_id in g.graph["chain_ids"]:

        # Find chain residues
        chain_residues = [
            (n, v) for n, v in g.nodes(data=True) if v["chain_id"] == chain_id
        ]

        if not n_to_c:
            chain_residues.reverse()

        # Iterate over every residue in chain
        for i, residue in enumerate(chain_residues):
            # Checks not at chain terminus - is this versatile enough?
            if i == len(chain_residues) - 1:
                residue[1][f"sequence_neighbour_vector_{suffix}"] = np.array(
                    [0, 0, 0]
                )
                continue
            # Asserts residues are on the same chain
            cond_1 = (
                residue[1]["chain_id"] == chain_residues[i + 1][1]["chain_id"]
            )
            # Asserts residue numbers are adjacent
            cond_2 = (
                abs(
                    residue[1]["residue_number"]
                    - chain_residues[i + 1][1]["residue_number"]
                )
                == 1
            )

            # If this checks out, we compute the vector
            if (cond_1) and (cond_2):
                vec = chain_residues[i + 1][1]["coords"] - residue[1]["coords"]

                if reverse:
                    vec = -vec
                if scale:
                    vec = vec / np.linalg.norm(vec)

            residue[1][f"sequence_neighbour_vector_{suffix}"] = vec
