"""Provides geometry-based featurisation functions."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import List

import networkx as nx
import numpy as np
from loguru import logger as log

from graphein.protein.utils import compute_rgroup_dataframe, filter_dataframe

VECTOR_FEATURE_NAMES: List[str] = [
    "sidechain_vector",
    "c_beta_vector",
    "sequence_neighbour_vector_n_to_c",
    "sequence_neighbour_vector_c_to_n",
    "virtual_c_beta_vector",
]
"""Names of all vector features from the module."""


def add_sidechain_vector(
    g: nx.Graph, scale: bool = True, reverse: bool = False
):
    """Adds vector from node to average position of sidechain atoms.

    We compute the mean of the sidechain atoms for each node. For this we use
    the ``rgroup_df`` dataframe. If the graph does not contain the ``rgroup_df``
    dataframe, we compute it from the ``raw_pdb_df``. If ``scale``, we scale
    the vector to the unit vector. If ``reverse`` is ``True``, we reverse the
    vector (``sidechain - node``). If reverse is false (default) we compute
    (``node - sidechain``).

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

    sc_centroid = (
        g.graph["rgroup_df"].groupby("node_id").mean(numeric_only=True)
    )

    # Iterate over nodes and compute vector
    for n, d in g.nodes(data=True):
        if d["residue_name"] == "GLY":
            # If GLY, set vector to 0
            vec = np.array([0.0, 0.0, 0.0])
        elif n not in sc_centroid.index:
            vec = np.array([0.0, 0.0, 0.0])
            log.warning(
                f"Non-glycine residue {n} does not have side-chain atoms."
            )
        else:
            if reverse:
                vec = d["coords"] - np.array(
                    sc_centroid.loc[n][["x_coord", "y_coord", "z_coord"]],
                    dtype=float,
                )
            else:
                vec = (
                    np.array(
                        sc_centroid.loc[n][["x_coord", "y_coord", "z_coord"]],
                        dtype=float,
                    )
                    - d["coords"]
                )

            if scale:
                vec = vec / np.linalg.norm(vec)

        d["sidechain_vector"] = vec


def add_beta_carbon_vector(
    g: nx.Graph, scale: bool = True, reverse: bool = False
):
    """Adds vector from node (typically alpha carbon) to position of beta
    carbon.

    Glycine does not have a beta carbon, so we set it to
    ``np.array([0., 0., 0.])``. We extract the position of the beta carbon from the
    unprocessed atomic PDB dataframe. For this we use the ``raw_pdb_df``
    DataFrame. If ``scale``, we scale the vector to the unit vector. If
    ``reverse`` is ``True``, we reverse the vector (``C beta - node``).
    If ``reverse`` is ``False`` (default) we compute (``node - C beta``).

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

    c_beta_coords = filter_dataframe(
        g.graph["rgroup_df"], "atom_name", ["CB"], boolean=True
    )
    c_beta_coords.index = c_beta_coords["node_id"]

    # Iterate over nodes and compute vector
    for n, d in g.nodes(data=True):
        if d["residue_name"] == "GLY":
            vec = np.array([0.0, 0.0, 0.0])
        elif n not in c_beta_coords.index:
            vec = np.array([0.0, 0.0, 0.0])
            log.warning(
                f"Non-glycine residue {n} does not have a beta-carbon."
            )
        else:
            if reverse:
                vec = d["coords"] - np.array(
                    c_beta_coords.loc[n][["x_coord", "y_coord", "z_coord"]],
                    dtype=float,
                )
            else:
                vec = (
                    np.array(
                        c_beta_coords.loc[n][
                            ["x_coord", "y_coord", "z_coord"]
                        ],
                        dtype=float,
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

    If ``n_to_c`` is ``True`` (default), we compute the vectors from the N
    terminus to the C terminus (canonical direction). If ``reverse`` is
    ``False`` (default), we compute ``Node_i - Node_{i+1}``. If ``reverse is
    ``True``, we compute ``Node_{i+1} - Node_i``.

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
                    [0.0, 0.0, 0.0]
                )
                continue

            # Get insertion codes
            ins_current = (
                residue[0].split(":")[3] if residue[0].count(":") > 2 else ""
            )
            ins_next = (
                chain_residues[i + 1][0].split(":")[3]
                if chain_residues[i + 1][0].count(":") > 2
                else ""
            )
            if not n_to_c:
                ins_current, ins_next = ins_next, ins_current

            # Get sequence distance
            dist = abs(
                residue[1]["residue_number"]
                - chain_residues[i + 1][1]["residue_number"]
            )

            # Asserts residues are adjacent
            cond_adjacent = (
                dist == 1
                or (dist == 0 and not ins_current and ins_next == "A")
                or (
                    dist == 0
                    and ins_current
                    and ins_next
                    and chr(ord(ins_current) + 1) == ins_next
                )
            )

            # If this checks out, we compute the non-zero vector
            if cond_adjacent:
                vec = chain_residues[i + 1][1]["coords"] - residue[1]["coords"]

                if reverse:
                    vec = -vec
                if scale:
                    vec = vec / np.linalg.norm(vec)
            else:
                vec = np.array([0.0, 0.0, 0.0])

            residue[1][f"sequence_neighbour_vector_{suffix}"] = vec


def add_virtual_beta_carbon_vector(
    g: nx.Graph, scale: bool = False, reverse: bool = False
):
    """For each node adds a vector from alpha carbon to virtual beta carbon.
    :param g: Graph to add vector to.
    :type g: nx.Graph
    :param scale: Scale vector to unit vector. Defaults to ``False``.
    :type scale: bool
    :param reverse: Reverse vector. Defaults to ``False``.
    :type reverse: bool
    """
    # Get coords of backbone atoms
    coord_dfs = {}
    for atom_type in ["N", "CA", "C"]:
        df = filter_dataframe(
            g.graph["raw_pdb_df"], "atom_name", [atom_type], boolean=True
        )
        df.index = df["node_id"]
        coord_dfs[atom_type] = df

    # Iterate over nodes and compute vector
    for n, d in g.nodes(data=True):
        if any([n not in df.index for df in coord_dfs.values()]):
            vec = np.array([0, 0, 0], dtype=float)
            log.warning(f"Missing backbone atom in residue {n}.")
        else:
            N = np.array(
                coord_dfs["N"].loc[n][["x_coord", "y_coord", "z_coord"]],
                dtype=float,
            )
            Ca = np.array(
                coord_dfs["CA"].loc[n][["x_coord", "y_coord", "z_coord"]],
                dtype=float,
            )
            C = np.array(
                coord_dfs["C"].loc[n][["x_coord", "y_coord", "z_coord"]],
                dtype=float,
            )
            b = Ca - N
            c = C - Ca
            a = np.cross(b, c)
            Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca
            vec = Cb - Ca

            if reverse:
                vec = -vec
            if scale:
                vec = vec / np.linalg.norm(vec)
        d["virtual_c_beta_vector"] = vec
