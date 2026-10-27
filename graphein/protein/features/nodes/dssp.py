"""Featurization functions for graph nodes using DSSP predicted features."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Charlie Harris
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
from Bio.PDB.DSSP import residue_max_acc
from loguru import logger

from graphein.protein import dssp_core

if TYPE_CHECKING:
    from graphein.protein.config import DSSPConfig

DSSP_COLS = [
    "chain",
    "resnum",
    "icode",
    "aa",
    "ss",
    "asa",
    "phi",
    "psi",
    "dssp_index",
    "NH_O_1_relidx",
    "NH_O_1_energy",
    "O_NH_1_relidx",
    "O_NH_1_energy",
    "NH_O_2_relidx",
    "NH_O_2_energy",
    "O_NH_2_relidx",
    "O_NH_2_energy",
]
"""Columns in DSSP Output."""

DSSP_SS = ["H", "B", "E", "G", "I", "T", "S"]
"""Secondary structure types detected by ``DSSP``"""


def parse_dssp_df(dssp: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse ``DSSP`` output (as read by Biopython's
    ``Bio.PDB.DSSP.dssp_dict_from_pdb_file``) to DataFrame.

    :param dssp: Dictionary containing ``DSSP`` output
    :type dssp: Dict[str, Any]
    :return: pd.DataFrame containing parsed ``DSSP`` output
    :rtype: pd.DataFrame
    """
    appender = []
    for k in dssp[1]:
        to_append = []
        y = dssp[0][k]
        chain = k[0]
        residue = k[1]
        # het = residue[0]
        resnum = residue[1]
        icode = residue[2]
        to_append.extend([chain, resnum, icode])
        to_append.extend(y)
        appender.append(to_append)

    return pd.DataFrame.from_records(appender, columns=DSSP_COLS)


def compute_dssp_df(df: pd.DataFrame) -> pd.DataFrame:
    """Computes DSSP for an atomic dataframe, following mkdssp 4.x.

    Produces the columns of mkdssp's classic output (:const:`DSSP_COLS`, as
    previously parsed with Biopython), without requiring the ``mkdssp``
    executable. Secondary structure, H-bond partners, phi/psi and
    accessibility are identical to mkdssp 4.6.1, except that polyproline
    (``P``, new in DSSP 4) is reported as loop (``-``).

    Rows are protein residues with a complete backbone (N, CA, C, O) in file
    order: ``ATOM`` residues, plus ``HETATM`` residues (e.g. ``MSE``)
    peptide-bonded into a chain. Hydrogens are ignored and, where atoms have
    alternate locations, the highest-occupancy one is used.

    :param df: Atomic dataframe, e.g. ``G.graph["raw_pdb_df"]``.
    :type df: pd.DataFrame
    :return: DataFrame with columns :const:`DSSP_COLS`.
    :rtype: pd.DataFrame
    """
    df = df.loc[
        df["record_name"].isin(["ATOM", "HETATM"])
        & ~df["element_symbol"].isin(["H", "D", "T"])
    ]
    # One conformer per atom (highest occupancy), then file order.
    df = (
        df.sort_values("occupancy", ascending=False, kind="stable")
        .drop_duplicates(
            ["chain_id", "residue_number", "insertion", "atom_name"]
        )
        .sort_values("atom_number", kind="stable")
    )
    keys = ["chain_id", "residue_number", "insertion"]
    res = df.groupby(keys, sort=False, dropna=False).ngroup().to_numpy()
    residues = df.drop_duplicates(keys)
    het = df["record_name"].eq("HETATM").groupby(res).any().to_numpy()

    # (residues, atoms, 3) with N, CA, C, O first, then side chain atoms.
    slot = df["atom_name"].map({"N": 0, "CA": 1, "C": 2, "O": 3})
    side = slot.isna().to_numpy()
    slot = slot.to_numpy(copy=True)
    slot[side] = 4 + df[side].groupby(res[side]).cumcount().to_numpy()
    slot = slot.astype(int)
    x = np.zeros((len(residues), slot.max(initial=3) + 1, 3), np.float32)
    present = np.zeros(x.shape[:2], dtype=bool)
    x[res, slot] = df[["x_coord", "y_coord", "z_coord"]].to_numpy()
    present[res, slot] = True

    # Keep complete residues; HETATM ones only if peptide-bonded in a chain.
    keep = present[:, :4].all(1)
    idx = np.flatnonzero(keep)
    chain = residues["chain_id"].to_numpy()
    bonded = np.zeros(len(residues), dtype=bool)
    link = ~dssp_core._dssp_breaks(x[idx, :4]) & (
        chain[idx[1:]] == chain[idx[:-1]]
    )
    bonded[idx[1:]] |= link
    bonded[idx[:-1]] |= link
    keep &= ~het | bonded
    x, present, residues = x[keep], present[keep], residues[keep]

    bb = x[:, :4]
    L = len(bb)
    aa = residues["residue_name"].to_numpy()
    donor_mask = aa != "PRO"  # proline N has no H to donate
    energies = dssp_core._dssp_energies(bb, np, donor_mask)
    ss = dssp_core._assign8(bb, np, donor_mask, energies)
    acc = dssp_core._accessibility(x, present).astype(np.float64)
    number = dssp_core._dssp_numbers(bb)
    phi, psi = dssp_core._dssp_phi_psi(bb)
    partners = dssp_core._dssp_partners(*energies, L)

    def one_dp(v):  # as printed by mkdssp and parsed back by Biopython
        return [float(f"{f:.1f}") for f in np.asarray(v, dtype=np.float64)]

    out = {
        "chain": residues["chain_id"].to_numpy(),
        "resnum": residues["residue_number"].to_numpy(),
        "icode": residues["insertion"].fillna("").replace("", " ").to_numpy(),
        "aa": aa,
        "ss": [dssp_core.DSSP8_STATES[i] for i in ss],
        "asa": np.floor(acc + 0.5).astype(int),
        "phi": one_dp(phi),
        "psi": one_dp(psi),
        "dssp_index": number,
    }
    for (partner, energy), name in zip(partners, ("NH_O", "O_NH")):
        for k in (0, 1):
            found = partner[:, k] >= 0
            rel = np.where(found, number[partner[:, k]] - number, 0)
            out[f"{name}_{k + 1}_relidx"] = rel
            out[f"{name}_{k + 1}_energy"] = one_dp(energy[:, k])
    return pd.DataFrame(out)[DSSP_COLS]


def add_dssp_df(
    G: nx.Graph,
    dssp_config: Optional[DSSPConfig] = None,
) -> nx.Graph:
    """
    Construct DSSP dataframe and add as graph level variable to protein graph.

    DSSP is computed natively (see :func:`compute_dssp_df`) from
    ``G.graph["raw_pdb_df"]``; the ``mkdssp`` executable is not required.

    :param G: Input protein graph
    :param G: nx.Graph
    :param dssp_config: Deprecated and ignored.
    :type dssp_config: DSSPConfig, optional
    :return: Protein graph with DSSP dataframe added
    :rtype: nx.Graph
    """
    if dssp_config is not None:
        warnings.warn(
            "DSSPConfig is deprecated and ignored: DSSP is now computed "
            "natively, without mkdssp.",
            DeprecationWarning,
            stacklevel=2,
        )
    config = G.graph["config"]
    dssp_dict = compute_dssp_df(G.graph["raw_pdb_df"])

    # Construct node IDs
    dssp_dict["node_id"] = (
        dssp_dict["chain"]
        + ":"
        + dssp_dict["aa"]
        + ":"
        + dssp_dict["resnum"].astype(str)
    )
    if config.insertions:
        dssp_dict["node_id"] = (
            dssp_dict["node_id"] + ":" + dssp_dict["icode"].apply(str)
        )
        # Replace trailing : for non insertions
        dssp_dict["node_id"] = dssp_dict["node_id"].str.replace(
            r":\s*$", "", regex=True
        )

    dssp_dict.set_index("node_id", inplace=True)

    if config.verbose:
        logger.debug(dssp_dict)

    # Assign DSSP Dict
    G.graph["dssp_df"] = dssp_dict

    return G


def add_dssp_feature(G: nx.Graph, feature: str) -> nx.Graph:
    """
    Adds specified amino acid feature as calculated
    by DSSP to every node in a protein graph

    :param G: Protein structure graph to add dssp feature to
    :type G: nx.Graph
    :param feature: string specifying name of DSSP feature to add:
        ``"chain"``,
        ``"resnum"``,
        ``"icode"``,
        ``"aa"``,
        ``"ss"``,
        ``"asa"``,
        ``"phi"``,
        ``"psi"``,
        ``"dssp_index"``,
        ``"NH_O_1_relidx"``,
        ``"NH_O_1_energy"``,
        ``"O_NH_1_relidx"``,
        ``"O_NH_1_energy"``,
        ``"NH_O_2_relidx"``,
        ``"NH_O_2_energy"``,
        ``"O_NH_2_relidx"``,
        ``"O_NH_2_energy"``,
        These names are accessible in the DSSP_COLS list
    :type feature: str
    :return: Protein structure graph with DSSP feature added to nodes
    :rtype: nx.Graph
    """
    if "dssp_df" not in G.graph:
        G = add_dssp_df(G, G.graph["config"].dssp_config)

    config = G.graph["config"]
    dssp_df = G.graph["dssp_df"]

    # Change to not allow for atom granularity?
    if config.granularity == "atom":
        # TODO confirm below is not needed and remove
        """
        # If granularity is atom, apply residue feature to every atom
        for n in G.nodes():
            residue = n.split(":")
            residue = residue[0] + ":" + residue[1] + ":" + residue[2]

            G.nodes[n][feature] = dssp_df.loc[residue, feature]
        """
        raise NameError(f"DSSP residue features ({feature}) \
            cannot be added to atom granularity graph")

    else:
        nx.set_node_attributes(G, dict(dssp_df[feature]), feature)

    if config.verbose:
        logger.debug("Added " + feature + " features to graph nodes")

    return G


def rsa(G: nx.Graph) -> nx.Graph:
    """
    Adds RSA (relative solvent accessibility) of each residue in protein graph
    as calculated by DSSP.

    :param G: Input protein graph
    :type G: nx.Graph
    :return: Protein graph with rsa values added
    :rtype: nx.Graph
    """

    # Calculate RSA
    try:
        dssp_df = G.graph["dssp_df"]
    except KeyError:
        G = add_dssp_df(G, G.graph["config"].dssp_config)
        dssp_df = G.graph["dssp_df"]
    dssp_df["max_acc"] = dssp_df["aa"].map(residue_max_acc["Sander"].get)
    dssp_df[["asa", "max_acc"]] = dssp_df[["asa", "max_acc"]].astype(float)
    dssp_df["rsa"] = dssp_df["asa"] / dssp_df["max_acc"]

    G.graph["dssp_df"] = dssp_df

    return add_dssp_feature(G, "rsa")


def asa(G: nx.Graph) -> nx.Graph:
    """
    Adds ASA of each residue in protein graph as calculated by DSSP.

    :param G: Input protein graph
    :type G: nx.Graph
    :return: Protein graph with asa values added
    :rtype: nx.Graph
    """
    return add_dssp_feature(G, "asa")


def phi(G: nx.Graph) -> nx.Graph:
    """
    Adds phi-angles of each residue in protein graph as calculated by DSSP.

    :param G: Input protein graph
    :type G: nx.Graph
    :return: Protein graph with phi-angles values added
    :rtype: nx.Graph
    """
    return add_dssp_feature(G, "phi")


def psi(G: nx.Graph) -> nx.Graph:
    """
    Adds psi-angles of each residue in protein graph as calculated by DSSP.

    :param G: Input protein graph
    :type G: nx.Graph
    :return: Protein graph with psi-angles values added
    :rtype: nx.Graph
    """
    return add_dssp_feature(G, "psi")


def secondary_structure(G: nx.Graph) -> nx.Graph:
    """
    Adds secondary structure of each residue in protein graph
    as calculated by DSSP in the form of a string

    :param G: Input protein graph
    :type G: nx.Graph
    :return: Protein graph with secondary structure added
    :rtype: nx.Graph
    """
    return add_dssp_feature(G, "ss")
