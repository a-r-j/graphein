"""Featurization functions for graph nodes using DSSP predicted features."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Charlie Harris
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Dict, Optional

import networkx as nx
import pandas as pd
from Bio.PDB.DSSP import dssp_dict_from_pdb_file, residue_max_acc
from loguru import logger

from graphein.protein.resi_atoms import STANDARD_AMINO_ACID_MAPPING_1_TO_3
from graphein.protein.utils import save_pdb_df_to_pdb
from graphein.utils.dependencies import is_tool, requires_external_dependencies

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
    Parse ``DSSP`` output to DataFrame.

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


@requires_external_dependencies("mkdssp")
def add_dssp_df(
    G: nx.Graph,
    dssp_config: Optional[DSSPConfig],
) -> nx.Graph:
    """
    Construct DSSP dataframe and add as graph level variable to protein graph.

    :param G: Input protein graph
    :param G: nx.Graph
    :param dssp_config: DSSPConfig object. Specifies which executable to run.
        Located in `:obj:graphein.protein.config`.
    :type dssp_config: DSSPConfig, optional
    :return: Protein graph with DSSP dataframe added
    :rtype: nx.Graph
    """
    # if dssp_config is None:
    config = G.graph["config"]
    pdb_code = G.graph["pdb_code"]
    path = G.graph["path"]
    pdb_name = G.graph["name"]

    # Extract DSSP executable
    executable = dssp_config.executable

    # Ensure that DSSP is on PATH and is marked as an executable.
    assert is_tool(
        executable
    ), "DSSP must be on PATH and marked as an executable"

    pdb_file = None
    if path:
        if os.path.isfile(path):
            pdb_file = path
    else:
        if config.pdb_dir:
            if os.path.isfile(config.pdb_dir / (pdb_code + ".pdb")):
                pdb_file = config.pdb_dir / (pdb_code + ".pdb")

    # get dssp version string
    dssp_version = re.search(
        r"version ([\d\.]+)", os.popen(f"{executable} --version").read()
    ).group(
        1
    )  # e.g. "4.0.4"
    # Check for existence of pdb file. If not, reconstructs it from the raw df.
    if pdb_file:
        dssp_dict = dssp_dict_from_pdb_file(
            pdb_file, DSSP=executable, dssp_version=dssp_version
        )
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_pdb_df_to_pdb(
                G.graph["raw_pdb_df"], tmpdirname + f"/{pdb_name}.pdb"
            )
            dssp_dict = dssp_dict_from_pdb_file(
                tmpdirname + f"/{pdb_name}.pdb",
                DSSP=executable,
                dssp_version=dssp_version,
            )

    if len(dssp_dict[0]) == 0:
        raise ValueError(
            "DSSP could not be calculated. Check DSSP version "
            f"({dssp_version}) orthat the input PDB file is valid."
        )

    if config.verbose:
        logger.debug(f"Using DSSP executable '{executable}'")

    dssp_dict = parse_dssp_df(dssp_dict)
    # Convert 1 letter aa code to 3 letter
    dssp_dict["aa"] = dssp_dict["aa"].map(STANDARD_AMINO_ACID_MAPPING_1_TO_3)

    # Resolve UNKs
    # NOTE: the original didn't work if HETATM residues exist in DSSP output
    _raw_pdb_df = G.graph["raw_pdb_df"].copy().drop_duplicates("node_id")
    _dssp_df_unk = dssp_dict.loc[dssp_dict["aa"] == "UNK"][
        ["chain", "resnum", "icode"]
    ]
    for chain, resnum, icode in _dssp_df_unk.values:
        dssp_dict.loc[
            (dssp_dict["chain"] == chain)
            & (dssp_dict["resnum"] == resnum)  # e.g. 'H'
            & (dssp_dict["icode"] == icode),  # e.g. 100  # e.g. 'E' or ' '
            "aa",
        ] = _raw_pdb_df.loc[
            (_raw_pdb_df["chain_id"] == chain)
            & (_raw_pdb_df["residue_number"] == resnum)
            & (
                _raw_pdb_df["insertion"] == icode.strip()
            )  # why strip?: dssp icode is a space, raw_pdb_df is empty string
        ][
            "residue_name"
        ].values[
            0
        ]

    # Construct node IDs
    dssp_dict["node_id"] = (
        dssp_dict["chain"]
        + ":"
        + dssp_dict["aa"]
        + ":"
        + dssp_dict["resnum"].astype(str)
    )
    if G.graph["config"].insertions:
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
        raise NameError(
            f"DSSP residue features ({feature}) \
            cannot be added to atom granularity graph"
        )

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
