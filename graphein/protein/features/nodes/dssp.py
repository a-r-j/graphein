"""Featurization functions for graph nodes using DSSP predicted features."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Charlie Harris
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import os
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
from Bio.Data.IUPACData import protein_letters_1to3
from Bio.PDB.DSSP import dssp_dict_from_pdb_file, residue_max_acc

from graphein.protein.utils import download_pdb

DSSP_COLS = [
    "chain",
    "resnum",
    "icode",
    "aa",
    "ss",
    "exposure_rsa",
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

DSSP_SS = ["H", "B", "E", "G", "I", "T", "S"]


def parse_dssp_df(dssp: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse DSSP output to DataFrame
    """
    appender = []
    for k in dssp[1]:
        to_append = []
        y = dssp[0][k]
        chain = k[0]
        residue = k[1]
        het = residue[0]
        resnum = residue[1]
        icode = residue[2]
        to_append.extend([chain, resnum, icode])
        to_append.extend(y)
        appender.append(to_append)

    return pd.DataFrame.from_records(appender, columns=DSSP_COLS)


def process_dssp_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DSSP DataFrame to make indexes align with node IDs
    """

    # Convert 1 letter aa code to 3 letter
    amino_acids = df["aa"].tolist()

    for i, amino_acid in enumerate(amino_acids):
        amino_acids[i] = protein_letters_1to3[amino_acid].upper()
    df["aa"] = amino_acids

    # Construct node IDs
    node_ids = []

    for i, row in df.iterrows():
        node_id = row["chain"] + ":" + row["aa"] + ":" + str(row["resnum"])
        node_ids.append(node_id)
    df["node_id"] = node_ids

    df.set_index("node_id", inplace=True)

    return df


def add_dssp_df(G: nx.Graph) -> nx.Graph:
    """
    Construct DSSP dataframe and add as graph level variable to protein grapgh
    :param G: Input protein graph
    :return: Protein graph with DSSP dataframe added
    """

    config = G.graph["config"]
    pdb_id = G.graph["pdb_id"]

    # To add - Check for DSSP installation

    # Check for existence of pdb file. If not, download it.
    if not os.path.isfile(config.pdb_dir / pdb_id):
        pdb_file = download_pdb(config, pdb_id)
    else:
        pdb_file = config.pdb_dir + pdb_id + ".pdb"

    if config.verbose:
        pass  # print DSSP executable

    # Todo - add executable from config
    dssp_dict = dssp_dict_from_pdb_file(pdb_file, DSSP="mkdssp")

    dssp_dict = parse_dssp_df(dssp_dict)
    dssp_dict = process_dssp_df(dssp_dict)

    if config.verbose:
        print(dssp_dict)

    G.graph["dssp_df"] = dssp_dict

    return G


def add_dssp_feature(G: nx.Graph, feature: str) -> nx.Graph:
    """
    Adds a certain amino acid feature as calculated by DSSP to every node in a protein graph
    """

    config = G.graph["config"]
    dssp_df = G.graph["dssp_df"]

    ## TO DO
    # Change to not allow for atom granuarlity?
    if config.granularity == "atom":
        # If granularity is atom, apply residue feature to every atom
        for n in G.nodes():
            residue = n.split(":")
            residue = residue[0] + ":" + residue[1] + ":" + residue[2]

            G.nodes[n][feature] = dssp_df.loc[residue, feature]

    else:
        nx.set_node_attributes(G, dict(dssp_df[feature]), feature)

    if config.verbose:
        print("Added " + feature + " features to graph nodes")

    return G


# TODO port ASA and RSA calculations from older version of graphein
# Check ASA
def asa(G: nx.Graph) -> nx.Graph:
    """
    Adds ASA of each residue in protein graph as calculated by DSSP.
        Note: DSSP dataframe must be precomputed and added as graph level variable "dssp_df".

    :param G: Input protein graph
    :return: Protein graph with asa values added
    """
    return add_dssp_feature(G, "exposure_rsa")


def phi(G: nx.Graph) -> nx.Graph:
    """
    Adds phi-angles of each residue in protein graph as calculated by DSSP.
        Note: DSSP dataframe must be precomputed and added as graph level variable "dssp_df".

    :param G: Input protein graph
    :return: Protein graph with phi-angles values added
    """
    return add_dssp_feature(G, "phi")


def psi(G: nx.Graph) -> nx.Graph:
    """
    Adds psi-angles of each residue in protein graph as calculated by DSSP.
        Note: DSSP dataframe must be precomputed and added as graph level variable "dssp_df".

    :param G: Input protein graph
    :return: Protein graph with psi-angles values added
    """
    return add_dssp_feature(G, "psi")


def secondary_structure(G: nx.Graph) -> nx.Graph:
    """
    Adds secondary structure of each residue in protein graph as calculated by DSSP in the form of a string
        Note: DSSP dataframe must be precomputed and added as graph level variable "dssp_df".

    :param G: Input protein graph
    :return: Protein graph with secondary structure added
    """
    return add_dssp_feature(G, "ss")
