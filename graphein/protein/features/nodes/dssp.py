"""Featurization functions for graph nodes using DSSP predicted features."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Charlie Harris
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

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


def add_dssp_df(G: nx.Graph, dssp_config: Optional[DSSPConfig]) -> nx.Graph:
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
    if "dssp_df" not in G.graph:
        G = add_dssp_df(G, G.graph["config"].dssp_config)

    config = G.graph["config"]
    dssp_df = G.graph["dssp_df"]

    # Change to not allow for atom granuarlity?
    if config.granularity == "atom":
        raise NameError(
            f"DSSP residue features ({feature}) cannot be added to atom granularity graph"
        )

        # TODO confirm below is not needed and remove
        """
        # If granularity is atom, apply residue feature to every atom
        for n in G.nodes():
            residue = n.split(":")
            residue = residue[0] + ":" + residue[1] + ":" + residue[2]

            G.nodes[n][feature] = dssp_df.loc[residue, feature]
        """

    else:
        nx.set_node_attributes(G, dict(dssp_df[feature]), feature)

    if config.verbose:
        print("Added " + feature + " features to graph nodes")

    return G


def rsa(G: nx.Graph) -> nx.Graph:
    """
    Adds RSA (relative solvent accessibility) of each residue in protein graph as calculated by DSSP.

    :param G: Input protein graph
    :return: Protein graph with rsa values added
    """

    # Calcualte RSA
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
    :return: Protein graph with asa values added
    """
    return add_dssp_feature(G, "asa")


def phi(G: nx.Graph) -> nx.Graph:
    """
    Adds phi-angles of each residue in protein graph as calculated by DSSP.

    :param G: Input protein graph
    :return: Protein graph with phi-angles values added
    """
    return add_dssp_feature(G, "phi")


def psi(G: nx.Graph) -> nx.Graph:
    """
    Adds psi-angles of each residue in protein graph as calculated by DSSP.

    :param G: Input protein graph
    :return: Protein graph with psi-angles values added
    """
    return add_dssp_feature(G, "psi")


def secondary_structure(G: nx.Graph) -> nx.Graph:
    """
    Adds secondary structure of each residue in protein graph as calculated by DSSP in the form of a string

    :param G: Input protein graph
    :return: Protein graph with secondary structure added
    """
    return add_dssp_feature(G, "ss")

    """
def _get_protein_features(
        self, pdb_code: Optional[str], file_path: Optional[str], chain_selection: str
) -> pd.DataFrame:
    :param file_path: (str) file path to PDB file
    :param pdb_code: (str) String containing four letter PDB accession
    :return df (pd.DataFrame): Dataframe containing output of DSSP (Solvent accessibility, secondary structure for each residue)

    # Run DSSP on relevant PDB file
    if pdb_code:
        d = dssp_dict_from_pdb_file(self.pdb_dir + pdb_code + ".pdb")
    if file_path:
        d = dssp_dict_from_pdb_file(file_path)

    # Subset dataframe to those in chain_selection
    if chain_selection != "all":
        df = df.loc[df["chain"].isin(chain_selection)]
    # Rename cysteines to 'C'
    df["aa"] = df["aa"].str.replace("[a-z]", "C")
    df = df[df["aa"].isin(list(aa1))]

    # Drop alt_loc residues
    df = df.loc[df["icode"] == " "]

    # Add additional Columns
    df["aa_three"] = df["aa"].apply(one_to_three)
    df["max_acc"] = df["aa_three"].map(residue_max_acc["Sander"].get)
    df[["exposure_rsa", "max_acc"]] = df[["exposure_rsa", "max_acc"]].astype(float)
    df["exposure_asa"] = df["exposure_rsa"] * df["max_acc"]
    df["index"] = df["chain"] + ":" + df["aa_three"] + ":" + df["resnum"].apply(str)
    return df
"""
