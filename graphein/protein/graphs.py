"""Functions for working with Protein Structure Graphs"""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import glob
import os
import re
import subprocess
from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional, Union, Callable
from dataenforce import Dataset

import networkx as nx
import numpy as np
import pandas as pd

from Bio.PDB import *
from Bio.PDB.DSSP import dssp_dict_from_pdb_file, residue_max_acc
from Bio.PDB.Polypeptide import aa1, one_to_three, three_to_one
from biopandas.pdb import PandasPdb

from rdkit.Chem import MolFromPDBFile


import matplotlib.pyplot as plt
from pydantic import BaseModel

from graphein import utils
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.visualisation import protein_graph_plot_3d

# Todo dataenforce types
# AtomicDF = Dataset[]


def read_pdb_to_dataframe(
    pdb_path: str, verbose: bool = False
) -> pd.DataFrame:
    """Reads PDB file to PandasPDB object
    :param pdb_path: path to PDB file
    :type pdb_path: str
    :param verbose: print dataframe?
    :type verbose: bool
    """
    atomic_df = PandasPdb().read_pdb(pdb_path)

    if verbose:
        print(atomic_df)
    return atomic_df


def process_dataframe(
    protein_df: pd.DataFrame,
    granularity: str = "centroids",
    chain_selection: str = "all",
    insertions: bool = False,
    deprotonate: bool = True,
    keep_hets: bool = False,
    exclude_waters: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Process ATOM and HETATM dataframes to produce singular dataframe used for graph construction
    :param protein_df:
    :param granularity:
    :param insertions:
    :param deprotonate:
    :param keep_hets:
    :param exclude_waters:
    :param verbose:
    :return:
    """
    atoms = protein_df.df["ATOM"]
    hetatms = protein_df.df["HETATM"]

    if granularity == "centroids":
        if deprotonate:
            atoms = atoms.loc[atoms["atom_name"] != "H"].reset_index()
        centroids = calculate_centroid_positions(atoms)
        # centroids["residue_id"] =
        atoms = atoms.loc[atoms["atom_name"] == "CA"].reset_index()
        atoms["x_coord"] = centroids["x_coord"]
        atoms["y_coord"] = centroids["y_coord"]
        atoms["z_coord"] = centroids["z_coord"]
    else:
        atoms = atoms.loc[atoms["atom_name"] == granularity]

    if keep_hets:
        # Todo this control flow needs improving.
        if exclude_waters:
            hetatms = hetatms.loc[hetatms["residue_name"] != "HOH"]
        if verbose:
            print(f"Detected {len(hetatms)} HETATOM nodes")
        protein_df = pd.concat([atoms, hetatms])
    else:
        protein_df = atoms

    # Remove alt_loc residues
    if not insertions:
        protein_df = protein_df.loc[protein_df["alt_loc"].isin(["", "A"])]

    # perform chain selection
    protein_df = select_chains(
        protein_df, chain_selection=chain_selection, verbose=verbose
    )

    if verbose:
        print(f"Detected {len(protein_df)} total nodes")

    return protein_df


def select_chains(
    protein_df: pd.DataFrame, chain_selection: str, verbose: bool = False
) -> pd.DataFrame:
    """
    Extracts relevant chains from protein_df
    :param protein_df: pandas dataframe of PDB subsetted to relevant atoms (CA, CB)
    :param chain_selection:
    :param verbose: Print dataframe
    :type verbose: bool
    :return
    """
    if chain_selection != "all":
        chains = [
            protein_df.loc[protein_df["chain_id"] == chain]
            for chain in chain_selection
        ]
    else:
        chains = [
            protein_df.loc[protein_df["chain_id"] == chain]
            for chain in protein_df["chain_id"].unique()
        ]
    protein_df = pd.concat([c for c in chains])

    if verbose:
        print(protein_df)

    return protein_df


def add_nodes_to_graph(
    protein_df: pd.DataFrame,
    pdb_id: str,
    granularity: str = "CA",
    verbose: bool = False,
) -> nx.Graph:
    G = nx.Graph()

    # Assign graph level attributes todo move to separate function
    G.graph["pdb_id"] = pdb_id
    G.graph["chain_ids"] = list(protein_df["chain_id"].unique())
    G.graph["pdb_df"] = protein_df
    # Add Sequences
    for c in G.graph["chain_ids"]:
        G.graph[f"sequence_{c}"] = (
            protein_df.loc[protein_df["chain_id"] == c]["residue_name"]
            .apply(three_to_one)
            .str.cat()
        )

    # Assign intrinsic node attributes
    chain_id = protein_df["chain_id"].apply(str)
    residue_name = protein_df["residue_name"]
    residue_number = protein_df["residue_number"].apply(str)
    coords = np.asarray(protein_df[["x_coord", "y_coord", "z_coord"]])
    b_factor = protein_df["b_factor"]

    # Name nodes
    nodes = protein_df["chain_id"] + ":" + residue_name + ":" + residue_number
    if granularity == "atom":
        nodes = nodes + ":" + protein_df["atom_name"]

    # add nodes to graph
    G.add_nodes_from(nodes)

    # Set intrinsic node attributes
    nx.set_node_attributes(G, dict(zip(nodes, chain_id)), "chain_id")
    nx.set_node_attributes(G, dict(zip(nodes, residue_name)), "residue_name")
    nx.set_node_attributes(
        G, dict(zip(nodes, residue_number)), "residue_number"
    )
    nx.set_node_attributes(G, dict(zip(nodes, coords)), "coords")
    nx.set_node_attributes(G, dict(zip(nodes, b_factor)), "b_factor")

    # Todo include charge, line_idx for traceability?

    if verbose:
        print(nx.info(G))
        print(G.nodes())

    return G


def calculate_centroid_positions(
    atoms: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """
    Calculates position of sidechain centroids
    :param atoms: ATOM df of protein structure
    :param verbose: bool
    :type verbose: bool
    :return: centroids (df)
    """
    centroids = (
        atoms.groupby("residue_number")
        .mean()[["x_coord", "y_coord", "z_coord"]]
        .reset_index()
    )
    if verbose:
        print(f"Calculated {len(centroids)} centroid nodes")
    return centroids


def annotate_node_metadata(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    for func in funcs:
        for n, d in G.nodes(data=True):
            func(n, d)
    return G


def compute_node_metadata(G: nx.Graph, funcs: List[Callable]) -> pd.Series:
    for func in funcs:
        for n, d in G.nodes(data=True):
            metadata = func(n, d)
    return metadata


def annotate_edge_metadata(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    raise NotImplementedError


def compute_edge_metadata(G: nx.Graph, funcs: List[Callable]) -> pd.Series:
    raise NotImplementedError


def annotate_graph_metadata(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    for func in funcs:
        func(G)
    return G


def compute_edges(
    G: nx.Graph, config: BaseModel, funcs: List[Callable]
) -> nx.Graph:

    for func in funcs:
        func(G)

    return G


def construct_graph(config: BaseModel, pdb_path: str, pdb_code: str):
    df = read_pdb_to_dataframe(pdb_path, verbose=config.verbose)
    df = process_dataframe(df)

    g = add_nodes_to_graph(df, pdb_code, config.granularity, config.verbose)
    g = annotate_node_metadata(g, [expasy_protein_scale, meiler_embedding])
    g = compute_edges(
        g, config, [peptide_bonds, salt_bridge, van_der_waals, pi_cation]
    )
    g = annotate_graph_metadata(g, [esm_sequence_embedding])

    return g


if __name__ == "__main__":
    from graphein.features.edges import peptide_bonds, salt_bridge, van_der_waals, pi_cation
    from graphein.features.amino_acid import expasy_protein_scale, meiler_embedding
    from graphein.features.sequence import (
        esm_sequence_embedding,
        biovec_sequence_embedding,
        molecular_weight,
    )

    configs = {
        "granularity": "CA",
        "keep_hets": False,
        "insertions": False,
        "contacts_dir": "../../examples/contacts/",
        "verbose": False,
    }
    config = ProteinGraphConfig(**configs)

    df = read_pdb_to_dataframe(
        pdb_path="../../examples/pdbs/3eiy.pdb",
        verbose=config.verbose,
    )
    df = process_dataframe(df)

    g = add_nodes_to_graph(df, "3eiy", config.granularity, config.verbose)

    g = annotate_node_metadata(g, [expasy_protein_scale, meiler_embedding])
    g = compute_edges(
        g, config, [peptide_bonds, salt_bridge, van_der_waals, pi_cation]
    )
    g = annotate_graph_metadata(
        g,
        [esm_sequence_embedding, biovec_sequence_embedding, molecular_weight],
    )

    print(nx.info(g))

    print(g.graph)
    colors = nx.get_edge_attributes(g, "color").values()

    """
    nx.draw(
        g,
        # pos = nx.circular_layout(g),
        edge_color=colors,
        with_labels=True,
    )
    plt.show()
    """
