"""Functions for working with Protein Structure Graphs"""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma # Todo
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import glob
import os
import re
import subprocess
from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional, Union

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch as torch
import torch.nn.functional as F
from Bio.PDB import *
from Bio.PDB.DSSP import dssp_dict_from_pdb_file, residue_max_acc
from Bio.PDB.Polypeptide import aa1, one_to_three
from biopandas.pdb import PandasPdb
from dgllife.utils import (
    BaseAtomFeaturizer,
    BaseBondFeaturizer,
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    mol_to_bigraph,
    mol_to_complete_graph,
    mol_to_nearest_neighbor_graph,
)
from pydantic import BaseModel
from rdkit.Chem import MolFromPDBFile
from scipy import spatial
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

from graphein import utils


class Config(BaseModel):
    granularity: str = "CA"
    keep_hets: bool = False
    insertions: bool = False
    """
    get_contacts_path: str
    pdb_dir: str
    contacts_dir: str
    exclude_waters: bool = True
    covalent_bonds: bool = True
    include_ss: bool = True
    include_ligand: bool = False
    intramolecular_interactions: Optional[List[str]] = None
    graph_constructor: Optional[str] = None
    edge_distance_cutoff: Optional[float] = None
    verbose: bool = True
    deprotonate: bool = False
    remove_string_labels: bool = False
    long_interaction_threshold: Optional[int] = None
    node_featuriser: Optional[
        Union[BaseAtomFeaturizer, CanonicalAtomFeaturizer, str]
    ] = None
    edge_featuriser: Optional[
        Union[BaseBondFeaturizer, CanonicalBondFeaturizer, str]
    ] = None
    """


def read_pdb_to_dataframe(
    pdb_path: str, verbose: bool = False
) -> pd.DataFrame:
    """Reads PDB file to PandasPDB object
    :param pdb_path: path to PDB file
    :type pdb_path: str
    :param verbose: print dataframe?
    :type verbose: bool
    """

    protein_df = PandasPdb().read_pdb(pdb_path)

    if verbose:
        print(protein_df)

    return protein_df


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

    # Remove alt_loc resdiues
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
    protein_df: pd.DataFrame, granularity: str = "CA", verbose: bool = False
) -> nx.Graph:
    G = nx.Graph()

    residue_name = protein_df["residue_name"]
    residue_number = protein_df["residue_number"].apply(str)
    coords = np.asarray(protein_df[["x_coord", "y_coord", "z_coord"]])

    nodes = protein_df["chain_id"] + ":" + residue_name + ":" + residue_number

    if granularity == "atom":
        nodes = nodes + ":" + protein_df["atom_name"]

    residue_name_dict = dict(zip(nodes, residue_name))
    residue_number_dict = dict(zip(nodes, residue_number))
    coords_dict = dict(zip(nodes, coords))

    # add nodes to graph
    G.add_nodes_from(nodes)
    # Set intrinsic node attributes
    nx.set_node_attributes(G, residue_name_dict, "residue_name")
    nx.set_node_attributes(G, residue_number_dict, "residue_number")
    nx.set_node_attributes(
        G, coords_dict, "coords"
    )  # Todo, maybe split into x_coord, y_coord, z_coord
    # Todo include charge, B factor, line_idx for traceability?

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


def annotate_node_metadata(G, funcs):
    for func in funcs:
        for n in G.nodes():
            func(G, n)
    return G


def annotate_node_metadata(G, n):
    G.node[n]["metadata_field"] = some_value
    return G


def construct_graph():
    pass


if __name__ == "__main__":
    configs = {"granularity": "CA", "keep_hets": False, "insertions": False}
    config = Config(**configs)

    df = read_pdb_to_dataframe(
        "/Users/arianjamasb/github/graphein/examples/pdbs/3eiy.pdb",
        verbose=True,
    )
    df = process_dataframe(df)
    g = add_nodes_to_graph(df, config.granularity, verbose=True)
    print(nx.info(g))
    print(nx.get_node_attributes(g, "coords"))
    print(nx.get_node_attributes(g, "residue_number"))
    print(nx.get_node_attributes(g, "residue_name"))
