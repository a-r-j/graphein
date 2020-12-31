"""Functions for working with Protein Structure Graphs"""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from Bio.PDB import *
from Bio.PDB.DSSP import dssp_dict_from_pdb_file, residue_max_acc
from Bio.PDB.Polypeptide import three_to_one
from biopandas.pdb import PandasPdb
from rdkit.Chem import MolFromPDBFile

from graphein import utils
from graphein.features.edges.distance import compute_distmat
from graphein.features.edges.intramolecular import get_contacts_df
from graphein.protein.config import ProteinGraphConfig

from ..utils import annotate_graph_metadata, annotate_node_metadata

# from graphein.protein.visualisation import protein_graph_plot_3d


logging.basicConfig(level="DEBUG")
log = logging.getLogger(__name__)


def read_pdb_to_dataframe(
    pdb_path: str, verbose: bool = False
) -> pd.DataFrame:
    """
    Reads PDB file to PandasPDB object
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
    keep_hets: List[str] = [],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Process ATOM and HETATM dataframes to produce singular dataframe used for graph construction

    :param protein_df: Dataframe to process.
        Should be the object returned from `read_pdb_to_dataframe`.
    :param granularity: The level of granualrity for the graph.
        This determines the node definition.
        Acceptable values include:
        - "centroids"
        - "atoms"
        - any of the atom_names in the PDB file (e.g. "CA", "CB", "OG", etc.)
    :param insertions: Whether or not to keep insertions.
    :param deprotonate: Whether or not to remove hydrogen atoms (i.e. deprotonation).
    :param keep_hets: Hetatoms to keep. Defaults to an empty list.
        To keep a hetatom, pass it inside a list of hetatom names to keep.
    :param verbose: Verbosity level.
    :paraam chain_selection: Which protein chain to select. Defaults to "all".
    :return: A protein dataframe that can be consumed by
        other graph construction functions.
    """
    atoms = protein_df.df["ATOM"]
    hetatms = protein_df.df["HETATM"]

    # Deprotonate structure by removing H atoms
    if deprotonate:
        log.debug(
            "Deprotonating protein. This removes H atoms from the pdb_df dataframe"
        )
        atoms = atoms.loc[atoms["atom_name"] != "H"].reset_index(drop=True)

    # Restrict DF to desired granularity
    if granularity == "centroids":
        centroids = calculate_centroid_positions(atoms)
        atoms = atoms.loc[atoms["atom_name"] == "CA"].reset_index(drop=True)
        atoms["x_coord"] = centroids["x_coord"]
        atoms["y_coord"] = centroids["y_coord"]
        atoms["z_coord"] = centroids["z_coord"]
    else:
        atoms = atoms.loc[atoms["atom_name"] == granularity]

    hetatms_to_keep = []
    for hetatm in keep_hets:
        hetatms_to_keep.append(hetatms.loc[hetatms["residue_name"] == hetatm])
    protein_df = pd.concat([atoms, hetatms_to_keep])

    # Remove alt_loc residues
    if not insertions:
        # Todo log.debug(f"Detected X insertions")
        protein_df = protein_df.loc[protein_df["alt_loc"].isin(["", "A"])]

    # perform chain selection
    protein_df = select_chains(
        protein_df, chain_selection=chain_selection, verbose=verbose
    )

    # Name nodes
    protein_df["node_id"] = (
        protein_df["chain_id"].apply(str)
        + ":"
        + protein_df["residue_name"]
        + ":"
        + protein_df["residue_number"].apply(str)
    )
    if granularity == "atom":
        protein_df["node_id"] = (
            protein_df["node_id"] + ":" + protein_df["atom_name"]
        )

    log.debug(f"Detected {len(protein_df)} total nodes")

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

    return protein_df


def add_nodes_to_graph(
    protein_df: pd.DataFrame,
    pdb_id: str,
    granularity: str = "CA",
    verbose: bool = False,
) -> nx.Graph:
    # Create graph and assign intrinsic graph-level metadata
    G = nx.Graph(
        name=pdb_id,
        pdb_id=pdb_id,
        node_type=granularity,
        chain_ids=list(protein_df["chain_id"].unique()),
        pdb_df=protein_df,
    )
    # Add Sequences to graph metadata
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

    nodes = protein_df["node_id"]
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
    log.debug(f"Calculated {len(centroids)} centroid nodes")
    return centroids


def compute_node_metadata(G: nx.Graph, funcs: List[Callable]) -> pd.Series:
    # TODO: This needs a clearer function definition.
    for func in funcs:
        for n, d in G.nodes(data=True):
            metadata = func(n, d)
    return metadata


def compute_edge_metadata(G: nx.Graph, funcs: List[Callable]) -> pd.Series:
    raise NotImplementedError


def compute_edges(
    G: nx.Graph, config: BaseModel, funcs: List[Callable]
) -> nx.Graph:
    # Todo move to edge computation
    G.graph["contacts_df"] = get_contacts_df(config, G.graph["pdb_id"])
    G.graph["dist_mat"] = compute_distmat(G.graph["pdb_df"])

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
    import graphein.features.sequence.propy
    from graphein.features.amino_acid import (
        expasy_protein_scale,
        meiler_embedding,
    )
    from graphein.features.edges.distance import (
        add_aromatic_interactions,
        add_aromatic_sulphur_interactions,
        add_cation_pi_interactions,
        add_delaunay_triangulation,
        add_distance_threshold,
        add_disulfide_interactions,
        add_hydrogen_bond_interactions,
        add_hydrophobic_interactions,
        add_ionic_interactions,
        add_k_nn_edges,
    )
    from graphein.features.edges.intramolecular import (
        peptide_bonds,
        pi_cation,
        salt_bridge,
        van_der_waals,
    )
    from graphein.features.sequence.embeddings import (
        biovec_sequence_embedding,
        esm_sequence_embedding,
    )
    from graphein.features.sequence.sequence import molecular_weight

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
        g, config, [partial(add_k_nn_edges, long_interaction_threshold=0)]
    )
    """
    g = annotate_graph_metadata(
        g,
        [
            # esm_sequence_embedding,
            # biovec_sequence_embedding,
            # partial(molecular_weight, aggregation_type=["sum", "mean", "max"]),
            partial(
                amino_acid_composition, aggregation_type=["sum", "mean", "max"]
            ),
            partial(
                dipeptide_composition, aggregation_type=["sum", "mean", "max"]
            ),
            partial(
                aa_dipeptide_composition,
                aggregation_type=["sum", "mean", "max"],
            ),
            all_composition_descriptors,
            composition_normalized_vdwv
        ],
    )
    """

    print(nx.info(g))
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
