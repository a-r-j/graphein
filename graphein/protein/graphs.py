"""Functions for working with Protein Structure Graphs"""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import logging
from typing import Callable, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from Bio.PDB.Polypeptide import three_to_one
from biopandas.pdb import PandasPdb

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import compute_distmat
from graphein.protein.edges.intramolecular import get_contacts_df
from graphein.protein.resi_atoms import BACKBONE_ATOMS
from graphein.protein.utils import (
    filter_dataframe,
    get_protein_name_from_filename,
)
from graphein.utils import (
    annotate_edge_metadata,
    annotate_graph_metadata,
    annotate_node_metadata,
    compute_edges,
)

# from rdkit.Chem import MolFromPDBFile
# from graphein.protein.visualisation import protein_graph_plot_3d


logging.basicConfig(level="DEBUG")
log = logging.getLogger(__name__)


def read_pdb_to_dataframe(
    pdb_path: Optional[str] = None,
    pdb_code: Optional[str] = None,
    verbose: bool = False,
    granularity: str = "CA",
) -> pd.DataFrame:
    """
    Reads PDB file to PandasPDB object.

    Returns `atomic_df`,
    which is a dataframe enumerating all atoms
    and their cartesian coordinates in 3D space.

    :param pdb_path: path to PDB file
    :param pdb_code: 4-character PDB accession
    :param verbose: print dataframe?
    """
    if pdb_code is None and pdb_path is None:
        raise NameError("One of pdb_code or pdb_path must be specified!")

    atomic_df = (
        PandasPdb().read_pdb(pdb_path)
        if pdb_path is not None
        else PandasPdb().fetch_pdb(pdb_code)
    )

    # Assign Node IDs to dataframes
    atomic_df.df["ATOM"]["node_id"] = (
        atomic_df.df["ATOM"]["chain_id"].apply(str)
        + ":"
        + atomic_df.df["ATOM"]["residue_name"]
        + ":"
        + atomic_df.df["ATOM"]["residue_number"].apply(str)
    )
    if granularity == "atom":
        atomic_df.df["ATOM"]["node_id"] = (
            atomic_df.df["ATOM"]["node_id"]
            + ":"
            + atomic_df.df["ATOM"]["atom_name"]
        )
    if verbose:
        print(atomic_df)
    return atomic_df


def deprotonate_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Remove protons from PDB dataframe.

    :param df: Atomic dataframe.
    :returns: Atomic dataframe with all atom_name == "H" removed.
    """
    log.debug(
        "Deprotonating protein. This removes H atoms from the pdb_df dataframe"
    )
    # return df.loc[df["atom_name"] != "H"].reset_index(drop=True)
    return filter_dataframe(
        df, by_column="atom_name", list_of_values=["H"], boolean=False
    )


def convert_structure_to_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Overwrite existing (x, y, z) coordinates with centroids of the amino acids."""
    centroids = calculate_centroid_positions(df)
    df = df.loc[df["atom_name"] == "CA"].reset_index(drop=True)
    df["x_coord"] = centroids["x_coord"]
    df["y_coord"] = centroids["y_coord"]
    df["z_coord"] = centroids["z_coord"]

    return df


def subset_structure_to_atom_type(
    df: pd.DataFrame, granularity: str
) -> pd.DataFrame:
    """Return a subset of atomic dataframe that contains only certain atom names."""
    return filter_dataframe(
        df, by_column="atom_name", list_of_values=[granularity], boolean=True
    )
    # return df.loc[df["atom_name"] == granularity]


def remove_insertions(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function removes insertions from PDB dataframes
    :param df:
    :type df:
    :return:
    :rtype:
    """
    """Remove insertions from structure."""
    # Remove alt_loc residues
    # Todo log.debug(f"Detected X insertions")
    # return df.loc[df["alt_loc"].isin(["", "A"])]
    return filter_dataframe(
        df, by_column="alt_loc", list_of_values=["", "A"], boolean=True
    )


def filter_hetatms(
    df: pd.DataFrame, keep_hets: List[str]
) -> List[pd.DataFrame]:
    """Return hetatms of interest."""
    hetatms_to_keep = []
    for hetatm in keep_hets:
        hetatms_to_keep.append(df.loc[df["residue_name"] == hetatm])
    return hetatms_to_keep


def compute_rgroup_dataframe(pdb_df: pd.DataFrame) -> pd.DataFrame:
    """Return the atoms that are in R-groups and not the backbone chain."""
    rgroup_df = filter_dataframe(pdb_df, "atom_name", BACKBONE_ATOMS, False)
    return rgroup_df


def process_dataframe(
    protein_df: pd.DataFrame,
    atom_df_processing_funcs: Optional[List[Callable]] = None,
    hetatom_df_processing_funcs: Optional[List[Callable]] = None,
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
    :param atom_df_processing_funcs: List of functions to process dataframe. These must take in a dataframe and return a dataframe
    :param hetatom_df_processing_funcs: List of functions to process dataframe. These must take in a dataframe and return a dataframe
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
    :param chain_selection: Which protein chain to select. Defaults to "all".
    :return: A protein dataframe that can be consumed by
        other graph construction functions.
    """
    # TODO: Need to properly define what "granularity" is supposed to do.
    atoms = protein_df.df["ATOM"]
    hetatms = protein_df.df["HETATM"]

    # This block enables processing via a list of supplied functions operating on the atom and hetatom dataframes
    # If these are provided, the dataframe returned will be computed only from these and the default workflow
    # below this block will not execute.
    if atom_df_processing_funcs is not None:
        for func in atom_df_processing_funcs:
            atoms = func(atoms)
        if hetatom_df_processing_funcs is None:
            return atoms

    if hetatom_df_processing_funcs is not None:
        for func in hetatom_df_processing_funcs:
            hetatms = func(hetatms)
        return pd.concat([atoms, hetatms])

    # Deprotonate structure by removing H atoms
    if deprotonate:
        atoms = deprotonate_structure(atoms)

    # Restrict DF to desired granularity
    if granularity == "centroids":
        atoms = convert_structure_to_centroids(atoms)
    elif granularity == "atom":
        atoms = atoms
    else:
        atoms = subset_structure_to_atom_type(atoms, granularity)

    protein_df = atoms

    if len(keep_hets) > 0:
        hetatms_to_keep = filter_hetatms(atoms, keep_hets)
        protein_df = pd.concat([atoms, hetatms_to_keep])

    # Remove alt_loc residues
    if not insertions:
        protein_df = remove_insertions(protein_df)

    # perform chain selection
    protein_df = select_chains(
        protein_df, chain_selection=chain_selection, verbose=verbose
    )

    """
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
    """

    log.debug(f"Detected {len(protein_df)} total nodes")

    return protein_df


def assign_node_id_to_dataframe(
    protein_df: pd.DataFrame, granularity: str
) -> pd.DataFrame:
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
        # chains = [
        #    protein_df.loc[protein_df["chain_id"] == chain]
        #    for chain in chain_selection
        # ]
        protein_df = filter_dataframe(
            protein_df, list(chain_selection), boolean=True
        )
    # else:
    # chains = [
    #    protein_df.loc[protein_df["chain_id"] == chain]
    #    for chain in protein_df["chain_id"].unique()
    # ]
    # protein_df = pd.concat([c for c in chains])

    return protein_df


def initialise_graph_with_metadata(
    protein_df, raw_pdb_df, pdb_id, granularity: str
) -> nx.Graph:
    G = nx.Graph(
        name=pdb_id,
        pdb_id=pdb_id,
        chain_ids=list(protein_df["chain_id"].unique()),
        pdb_df=protein_df,
        raw_pdb_df=raw_pdb_df,
        rgroup_df=compute_rgroup_dataframe(raw_pdb_df),
    )

    # Create graph and assign intrinsic graph-level metadata
    G.graph["node_type"] = granularity

    # Add Sequences to graph metadata
    for c in G.graph["chain_ids"]:
        G.graph[f"sequence_{c}"] = (
            protein_df.loc[protein_df["chain_id"] == c]["residue_name"]
            .apply(three_to_one)
            .str.cat()
        )
    return G


def add_nodes_to_graph(
    G: nx.Graph,
    protein_df: Optional[pd.DataFrame] = None,
    verbose: bool = False,
) -> nx.Graph:
    """Add nodes into protein graph."""

    # If no protein dataframe is supplied, use the one stored in the Graph object
    if protein_df is None:
        protein_df = G.graph["pdb_df"]
    # Assign intrinsic node attributes
    chain_id = protein_df["chain_id"].apply(str)
    residue_name = protein_df["residue_name"]
    residue_number = protein_df["residue_number"]  # .apply(str)
    coords = np.asarray(protein_df[["x_coord", "y_coord", "z_coord"]])
    b_factor = protein_df["b_factor"]
    atom_type = protein_df["atom_name"]
    nodes = protein_df["node_id"]
    element_symbol = protein_df["element_symbol"]
    G.add_nodes_from(nodes)

    # Set intrinsic node attributes
    nx.set_node_attributes(G, dict(zip(nodes, chain_id)), "chain_id")
    nx.set_node_attributes(G, dict(zip(nodes, residue_name)), "residue_name")
    nx.set_node_attributes(
        G, dict(zip(nodes, residue_number)), "residue_number"
    )
    nx.set_node_attributes(G, dict(zip(nodes, atom_type)), "atom_type")
    nx.set_node_attributes(
        G, dict(zip(nodes, element_symbol)), "element_symbol"
    )
    nx.set_node_attributes(G, dict(zip(nodes, coords)), "coords")
    nx.set_node_attributes(G, dict(zip(nodes, b_factor)), "b_factor")

    # TODO: include charge, line_idx for traceability?
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


def compute_edges(
    G: nx.Graph,
    get_contacts_config: Optional[GetContactsConfig],
    funcs: List[Callable],
) -> nx.Graph:
    """Compute edges."""
    # Todo move to edge computation
    if get_contacts_config is not None:
        G.graph["contacts_df"] = get_contacts_df(
            get_contacts_config, G.graph["pdb_id"]
        )

    G.graph["atomic_dist_mat"] = compute_distmat(G.graph["raw_pdb_df"])
    G.graph["dist_mat"] = compute_distmat(G.graph["pdb_df"])

    for func in funcs:
        func(G)

    return G


def construct_graph(
    config: Optional[ProteinGraphConfig],
    pdb_path: Optional[str] = None,
    pdb_code: Optional[str] = None,
    chain_selection: str = "all",
    df_processing_funcs: Optional[List[Callable]] = None,
    edge_construction_funcs: Optional[List[Callable]] = None,
    edge_annotation_funcs: Optional[List[Callable]] = None,
    node_annotation_funcs: Optional[List[Callable]] = None,
    graph_annotation_funcs: Optional[List[Callable]] = None,
) -> nx.Graph:
    """
    Constructs protein structure graph from a pdb_code or pdb_path. Users can provide a ProteinGraphConfig object.

    However, config parameters can be overridden by passing arguments directly to the function.

    :param config: ProteinGraphConfig object. If None, defaults to config in graphein.protein.config
    :param pdb_path: Path to pdb_file to build graph from
    :param pdb_code: 4-character PDB accession pdb_code to build graph from
    :param df_processing_funcs: List of dataframe processing functions
    :param edge_construction_funcs: List of edge construction functions
    :param edge_annotation_funcs: List of edge annotation functions
    :param node_annotation_funcs: List of node annotation functions
    :param graph_annotation_funcs: List of graph annotation function
    :return: Protein Structure Graph
    """

    # If no config is provided, use default
    if config is None:
        config = ProteinGraphConfig()

    # Get name from pdb_file is no pdb_code is provided
    if pdb_path and (pdb_code is None):
        pdb_code = get_protein_name_from_filename(pdb_path)

    # If config params are provided, overwrite them
    config.protein_df_processing_functions = (
        df_processing_funcs
        if config.protein_df_processing_functions is None
        else config.protein_df_processing_functions
    )
    config.edge_construction_functions = (
        edge_construction_funcs
        if config.edge_construction_functions is None
        else config.edge_construction_functions
    )
    config.node_metadata_functions = (
        node_annotation_funcs
        if config.node_metadata_functions is None
        else config.node_metadata_functions
    )
    config.graph_metadata_functions = (
        graph_annotation_funcs
        if config.graph_metadata_functions is None
        else config.graph_metadata_functions
    )
    config.edge_metadata_functions = (
        edge_annotation_funcs
        if config.edge_metadata_functions is None
        else config.edge_metadata_functions
    )

    raw_df = read_pdb_to_dataframe(
        pdb_path,
        pdb_code,
        verbose=config.verbose,
        granularity=config.granularity,
    )
    protein_df = process_dataframe(
        raw_df, chain_selection=chain_selection, granularity=config.granularity
    )

    # Initialise graph with metadata
    g = initialise_graph_with_metadata(
        protein_df=protein_df,
        raw_pdb_df=raw_df.df["ATOM"],
        pdb_id=pdb_code,
        granularity=config.granularity,
    )
    # Add nodes to graph
    g = add_nodes_to_graph(g)
    # Annotate additional node metadata
    if config.node_metadata_functions is not None:
        g = annotate_node_metadata(g, config.node_metadata_functions)

    # Compute graph edges
    g = compute_edges(
        g, config.get_contacts_config, config.edge_construction_functions
    )

    # Annotate additional graph metadata
    if config.graph_metadata_functions is not None:
        g = annotate_graph_metadata(g, config.graph_metadata_functions)

    # Annotate additional edge metadata
    if config.edge_metadata_functions is not None:
        g = annotate_edge_metadata(g, config.edge_metadata_functions)

    # Add config to graph
    g.graph["config"] = config

    return g


if __name__ == "__main__":
    from functools import partial

    import graphein.protein.features.sequence.propy
    from graphein.protein.edges.atomic import (
        add_atomic_edges,
        add_bond_order,
        add_ring_status,
    )
    from graphein.protein.edges.distance import (
        add_delaunay_triangulation,
        add_hydrogen_bond_interactions,
        add_k_nn_edges,
    )
    from graphein.protein.edges.intramolecular import (
        peptide_bonds,
        salt_bridge,
    )
    from graphein.protein.features.nodes.amino_acid import (
        expasy_protein_scale,
        meiler_embedding,
    )
    from graphein.protein.features.sequence.embeddings import (
        biovec_sequence_embedding,
        esm_sequence_embedding,
    )
    from graphein.protein.features.sequence.sequence import molecular_weight

    configs = {
        "granularity": "atom",
        "keep_hets": False,
        "insertions": False,
        "verbose": False,
    }
    config = ProteinGraphConfig(**configs)
    config.edge_construction_functions = [
        add_atomic_edges,
        add_bond_order,
        add_ring_status,
    ]
    # Test High-level API
    g = construct_graph(config=config, pdb_path="../../examples/pdbs/3eiy.pdb")
    print(nx.info(g))
    print(g.edges())
    """
    # Test Low-level API
    raw_df = read_pdb_to_dataframe(
        pdb_path="../../examples/pdbs/3eiy.pdb",
        verbose=config.verbose,
    )

    processed_pdb_df = process_dataframe(
        protein_df=raw_df,
        atom_df_processing_funcs=None,
        hetatom_df_processing_funcs=None,
        granularity="centroids",
        chain_selection="all",
        insertions=False,
        deprotonate=True,
        keep_hets=[],
        verbose=False,
    )

    g = initialise_graph_with_metadata(
        protein_df=processed_pdb_df,
        raw_pdb_df=raw_df.df["ATOM"],
        pdb_id="3eiy",
        granularity=config.granularity,
    )

    g = add_nodes_to_graph(g)

    g = annotate_node_metadata(g, [expasy_protein_scale, meiler_embedding])
    g = compute_edges(
        g,
        config.get_contacts_config,
        [
            add_delaunay_triangulation,
            peptide_bonds,
            salt_bridge,
            add_hydrogen_bond_interactions,
        ],
    )

    g = annotate_graph_metadata(
        g,
        [
            esm_sequence_embedding,
            biovec_sequence_embedding,
            molecular_weight,
        ],
    )

    print(nx.info(g))
    colors = nx.get_edge_attributes(g, "color").values()
    """
    """
    nx.draw(
        g,
        # pos = nx.circular_layout(g),
        edge_color=colors,
        with_labels=True,
    )
    plt.show()
    """
