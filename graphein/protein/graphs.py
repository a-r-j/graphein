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

from graphein.features.edges.distance import compute_distmat
from graphein.features.edges.intramolecular import get_contacts_df
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.utils import get_protein_name_from_filename
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
        PandasPdb().fetch_pdb(pdb_code)
        if pdb_code is None
        else PandasPdb().read_pdb(pdb_path)
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
    return df.loc[df["atom_name"] != "H"].reset_index(drop=True)


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
    return df.loc[df["atom_name"] == granularity]


def remove_insertions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove insertions from structure."""
    # Remove alt_loc residues
    # Todo log.debug(f"Detected X insertions")
    return df.loc[df["alt_loc"].isin(["", "A"])]


def filter_hetatms(
    df: pd.DataFrame, keep_hets: List[str]
) -> List[pd.DataFrame]:
    """Return hetatms of interest."""
    hetatms_to_keep = []
    for hetatm in keep_hets:
        hetatms_to_keep.append(df.loc[df["residue_name"] == hetatm])
    return hetatms_to_keep


def process_dataframe(
    protein_df: pd.DataFrame,
    config: Optional[ProteinGraphConfig],
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
    :param chain_selection: Which protein chain to select. Defaults to "all".
    :return: A protein dataframe that can be consumed by
        other graph construction functions.
    """
    # TODO: Need to properly define what "granularity" is supposed to do.
    atoms = protein_df.df["ATOM"]
    hetatms = protein_df.df["HETATM"]

    # Deprotonate structure by removing H atoms
    if deprotonate:
        atoms = deprotonate_structure(atoms)

    # Restrict DF to desired granularity
    if granularity == "centroids":
        atoms = convert_structure_to_centroids(atoms)
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
    """Add nodes into protein graph."""
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
    G: nx.Graph, config: ProteinGraphConfig, funcs: List[Callable]
) -> nx.Graph:
    """Compute edges."""
    # Todo move to edge computation
    G.graph["contacts_df"] = get_contacts_df(config, G.graph["pdb_id"])
    G.graph["dist_mat"] = compute_distmat(G.graph["pdb_df"])

    for func in funcs:
        func(G)

    return G


def construct_graph(
    config: Optional[ProteinGraphConfig],
    pdb_path: Optional[str] = None,
    pdb_code: Optional[str] = None,
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

    df = read_pdb_to_dataframe(pdb_path, pdb_code, verbose=config.verbose)
    df = process_dataframe(df, config)

    # Add nodes to graph
    g = add_nodes_to_graph(df, pdb_code, config.granularity, config.verbose)

    # Annotate additional node metadata
    if config.node_metadata_functions is not None:
        g = annotate_node_metadata(g, config.node_metadata_functions)

    # Compute graph edges
    g = compute_edges(g, config, config.edge_construction_functions)

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
    print(config)
    g = construct_graph(config=config, pdb_path="../../examples/pdbs/3eiy.pdb")

    """
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
