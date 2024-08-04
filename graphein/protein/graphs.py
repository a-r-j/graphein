"""Functions for working with Protein Structure Graphs."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma, Charlie Harris
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import os
import traceback
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cpdb
import networkx as nx
import numpy as np
import pandas as pd
from biopandas.mmcif import PandasMmcif
from biopandas.mmtf import PandasMmtf
from biopandas.pdb import PandasPdb
from loguru import logger as log
from rich.progress import Progress
from tqdm.contrib.concurrent import process_map

from graphein.protein.config import GetContactsConfig, ProteinGraphConfig
from graphein.protein.edges.distance import (
    add_distance_to_edges,
    compute_distmat,
)
from graphein.protein.resi_atoms import BACKBONE_ATOMS, RESI_THREE_TO_1
from graphein.protein.subgraphs import extract_subgraph_from_chains
from graphein.protein.utils import (
    ProteinGraphConfigurationError,
    compute_rgroup_dataframe,
    filter_dataframe,
    get_protein_name_from_filename,
    three_to_one_with_mods,
)
from graphein.rna.constants import RNA_ATOMS
from graphein.utils.utils import (
    annotate_edge_metadata,
    annotate_graph_metadata,
    annotate_node_metadata,
    compute_edges,
)

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def subset_structure_to_rna(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a subset of atomic DataFrame that contains only certain atom names
    relevant for RNA structures.

    :param df: Protein Structure DataFrame to subset.
    :type df: pd.DataFrame
    :returns: Subsetted protein structure DataFrame.
    :rtype: pd.DataFrame
    """
    return filter_dataframe(
        df, by_column="atom_name", list_of_values=RNA_ATOMS, boolean=True
    )


def read_pdb_to_dataframe(
    path: Optional[Union[str, os.PathLike]] = None,
    pdb_code: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    model_index: int = 1,
) -> pd.DataFrame:
    """
    Reads PDB file to ``PandasPDB`` object.

    Returns ``atomic_df``, which is a DataFrame enumerating all atoms and
    their cartesian coordinates in 3D space. Also contains associated metadata
    from the PDB file.

    :param path: path to PDB or MMTF file. Defaults to ``None``.
    :type path: str, optional
    :param pdb_code: 4-character PDB accession. Defaults to ``None``.
    :type pdb_code: str, optional
    :param uniprot_id: UniProt ID to build graph from AlphaFoldDB. Defaults to
        ``None``.
    :type uniprot_id: str, optional
    :param model_index: Index of model to read. Only relevant for structures
        containing ensembles. Defaults to ``1``.
    :type model_index: int, optional
    :returns: ``pd.DataFrame`` containing protein structure
    :rtype: pd.DataFrame
    """
    if pdb_code is None and path is None and uniprot_id is None:
        raise NameError(
            "One of pdb_code, path or uniprot_id must be specified!"
        )

    if path is not None:
        if isinstance(path, Path):
            path = os.fsdecode(path)
        if (
            path.endswith(".pdb")
            or path.endswith(".pdb.gz")
            or path.endswith(".ent")
        ):
            atomic_df = cpdb.parse(path)
        elif path.endswith(".mmtf") or path.endswith(".mmtf.gz"):
            atomic_df = PandasMmtf().read_mmtf(path)
            atomic_df = atomic_df.get_model(model_index)
            atomic_df = pd.concat(
                [atomic_df.df["ATOM"], atomic_df.df["HETATM"]]
            )
        elif (
            path.endswith(".cif")
            or path.endswith(".cif.gz")
            or path.endswith(".mmcif")
            or path.endswith(".mmcif.gz")
        ):
            atomic_df = PandasMmcif().read_mmcif(path)
            atomic_df = atomic_df.get_model(model_index)
            atomic_df = atomic_df.convert_to_pandas_pdb()
            atomic_df = pd.concat(
                [atomic_df.df["ATOM"], atomic_df.df["HETATM"]]
            )
        else:
            raise ValueError(
                f"File {path} must be either .pdb(.gz), .mmtf(.gz), .(mm)cif(.gz) or .ent, not {path.split('.')[-1]}"
            )
    elif uniprot_id is not None:
        atomic_df = cpdb.parse(uniprot_id=uniprot_id)
    else:
        atomic_df = cpdb.parse(pdb_code=pdb_code)

    if "model_idx" in atomic_df.columns:
        atomic_df = atomic_df.loc[atomic_df["model_idx"] == model_index]

    if len(atomic_df) == 0:
        raise ValueError(f"No model found for index: {model_index}")

    return atomic_df


def label_node_id(
    df: pd.DataFrame, granularity: str, insertions: bool = False
) -> pd.DataFrame:
    """Assigns a ``node_id`` column to the atomic dataframe. Node IDs are of the
    form: ``"<CHAIN>:<RESIDUE_NAME>:<RESIDUE_NUMBER>:<ATOM_NAME>"`` for atomic
    graphs or ``"<CHAIN>:<RESIDUE_NAME>:<RESIDUE_NUMBER>"`` for residue graphs.

    If ``insertions=True``, the insertion code will be appended to the end of
    the node_id (e.g. ``"<CHAIN>:<RESIDUE_NAME>:<RESIDUE_NUMBER>:<ATOM_NAME>:"``)

    :param df: Protein structure DataFrame.
    :type df: pd.DataFrame
    :param granularity: Granularity of graph. Atom-level,
        residue (e.g. ``CA``) or ``centroids``. See:
        :const:`~graphein.protein.config.GRAPH_ATOMS` and
        :const:`~graphein.protein.config.GRANULARITY_OPTS`.
    :type granularity: str
    :param insertions: Whether or not to include insertion codes in the node id.
        Default is ``False``.
    :type insertions: bool
    :return: Protein structure DataFrame with ``node_id`` column.
    :rtype: pd.DataFrame
    """
    df["node_id"] = (
        df["chain_id"].apply(str)
        + ":"
        + df["residue_name"]
        + ":"
        + df["residue_number"].apply(str)
    )

    if insertions:
        df["node_id"] = df["node_id"] + ":" + df["insertion"].apply(str)
        # Replace trailing : for non insertions
        df["node_id"] = df["node_id"].str.replace(":$", "", regex=True)
    # Add Alt Loc identifiers
    df["node_id"] = df["node_id"] + ":" + df["alt_loc"].apply(str)
    df["node_id"] = df["node_id"].str.replace(":$", "", regex=True)
    df["residue_id"] = df["node_id"]
    if granularity == "atom":
        df["node_id"] = df["node_id"] + ":" + df["atom_name"]
    elif granularity in {"rna_atom", "rna_centroid"}:
        df["node_id"] = (
            df["node_id"]
            + ":"
            + df["atom_number"].apply(str)
            + ":"
            + df["atom_name"]
        )
    return df


def deprotonate_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Remove protons from PDB DataFrame.

    :param df: Atomic dataframe.
    :type df: pd.DataFrame
    :returns: Atomic dataframe with all ``element_symbol == "H" or "D" or "T"`` removed.
    :rtype: pd.DataFrame
    """
    log.debug(
        "Deprotonating protein. This removes H atoms from the pdb_df dataframe"
    )
    return filter_dataframe(
        df,
        by_column="element_symbol",
        list_of_values=["H", "D", "T"],
        boolean=False,
    )


def convert_structure_to_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Overwrite existing ``(x, y, z)`` coordinates with centroids of the amino
    acids.

    :param df: Pandas DataFrame protein structure to convert into a dataframe of
        centroid positions.
    :type df: pd.DataFrame
    :return: pd.DataFrame with atoms/residues positions converted into centroid
        positions.
    :rtype: pd.DataFrame
    """
    log.debug(
        "Converting dataframe to centroids. This averages XYZ coords of the \
            atoms in a residue"
    )

    centroids = calculate_centroid_positions(df)
    df = df.loc[df["atom_name"] == "CA"].reset_index(drop=True)
    df["x_coord"] = centroids["x_coord"]
    df["y_coord"] = centroids["y_coord"]
    df["z_coord"] = centroids["z_coord"]

    return df


def subset_structure_to_atom_type(
    df: pd.DataFrame, granularity: str
) -> pd.DataFrame:
    """
    Return a subset of atomic dataframe that contains only certain atom names.

    :param df: Protein Structure dataframe to subset.
    :type df: pd.DataFrame
    :returns: Subset protein structure dataframe.
    :rtype: pd.DataFrame
    """
    return filter_dataframe(
        df, by_column="atom_name", list_of_values=[granularity], boolean=True
    )


def remove_alt_locs(
    df: pd.DataFrame, keep: str = "max_occupancy"
) -> pd.DataFrame:
    """
    This function removes alternatively located atoms from PDB DataFrames
    (see https://proteopedia.org/wiki/index.php/Alternate_locations). Among the
    alternative locations the ones with the highest occupancies are left.

    :param df: Protein Structure dataframe to remove alternative located atoms
        from.
    :type df: pd.DataFrame
    :param keep: Controls how to remove altlocs. Default is ``"max_occupancy"``.
    :type keep: Literal["max_occupancy", "min_occupancy", "first", "last"]
    :return: Protein structure dataframe with alternative located atoms removed
    :rtype: pd.DataFrame
    """
    # Sort accordingly
    if keep == "max_occupancy":
        df = df.sort_values("occupancy")
        keep = "last"
    elif keep == "min_occupancy":
        df = df.sort_values("occupancy")
        keep = "first"
    elif keep == "exclude":
        keep = False

    # Filter
    duplicates = df.duplicated(
        subset=["chain_id", "residue_number", "atom_name", "insertion"],
        keep=keep,
    )
    df = df[~duplicates]

    # Unsort
    if keep in ["max_occupancy", "min_occupancy"]:
        df = df.sort_index()
    df = df.reset_index(drop=True)
    return df


def remove_insertions(
    df: pd.DataFrame, keep: Literal["first", "last"] = "first"
) -> pd.DataFrame:
    """
    This function removes insertions from PDB DataFrames.

    :param df: Protein Structure dataframe to remove insertions from.
    :type df: pd.DataFrame
    :param keep: Specifies which insertion to keep. Options are ``"first"`` or
        ``"last"``. Default is ``"first"``.
    :type keep: Literal["first", "last"]
    :return: Protein structure dataframe with insertions removed
    :rtype: pd.DataFrame
    """
    # Catches unnamed insertions
    duplicates = df.duplicated(
        subset=["chain_id", "residue_number", "atom_name", "alt_loc"],
        keep=keep,
    )
    df = df[~duplicates]

    return filter_dataframe(
        df, by_column="insertion", list_of_values=[""], boolean=True
    )


def filter_hetatms(
    df: pd.DataFrame, keep_hets: List[str]
) -> List[pd.DataFrame]:
    """Return hetatms of interest.

    :param df: Protein Structure dataframe to filter hetatoms from.
    :type df: pd.DataFrame
    :param keep_hets: List of hetero atom names to keep.
    :type keep_hets: List[str]
    :returns: Protein structure dataframe with heteroatoms removed
    :rtype: pd.DataFrame
    """
    return [df.loc[df["residue_name"] == hetatm] for hetatm in keep_hets]


def process_dataframe(
    protein_df: pd.DataFrame,
    atom_df_processing_funcs: Optional[List[Callable]] = None,
    hetatom_df_processing_funcs: Optional[List[Callable]] = None,
    granularity: str = "centroids",
    chain_selection: str = "all",
    insertions: bool = False,
    alt_locs: bool = False,
    deprotonate: bool = True,
    keep_hets: List[str] = [],
) -> pd.DataFrame:
    """
    Process ATOM and HETATM dataframes to produce singular dataframe used for
    graph construction.

    :param protein_df: Dataframe to process.
        Should be the object returned from
        :func:`~graphein.protein.graphs.read_pdb_to_dataframe`.
    :type protein_df: pd.DataFrame
    :param atom_df_processing_funcs: List of functions to process DataFrame.
        These must take in a DataFrame and return a DataFrame. Defaults to
        ``None``.
    :type atom_df_processing_funcs: List[Callable], optional
    :param hetatom_df_processing_funcs: List of functions to process the hetatom
        dataframe. These must take in a DataFrame and return a DataFrame.
    :type hetatom_df_processing_funcs: List[Callable], optional
    :param granularity: The level of granularity for the graph. This determines
        the node definition. Acceptable values include: ``"centroids"``,
        ``"atoms"``, any of the atom_names in the PDB file (e.g. ``"CA"``,
        ``"CB"``, ``"OG"``, etc.).
        See: :const:`~graphein.protein.config.GRAPH_ATOMS` and
        :const:`~graphein.protein.config.GRANULARITY_OPTS`.
    :type granularity: str
    :param insertions: Whether or not to keep insertions. Defaults to ``False``.
    :param insertions: bool
    :param alt_locs: Whether or not to keep alternatively located atoms.
    :param alt_locs: bool
    :param deprotonate: Whether or not to remove hydrogen atoms (i.e.
        deprotonation).
    :type deprotonate: bool
    :param keep_hets: Hetatoms to keep. Defaults to an empty list (``[]``).
        To keep a hetatom, pass it inside a list of hetatom names to keep.
    :type keep_hets: List[str]
    :param chain_selection: Which protein chain to select. Defaults to
        ``"all"``. Eg can use ``"ACF"`` to select 3 chains (``A``, ``C`` &
        ``F``)
    :type chain_selection: str
    :return: A protein dataframe that can be consumed by other graph
        construction functions.
    :rtype: pd.DataFrame
    """
    protein_df = label_node_id(
        protein_df, granularity=granularity, insertions=insertions
    )
    # TODO: Need to properly define what "granularity" is supposed to do.
    atoms = filter_dataframe(
        protein_df,
        by_column="record_name",
        list_of_values=["ATOM"],
        boolean=True,
    )
    hetatms = filter_dataframe(
        protein_df,
        by_column="record_name",
        list_of_values=["HETATM"],
        boolean=True,
    )

    # This block enables processing via a list of supplied functions operating
    # on the atom and hetatom DataFrames. If these are provided, the DataFrame
    # returned will be computed only from these and the default workflow
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

    if keep_hets:
        hetatms_to_keep = filter_hetatms(hetatms, keep_hets)
        atoms = pd.concat([atoms] + hetatms_to_keep)

    # Deprotonate structure by removing H atoms
    if deprotonate:
        atoms = deprotonate_structure(atoms)

    # Restrict DF to desired granularity
    if granularity == "atom":
        pass
    elif granularity in {"centroids", "rna_centroid"}:
        atoms = convert_structure_to_centroids(atoms)
    elif granularity == "rna_atom":
        atoms = subset_structure_to_rna(atoms)
    else:
        atoms = subset_structure_to_atom_type(atoms, granularity)

    protein_df = atoms

    # Remove alt_loc residues
    if alt_locs != "include":
        protein_df = remove_alt_locs(protein_df, keep=alt_locs)

    # Remove inserted residues
    if not insertions:
        protein_df = remove_insertions(protein_df)

    # perform chain selection
    protein_df = select_chains(protein_df, chain_selection=chain_selection)

    log.debug(f"Detected {len(protein_df)} total nodes")

    # Sort dataframe to place HETATMs
    protein_df = sort_dataframe(protein_df)

    return protein_df


def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts a protein dataframe by chain->residue number->atom number

    This is useful for distributing hetatms/modified residues through the DF.

    :param df: Protein dataframe to sort.
    :type df: pd.DataFrame
    :return: Sorted protein dataframe.
    :rtype: pd.DataFrame
    """
    return df.sort_values(
        by=["chain_id", "residue_number", "atom_number", "insertion"]
    )


def select_chains(
    protein_df: pd.DataFrame,
    chain_selection: Union[str, List[str]],
) -> pd.DataFrame:
    """
    Extracts relevant chains from ``protein_df``.

    :param protein_df: pandas DataFrame of PDB subsetted to relevant atoms
        (``CA``, ``CB``).
    :type protein_df: pd.DataFrame
    :param chain_selection: Specifies chains that should be extracted from
        the larger complexed structure. If chain_selection is ``"all"``, all
        chains will be selected. Otherwise, provide a list of strings.
    :type chain_selection: Union[str, List[str]]
    :return: Protein structure dataframe containing only entries in the
        chain selection.
    :rtype: pd.DataFrame
    """
    if chain_selection != "all":
        if isinstance(chain_selection, str):
            raise ValueError(
                "Only 'all' is a valid string for chain selection. Otherwise use a list of strings: e.g. ['A', 'B', 'C']"
            )
        protein_df = filter_dataframe(
            protein_df,
            by_column="chain_id",
            list_of_values=chain_selection,
            boolean=True,
        )

    return protein_df


def initialise_graph_with_metadata(
    protein_df: pd.DataFrame,
    raw_pdb_df: pd.DataFrame,
    granularity: str,
    name: Optional[str] = None,
    pdb_code: Optional[str] = None,
    path: Optional[str] = None,
) -> nx.Graph:
    """
    Initializes the nx Graph object with initial metadata.

    :param protein_df: Processed DataFrame of protein structure.
    :type protein_df: pd.DataFrame
    :param raw_pdb_df: Unprocessed dataframe of protein structure for comparison
        and traceability downstream.
    :type raw_pdb_df: pd.DataFrame
    :param granularity: Granularity of the graph (eg ``"atom"``, ``"CA"``,
        ``"CB"`` etc or ``"centroid"``). See:
        :const:`~graphein.protein.config.GRAPH_ATOMS` and
        :const:`~graphein.protein.config.GRANULARITY_OPTS`.
    :type granularity: str
    :param name: specified given name for the graph. If None, the PDB code or
        the file name will be used to name the graph.
    :type name: Optional[str], defaults to ``None``
    :param pdb_code: PDB ID / Accession code, if the PDB is available on the
        PDB database.
    :type pdb_code: Optional[str], defaults to ``None``.
    :param path: path to local PDB or MMTF file, if constructing a graph from a
        local file.
    :type path: Optional[str], defaults to ``None``.
    :return: Returns initial protein structure graph with metadata.
    :rtype: nx.Graph
    """
    if path is not None and isinstance(path, Path):
        path = os.fsdecode(path)

    # Get name for graph if no name was provided
    if name is None:
        if path is not None:
            name = get_protein_name_from_filename(path)
        else:
            name = pdb_code

    G = nx.Graph(
        name=name,
        pdb_code=pdb_code,
        path=path,
        chain_ids=list(protein_df["chain_id"].unique()),
        pdb_df=protein_df,
        raw_pdb_df=raw_pdb_df,
        rgroup_df=compute_rgroup_dataframe(raw_pdb_df),
        coords=np.asarray(protein_df[["x_coord", "y_coord", "z_coord"]]),
    )

    # Create graph and assign intrinsic graph-level metadata
    G.graph["node_type"] = granularity

    # Add Sequences to graph metadata
    for c in G.graph["chain_ids"]:
        if granularity == "rna_atom":
            sequence = protein_df.loc[protein_df["chain_id"] == c][
                "residue_name"
            ].str.cat()
        elif granularity == "atom":
            sequence = (
                protein_df.loc[
                    (protein_df["chain_id"] == c)
                    & (protein_df["atom_name"] == "CA")
                ]["residue_name"]
                .apply(three_to_one_with_mods)
                .str.cat()
            )
        else:
            sequence = (
                protein_df.loc[protein_df["chain_id"] == c]["residue_name"]
                .apply(three_to_one_with_mods)
                .str.cat()
            )
        G.graph[f"sequence_{c}"] = sequence
    return G


def add_nodes_to_graph(
    G: nx.Graph,
    protein_df: Optional[pd.DataFrame] = None,
    verbose: bool = False,
) -> nx.Graph:
    """Add nodes into protein graph.

    :param G: ``nx.Graph`` with metadata to populate with nodes.
    :type G: nx.Graph
    :param protein_df: DataFrame of protein structure containing nodes & initial
        node metadata to add to the graph. Defaults to ``None``.
    :type protein_df: pd.DataFrame, optional
    :param verbose: Controls verbosity of this step. Defaults to ``False``.
    :type verbose: bool
    :returns: nx.Graph with nodes added.
    :rtype: nx.Graph
    """

    # If no protein dataframe is supplied, use the one stored in the Graph
    # object
    if protein_df is None:
        protein_df: pd.DataFrame = G.graph["pdb_df"]
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
    Calculates position of sidechain centroids.

    :param atoms: ATOM df of protein structure.
    :type atoms: pd.DataFrame
    :param verbose: bool controlling verbosity.
    :type verbose: bool
    :return: centroids (df).
    :rtype: pd.DataFrame
    """
    centroids = (
        atoms.groupby(
            ["residue_number", "chain_id", "residue_name", "insertion"]
        )
        .mean(numeric_only=True)[["x_coord", "y_coord", "z_coord"]]
        .reset_index()
    )
    if verbose:
        print(f"Calculated {len(centroids)} centroid nodes")
    log.debug(f"Calculated {len(centroids)} centroid nodes")
    return centroids


def compute_edges(
    G: nx.Graph,
    funcs: List[Callable],
    get_contacts_config: Optional[GetContactsConfig] = None,
) -> nx.Graph:
    """
    Computes edges for the protein structure graph. Will compute a pairwise
    distance matrix between nodes which is
    added to the graph metadata to facilitate some edge computations.

    :param G: nx.Graph with nodes to add edges to.
    :type G: nx.Graph
    :param funcs: List of edge construction functions.
    :type funcs: List[Callable]
    :param get_contacts_config: Config object for ``GetContacts`` if
        intramolecular edges are being used.
    :type get_contacts_config: graphein.protein.config.GetContactsConfig
    :return: Graph with added edges.
    :rtype: nx.Graph
    """
    # This control flow prevents unnecessary computation of the distance
    # matrices
    if "config" in G.graph:
        if G.graph["config"].granularity == "atom":
            G.graph["atomic_dist_mat"] = compute_distmat(G.graph["pdb_df"])
        else:
            G.graph["dist_mat"] = compute_distmat(G.graph["pdb_df"])

    for func in funcs:
        func(G)

    return add_distance_to_edges(G)


def construct_graph(
    config: Optional[ProteinGraphConfig] = None,
    name: Optional[str] = None,
    path: Optional[Union[str, os.PathLike]] = None,
    uniprot_id: Optional[str] = None,
    pdb_code: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    chain_selection: Union[str, List[str]] = "all",
    model_index: int = 1,
    df_processing_funcs: Optional[List[Callable]] = None,
    edge_construction_funcs: Optional[List[Callable]] = None,
    edge_annotation_funcs: Optional[List[Callable]] = None,
    node_annotation_funcs: Optional[List[Callable]] = None,
    graph_annotation_funcs: Optional[List[Callable]] = None,
    verbose: bool = True,
) -> nx.Graph:
    """
    Constructs protein structure graph from a ``pdb_code``, ``path``,
    ``uniprot_id`` or a BioPandas DataFrame containing ``ATOM`` data.

    Users can provide a :class:`~graphein.protein.config.ProteinGraphConfig`
    object to specify construction parameters.

    However, config parameters can be overridden by passing arguments directly
    to the function.

    :param config: :class:`~graphein.protein.config.ProteinGraphConfig` object.
        If ``None``, defaults to config in ``graphein.protein.config``.
    :type config: graphein.protein.config.ProteinGraphConfig, optional
    :param name: an optional given name for the graph. the PDB ID or PDB file
        name will be used if not specified.
    :type name: str, optional
    :param path: Path to PDB or MMTF file when constructing a graph from a
        local pdb file. Default is ``None``.
    :type path: Optional[str], defaults to ``None``
    :param pdb_code: A 4-character PDB ID / accession to be used to construct
        the graph, if available. Default is ``None``.
    :type pdb_code: Optional[str], defaults to ``None``
    :param uniprot_id: UniProt accession ID to build graph from AlphaFold2DB.
        Default is ``None``.
    :type uniprot_id: str, optional
    :param df: Pandas dataframe containing ATOM data to build graph from.
        Default is ``None``.
    :type df: pd.DataFrame, optional
    :param chain_selection: List of strings denoting polypeptide chains to
        include in graph. E.g ``["A", "B", "D", "F"]`` or ``"all"``. Default is ``"all"``.
    :type chain_selection: str
    :param model_index: Index of model to use in the case of structural
        ensembles. Default is ``1``.
    :type model_index: int
    :param df_processing_funcs: List of dataframe processing functions.
        Default is ``None``.
    :type df_processing_funcs: List[Callable], optional
    :param edge_construction_funcs: List of edge construction functions.
        Default is ``None``.
    :type edge_construction_funcs: List[Callable], optional
    :param edge_annotation_funcs: List of edge annotation functions.
        Default is ``None``.
    :type edge_annotation_funcs: List[Callable], optional
    :param node_annotation_funcs: List of node annotation functions.
        Default is ``None``.
    :type node_annotation_funcs: List[Callable], optional
    :param graph_annotation_funcs: List of graph annotation function.
        Default is ``None``.
    :type graph_annotation_funcs: List[Callable]
    :param verbose: Controls the verbosity.
        Default is ``True``.
    :type verbose: bool
    :return: Protein Structure Graph
    :rtype: nx.Graph
    """

    if pdb_code is None and path is None and uniprot_id is None and df is None:
        raise ValueError(
            "Either a PDB ID, UniProt ID, a dataframe or a path to a local PDB file"
            " must be specified to construct a graph"
        )
    if path is not None and isinstance(path, Path):
        path = os.fsdecode(path)

    # If no config is provided, use default
    if config is None:
        config = ProteinGraphConfig()

    # Use progress tracking context if in verbose mode
    context = Progress(transient=True) if verbose else nullcontext()
    with context as progress:
        if verbose:
            task1 = progress.add_task("Reading PDB file...", total=1)
            progress.advance(task1)

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
        if df is None:
            raw_df = read_pdb_to_dataframe(
                path,
                pdb_code,
                uniprot_id,
                model_index=model_index,
            )
        else:
            raw_df = df

        if verbose:
            task2 = progress.add_task("Processing PDB dataframe...", total=1)
        raw_df = sort_dataframe(raw_df)
        protein_df = process_dataframe(
            raw_df,
            chain_selection=chain_selection,
            granularity=config.granularity,
            insertions=config.insertions,
            alt_locs=config.alt_locs,
            keep_hets=config.keep_hets,
            atom_df_processing_funcs=config.protein_df_processing_functions,
            hetatom_df_processing_funcs=config.protein_df_processing_functions,
            deprotonate=config.deprotonate,
        )

        if verbose:
            progress.advance(task2)

            task3 = progress.add_task("Initializing graph...", total=1)
        # Initialise graph with metadata
        g = initialise_graph_with_metadata(
            protein_df=protein_df,
            raw_pdb_df=raw_df,
            name=name,
            pdb_code=pdb_code,
            path=path,
            granularity=config.granularity,
        )
        # Add nodes to graph
        g = add_nodes_to_graph(g)

        # Add config to graph
        g.graph["config"] = config

        # Annotate additional node metadata
        if config.node_metadata_functions is not None:
            g = annotate_node_metadata(g, config.node_metadata_functions)

        if verbose:
            progress.advance(task3)
            task4 = progress.add_task("Constructing edges...", total=1)
        # Compute graph edges
        g = compute_edges(
            g,
            funcs=config.edge_construction_functions,
            get_contacts_config=None,
        )

        if verbose:
            progress.advance(task4)

    # Annotate additional graph metadata
    if config.graph_metadata_functions is not None:
        g = annotate_graph_metadata(g, config.graph_metadata_functions)

    # Annotate additional edge metadata
    if config.edge_metadata_functions is not None:
        g = annotate_edge_metadata(g, config.edge_metadata_functions)

    return g


def _mp_graph_constructor(
    args: Tuple[str, str, int], source: str, config: ProteinGraphConfig
) -> Union[nx.Graph, None]:
    """
    Protein graph constructor for use in multiprocessing several protein
    structure graphs.

    :param args: Tuple of pdb code/path and the chain selection for that PDB.
    :type args: Tuple[str, str]
    :param use_pdb_code: Whether we are using ``"pdb_code"``s, ``path``s
        (to PDB or MMTF files) or ``"uniprot_id"``s.
    :type use_pdb_code: bool
    :param config: Protein structure graph construction config
        (see: :class:`graphein.protein.config.ProteinGraphConfig`).
    :type config: ProteinGraphConfig
    :return: Protein structure graph or ``None`` if an error is encountered.
    :rtype: Union[nx.Graph, None]
    """
    log.info(
        f"Constructing graph for: {args[0]}. Chain selection: {args[1]}. \
            Model index: {args[2]}"
    )
    func = partial(construct_graph, config=config)
    try:
        if source == "pdb_code":
            return func(
                pdb_code=args[0], chain_selection=args[1], model_index=args[2]
            )
        elif source == "path":
            return func(
                path=args[0], chain_selection=args[1], model_index=args[2]
            )
        elif source == "uniprot_id":
            return func(
                uniprot_id=args[0],
                chain_selection=args[1],
                model_index=args[2],
            )

    except Exception as ex:
        log.info(
            f"Graph construction error (PDB={args[0]})! \
                {traceback.format_exc()}"
        )
        log.info(ex)
        return None


def construct_graphs_mp(
    pdb_code_it: Optional[List[str]] = None,
    path_it: Optional[List[str]] = None,
    uniprot_id_it: Optional[List[str]] = None,
    chain_selections: Optional[Union[List[List[str]], List[str]]] = None,
    model_indices: Optional[List[str]] = None,
    config: ProteinGraphConfig = ProteinGraphConfig(),
    num_cores: int = 16,
    return_dict: bool = True,
    out_path: Optional[str] = None,
) -> Union[List[nx.Graph], Dict[str, nx.Graph]]:
    """
    Constructs protein graphs for a list of pdb codes or pdb paths using
    multiprocessing.

    :param pdb_code_it: List of pdb codes to use for protein graph construction
    :type pdb_code_it: Optional[List[str]], defaults to ``None``
    :param path_it: List of paths to PDB or MMTF files to use for protein graph
        construction.
    :type path_it: Optional[List[str]], defaults to ``None``
    :param chain_selections: List of chains to select from the protein
        structures (e.g. ``[["A", "B" "C"], ["A"], ["L"], ["C", "D"]...]``).
    :type chain_selections: Optional[List[str]], defaults to ``None``
    :param model_indices: List of model indices to use for protein graph
        construction. Only relevant for structures containing ensembles of
        models.
    :type model_indices: Optional[List[str]], defaults to ``None``
    :param config: ProteinGraphConfig to use.
    :type config: graphein.protein.config.ProteinGraphConfig, defaults to
        default config params.
    :param num_cores: Number of cores to use for multiprocessing. The more the
        merrier.
    :type num_cores: int, defaults to ``16``
    :param return_dict: Whether or not to return a dictionary
        (indexed by pdb codes/paths) or a list of graphs.
    :type return_dict: bool, default to ``True``
    :param out_path: Path to save the graphs to. If ``None``, graphs are not
        saved to disk.
    :type out_path: Optional[str], defaults to ``None``
    :return: Iterable of protein graphs. ``None`` values indicate there was a
        problem in constructing the graph for this particular pdb.
    :rtype: Union[List[nx.Graph], Dict[str, nx.Graph]]
    """
    assert (
        pdb_code_it is not None or path_it is not None
    ), "Iterable of pdb codes, pdb paths or uniprot IDs required."

    if pdb_code_it is not None:
        pdbs = pdb_code_it
        source = "pdb_code"

    if path_it is not None:
        pdbs = path_it
        source = "path"

    if uniprot_id_it is not None:
        pdbs = uniprot_id_it
        source = "uniprot_id"

    if chain_selections is None:
        chain_selections = ["all"] * len(pdbs)

    if model_indices is None:
        model_indices = [1] * len(pdbs)

    constructor = partial(_mp_graph_constructor, source=source, config=config)

    graphs = list(
        process_map(
            constructor,
            [
                (pdb, chain_selections[i], model_indices[i])
                for i, pdb in enumerate(pdbs)
            ],
            max_workers=num_cores,
        )
    )
    if out_path is not None:
        [
            nx.write_gpickle(
                g, str(f"{out_path}/" + f"{g.graph['name']}.pickle")
            )
            for g in graphs
        ]

    if return_dict:
        graphs = {pdb: graphs[i] for i, pdb in enumerate(pdbs)}

    return graphs


def compute_chain_graph(
    g: nx.Graph,
    chain_list: Optional[List[str]] = None,
    remove_self_loops: bool = False,
    return_weighted_graph: bool = False,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Computes a chain-level graph from a protein structure graph.

    This graph features nodes as individual chains in a complex and edges as
    the interactions between constituent nodes in each chain. You have the
    option of returning an unweighted graph (multigraph,
    ``return_weighted_graph=False``) or a weighted graph
    (``return_weighted_graph=True``). The difference between these is the
    unweighted graph features and edge for each interaction between chains
    (ie the number of edges will be equal to the number of edges in the input
    protein structure graph), while the weighted graph sums these interactions
    to a single edge between chains with the counts stored as features.

    :param g: A protein structure graph to compute the chain graph of.
    :type g: nx.Graph
    :param chain_list: A list of chains to extract from the input graph.
        If ``None``, all chains will be used. This is provided as input to
        ``extract_subgraph_from_chains``. Default is ``None``.
    :type chain_list: Optional[List[str]]
    :param remove_self_loops: Whether to remove self-loops from the graph.
        Default is False.
    :type remove_self_loops: bool
    :return: A chain-level graph.
    :rtype: Union[nx.Graph, nx.MultiGraph]
    """
    # If we are extracting specific chains, do it here.
    if chain_list is not None:
        g = extract_subgraph_from_chains(g, chain_list)

    # Initialise new graph with Metadata
    h = nx.MultiGraph()
    h.graph = g.graph
    h.graph["node_type"] = "chain"

    # Set nodes
    nodes_per_chain = {chain: 0 for chain in g.graph["chain_ids"]}
    sequences = {chain: "" for chain in g.graph["chain_ids"]}
    for n, d in g.nodes(data=True):
        nodes_per_chain[d["chain_id"]] += 1
        sequences[d["chain_id"]] += RESI_THREE_TO_1[d["residue_name"]]

    h.add_nodes_from(g.graph["chain_ids"])

    for n, d in h.nodes(data=True):
        d["num_residues"] = nodes_per_chain[n]
        d["sequence"] = sequences[n]

    # Add edges
    for u, v, d in g.edges(data=True):
        h.add_edge(
            g.nodes[u]["chain_id"], g.nodes[v]["chain_id"], kind=d["kind"]
        )
    # Remove self-loops if necessary. Checks for equality between nodes in a
    # given edge.
    if remove_self_loops:
        edges_to_remove: List[Tuple[str]] = [
            (u, v) for u, v in h.edges() if u == v
        ]
        h.remove_edges_from(edges_to_remove)

    # Compute a weighted graph if required.
    if return_weighted_graph:
        return compute_weighted_graph_from_multigraph(h)
    return h


def compute_weighted_graph_from_multigraph(g: nx.MultiGraph) -> nx.Graph:
    """Computes a weighted graph from a multigraph.

    This function is used to convert a multigraph to a weighted graph. The
    weights of the edges are the number of interactions between the nodes.

    :param g: A multigraph.
    :type g: nx.MultiGraph
    :return: A weighted graph.
    :rtype: nx.Graph
    """
    H = nx.Graph()
    H.graph = g.graph
    H.add_nodes_from(g.nodes(data=True))
    for u, v, d in g.edges(data=True):
        if H.has_edge(u, v):
            H[u][v]["weight"] += len(d["kind"])
            H[u][v]["kind"].update(d["kind"])
            for kind in list(d["kind"]):
                try:
                    H[u][v][kind] += 1
                except KeyError:
                    H[u][v][kind] = 1
        else:
            H.add_edge(u, v, weight=len(d["kind"]), kind=d["kind"])
            for kind in list(d["kind"]):
                H[u][v][kind] = 1
    return H


def number_groups_of_runs(list_of_values: List[Any]) -> List[str]:
    """Numbers groups of runs in a list of values.

    E.g. ``["A", "A", "B", "A", "A", "A", "B", "B"] ->
    ["A1", "A1", "B1", "A2", "A2", "A2", "B2", "B2"]``

    :param list_of_values: List of values to number.
    :type list_of_values: List[Any]
    :return: List of numbered values.
    :rtype: List[str]
    """
    df = pd.DataFrame({"val": list_of_values})
    df["idx"] = df["val"].shift() != df["val"]
    df["sum"] = df.groupby("val")["idx"].cumsum(numeric_only=True)
    return list(df["val"].astype(str) + df["sum"].astype(str))


def compute_secondary_structure_graph(
    g: nx.Graph,
    allowable_ss_elements: Optional[List[str]] = None,
    remove_non_ss: bool = True,
    remove_self_loops: bool = False,
    return_weighted_graph: bool = False,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Computes a secondary structure graph from a protein structure graph.

    :param g: A protein structure graph to compute the secondary structure
        graph of.
    :type g: nx.Graph
    :param remove_non_ss: Whether to remove non-secondary structure nodes from
        the graph. These are denoted as ``"-"`` by DSSP. Default is True.
    :type remove_non_ss: bool
    :param remove_self_loops: Whether to remove self-loops from the graph.
        Default is ``False``.
    :type remove_self_loops: bool
    :param return_weighted_graph: Whether to return a weighted graph.
        Default is False.
    :type return_weighted_graph: bool
    :raises ProteinGraphConfigurationError: If the protein structure graph is
        not configured correctly with secondary structure assignments on all
        nodes.
    :return: A secondary structure graph.
    :rtype: Union[nx.Graph, nx.MultiGraph]
    """
    # Initialise list of secondary structure elements we use to build the graph
    ss_list: List[str] = []

    # Check nodes have secondary structure assignment & store them in list
    for _, d in g.nodes(data=True):
        if "ss" not in d.keys():
            raise ProteinGraphConfigurationError(
                "Secondary structure not defined for all nodes."
            )
        ss_list.append(d["ss"])

    # Number SS elements
    ss_list: pd.Series = pd.Series(number_groups_of_runs(ss_list))
    ss_list.index = list(g.nodes())

    # Remove unstructured elements if necessary
    if remove_non_ss:
        ss_list = ss_list[~ss_list.str.contains("-")]
    # Subset to only allowable SS elements if necessary
    if allowable_ss_elements:
        ss_list = ss_list[
            ss_list.str.contains("|".join(allowable_ss_elements))
        ]

    constituent_residues: Dict[str, List[str]] = ss_list.index.groupby(
        ss_list.values
    )
    constituent_residues = {
        k: list(v) for k, v in constituent_residues.items()
    }
    residue_counts: Dict[str, int] = ss_list.groupby(ss_list).count().to_dict()

    # Add Nodes from secondary structure list
    h = nx.MultiGraph()
    h.add_nodes_from(ss_list)
    nx.set_node_attributes(h, residue_counts, "residue_counts")
    nx.set_node_attributes(h, constituent_residues, "constituent_residues")
    # Assign ss
    for n, d in h.nodes(data=True):
        d["ss"] = n[0]

    # Add graph-level metadata
    h.graph = g.graph
    h.graph["node_type"] = "secondary_structure"

    # Iterate over edges in source graph and add SS-SS edges to new graph.
    for u, v, d in g.edges(data=True):
        try:
            h.add_edge(
                ss_list[u], ss_list[v], kind=d["kind"], source=f"{u}_{v}"
            )
        except KeyError as e:
            log.debug(
                f"Edge {u}-{v} not added to secondary structure graph. \
                Reason: {e} not in graph"
            )

    # Remove self-loops if necessary.
    # Checks for equality between nodes in a given edge.
    if remove_self_loops:
        edges_to_remove: List[Tuple[str]] = [
            (u, v) for u, v in h.edges() if u == v
        ]
        h.remove_edges_from(edges_to_remove)

    # Create weighted graph from h
    if return_weighted_graph:
        return compute_weighted_graph_from_multigraph(h)
    return h


def compute_line_graph(g: nx.Graph, repopulate_data: bool = True) -> nx.Graph:
    """Computes the line graph of a graph.

    The line graph of a graph G has a node for each edge in G and an edge
    joining those nodes if the two edges in G share a common node. For directed
    graphs, nodes are adjacent exactly when the edges they represent form a
    directed path of length two.

    The nodes of the line graph are 2-tuples of nodes in the original graph (or
    3-tuples for multigraphs, with the key of the edge as the third element).

    :param g: Graph to compute the line graph of.
    :type g: nx.Graph
    :param repopulate_data: Whether or not to map node and edge data to edges
        and nodes of the line graph, defaults to True
    :type repopulate_data: bool, optional
    :return: Line graph of g.
    :rtype: nx.Graph
    """
    l_g = nx.generators.line_graph(g)
    l_g.graph = g.graph

    if repopulate_data:
        source_edge_data = {(u, v): d for u, v, d in g.edges(data=True)}
        nx.set_node_attributes(l_g, source_edge_data)

        node_list = {}
        for u, v, d in l_g.edges(data=True):
            node_union = u + v
            for n in node_union:
                if node_union.count(n) > 1:
                    node_list[(u, v)] = n
                    break

        source_node_data = {k: g.nodes[v] for k, v in node_list.items()}
        nx.set_edge_attributes(l_g, source_node_data)
    return l_g
