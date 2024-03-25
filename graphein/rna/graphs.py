"""Functions for working with RNA Secondary Structure Graphs."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Emmanuele Rossi, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
# This submodule is heavily inspired by: https://github.com/emalgorithm/rna-design/blob/aec77a18abe4850958d6736ec185a6f8cbfdf20c/src/util.py#L9

import os
from typing import Callable, List, Optional, Union

import networkx as nx
from loguru import logger as log

import graphein.protein.graphs as gp
from graphein.rna.config import BpRNAConfig, RNAGraphConfig
from graphein.rna.constants import (
    RNA_BASE_COLORS,
    RNA_BASES,
    SUPPORTED_DOTBRACKET_NOTATION,
)
from graphein.rna.nussinov import nussinov
from graphein.rna.utils import read_dbn_file
from graphein.utils.utils import (
    annotate_edge_metadata,
    annotate_graph_metadata,
    annotate_node_metadata,
    compute_edges,
)


def validate_rna_sequence(s: str) -> None:
    """
    Validate RNA sequence. This ensures that it only contains supported bases.

    Supported bases are: ``"A", "U", "G", "C", "I"``
    Supported bases can be accessed in
        :const:`~graphein.rna.constants.RNA_BASES`

    :param s: Sequence to validate
    :type s: str
    :raises ValueError: Raises ValueError if the sequence contains an
        unsupported base character
    """
    letters_used = set(s)
    if not letters_used.issubset(RNA_BASES):
        offending_letter = letters_used.difference(RNA_BASES)
        position = s.index(offending_letter)
        raise ValueError(
            f"Invalid letter {offending_letter} found at position {position} \
                in the sequence {s}."
        )


def validate_lengths(db: str, seq: str) -> None:
    """
    Check lengths of dotbracket and sequence match.

    :param db: Dotbracket string to check
    :type db: str
    :param seq: RNA nucleotide sequence to check.
    :type seq: str
    :raises ValueError: Raises ``ValueError`` if lengths of dotbracket and
        sequence do not match.
    """
    if len(db) != len(seq):
        raise ValueError(
            f"Length of dotbracket ({len(db)}) does not match length of \
                sequence ({len(seq)})."
        )


def validate_dotbracket(db: str):
    """
    Sanitize dotbracket string. This ensures that it only has supported symbols.

    See: :const:`~graphein.rna.constants.SUPPORTED_DOTBRACKET_NOTATION`

    :param db: Dotbracket notation string
    :type db: str
    :raises ValueError: Raises ValueError if dotbracket notation contains
        unsupported symbols
    """
    chars_used = set(db)
    if not chars_used.issubset(SUPPORTED_DOTBRACKET_NOTATION):
        offending_letter = chars_used.difference(SUPPORTED_DOTBRACKET_NOTATION)
        position = db.index(offending_letter)
        raise ValueError(
            f"Invalid letter {offending_letter} found at position {position} \
                in the sequence {db}."
        )


def construct_rna_graph_3d(
    config: Optional[RNAGraphConfig] = None,
    name: Optional[str] = None,
    path: Optional[Union[str, os.PathLike]] = None,
    pdb_code: Optional[str] = None,
    chain_selection: str = "all",
    model_index: int = 1,
    rna_df_processing_funcs: Optional[List[Callable]] = None,
    edge_construction_funcs: Optional[List[Callable]] = None,
    edge_annotation_funcs: Optional[List[Callable]] = None,
    node_annotation_funcs: Optional[List[Callable]] = None,
    graph_annotation_funcs: Optional[List[Callable]] = None,
) -> nx.Graph:
    """
    Constructs RNA structure graph from a ``pdb_code`` or ``path``.

    Users can provide a :class:`~graphein.rna.config.RNAGraphConfig`
    object to specify construction parameters.

    However, config parameters can be overridden by passing arguments directly
    to the function.

    :param config: :class:`~graphein.rna.config.RNAGraphConfig` object. If
        ``None``, defaults to config in ``graphein.rna.config``.
    :type config: graphein.protein.config.RNAGraphConfig, optional
    :param path: Path to PDB or MMTF to build graph from. Default is
        ``None``.
    :type path: str, optional
    :param pdb_code: 4-character PDB accession pdb_code to build graph from.
        Default is ``None``.
    :type pdb_code: str, optional
    :param chain_selection: String of nucleotide chains to include in graph.
        E.g ``"ABDF"`` or ``"all"``. Default is ``"all"``.
    :type chain_selection: str
    :param model_index: Index of model to use in the case of structural
        ensembles. Default is ``1``.
    :type model_index: int
    :param df_processing_funcs: List of DataFrame processing functions.
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
    :return: RNA Structure Graph
    :type: nx.Graph
    """

    # If no config is provided, use default
    if config is None:
        config = RNAGraphConfig()

    # Get name from pdb_file is no pdb_code is provided
    if path and (pdb_code is None):
        pdb_code = gp.get_protein_name_from_filename(path)

    # If config params are provided, overwrite them
    config.rna_df_processing_functions = (
        rna_df_processing_funcs
        if config.rna_df_processing_functions is None
        else config.rna_df_processing_functions
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

    raw_df = gp.read_pdb_to_dataframe(
        path,
        pdb_code,
        model_index=model_index,
    )

    raw_df = gp.sort_dataframe(raw_df)
    protein_df = gp.process_dataframe(
        raw_df,
        chain_selection=chain_selection,
        granularity=config.granularity,
        insertions=config.insertions,
        keep_hets=config.keep_hets,
    )

    # Initialise graph with metadata
    g = gp.initialise_graph_with_metadata(
        protein_df=protein_df,
        raw_pdb_df=raw_df,
        name=name,
        pdb_code=pdb_code,
        path=path,
        granularity=config.granularity,
    )
    # Add nodes to graph
    g = gp.add_nodes_to_graph(g)

    # Add config to graph
    g.graph["config"] = config

    # Annotate additional node metadata
    if config.node_metadata_functions is not None:
        g = annotate_node_metadata(g, config.node_metadata_functions)

    # Compute graph edges
    g = compute_edges(
        g,
        funcs=config.edge_construction_functions,
    )

    # Annotate additional graph metadata
    if config.graph_metadata_functions is not None:
        g = annotate_graph_metadata(g, config.graph_metadata_functions)

    # Annotate additional edge metadata
    if config.edge_metadata_functions is not None:
        g = annotate_edge_metadata(g, config.edge_metadata_functions)

    return g


def construct_rna_graph_2d(
    dotbracket: Optional[str],
    sequence: Optional[str],
    name: Optional[str] = None,
    bprna_id: Optional[str] = None,
    edge_construction_funcs: Optional[List[Callable]] = None,
    edge_annotation_funcs: Optional[List[Callable]] = None,
    node_annotation_funcs: Optional[List[Callable]] = None,
    graph_annotation_funcs: Optional[List[Callable]] = None,
    use_nussinov: bool = False,
    config: Optional[RNAGraphConfig] = None,
) -> nx.Graph:
    """
    Constructs an RNA secondary structure graph from dotbracket notation.

    :param dotbracket: Dotbracket notation representation of secondary
        structure.
    :type dotbracket: str, optional
    :param sequence: Corresponding sequence RNA bases
    :type sequence: str, optional
    :param bprna_id: bp RNA ID of the RNA secondary structure from which to
        construct the graph. Defaults to ``None``.
    :type bprna_id: str, optional
    :param edge_construction_funcs: List of edge construction functions.
        Defaults to ``None``.
    :type edge_construction_funcs: List[Callable], optional
    :param edge_annotation_funcs: List of edge metadata annotation functions.
        Defaults to ``None``.
    :type edge_annotation_funcs: List[Callable], optional
    :param node_annotation_funcs: List of node metadata annotation functions.
        Defaults to ``None``.
    :type node_annotation_funcs: List[Callable], optional
    :param graph_annotation_funcs: List of graph metadata annotation functions.
        Defaults to ``None``.
    :type graph_annotation_funcs: List[Callable], optional
    :param config: BpRNA Configuration object. Defaults to ``None``.
        Unused unless using a bpRNA to compute a graph.
    :type config: BpRNAConfig, optional
    :return: nx.Graph of RNA secondary structure
    :rtype: nx.Graph
    """
    G = nx.Graph(name=name)

    if bprna_id is not None:
        if config is None:
            config = BpRNAConfig()
        sequence, dotbracket = read_dbn_file(config.path)

    # Build node IDs first.
    node_ids = (
        list(range(len(sequence)))
        if sequence
        else list(range(len(dotbracket)))
    )

    if use_nussinov:
        dotbracket = nussinov(sequence)

    # Check sequence and dotbracket lengths match
    if dotbracket and sequence:
        validate_lengths(dotbracket, sequence)

    # add nodes
    G.add_nodes_from(node_ids)
    log.debug(f"Added {len(node_ids)} nodes")

    # Add dotbracket symbol if dotbracket is provided
    if dotbracket:
        validate_dotbracket(dotbracket)
        G.graph["dotbracket"] = dotbracket

        nx.set_node_attributes(
            G,
            dict(zip(node_ids, dotbracket)),
            "dotbracket_symbol",
        )

    # Add nucleotide base info if sequence is provided
    if sequence:
        validate_rna_sequence(sequence)
        G.graph["sequence"] = sequence
        nx.set_node_attributes(G, dict(zip(node_ids, sequence)), "nucleotide")
        colors = [RNA_BASE_COLORS[i] for i in sequence]
        nx.set_node_attributes(G, dict(zip(node_ids, colors)), "color")

    # Annotate additional graph metadata
    if graph_annotation_funcs is not None:
        G = annotate_graph_metadata(G, graph_annotation_funcs)

    # Annotate additional node metadata
    if node_annotation_funcs is not None:
        G = annotate_node_metadata(G, node_annotation_funcs)

    # Add edges
    G = compute_edges(G, edge_construction_funcs)

    # Annotate additional edge metadata
    if edge_annotation_funcs is not None:
        G = annotate_edge_metadata(G, edge_annotation_funcs)

    return G


def construct_graph(
    dotbracket: Optional[str] = None,
    sequence: Optional[str] = None,
    bprna_id: Optional[str] = None,
    use_nussinov: bool = False,
    name: Optional[str] = None,
    config: Optional[RNAGraphConfig] = None,
    path: Optional[str] = None,
    pdb_code: Optional[str] = None,
    chain_selection: str = "all",
    rna_df_processing_funcs: Optional[List[Callable]] = None,
    edge_construction_funcs: Optional[List[Callable]] = None,
    edge_annotation_funcs: Optional[List[Callable]] = None,
    node_annotation_funcs: Optional[List[Callable]] = None,
    graph_annotation_funcs: Optional[List[Callable]] = None,
) -> nx.Graph:
    # TODO Docstring

    if path is not None or pdb_code is not None:
        return construct_rna_graph_3d(
            config=config,
            name=name,
            path=path,
            pdb_code=pdb_code,
            chain_selection=chain_selection,
            rna_df_processing_funcs=rna_df_processing_funcs,
            edge_construction_funcs=edge_construction_funcs,
            edge_annotation_funcs=edge_annotation_funcs,
            node_annotation_funcs=node_annotation_funcs,
            graph_annotation_funcs=graph_annotation_funcs,
        )
    elif (
        dotbracket is not None or sequence is not None or bprna_id is not None
    ):
        return construct_rna_graph_2d(
            dotbracket=dotbracket,
            sequence=sequence,
            name=name,
            bprna_id=bprna_id,
            use_nussinov=use_nussinov,
            edge_construction_funcs=edge_construction_funcs,
            edge_annotation_funcs=edge_annotation_funcs,
            node_annotation_funcs=node_annotation_funcs,
            graph_annotation_funcs=graph_annotation_funcs,
        )
