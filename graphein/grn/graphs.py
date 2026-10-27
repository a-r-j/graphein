"""Graph construction utilities for Gene Regulatory Networks."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Ramon Vinas
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import Callable, List, Optional

import networkx as nx
from loguru import logger as log

from graphein.grn.config import GRNGraphConfig
from graphein.utils.utils import (
    annotate_edge_metadata,
    annotate_graph_metadata,
    annotate_node_metadata,
    compute_edges,
)

EDGE_COLOR_MAPPING = {"trrust": "r", "regnetwork": "b", "abasy": "g"}


def parse_kwargs_from_config(config: GRNGraphConfig) -> GRNGraphConfig:
    """
    If configs for specific dataset are provided in the Global GRNGraphConfig, we update the kwargs

    :param config: GRN graph configuration object.
    :type config: graphein.grn.GRNGraphConfig
    :return: config with updated config.kwargs
    :rtype: graphein.grn.GRNGraphConfig
    """
    if config.trrust_config.kwargs is not None:
        trrust_config_dict = {
            "TRRUST_" + k: v
            for k, v in dict(config.trrust_config.kwargs.items())
        }
        config.kwargs = config.kwargs.update(trrust_config_dict)

    if config.regnetwork_config.kwargs is not None:
        regnetwork_config_dict = {
            "RegNetwork_" + k: v
            for k, v in dict(config.regnetwork_config.kwargs.items())
        }
        config.kwargs = config.kwargs.update(regnetwork_config_dict)
    return config


def compute_grn_graph(
    gene_list: List[str],
    edge_construction_funcs: List[Callable],
    graph_annotation_funcs: Optional[List[Callable]] = None,
    node_annotation_funcs: Optional[List[Callable]] = None,
    edge_annotation_funcs: Optional[List[Callable]] = None,
    config: Optional[GRNGraphConfig] = None,
) -> nx.Graph:
    """
    Computes a Gene Regulatory Network Graph from a list of gene IDs

    :param gene_list: List of gene identifiers
    :type gene_list: List[str]
    :param edge_construction_funcs:  List of functions to construct edges with
    :type edge_construction_funcs: List[Callable]
    :param graph_annotation_funcs: List of functions functools annotate graph metadata, defaults to None
    :type graph_annotation_funcs: List[Callable], optional
    :param node_annotation_funcs: List of functions to annotate node metadata, defaults to None
    :type node_annotation_funcs: List[Callable], optional
    :param edge_annotation_funcs: List of functions to annotate edge metadata, defaults to None
    :type edge_annotation_funcs: List[Callable], optional
    :param config: Config specifying additional parameters for STRING and BIOGRID, defaults to None
    :type config: graphein.grn.GRNGraphConfig, optional
    :return: nx.Graph of PPI network
    :rtype: nx.Graph
    """

    # Load default config if none supplied
    if config is None:
        config = GRNGraphConfig()

    # Parse kwargs from config
    config = parse_kwargs_from_config(config)

    # Create *directed* graph and add genes as nodes
    G = nx.DiGraph(
        gene_list=gene_list,
        sources=[],
        # ncbi_taxon_id=config.ncbi_taxon_id,
    )
    G.add_nodes_from(gene_list)
    log.debug(f"Added {len(gene_list)} nodes to graph")

    nx.set_node_attributes(
        G,
        dict(zip(gene_list, gene_list)),
        "gene_id",
    )

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
