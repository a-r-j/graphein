"""Functions for constructing a PPI PPIGraphConfig from STRINGdb and BIOGRID."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Ramon Vinas
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import Callable, List, Optional

import networkx as nx
from loguru import logger as log

from graphein.ppi.config import PPIGraphConfig
from graphein.utils.utils import (
    annotate_edge_metadata,
    annotate_graph_metadata,
    annotate_node_metadata,
    compute_edges,
)

EDGE_COLOR_MAPPING = {"string": "r", "biogrid": "b"}


def parse_kwargs_from_config(config: PPIGraphConfig) -> PPIGraphConfig:
    """
    If configs for STRING and BIOGRID are provided in the Global
        :ref:`~graphein.ppi.config.PPIGraphConfig`, we update the kwargs

    :param config: PPI graph configuration object.
    :type config: PPIGraphConfig
    :return: config with updated config.kwargs
    :rtype: PPIGraphConfig
    """
    if config.string_config is not None:
        string_config_dict = {
            f"STRING_{k}": v for k, v in dict(config.string_config.items())
        }

        config.kwargs = config.kwargs.update(string_config_dict)

    if config.biogrid_config is not None:
        biogrid_config_dict = {
            f"BIOGRID_{k}": v for k, v in dict(config.biogrid_config.items())
        }

        config.kwargs = config.kwargs.update(biogrid_config_dict)
    return config


def compute_ppi_graph(
    protein_list: List[str],
    edge_construction_funcs: List[Callable],
    graph_annotation_funcs: Optional[List[Callable]] = None,
    node_annotation_funcs: Optional[List[Callable]] = None,
    edge_annotation_funcs: Optional[List[Callable]] = None,
    config: Optional[PPIGraphConfig] = None,
) -> nx.Graph:
    """
    Computes a PPI Graph from a list of protein IDs. This is the core function
        for PPI graph construction.

    :param protein_list: List of protein identifiers
    :type protein_list: List[str]
    :param edge_construction_funcs:  List of functions to construct edges with
    :type edge_construction_funcs: List[Callable], optional
    :param graph_annotation_funcs: List of functions to annotate graph metadata
    :type graph_annotation_funcs: List[Callable], optional
    :param node_annotation_funcs: List of functions to annotate node metadata
    :type node_annotation_funcs: List[Callable], optional
    :param edge_annotation_funcs: List of function to annotate edge metadata
    :type edge_annotation_funcs: List[Callable], optional
    :param config: Config object specifying additional parameters for STRING
        and BIOGRID API calls
    :type config: PPIGraphConfig, optional
    :return: ``nx.Graph`` of PPI network
    :rtype: nx.Graph
    """

    # Load default config if none supplied
    if config is None:
        config = PPIGraphConfig()

    # Parse kwargs from config
    config = parse_kwargs_from_config(config)

    # Create graph and add proteins as nodes
    G = nx.Graph(
        protein_list=protein_list,
        sources=[],
        ncbi_taxon_id=config.ncbi_taxon_id,
    )
    G.add_nodes_from(protein_list)
    log.debug(f"Added {len(protein_list)} nodes to graph")

    nx.set_node_attributes(
        G,
        dict(zip(protein_list, protein_list)),
        "protein_id",
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


if __name__ == "__main__":
    from functools import partial

    import matplotlib.pyplot as plt

    from graphein.ppi.edges import add_biogrid_edges, add_string_edges
    from graphein.ppi.features.node_features import add_sequence_to_nodes
    from graphein.protein.features.sequence.sequence import molecular_weight

    protein_list = [
        "CDC42",
        "CDK1",
        "KIF23",
        "PLK1",
        "RAC2",
        "RACGAP1",
        "RHOA",
        "RHOB",
    ]

    config = PPIGraphConfig()
    kwargs = config.kwargs

    g = compute_ppi_graph(
        protein_list=protein_list,
        edge_construction_funcs=[
            partial(add_string_edges, kwargs=kwargs),
            partial(add_biogrid_edges, kwargs=kwargs),
        ],
        node_annotation_funcs=[add_sequence_to_nodes, molecular_weight],
    )

    edge_colors = [
        (
            "r"
            if g[u][v]["kind"] == {"string"}
            else "b" if g[u][v]["kind"] == {"biogrid"} else "y"
        )
        for u, v in g.edges()
    ]

    print(g.nodes())

    print(nx.info(g))
    nx.draw(g, with_labels=True, edge_color=edge_colors)
    plt.show()
