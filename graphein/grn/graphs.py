# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Ramon Vinas
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import logging
from typing import Callable, List, Optional

import networkx as nx

from graphein.grn.config import GRNGraphConfig
from graphein.utils import (
    annotate_edge_metadata,
    annotate_graph_metadata,
    annotate_node_metadata,
    compute_edges,
)

log = logging.getLogger(__name__)


EDGE_COLOR_MAPPING = {"trrust": "r", "regnetwork": "b", "abasy": "g"}


def parse_kwargs_from_config(config: GRNGraphConfig) -> GRNGraphConfig:
    """
    If configs for specific dataset are provided in the Global GRNGraphConfig, we update the kwargs
    :param config: GRN graph configuration object.
    :return: config with updated config.kwargs
    """
    if config.trrust_config is not None:
        trrust_config_dict = {
            "TRRUST_" + k: v for k, v in dict(config.trrust_config.items())
        }
        config.kwargs = config.kwargs.update(trrust_config_dict)

    if config.regnetwork_config is not None:
        regnetwork_config_dict = {
            "RegNetwork_" + k: v
            for k, v in dict(config.regnetwork_config.items())
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
    Computes a GRN Graph from a list of gene IDs
    :param gene_list: List of gene identifiers
    :param edge_construction_funcs:  List of functions to construct edges with
    :param graph_annotation_funcs: List of functions functools annotate graph metadata
    :param node_annotation_funcs: List of functions to annotate node metadata
    :param edge_annotation_funcs: List of functions to annotate edge metadata
    :param config: Config specifying additional parameters for STRING and BIOGRID
    :return: nx.Graph of PPI network
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


if __name__ == "__main__":
    from functools import partial

    import matplotlib.pyplot as plt

    from graphein.grn.edges import add_trrust_edges, add_regnetwork_edges

    gene_list = ["AATF", "MYC", "USF1", "SP1", "TP53", "DUSP1"]

    config = GRNGraphConfig()
    kwargs = config.kwargs

    def edge_ann_fn(u, v, d):
        if "+" in d["regtype"]:
            d["regtype"] = "+"
        elif "-" in d["regtype"]:
            d["regtype"] = "-"
        elif "?" in d["regtype"]:
            d["regtype"] = "?"

    g = compute_grn_graph(
        gene_list=gene_list,
        edge_construction_funcs=[
            partial(add_trrust_edges, kwargs=kwargs),
            partial(add_regnetwork_edges, kwargs=kwargs),
        ],
        # node_annotation_funcs=[add_sequence_to_nodes, molecular_weight],
        edge_annotation_funcs=[edge_ann_fn],
    )
    print(g.edges(data=True))

    edge_colors = [
        "r"
        if g[u][v]["kind"] == {"trrust"}
        else "b"
        if g[u][v]["kind"] == {"regnetwork"}
        else "y"
        for u, v in g.edges()
    ]

    print(nx.info(g))

    pos = nx.spring_layout(g)
    nx.draw(g, pos=pos, with_labels=True, edge_color=edge_colors)
    edge_labels = {(u, v): g[u][v]["regtype"] for u, v in g.edges}
    nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=edge_labels)
    plt.show()
