# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Ramon Vinas
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import logging

import networkx as nx
import pandas as pd

from graphein.grn.parse_regnetwork import RegNetwork_df
from graphein.grn.parse_trrust import TRRUST_df

log = logging.getLogger(__name__)


def add_trrust_edges(G: nx.Graph, kwargs) -> nx.Graph:
    """
    Adds edges from TRRUST to GRNGraph
    :param G: Graph to edges to (populated with gene_id nodes)
    :param kwargs:  Additional parameters to pass to TRRUST
    :return: nx.Graph GRNGraph with TRRUST regulatory interactions added as edges
    """
    G.graph["sources"].append("trrust")
    G.graph["trrust_df"] = TRRUST_df(
        G.graph["gene_list"],
        kwargs=kwargs,
    )
    add_interacting_genes(G, df=G.graph["trrust_df"], kind="trrust")

    return G


def add_regnetwork_edges(G: nx.Graph, kwargs) -> nx.Graph:
    """
    Adds edges from RegNetwork to GRNGraph
    :param G: Graph to edges to (populated with gene_id nodes)
    :param kwargs:  Additional parameters to pass to RegNetwork
    :return: nx.Graph GRNGraph with RegNetwork regulatory interactions added as edges
    """
    G.graph["sources"].append("regnetwork")
    G.graph["regnetwork_df"] = RegNetwork_df(
        G.graph["gene_list"],
        kwargs=kwargs,
    )
    add_interacting_genes(G, df=G.graph["regnetwork_df"], kind="regnetwork")

    return G


def add_interacting_genes(
    G: nx.Graph, df: pd.DataFrame, kind: str
) -> nx.Graph:
    """
    Generic function for adding interaction edges to GRNGraph
    :param G: GRNGraph to populate with edges
    :param df: Dataframe containing edgelist
    :param kind: name of interaction type
    """

    gene_1 = df["g1"].values
    gene_2 = df["g2"].values
    reg_type = df["regtype"].values

    interacting_genes = set(list(zip(gene_1, gene_2, reg_type)))

    for g1, g2, r in interacting_genes:
        if G.has_edge(g1, g2):
            G.edges[g1, g2]["kind"].add(kind)
            G.edges[g1, g2]["regtype"].add(r)
        else:
            G.add_edge(g1, g2, kind={kind}, regtype={r})
    log.debug(f"Added {len(df)} {kind} interaction edges")
