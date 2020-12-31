from typing import Callable, List

import networkx as nx


def onek_encoding_unk(x, allowable_set):
    """
    Function for one hot encoding
    :param x: value to one-hot
    :param allowable_set: set of options to encode
    :return: one-hot encoding as torch tensor
    """
    # if x not in allowable_set:
    #    x = allowable_set[-1]
    return [x == s for s in allowable_set]


def annotate_graph_metadata(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    """
    Annotates graph with graph-level metadata

    :param G: Graph on which to add graph-level metadata to
    :param funcs: List of graph metadata annotation functions
    :return: Graph on which with node metadata added
    """
    for func in funcs:
        func(G)
    return G


def annotate_edge_metadata(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    """
    Annotates Graph edges with edge metadata
    :param G: Graph to add edge metadata to
    :param funcs: List of edge metadata annotation functions
    :return: Graph with edge metadata added
    """
    for func in funcs:
        for u, v, d in G.edges(data=True):
            func(u, v, d)
    return G


def annotate_node_metadata(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    """
    Annotates nodes with metadata
    :param G: Graph to add node metadata to
    :param funcs: List of node metadata annotation functions
    :return: Graph with node metadata added
    """
    for func in funcs:
        for n, d in G.nodes(data=True):
            func(n, d)
    return G


def compute_edges(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    """
    Computes edges for an Graph from a list of edge construction functions

    :param G: Graph to add features to
    :param funcs: List of edge construction functions
    :return: Graph with edges added
    """
    for func in funcs:
        func(G)
    return G
