"""Utilities for working with graph objects"""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any, Callable, Iterable, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from Bio.Data.IUPACData import protein_letters_3to1


def onek_encoding_unk(
    x: Iterable[Any], allowable_set: List[Any]
) -> List[bool]:
    """
    Function for perfroming one hot encoding

    :param x: values to one-hot
    :type x: Iterable[Any]
    :param allowable_set: set of options to encode
    :type allowable_set: List[Any]
    :return: one-hot encoding as list
    :rtype: List[bool]
    """
    # if x not in allowable_set:
    #    x = allowable_set[-1]
    return [x == s for s in allowable_set]


def filter_dataframe(df: pd.DataFrame, funcs: List[Callable]) -> pd.DataFrame:
    """
    Applies transformation functions to a dataframe. Each function in ``funcs`` must accept a ``pd.DataFrame`` and return a ``pd.DataFrame``.

    Additional parameters can be provided by using partial functions.

    :param df: Dataframe to apply transformations to.
    :type df: pd.DataFrame
    :param funcs: List of transformation functions.
    :type funcs: List[Callable]
    :rtype: nx.Graph
    """
    for func in funcs:
        func(df)
    return df


def annotate_graph_metadata(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    """
    Annotates graph with graph-level metadata

    :param G: Graph on which to add graph-level metadata to
    :type G: nx.Graph
    :param funcs: List of graph metadata annotation functions
    :type funcs: List[Callable]
    :return: Graph on which with node metadata added
    :rtype: nx.Graph
    """
    for func in funcs:
        func(G)
    return G


def annotate_edge_metadata(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    """
    Annotates Graph edges with edge metadata. Each function in ``funcs`` must take the three arguments ``u``, ``v`` and ``d``, where ``u`` and ``v`` are the nodes of the edge, and ``d`` is the edge data dictionary.

    Additional parameters can be provided by using partial functions.

    :param G: Graph to add edge metadata to
    :type G: nx.Graph
    :param funcs: List of edge metadata annotation functions
    :type funcs: List[Callable]
    :return: Graph with edge metadata added
    :rtype: nx.Graph
    """
    for func in funcs:
        for u, v, d in G.edges(data=True):
            func(u, v, d)
    return G


def annotate_node_metadata(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    """
    Annotates nodes with metadata. Each function in ``funcs`` must take two arguments ``n`` and ``d``, where ``n`` is the node and ``d`` is the node data dictionary.

    Additional parameters can be provided by using partial functions.

    :param G: Graph to add node metadata to
    :type G: nx.Graph
    :param funcs: List of node metadata annotation functions
    :type funcs: List[Callable]
    :return: Graph with node metadata added
    :rtype: nx.Graph
    """

    for func in funcs:
        for n, d in G.nodes(data=True):
            func(n, d)
    return G


def annotate_node_features(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    """
    Annotates nodes with features data. Note: passes whole graph to function.

    :param G: Graph to add node features to
    :type G: nx.Graph
    :param funcs: List of node feature annotation functions
    :type funcs: List[Callable]
    :return: Graph with node features added
    :rtype: nx.Graph
    """
    for func in funcs:
        func(G)
    return G


def compute_edges(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    """
    Computes edges for an Graph from a list of edge construction functions. Each func in ``funcs`` must take an ``nx.Graph`` and return an ``nx.Graph``.

    :param G: Graph to add features to
    :type G: nx.Graph
    :param funcs: List of edge construction functions
    :type funcs: List[Callable]
    :return: Graph with edges added
    :rtype: nx.Graph
    """
    for func in funcs:
        func(G)
    return G


def generate_feature_dataframe(
    G: nx.Graph,
    funcs: List[Callable],
    return_array=False,
) -> pd.DataFrame:
    """
    Return a pandas DataFrame representation of node metadata.

    ``funcs`` has to be list of callables whose signature is

        f(n, d) -> pd.Series

    where ``n`` is the graph node,
    ``d`` is the node metadata dictionary.
    The function must return a pandas Series whose name is the node.

    Example function:

    .. code-block:: python

        def x_vec(n: Hashable, d: Dict[Hashable, Any]) -> pd.Series:
            return pd.Series({"x_coord": d["x_coord"]}, name=n)

    One fairly strong assumption is that each func
    has all the information it needs to act
    stored on the metadata dictionary.
    If you need to reference an external piece of information,
    such as a dictionary to look up values,
    set up the function to accept the dictionary,
    and use ``functools.partial``
    to "reduce" the function signature to just ``(n, d)``.
    An example below:

    .. code-block:: python

        from functools import partial
        def get_molweight(n, d, mw_dict):
            return pd.Series({"mw": mw_dict[d["amino_acid"]]}, name=n)

        mw_dict = {"PHE": 165, "GLY": 75, ...}
        get_molweight_func = partial(get_molweight, mw_dict=mw_dict)

        generate_feature_dataframe(G, [get_molweight_func])

    The ``name=n`` piece is important;
    the ``name`` becomes the row index in the resulting dataframe.

    The series that is returned from each function
    need not only contain one key-value pair.
    You can have two or more, and that's completely fine;
    each key becomes a column in the resulting dataframe.

    A key design choice: We default to returning DataFrames,
    to make inspecting the data easy,
    but for consumption in tensor libraries,
    you can turn on returning a NumPy array
    by switching ``return_array=True``.


    :param G: A NetworkX-compatible graph object.
    :type G: nx.Graph
    :param funcs: A list of functions.
    :type funcs: List[Callable]
    :param return_array: Whether or not to return
        a NumPy array version of the data.
        Useful for consumption in tensor libs, like PyTorch or JAX.
    :type return_array: bool
    :return: pandas DataFrame representation of node metadata.
    :rtype: pd.DataFrame
    """
    matrix = []
    for n, d in G.nodes(data=True):
        series = []
        for func in funcs:
            res = func(n, d)
            if res.name != n:
                raise NameError(
                    f"function {func.__name__} returns a series "
                    "that is not named after the node."
                )
            series.append(res)
        matrix.append(pd.concat(series))

    df = pd.DataFrame(matrix)
    if return_array:
        return df.values
    return df


def format_adjacency(G: nx.Graph, adj: np.ndarray, name: str) -> xr.DataArray:
    """
    Format adjacency matrix nicely.

    Intended to be used when computing an adjacency-like matrix
    of a graph object ``G``.
    For example, in defining a func:

    .. code-block:: python

        def my_adj_matrix_func(G):
            adj = some_adj_func(G)
            return format_adjacency(G, adj, "xarray_coord_name")

    **Assumptions**

    #. ``adj`` should be a 2D matrix of shape ``(n_nodes, n_nodes)``
    #. ``name`` is something that is unique amongst all names used
    in the final adjacency tensor.

    :param G: NetworkX-compatible Graph
    :type param: nx.Graph
    :param adj: 2D numpy array of shape ``(n_nodes, n_nodes)``
    :type adj: np.ndarray
    :param name: A unique name for the kind of adjacency matrix
        being constructed.
        Gets used in xarray as a coordinate in the ``"name"`` dimension.
    :type name: str
    :returns: An XArray DataArray of shape ``(n_nodes, n_nodes, 1)``
    :rtype: xr.DataArray
    """
    expected_shape = (len(G), len(G))
    if adj.shape != expected_shape:
        raise ValueError(
            "Adjacency matrix is not shaped correctly, "
            f"should be of shape {expected_shape}, "
            f"instead got shape {adj.shape}."
        )
    adj = np.expand_dims(adj, axis=-1)
    nodes = list(G.nodes())
    return xr.DataArray(
        adj,
        dims=["n1", "n2", "name"],
        coords={"n1": nodes, "n2": nodes, "name": [name]},
    )


def generate_adjacency_tensor(
    G: nx.Graph, funcs: List[Callable], return_array=False
) -> xr.DataArray:
    """
    Generate adjacency tensor for a graph.

    Uses the collection of functions in ``funcs``
    to build an xarray DataArray
    that houses the resulting "adjacency tensor".

    A key design choice:
    We default to returning xarray DataArrays,
    to make inspecting the data easy,
    but for consumption in tensor libraries,
    you can turn on returning a NumPy array
    by switching ``return_array=True``.

    :param G: NetworkX Graph.
    :type G: nx.Graph
    :param funcs: A list of functions that take in G
        and return an xr.DataArray
    :type funcs: List[Callable]
    :returns: xr.DataArray,
        which is of shape ``(n_nodes, n_nodes, n_funcs)``.
    :rtype: xr.DataArray
    """
    mats = [func(G) for func in funcs]
    da = xr.concat(mats, dim="name")
    if return_array:
        return da.data
    return da


def protein_letters_3to1_all_caps(amino_acid: str) -> str:
    """
    Converts capitalised 3 letter amino acid code to single letter. Not provided in default biopython.

    :param amino_acid: Capitalised 3-letter amino acid code (eg. ``"GLY"``)
    :type amino_acid: str
    :returns: Single-letter amino acid code
    :rtype: str
    """
    amino_acid = amino_acid[0] + amino_acid[1:].lower()
    return protein_letters_3to1[amino_acid]


def import_message(
    submodule: str,
    package: str,
    conda_channel: Optional[str] = None,
    pip_install: bool = False,
):
    """
    Return warning if package is not found.
    Generic message for indicating to the user when a function relies on an
    optional module / package that is not currently installed. Includes
    installation instructions. Typically used in conjunction without optional featurisation libraries

    :param submodule: graphein submodule that needs an external dependency.
    :type submodule: str
    :param package: External package this submodule relies on.
    :type package: str
    :param conda_channel: Conda channel package can be installed from, if at all. Defaults to None
    :type conda_channel: str, optional
    :param pip_install: Whether package can be installed via pip. Defaults to False
    :type pip_install: bool
    """
    is_conda = os.path.exists(os.path.join(sys.prefix, "conda-meta"))
    installable = True
    if is_conda:
        if conda_channel is None:
            installable = False
            installation = f"{package} cannot be installed via conda"
        else:
            installation = f"conda install -c {conda_channel} {package}"
    elif pip_install:
        installation = f"pip install {package}"
    else:
        installable = False
        installation = f"{package} cannot be installed via pip"

    print(
        f"To use the Graphein submodule {submodule}, you need to install "
        f"{package}."
    )
    print()
    if installable:
        print("To do so, use the following command:")
        print()
        print(f"    {installation}")
    else:
        print(f"{installation}")


def ping(host: str) -> bool:
    """
    Returns ``True`` if host (str) responds to a ping request.
    Remember that a host may not respond to a ping (ICMP) request even if the host name is valid.

    :param host: IP or hostname
    :type host: str
    :returns: True if host responds to a ping request.
    :rtype: bool
    """

    # Option for the number of packets as a function of
    param = "-n" if platform.system().lower() == "windows" else "-c"

    # Building the command. Ex: "ping -c 1 google.com"
    command = ["ping", param, "1", host]

    return subprocess.call(command) == 0
