from typing import Callable, List

import networkx as nx
import pandas as pd
import numpy as np
import xarray as xr


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


def generate_feature_dataframe(
    G: nx.Graph,
    funcs: List[Callable],
    return_array=False,
) -> pd.DataFrame:
    """
    Return a pandas DataFrame representation of node metadata.

    `funcs` has to be list of callables whose signature is

        f(n, d) -> pd.Series

    where `n` is the graph node,
    `d` is the node metadata dictionary.
    The function must return a pandas Series whose name is the node.

    Example function:

    ```python
    def x_vec(n: Hashable, d: Dict[Hashable, Any]) -> pd.Series:
        return pd.Series({"x_coord": d["x_coord"]}, name=n)
    ```

    One fairly strong assumption is that each func
    has all the information it needs to act
    stored on the metadata dictionary.
    If you need to reference an external piece of information,
    such as a dictionary to look up values,
    set up the function to accept the dictionary,
    and use `functools.partial`
    to "reduce" the function signature to just `(n, d)`.
    An example below:

    ```python
    from functools import partial
    def get_molweight(n, d, mw_dict):
        return pd.Series({"mw": mw_dict[d["amino_acid"]]}, name=n)

    mw_dict = {"PHE": 165, "GLY": 75, ...}
    get_molweight_func = partial(get_molweight, mw_dict=mw_dict)

    generate_feature_dataframe(G, [get_molweight_func])
    ```

    The `name=n` piece is important;
    the `name` becomes the row index in the resulting dataframe.

    The series that is returned from each function
    need not only contain one key-value pair.
    You can have two or more, and that's completely fine;
    each key becomes a column in the resulting dataframe.

    A key design choice: We default to returning DataFrames,
    to make inspecting the data easy,
    but for consumption in tensor libraries,
    you can turn on returning a NumPy array
    by switching `return_array` to True.

    ## Parameters

    - `G`: A NetworkX-compatible graph object.
    - `funcs`: A list of functions.
    - `return_array`: Whether or not to return
        a NumPy array version of the data.
        Useful for consumption in tensor libs, like PyTorch or JAX.

    ## Returns

    - A pandas DataFrame.
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
    off a graph object G.
    For example, in defining a func:

    ```python
    def my_adj_matrix_func(G):
        adj = some_adj_func(G)
        return format_adjacency(G, adj, "xarray_coord_name")
    ```

    ## Assumptions

    1. `adj` should be a 2D matrix of shape (n_nodes, n_nodes)
    1. `name` is something that is unique amongst all names used
    in the final adjacency tensor.

    ## Parameters

    - `G`: NetworkX-compatible Graph
    - `adj`: 2D numpy array
    - `name`: A unique name for the kind of adjacency matrix
        being constructed.
        Gets used in xarray as a coordinate in the "name" dimension.

    ## Returns

    - An XArray DataArray of shape (n_nodes, n_nodes, 1)
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

    Uses the collection of functions in `funcs`
    to build an xarray DataArray
    that houses the resulting "adjacency tensor".

    A key design choice:
    We default to returning xarray DataArrays,
    to make inspecting the data easy,
    but for consumption in tensor libraries,
    you can turn on returning a NumPy array
    by switching `return_array` to True.

    ## Parameters

    - G: NetworkX Graph.
    - funcs: A list of functions that take in G
        and return an xr.DataArray

    ## Returns
    - xr.DataArray,
        which is of shape (n_nodes, n_nodes, n_funcs).
    """
    mats = []
    for func in funcs:
        mats.append(func(G))
    da = xr.concat(mats, dim="name")
    if return_array:
        return da.data
    return da
