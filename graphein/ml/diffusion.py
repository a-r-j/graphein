"""Generate diffusion matrices based on NetworkX graph objects.

Author: Arian Jamasb & Eric Ma

Each of the functions here accepts a NetworkX graph object
and returns a 2D xarray.
These arrays can then be stacked to generate a diffusion tensor.
"""

from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

from graphein.protein.edges.distance import compute_distmat
from graphein.utils.utils import format_adjacency, generate_feature_dataframe


def identity_matrix(G: nx.Graph) -> xr.DataArray:
    """Return the identity diffusion matrix.

    This is nothing more than the identity matrix
    with diagonals of ``1``.

    :param G: Graph to retrieve identity diffusion matrix from
    :type G: nx.Graph
    :returns: Identity diffusion matrix
    :rtype: xr.DataArray
    """
    return format_adjacency(G, np.eye(len(G)), "identity")


def adjacency_matrix_power(
    G: nx.Graph,
    amat_kwargs: Dict = {},
    with_identity: bool = True,
    power: float = 1,
) -> xr.DataArray:
    """
    Return the matrix power of the adjacency matrix.

    :param amat_kwargs: Keyword arguments to configure NetworkX's ``adjacency_matrix`` function. Defaults to ``{}``
    :type amat_kwargs: Dict[str, Any]
    :param with_identity: Whether or not to add in the identity matrix to the adjacency matrix. Effectively adding "self loops". Defaults to True.
    :type with_identity: bool
    :param power: Matrix power to raise the adjacency matrix, defaults to 1.
    :type power: float
    :returns: the matrix power of the adjacency matrix
    :rtype: xr.DataArray
    """
    adjacency_matrix_kwargs = {"weight": "weight", "nodelist": None}
    adjacency_matrix_kwargs.update(amat_kwargs)
    amat = nx.adjacency_matrix(G, **adjacency_matrix_kwargs)
    if with_identity:
        amat = amat + np.eye(len(G))
    amat = np.linalg.matrix_power(amat, power)
    return format_adjacency(
        G,
        amat,
        f"adjacency_matrix_power_{power}",
    )


def inverse_distance_matrix(G: nx.Graph, power: float) -> xr.DataArray:
    """
    Return the inverse distance matrix.

    Using the coordinates present on the graph object,
    calculate the inverse power distance matrix.

    The resultant array will be symmetric,
    with entries [i, j] corresponding to 1 / (distance**power),
    where distance == euclidean distance between the (x, y, z) coordinates
    of the two graphs.
    Diagonals (entries [i, i]) are set to 0.

    :param G: NetworkX Graph object. Assumes that each node has ``x_coord``, ``y_coord``, and ``z_coord`` as node attributes.
    :type G: nx.Graph
    :param power: The power for the distance calculation.
    :type power: float
    :returns: the inverse distance matrix.
    :type: xr.DataArray
    """

    def extract_coords(n, d) -> pd.Series:
        """
        Extracts xyz coordinates from a node.

        :param n: node id
        :param d: node data
        :return: pd.Series containing x,y,z coordinates
        :rtype: pd.Series
        """
        coord_names = ("x_coord", "y_coord", "z_coord")
        return pd.Series(
            {coord_name: d[coord_name] for coord_name in coord_names}, name=n
        )

    coords = generate_feature_dataframe(G, funcs=[extract_coords])
    distmat = compute_distmat(coords).values

    I = np.eye(len(G))
    distmat = (distmat + I) ** power
    distmat = 1 / distmat
    distmat -= I
    return format_adjacency(
        G, distmat, f"inverse_distance_matrix_power_{power}"
    )
