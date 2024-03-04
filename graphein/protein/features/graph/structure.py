"""Functions for computing graph-level features based on structure."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import math
from typing import Union

import networkx as nx
import numpy as np

from graphein.protein.resi_atoms import ATOMIC_MASSES


def add_radius_of_gyration(G: nx.Graph, round: bool = False) -> nx.Graph:
    """Adds radius of gyration (Rg) to graph as a graph attribute  (``G.graph["radius_of_gyration"]``).

    Atomic masses are defined in :ref:`graphein.protein.resi_atoms.ATOMIC_MASSES`.

    :param G: Structure graph to add radius of gyration to.
    :type G: nx.Graph
    :param round: Whether to round the result to the nearest integer, defaults to ``False``
    :type round: bool
    :return: Graph with radius of gyration added (in angstroms).
    :rtype: nx.Graph
    """

    G.graph["radius_of_gyration"] = radius_of_gyration(G, round)
    return G


def radius_of_gyration(G: nx.Graph, round: bool = False) -> Union[float, int]:
    """Calculates the radius of gyration of a structure graph in angstroms.

    Atomic masses are defined in :ref:`graphein.protein.resi_atoms.ATOMIC_MASSES`.

    :param G: Graph to calculate radius of gyration of.
    :type G: nx.Graph
    :param round: Whether to round the result to the nearest integer.
    :type round: bool
    :return: Radius of gyration in angstroms.
    :rtype: float
    """
    masses = [
        ATOMIC_MASSES[d["element_symbol"]] for _, d in G.nodes(data=True)
    ]
    total_mass = sum(masses)

    coords = [d["coords"] for _, d in G.nodes(data=True)]
    weighted_coords = [coord * mass for coord, mass in zip(coords, masses)]

    rr = sum(
        np.sum(w_coord * coord)
        for w_coord, coord in zip(weighted_coords, coords)
    )
    mm = sum((sum(w_coord) / total_mass) ** 2 for w_coord in weighted_coords)

    rg = math.sqrt(rr / total_mass - mm)
    return round(rg, 3) if round else rg
