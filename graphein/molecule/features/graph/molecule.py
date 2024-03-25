"""Functions for featurising Small Molecule Graphs."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from graphein.molecule.utils import (
    count_fragments,
    get_center,
    get_max_ring_size,
    get_mol_weight,
    get_morgan_fp,
    get_morgan_fp_np,
    get_qed_score,
    get_shape_moments,
)
from graphein.utils.dependencies import import_message

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:
    import_message(
        "graphe in.molecule.features.graph.molecule",
        "rdkit",
        "rdkit",
        True,
        extras=True,
    )


def mol_descriptors(
    g: nx.Graph,
    descriptor_list: Optional[List[str]] = None,
    return_array: bool = False,
    return_series: bool = False,
) -> Union[np.ndarray, pd.Series, Dict[str, Union[float, int]]]:
    """Adds global molecular descriptors to the graph.

    :param g: The graph to add the descriptors to.
    :type g: nx.Graph
    :param descriptor_list: The list of descriptors to add. If ``None``, all descriptors are added.
    :type descriptor_list: Optional[List[str]]
    :param return_array: If ``True``, the descriptors are returned as a ``np.ndarray``.
    :type return_array: bool
    :param return_series: If ``True``, the descriptors are returned as a ``pd.Series``.
    :return: The descriptors as a dictionary (default) ``np.ndarray`` or ``pd.Series``.
    :rtype: Union[np.ndarray, pd.Series,  Dict[str, Union[float, int]]]
    """

    mol = g.graph["rdmol"]
    # Retrieve list of possible descriptors
    descriptors = {d[0]: d[1] for d in Descriptors.descList}

    # Subset descriptors to those provided
    if descriptor_list is not None:
        descriptors = {
            k: v for k, v in descriptors.items() if k in descriptor_list
        }

    # Compute descriptors
    desc = {d: descriptors[d](mol) for d in descriptors}

    # Process Outformat
    if return_array:
        desc = np.array(desc.values())
        g.graph["descriptors"] = desc
    elif return_series:
        desc = pd.Series(desc)
        g.graph["descriptors"] = desc
    else:
        g.graph.update(desc)

    return desc


def add_center_of_mol(
    g: nx.Graph, weights: Optional[np.ndarray] = None
) -> nx.Graph:
    """Compute the centroid of the conformation.

    Hydrogens are ignored and no attention is paid to the difference in sizes
    of the heavy atoms; however, an optional vector of weights can be passed.

    :param g: Molecular Graph or RDkit Mol to compute the center of.
    :type g: nx.Graph
    :return: Graph with the centroid of the molecule added as a graph feature.
    :rtype: np.ndarray
    """
    g.graph["center"] = get_center(g, weights=weights)
    return g


def add_shape_moments(g: nx.Graph):
    """Add principal moments of inertia as defined in https://pubs.acs.org/doi/10.1021/ci025599w

    :param mol: Molecular Graph or RDkit Mol to compute the moments of intertia of.
    :type mol: Union[nx.Graph, Chem.Mol]
    :return: First 2 moments as a tuple.
    :rtype: Tuple[float, float]
    """
    g.graph["shape_moments"] = get_shape_moments(g)
    return g


def add_max_ring_size(g: nx.Graph) -> nx.Graph:
    g.graph["max_ring_size"] = get_max_ring_size(g)
    return g


def add_morgan_fingerprint(
    g: nx.Graph, radius: int = 2, n_bits: int = 2048
) -> nx.Graph:
    g.graph["morgan_fingerprint"] = get_morgan_fp(
        g, radius=radius, n_bits=n_bits
    )
    return g


def add_morgan_fingerprint_np(
    g: nx.Graph, radius: int = 2, n_bits: int = 2048
) -> nx.Graph:
    g.graph["morgan_fingerprint_np"] = get_morgan_fp_np(
        g, radius=radius, n_bits=n_bits
    )
    return g


def add_fragment_counts(g: nx.Graph) -> nx.Graph:
    g.graph["fragment_counts"] = count_fragments(g)
    return g


def add_mol_weight(g: nx.Graph) -> nx.Graph:
    g.graph["mol_weight"] = get_mol_weight(g)
    return g


def add_qed_score(g: nx.Graph) -> nx.Graph:
    g.graph["qed"] = get_qed_score(g)
    return g
