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

from graphein.utils.utils import import_message

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:
    import_message(
        "graphein.molecule.features.graph.molecule", "rdkit", "rdkit", True
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
