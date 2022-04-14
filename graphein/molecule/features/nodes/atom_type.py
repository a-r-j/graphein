"""Functions for featurising Small Molecule Graphs."""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Yuanqi Du
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from graphein.molecule.atoms import BASE_ATOMS
from graphein.utils.utils import onek_encoding_unk


def atom_type_one_hot(
    n,
    d: Dict[str, Any],
    return_array: bool = True,
    allowable_set: Optional[List[str]] = None,
) -> np.ndarray:
    """Adds a one-hot encoding of atom types as a node attribute.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :param return_array: If ``True``, returns a numpy ``np.ndarray`` of one-hot encoding, otherwise returns a ``pd.Series``. Default is ``True``.
    :type return_array: bool
    :param allowable_set: Specifies vocabulary of amino acids. Default is ``None`` (which uses `graphein.molecule.atoms.BASE_ATOMS`).
    :return: One-hot encoding of amino acid types.
    :rtype: Union[pd.Series, np.ndarray]
    """

    if allowable_set is None:
        allowable_set = BASE_ATOMS

    features = onek_encoding_unk(d["atomic_num"], allowable_set)

    if return_array:
        features = np.array(features).astype(int)
    else:
        features = pd.Series(features).astype(int)
        features.index = allowable_set

    d["atom_type_one_hot"] = features
    return features
