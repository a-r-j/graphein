import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from graphein.molecule.atoms import (
    BASE_ATOMS,
)

from graphein.utils.utils import onek_encoding_unk

def atom_type_one_hot(
    n,
    d: Dict[str, Any],
    return_array: bool = True,
    allowable_set: Optional[List[str]] = None,
) -> np.ndarray:
    """Adds a one-hot encoding of amino acid types as a node attribute.

    :param n: node name, this is unused and only included for compatibility with the other functions
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :param return_array: If True, returns a numpy array of one-hot encoding, otherwise returns a pd.Series. Default is True.
    :type return_array: bool
    :param allowable_set: Specifies vocabulary of amino acids. Default is None (which uses `graphein.protein.resi_atoms.STANDARD_AMINO_ACIDS`).
    :return: One-hot encoding of amino acid types
    :rtype: Union[pd.Series, np.ndarray]
    """

    if allowable_set is None:
        allowable_set = BASE_ATOMS

    features = onek_encoding_unk(
        d["atomic_num"], allowable_set
    )

    if return_array:
        features = np.array(features).astype(int)
    else:
        features = pd.Series(features).astype(int)
        features.index = allowable_set

    d["atom_type_one_hot"] = features
    return features