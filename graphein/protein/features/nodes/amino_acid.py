"""Featurization functions for amino acids."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from graphein.protein.resi_atoms import BASE_AMINO_ACIDS, RESI_THREE_TO_1
from graphein.utils.utils import onek_encoding_unk

log = logging.getLogger(__name__)


@lru_cache()
def load_expasy_scales() -> pd.DataFrame:
    """
    Load pre-downloaded EXPASY scales.

    This helps with node featuarization.

    The function is LRU-cached in memory for fast access
    on each function call.

    :returns: pd.DataFrame containing expasy scales
    :rtype: pd.DataFrame
    """
    fpath = Path(__file__).parent / "amino_acid_properties.csv"
    log.debug(f"Reading Expasy protein scales from: {fpath}")
    df = pd.read_csv(fpath, index_col=0)
    return df


@lru_cache()
def load_meiler_embeddings() -> pd.DataFrame:
    """
    Load pre-downloaded Meiler embeddings.

    This helps with node featurization.

    The function is LRU-cached in memory for fast access
    on each function call.

    :returns: pd.DataFrame containing Meiler Embeddings from Meiler et al. 2001
    :rtype: pd.DataFrame
    """
    fpath = Path(__file__).parent / "meiler_embeddings.csv"
    log.debug(f"Reading meiler embeddings from: {fpath}")
    df = pd.read_csv(fpath, index_col=0)
    return df


def expasy_protein_scale(
    n,
    d,
    selection: Optional[List[str]] = None,
    add_separate: bool = False,
    return_array: bool = False,
) -> Union[pd.Series, np.ndarray]:
    """
    Return amino acid features that come from the EXPASY protein scale.

    Source: https://web.expasy.org/protscale/

    :param n: Node in a NetworkX graph
    :param d: NetworkX node attributes.
    :param selection: List of columns to select. Viewable in graphein.protein.features.nodes.meiler_embeddings
    :type selection: List[str], optional
    :param add_separate: Whether or not to add the expasy features as indvidual entries or as a series.
    :param return_array: Bool indicating whether or not to return a np.ndarray of the features. Default is pd.Series
    :type return_array: bool
    :returns: pd.Series of amino acid features
    :rtype: pd.Series
    """
    df = load_expasy_scales()
    amino_acid = d["residue_name"]
    try:
        features = df[amino_acid]
        if selection is not None:
            features = features.filter(selection)
    except:
        features = pd.Series(np.zeros(len(df)))

    if return_array:
        features = np.array(features)

    if add_separate:
        for k, v in features.to_dict().items():
            d[k] = v
    else:
        d["expasy"] = features

    return features


def meiler_embedding(
    n, d, return_array: bool = False
) -> Union[pd.Series, np.array]:
    """
    Return amino acid features from reduced dimensional embeddings of amino acid physicochemical properties.

    Source: https://link.springer.com/article/10.1007/s008940100038
    doi: https://doi.org/10.1007/s008940100038

    :param n: Node in a NetworkX graph
    :param d: NetworkX node attributes.
    :returns: pd.Series of amino acid features
    :rtype: pd.Series
    """
    df = load_meiler_embeddings()
    amino_acid = d["residue_name"]
    try:
        features = df[amino_acid]
    except:
        features = pd.Series(np.zeros(len(df)))

    if return_array:
        features = np.array(features)

    d["meiler"] = features

    return features


def amino_acid_one_hot(
    n,
    d: Dict[str, Any],
    return_array: bool = True,
    allowable_set: Optional[List[str]] = None,
) -> Union[pd.Series, np.ndarray]:
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
        allowable_set = BASE_AMINO_ACIDS

    features = onek_encoding_unk(
        RESI_THREE_TO_1[d["residue_name"]], allowable_set
    )

    if return_array:
        features = np.array(features).astype(int)
    else:
        features = pd.Series(features).astype(int)
        features.index = allowable_set

    d["amino_acid_one_hot"] = features
    return features
