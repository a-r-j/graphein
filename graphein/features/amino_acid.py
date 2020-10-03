"""
Featurization functions for amino acids.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache
def load_expasy_scales() -> pd.DataFrame:
    """
    Load pre-downloaded EXPASY scales.

    This helps with node featurization.

    The function is LRU-cached in memory for fast access
    on each function call.
    """
    df = pd.read_csv(
        Path(__file__).parent / "amino_acid_properties.csv", index_col=0
    )
    return df


@lru_cache
def load_meiler_embeddings() -> pd.DataFrame:
    """
    Load pre-downloaded EXPASY scales.

    This helps with node featurization.

    The function is LRU-cached in memory for fast access
    on each function call.
    """
    df = pd.read_csv(
        Path(__file__).parent / "meiler_embeddings.csv", index_col=0
    )
    return df


def expasy_protein_scale(n, d) -> pd.Series:
    """
    Return amino acid features that come from the EXPASY protein scale.

    Source: https://web.expasy.org/protscale/

    :param n: Node in a NetworkX graph
    :param d: NetworkX node attributes.
    """
    df = load_expasy_scales()
    amino_acid = d["residue_name"]
    return df[amino_acid]


def meiler_embbeding(n, d) -> pd.Series:
    """
    Return amino acid features that come from the reduced dimensional embeddings of amino acid physicochemical properties.

    Source: https://link.springer.com/article/10.1007/s008940100038
    doi: https://doi.org/10.1007/s008940100038

    :param n: Node in a NetworkX graph
    :param d: NetworkX node attributes.
    """
    df = load_meiler_embeddings()
    amino_acid = d["residue_name"]
    return df[amino_acid]
