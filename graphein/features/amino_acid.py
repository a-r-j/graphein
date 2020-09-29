"""
Featurization functions for amino acids.
"""
import pandas as pd
from functools import lru_cache


@lru_cache
def load_expasy_scales() -> pd.DataFrame:
    """
    Load pre-downloaded EXPASY scales.

    This helps with node featurization.

    The function is LRU-cached in memory for fast access
    on each function call.
    """
    df = pd.read_csv("amino_acid_properties.csv", index_col=0)
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
