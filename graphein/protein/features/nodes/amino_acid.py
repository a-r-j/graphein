"""Featurization functions for amino acids."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache()
def load_expasy_scales() -> pd.DataFrame:
    """
    Load pre-downloaded EXPASY scales.

    This helps with node featuarization.

    The function is LRU-cached in memory for fast access
    on each function call.
    """
    df = pd.read_csv(
        Path(__file__).parent / "amino_acid_properties.csv", index_col=0
    )
    return df


@lru_cache()
def load_meiler_embeddings() -> pd.DataFrame:
    """
    Load pre-downloaded Meiler embeddings.

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


def meiler_embedding(n, d) -> pd.Series:
    """
    Return amino acid features from reduced dimensional embeddings of amino acid physicochemical properties.

    Source: https://link.springer.com/article/10.1007/s008940100038
    doi: https://doi.org/10.1007/s008940100038

    :param n: Node in a NetworkX graph
    :param d: NetworkX node attributes.
    """
    df = load_meiler_embeddings()
    amino_acid = d["residue_name"]
    return df[amino_acid]


def aaindex_1_feat(n, d, feature_name: str) -> pd.Series:
    # TODO: This function needs a docstring.

    # TODO: Decide whether these imports should go at the top of the file or not.
    # If not, document why they are imported inside the function,
    # as this is not a common practice.
    # Best to minimize the number of optional dependencies.
    from Bio.PDB.Polypeptide import three_to_one
    from propy.AAIndex import GetAAIndex1

    df = GetAAIndex1(feature_name)
    df = pd.Series(df).loc[three_to_one(d["residue_name"])]
    return df


def load_feature_dataframe(n, d) -> pd.Series:
    """
    Generic function for loading features from an on-disk file
    :param n:
    :param d:
    :return:
    """
    # TODO: Is this function still needed?
    raise NotImplementedError


def load_esm_embedding_residue(n, d) -> pd.Series:
    raise NotImplementedError


# TODO: Remove this from the script, as this should be instead present as a test.
if __name__ == "__main__":
    aaindex_1_feat(feature_name="KRIW790103")
