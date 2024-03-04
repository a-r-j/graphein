"""Utilities for parsing the TRRUST database."""

# %%
# Graphein
# Author: Ramon Vinas, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import wget
from loguru import logger as log

from graphein.utils.utils import filter_dataframe


def _download_TRRUST(root_dir: Optional[Path] = None) -> str:
    """
    Downloads TRRUST from
    https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv

    :param root_dir: Path to desired output directory to download TRRUST to.
        Defaults to None (downloads to graphein/datasets/trrust/)
    :type root_dir: pathlib.Path, optional
    :returns: Path to downloaded TRRUST Dataset.
    :rtype: str
    """
    url = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"

    if root_dir is None:
        root_dir = Path(__file__).parent.parent.parent
    trrust_dir = f"{root_dir}/datasets/trrust"
    Path(trrust_dir).mkdir(parents=False, exist_ok=True)
    file = f"{trrust_dir}/human.tsv"

    # Download data
    if not os.path.exists(file):
        log.info("Downloading TRRUST ...")
        wget.download(url, file)

    return file


@lru_cache()
def load_TRRUST(root_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the TRRUST datafile. If file not found, it is downloaded.

    :param root_dir: Root directory path to either find or download TRRUST
    :type root_dir: pathlib.Path, optional
    :returns: TRRUST database as a dataframe
    :rtype: pd.DataFrame
    """
    file = _download_TRRUST(root_dir)

    return pd.read_csv(
        file,
        delimiter="\t",
        header=None,
        names=["g1", "g2", "regtype", "references"],
    )


def parse_TRRUST(
    gene_list: List[str], root_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Parser for TRRUST regulatory interactions. If the TRRUST dataset is not
    found in the specified root_dir, it is downloaded

    :param gene_list: List of gene identifiers to restrict dataframe to.
    :type gene_list: List[str]
    :param root_dir: Root directory path to either find or download TRRUST.
        Defaults to ``None`` (downloads dataset to ``graphein/datasets/trrust``)
    :type root_dir: pathlib.Path, optional
    :returns: Pandas DataFrame with the regulatory interactions between genes
        in the gene list.
    :rtype: pd.DataFrame
    """
    df = load_TRRUST(root_dir=root_dir)
    # Select input genes
    df = df[df["g1"].isin(gene_list) & df["g2"].isin(gene_list)]

    return df


def filter_TRRUST(
    df: pd.DataFrame, funcs: Optional[List[Callable]]
) -> pd.DataFrame:
    """
    Filters results of TRRUST call according to user kwargs.

    :param df: Source specific Pandas dataframe (TRRUST) with results of the
        API call
    :type df: pd.DataFrame
    :param funcs: User functions to filter the results.
    :type funcs: List[Callable]
    :return: Source specific Pandas dataframe with filtered results
    :rtype: pd.DataFrame
    """
    if funcs is not None:
        df = filter_dataframe(df, funcs)

    return df


def standardise_TRRUST(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters results of TRRUST call by providing a list of
    user-defined functions that accept a dataframe and return a dataframe.

    :param df: pd.Dataframe to filter. Must contain columns:
        ``["g1", "g2", "regtype"]``.
    :type df: pd.DataFrame
    :param funcs: List of functions that carry out dataframe processing.
    :type funcs: List[Callable]
    :return: Processed DataFrame.
    :rtype: pd.DataFrame
    """
    # Rename & delete columns
    df = df[["g1", "g2", "regtype"]]

    # Rename type of regulatory interaction
    df["regtype"].replace(
        {"Activation": "+", "Repression": "-", "Unknown": "?"}, inplace=True
    )

    # Add source column
    df["source"] = "TRRUST"

    return df


def TRRUST_df(
    gene_list: List[str], filtering_funcs: Optional[List[Callable]] = None
) -> pd.DataFrame:
    """
    Generates standardised dataframe with TRRUST protein-protein interactions,
    filtered according to user's input.

    :param gene_list: List of gene identifiers to restrict dataframe to.
    :type gene_list: List[str]
    :param filtering_funcs: Functions with which to filter the dataframe.
    :type filtering_funcs: List[Callable]
    :return: Standardised dataframe with TRRUST interactions.
    :rtype: pd.DataFrame
    """
    df = parse_TRRUST(gene_list=gene_list)
    df = filter_TRRUST(df, filtering_funcs)
    df = standardise_TRRUST(df)

    return df


if __name__ == "__main__":
    df = TRRUST_df(["AATF", "MYC", "USF1", "SP1", "TP53", "DUSP1"])
    print(df.head())
