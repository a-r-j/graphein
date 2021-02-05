# %%
# Graphein
# Author: Ramon Vinas, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import logging
from typing import Any, Dict, List, Union

from pathlib import Path
import pandas as pd
import wget
import os
import zipfile

log = logging.getLogger(__name__)


def _download_TRRUST():
    """
    Downloads TRRUST
    """
    url = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"

    # TODO: perhaps this would be better?
    #  https://stackoverflow.com/questions/25389095/python-get-path-of-root-project-structure
    root_dir = os.path.dirname(os.path.realpath("../"))
    trrust_dir = "{}/datasets/trrust".format(root_dir)
    Path(trrust_dir).mkdir(parents=False, exist_ok=True)
    file = "{}/human.tsv".format(trrust_dir)

    # Download data
    if not os.path.exists(file):
        print("Downloading TRRUST ...")
        wget.download(url, file)

    return file


def parse_TRRUST(gene_list: List[str], **kwargs) -> pd.DataFrame:
    """
    Parser for TRRUST regulatory interactions
    :param gene_list: List of gene identifiers
    :return Pandas dataframe with the regulatory interactions between genes in the gene list
    """
    file = _download_TRRUST()
    df = pd.read_csv(
        file,
        delimiter="\t",
        header=None,
        names=["g1", "g2", "regtype", "references"],
    )

    # Select input genes
    df = df[df["g1"].isin(gene_list) & df["g2"].isin(gene_list)]

    return df


def filter_TRRUST(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Filters results of TRRUST call according to user kwargs,
    :param df: Source specific Pandas dataframe (TRRUST) with results of the API call
    :param kwargs: User thresholds used to filter the results. The parameter names are of the form TRRUST_<param>,
                   where <param> is the name of the parameter. All the parameters are numerical values.
    :return: Source specific Pandas dataframe with filtered results
    """
    return df


def standardise_TRRUST(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises STRING dataframe, e.g. puts everything into a common format
    :param df: Source specific Pandas dataframe
    :return: Standardised dataframe
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


def TRRUST_df(gene_list: List[str], **kwargs) -> pd.DataFrame:
    """
    Generates standardised dataframe with TRRUST protein-protein interactions, filtered according to user's input
    :return: Standardised dataframe with TRRUST interactions
    """
    df = parse_TRRUST(gene_list=gene_list)
    df = filter_TRRUST(df, **kwargs)
    df = standardise_TRRUST(df)

    return df


if __name__ == "__main__":
    df = TRRUST_df(["AATF", "MYC", "USF1", "SP1", "TP53", "DUSP1"])
    print(df.head())
