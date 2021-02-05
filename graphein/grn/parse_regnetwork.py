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


def _download_RegNetwork():
    """
    Downloads RegNetwork regulatory interactions
    """
    url = "http://www.regnetworkweb.org/download/human.zip"

    # TODO: perhaps this would be better?
    #  https://stackoverflow.com/questions/25389095/python-get-path-of-root-project-structure
    root_dir = os.path.dirname(os.path.realpath("../"))
    regnetwork_dir = "{}/datasets/regnetwork".format(root_dir)
    Path(regnetwork_dir).mkdir(parents=False, exist_ok=True)
    compressed_file = "{}/human.zip".format(regnetwork_dir)
    out_dir = "{}/human".format(regnetwork_dir)
    file = "{}/human.source".format(out_dir)

    # Download data and unzip
    if not os.path.exists(file):
        print("Downloading RegNetwork ...")
        wget.download(url, compressed_file)

        with zipfile.ZipFile(compressed_file, "r") as zip_ref:
            zip_ref.extractall(out_dir)

    return file


def _download_RegNetwork_regtypes():
    """
    Downloads RegNetwork regulatory interactions types
    """
    url = "http://www.regnetworkweb.org/download/RegulatoryDirections.zip"

    root_dir = os.path.dirname(os.path.realpath("../"))
    regnetwork_dir = "{}/datasets/regnetwork".format(root_dir)
    Path(regnetwork_dir).mkdir(parents=False, exist_ok=True)
    compressed_file = "{}/RegulatoryDirections.zip".format(regnetwork_dir)
    out_dir = "{}/human".format(regnetwork_dir)
    file = "{}/new_kegg.human.reg.direction.txt".format(out_dir)

    # Download data and unzip
    if not os.path.exists(file):
        print("Downloading RegNetwork reg types ...")
        wget.download(url, compressed_file)

        with zipfile.ZipFile(compressed_file, "r") as zip_ref:
            zip_ref.extractall(out_dir)

    return file


def parse_RegNetwork(gene_list: List[str], **kwargs) -> pd.DataFrame:
    """
    Parser for RegNetwork interactions
    :param gene_list: List of gene identifiers
    :return Pandas dataframe with the regulatory interactions between genes in the gene list
    """
    file = _download_RegNetwork()
    df = pd.read_csv(
        file, delimiter="\t", header=None, names=["g1", "id1", "g2", "id2"]
    )
    print(df.head())

    # Add regulatory types
    file = _download_RegNetwork_regtypes()
    reg_df = pd.read_csv(
        file,
        delimiter=" ",
        header=None,
        names=["tf", "id1", "target", "id2", "regtype"],
        skiprows=1,
    )
    df = pd.merge(
        df,
        reg_df,
        how="outer",
        left_on=["g1", "g2"],
        right_on=["tf", "target"],
    )
    df["g1"] = df["g1"].combine_first(df["tf"])
    df["g2"] = df["g2"].combine_first(df["target"])

    # Select input genes
    df = df[df["g1"].isin(gene_list) & df["g2"].isin(gene_list)]

    return df


def filter_RegNetwork(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Filters results of RegNetwork call according to user kwargs,
    :param df: Source specific Pandas dataframe (RegNetwork) with results of the API call
    :param kwargs: User thresholds used to filter the results. The parameter names are of the form RegNetwork_<param>,
                   where <param> is the name of the parameter. All the parameters are numerical values.
    :return: Source specific Pandas dataframe with filtered results
    """
    return df


def standardise_RegNetwork(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises STRING dataframe, e.g. puts everything into a common format
    :param df: Source specific Pandas dataframe
    :return: Standardised dataframe
    """
    # Rename & delete columns
    df = df[["g1", "g2", "regtype"]]

    # Add source column
    df["source"] = "RegNetwork"

    # Standardise regulatory types
    df["regtype"].replace(
        {"-->": "+", "--|": "-", None: "?", "-p": "?", "-/-": "?"},
        inplace=True,
    )

    return df


def RegNetwork_df(gene_list: List[str], **kwargs) -> pd.DataFrame:
    """
    Generates standardised dataframe with RegNetwork protein-protein interactions, filtered according to user's input
    :return: Standardised dataframe with RegNetwork interactions
    """
    df = parse_RegNetwork(gene_list=gene_list)
    df = filter_RegNetwork(df, **kwargs)
    df = standardise_RegNetwork(df)

    return df


if __name__ == "__main__":
    df = RegNetwork_df(["AATF", "MYC", "USF1", "SP1", "TP53", "DUSP1"])
    print(df)
