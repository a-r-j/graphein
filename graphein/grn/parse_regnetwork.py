"""Functions for parsing GRNs from RegNetwork."""

# %%
# Graphein
# Author: Ramon Vinas, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import functools
import os
import ssl
import zipfile
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import wget
from loguru import logger as log

from graphein.utils.utils import deprecated, filter_dataframe, ping


@deprecated(
    "RegNetwork appears to be down. This warning will be removed in a future release if the service is restored."
)
def _download_RegNetwork(
    root_dir: Optional[Path] = None, network_type: str = "human"
) -> str:
    """
    Downloads RegNetwork regulatory interactions to the root directory.
    Returns the filepath.

    :param root_dir: ``Path`` object specifying the location to download
        RegNetwork to. Default is ``None`` which downloads to the ``dataset/``
        directory inside graphein.
    :type root_dir: pathlib.Path, optional
    :param network_type: Specifies whether to download human or mouse
        regulatory network. Supported values: ``"human"`` (default),
        ``"mouse"``.
    :type network_type: str
    :returns: path to downloaded RegNetwork
    :rtype: str
    """

    # Ping server to check if file is available
    ping_result = ping("regnetworkweb.org")
    if not ping_result:
        log.warning(
            "RegNetwork is not available. Please check your internet \
            connection or verify at: http://www.regnetworkweb.org"
        )

    mouse_url = "https://regnetworkweb.org/download/mouse.zip"

    if network_type == "human":
        human_url = "https://regnetworkweb.org/download/human.zip"
        url = human_url
    elif network_type == "mouse":
        url = mouse_url
    else:
        raise ValueError(
            f"network_type: {network_type} is unsupported. Please use 'human' \
            or 'mouse'"
        )

    # If no root dir is provided, use the dataset directory inside graphein.
    if root_dir is None:
        root_dir = Path(__file__).parent.parent.parent / "datasets"

    regnetwork_dir = f"{root_dir}/regnetwork"
    Path(regnetwork_dir).mkdir(parents=False, exist_ok=True)
    compressed_file = f"{regnetwork_dir}/human.zip"
    out_dir = f"{regnetwork_dir}/human"
    file = f"{out_dir}/human.source"

    # Download data and unzip
    if not os.path.exists(file):
        log.info("Downloading RegNetwork ...")
        # switch ssl context for unverified download
        default_https_context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        wget.download(url, compressed_file)
        # switch ssl context back to default
        ssl._create_default_https_context = default_https_context
        with zipfile.ZipFile(compressed_file, "r") as zip_ref:
            zip_ref.extractall(out_dir)

    return file


@deprecated(
    "RegNetwork appears to be down. This warning will be removed in a future release if the service is restored."
)
def _download_RegNetwork_regtypes(root_dir: Optional[Path] = None) -> str:
    """
    Downloads RegNetwork regulatory interactions types to the root directory.

    :param root_dir: ``Path`` object specifying the location to download
        RegNetwork to
    :type root_dir: pathlib.Path, optional. Default is ``None`` which downloads
        to the ``dataset/`` directory inside graphein.
    :returns: Path to downloaded RegNetwork
    :rtype: str
    """
    url = "https://regnetworkweb.org/download/RegulatoryDirections.zip"

    if root_dir is None:
        root_dir = Path(__file__).parent.parent.parent / "datasets"

    regnetwork_dir = f"{root_dir}/regnetwork"
    Path(regnetwork_dir).mkdir(parents=False, exist_ok=True)
    compressed_file = f"{regnetwork_dir}/RegulatoryDirections.zip"
    out_dir = f"{regnetwork_dir}/human"
    file = f"{out_dir}/new_kegg.human.reg.direction.txt"

    # Download data and unzip
    if not os.path.exists(file):
        log.info("Downloading RegNetwork reg types ...")
        # switch ssl context for unverified download
        default_https_context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        wget.download(url, compressed_file)
        # switch ssl context back to default
        ssl._create_default_https_context = default_https_context

        with zipfile.ZipFile(compressed_file, "r") as zip_ref:
            zip_ref.extractall(out_dir)

    return file


@deprecated(
    "RegNetwork appears to be down. This warning will be removed in a future release if the service is restored."
)
@functools.lru_cache()
def load_RegNetwork_interactions(
    root_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Loads RegNetwork interaction datafile. Downloads the file first if not
    already present.

    :param root_dir: ``Path`` object specifying the location of the
        RegNetwork download.
    :type root_dir: pathlib.Path, optional. Default is ``None``.
    :returns: RegNetwork interaction data as a pd.DataFrame.
    :rtype: pd.DataFrame
    """
    file = _download_RegNetwork(root_dir)
    return pd.read_csv(
        file, delimiter="\t", header=None, names=["g1", "id1", "g2", "id2"]
    )


@deprecated(
    "RegNetwork appears to be down. This warning will be removed in a future release if the service is restored."
)
@functools.lru_cache()
def load_RegNetwork_regulation_types(
    root_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Loads RegNetwork regulation types. Downloads the file first if not already
        present.

    :param root_dir: ``Path`` object specifying the location of the RegNetwork
        download.
    :type root_dir: pathlib.Path, optional. Default is ``None``.
    :returns: RegNetwork regulation types as a pd.DataFrame.
    :rtype: pd.DataFrame
    """
    file = _download_RegNetwork_regtypes(root_dir)
    return pd.read_csv(
        file,
        delimiter=" ",
        header=None,
        names=["tf", "id1", "target", "id2", "regtype"],
        skiprows=1,
    )


@deprecated(
    "RegNetwork appears to be down. This warning will be removed in a future release if the service is restored."
)
def parse_RegNetwork(
    gene_list: List[str], root_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Parser for RegNetwork interactions

    :param gene_list: List of gene identifiers
    :type gene_list: List[str]
    :param root_dir: ``Path`` object specifying the location of the RegNetwork
        download.
    :return: Pandas DataFrame with the regulatory interactions between genes
        in the gene list.
    :rtype: pd.DataFrame
    """
    # Load dataframes
    df = load_RegNetwork_interactions(root_dir)
    reg_df = load_RegNetwork_regulation_types(root_dir)

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


def filter_RegNetwork(
    df: pd.DataFrame, funcs: Optional[List[Callable]] = None
) -> pd.DataFrame:
    """
    Filters results of RegNetwork call by providing a list of user-defined
    functions that accept a DataFrame and return a DataFrame

    :param df: pd.DataFrame to filter
    :param funcs: List of functions that carry out DataFrame processing.
    :return: Processed DataFrame.
    :rtype: pd.DataFrame
    """
    if funcs is not None:
        df = filter_dataframe(df, funcs)

    return df


def standardise_RegNetwork(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises RegNetwork dataframe, e.g. puts everything into a common format

    :param df: Source specific Pandas dataframe
    :type df: pd.DataFrame
    :return: Standardised dataframe
    :rtype: pd.DataFrame
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


@deprecated(
    "RegNetwork appears to be down. This warning will be removed in a future release if the service is restored."
)
def RegNetwork_df(
    gene_list: List[str],
    root_dir: Optional[Path] = None,
    filtering_funcs: Optional[List[Callable]] = None,
) -> pd.DataFrame:
    """
    Generates standardised dataframe with RegNetwork protein-protein
    interactions, filtered according to user's input.

    :param gene_list: List of gene identifiers.
    :type gene_list: List[str]
    :param root_dir: ``Path`` object specifying the location of the RegNetwork
        dataset.
    :type root_dir: pathlib.Path, optional. Default is ``None``.
    :param filtering_funcs: List of (optional) functions that accept a dataframe
        and return a dataframe. These functions are applied to the dataframe to
        filter entries according to the user's input.
    :type filtering_funcs: List[Callable], optional. Default is ``None``.
    :return: Standardised dataframe with RegNetwork interactions.
    :rtype: pd.DataFrame
    """
    df = parse_RegNetwork(gene_list=gene_list, root_dir=root_dir)
    df = filter_RegNetwork(df, filtering_funcs)
    df = standardise_RegNetwork(df)

    return df
