"""Provides utility functions for use across Graphein."""
import logging

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import wget
from Bio.PDB import PDBList

from .resi_atoms import RESI_THREE_TO_1

log = logging.getLogger(__name__)


def download_pdb(config, pdb_code: str) -> Path:
    """
    Download PDB structure from PDB

    :param pdb_code: 4 character PDB accession code
    :type pdb_code: str
    :return: returns filepath to downloaded structure
    :rtype: str
    """
    if not config.pdb_dir:
        config.pdb_dir = Path("/tmp/")

    # Initialise class and download pdb file
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(
        pdb_code, pdir=config.pdb_dir, overwrite=True, file_format="pdb"
    )
    # Rename file to .pdb from .ent
    os.rename(
        config.pdb_dir / ("pdb" + pdb_code + ".ent"),
        config.pdb_dir / (pdb_code + ".pdb"),
    )
    # Assert file has been downloaded
    assert any(pdb_code in s for s in os.listdir(config.pdb_dir))
    log.info(f"Downloaded PDB file for: {pdb_code}")
    return config.pdb_dir / (pdb_code + ".pdb")


def get_protein_name_from_filename(pdb_path: str) -> str:
    """
    Extracts a filename from a pdb_path

    :param pdb_path: Path to extract filename from
    :type pdb_path: str
    :return: file name
    :rtype: str
    """
    _, tail = os.path.split(pdb_path)
    tail = os.path.splitext(tail)[0]
    return tail


def filter_dataframe(
    dataframe: pd.DataFrame,
    by_column: str,
    list_of_values: List[Any],
    boolean: bool,
) -> pd.DataFrame:
    """
    Filter function for dataframe.
    Filters the [dataframe] such that the [by_column] values have to be
    in the [list_of_values] list if boolean == True, or not in the list
    if boolean == False

    :param dataframe: pd.DataFrame to filter
    :type dataframe: pd.DataFrame
    :param by_column: str denoting by_column of dataframe to filter
    :type by_column: str
    :param list_of_values: List of values to filter with
    :type list_of_values: List[Any]
    :param boolean: indicates whether to keep or exclude matching list_of_values. True -> in list, false -> not in list
    :type boolean: bool
    :returns: Filtered dataframe
    :rtype: pd.DataFrame
    """
    df = dataframe.copy()
    df = df[df[by_column].isin(list_of_values) == boolean]
    df.reset_index(inplace=True, drop=True)

    return df


def download_alphafold_structure(
    uniprot_id: str,
    out_dir: str = ".",
    pdb: bool = True,
    mmcif: bool = False,
    aligned_score: bool = True,
) -> Union[str, Tuple[str, str]]:
    BASE_URL = "https://alphafold.ebi.ac.uk/files/"
    """
    Downloads a structure from the Alphafold EBI database.

    :param uniprot_id: UniProt ID of desired protein
    :type uniprot_id: str
    :param out_dir: string specifying desired output location. Default is pwd.
    :type out_dir: str
    :param mmcif: Bool specifying whether to download MMCiF or PDB. Default is false (downloads pdb)
    :type mmcif: bool
    :param retrieve_aligned_score: Bool specifying whether or not to download score alignment json
    :type retrieve_aligned_score: bool
    :return: path to output. Tuple if several outputs specified.
    :rtype: Union[str, Tuple[str, str]]
    """
    if not mmcif and not pdb:
        raise ValueError("Must specify either mmcif or pdb.")
    if mmcif:
        query_url = BASE_URL + "AF-" + uniprot_id + "F1-model_v1.cif"
    if pdb:
        query_url = BASE_URL + "AF-" + uniprot_id + "-F1-model_v1.pdb"

    structure_filename = wget.download(query_url, out=out_dir)

    if aligned_score:
        score_query = (
            BASE_URL
            + "AF-"
            + uniprot_id
            + "-F1-predicted_aligned_error_v1.json"
        )
        score_filename = wget.download(score_query, out=out_dir)
        return structure_filename, score_filename

    return structure_filename


def three_to_one_with_mods(res: str) -> str:
    """
    Converts three letter AA codes into 1 letter. Allows for modified residues.

    :param res: Three letter residue code str:
    :type res: str
    :return: 1-letter residue code
    :rtype: str
    """
    return RESI_THREE_TO_1[res]
