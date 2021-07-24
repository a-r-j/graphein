"""Provides utility functions for use across Graphein"""
import os
from pathlib import Path

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import Any, List, Tuple, Union

import pandas as pd
import wget
from Bio.PDB import PDBList


def download_pdb(config, pdb_code: str) -> str:
    """
    Download PDB structure from PDB

    :param pdb_code: 4 character PDB accession code
    :type pdb_code: str
    :return: # todo impl return
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
    print(f"Downloaded PDB file for: {pdb_code}")
    return config.pdb_dir / (pdb_code + ".pdb")


def get_protein_name_from_filename(pdb_path: str) -> str:
    """
    Extracts a filename from a pdb_path
    :param pdb_path: Path to extract filename from
    :return: file name
    """
    head, tail = os.path.split(pdb_path)
    tail = os.path.splitext(tail)[0]
    return tail


def filter_dataframe(
    dataframe: pd.DataFrame,
    by_column: str,
    list_of_values: List[Any],
    boolean: bool,
):
    """
    Filter function for dataframe.
    Filters the [dataframe] such that the [by_column] values have to be
    in the [list_of_values] list if boolean == True, or not in the list
    if boolean == False

    :param dataframe: pd.DataFrame to filter
    :param by_column: str denoting by_column of dataframe to filter
    :param list_of_values: List of values to filter with
    :param bool: indicates whether to keep or exclude matching list_of_values. True -> in list, false -> not in list
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
    :param uniprot_id: UniProt ID of desirec protein
    :param out_dir: string specifying desired otput location. Default is pwd.
    :param mmcif: Bool specifying whether to download MMCiF or PDB. Default is false (downloads pdb)
    :param retrieve_aligned_score: Bool specifying whether or not to download score alignment json
    :return: path to output. Tuple if several outputs specified.
    """
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


if __name__ == "__main__":
    download_alphafold_structure(uniprot_id="Q8W3K0", aligned_score=True)
