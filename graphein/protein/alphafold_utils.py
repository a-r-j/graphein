"""Provides utility functions for working with the AlphaFold Database."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import os
from functools import lru_cache, partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import wget
from loguru import logger as log
from tqdm.contrib.concurrent import process_map


def download_alphafold_structure(
    uniprot_id: str,
    version: int = 2,
    out_dir: str = ".",
    rename: bool = True,
    pdb: bool = True,
    mmcif: bool = False,
    aligned_score: bool = True,
) -> Union[str, Tuple[str, str]]:
    """
    Downloads a structure from the Alphafold EBI database (https://alphafold.ebi.ac.uk/files/").

    :param uniprot_id: UniProt ID of desired protein.
    :type uniprot_id: str
    :param version: Version of the structure to download
    :type version: int
    :param out_dir: string specifying desired output location. Default is pwd.
    :type out_dir: str
    :param rename: boolean specifying whether to rename the output file to
        ``$uniprot_id.pdb``. Default is ``True``.
    :type rename: bool
    :param pdb: boolean specifying whether to download the PDB file. Default is
        ``True``.
    :type pdb: bool
    :param mmcif: Bool specifying whether to download MMCiF or PDB. Default is
        ``False`` (downloads pdb).
    :type mmcif: bool
    :param retrieve_aligned_score: Bool specifying whether or not to download
        score alignment json.
    :type retrieve_aligned_score: bool
    :return: path to output. Tuple if several outputs specified.
    :rtype: Union[str, Tuple[str, str]]
    """
    BASE_URL = "https://alphafold.ebi.ac.uk/files/"
    uniprot_id = uniprot_id.upper()

    if (not mmcif and not pdb) or (mmcif and pdb):
        raise ValueError("Must specify either mmcif or pdb.")
    if mmcif:
        query_url = f"{BASE_URL}AF-{uniprot_id}-F1-model_v{version}.cif"
    if pdb:
        query_url = f"{BASE_URL}AF-{uniprot_id}-F1-model_v{version}.pdb"
    structure_filename = wget.download(query_url, out=out_dir)

    if rename:
        extension = ".pdb" if pdb else ".cif"
        os.rename(
            structure_filename, Path(out_dir) / f"{uniprot_id}{extension}"
        )
        structure_filename = str(
            (Path(out_dir) / f"{uniprot_id}{extension}").resolve()
        )

    log.info(f"Downloaded AlphaFold PDB file for: {uniprot_id}")
    if aligned_score:
        score_query = (
            BASE_URL
            + "AF-"
            + uniprot_id
            + f"-F1-predicted_aligned_error_v{version}.json"
        )
        score_filename = wget.download(score_query, out=out_dir)
        if rename:
            os.rename(score_filename, Path(out_dir) / f"{uniprot_id}.json")
            score_filename = str(
                (Path(out_dir) / f"{uniprot_id}.json").resolve()
            )
        return structure_filename, score_filename

    return structure_filename


def download_alphafold_multiprocessing(
    uniprot_ids: List[str],
    version: int = 3,
    out_dir: str = ".",
    rename: bool = True,
    pdb: bool = True,
    mmcif: bool = False,
    aligned_score: bool = False,
    overwrite: bool = True,
    max_workers: int = 16,
    chunksize: int = 32,
) -> List[Union[str, Tuple[str, str]]]:
    """
    Parallelised download of AF2 structures from AlphaFold EBI database.

    :param uniprot_ids: List of UniProt IDs of desired proteins.
    :type uniprot_ids: List[str]
    :param version: AF2 release version to use
    :type version: int
    :param out_dir: string specifying desired output location. Default is pwd
        (``"."``).
    :type out_dir: str
    :param rename: boolean specifying whether to rename the output file to the
        uniprot_id. Default is ``True``.
    :type rename: bool
    :param pdb: boolean specifying whether to download the PDB file. Default is
        ``True``.
    :type pdb: bool
    :param mmcif: Bool specifying whether to download MMCiF or PDB. Default is
        ``False``.
    :type mmcif: bool
    :param aligned_score: Bool specifying whether or not to download score.
        Default is ``False``.
    :type aligned_score: bool
    :param overwrite: Bool specifying whether to overwrite existing files.
        Currently unused. #TODO
    :type overwrite: bool
    :param max_workers: Number of workers to use for multiprocessing. Default is
        ``16``.
    :type max_workers: int
    :param chunksize: Number of structures to download per worker. Default is
        ``32``.
    :type chunksize: int
    :return: List of paths to downloaded structures. List of tuples if
        downloading both PDB and score.
    :rtype: List[Union[str, Tuple[str, str]]]
    """
    func = partial(
        download_alphafold_structure,
        out_dir=out_dir,
        version=version,
        rename=rename,
        pdb=pdb,
        mmcif=mmcif,
        aligned_score=aligned_score,
    )
    return process_map(
        func, uniprot_ids, max_workers=max_workers, chunksize=chunksize
    )


@lru_cache()
def load_af2_metadata(path: Optional[str] = None) -> pd.DataFrame:
    """
    Download the download_metadata.json file from the AlphaFoldDB FTP server:
    https://ftp.ebi.ac.uk/pub/databases/alphafold/

    :param path: Path to save the metadata file to.
    :type path: Optional[str]
    :return: Metadata dataframe describing the proteome info.
    :rtype: pd.DataFrame
    """
    if path is None:
        return pd.read_json(
            "http://ftp.ebi.ac.uk/pub/databases/alphafold/download_metadata.json"
        )
    out_path = Path(path)
    if not os.path.exists(out_path / "download_metadata.csv"):
        print(
            f"Downloading metadata... to: {str(out_path / 'download_metadata.csv')}"
        )
        df = pd.read_json(
            "http://ftp.ebi.ac.uk/pub/databases/alphafold/download_metadata.json"
        )
        df.to_csv(out_path / "download_metadata.csv")
        return df
    else:
        print(
            f"Loading Metadata from disk... ({str(out_path / 'download_metadata.csv')})"
        )
        return pd.read_csv(out_path / "download_metadata.csv")
