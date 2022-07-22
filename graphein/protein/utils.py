"""Provides utility functions for use across Graphein."""
from __future__ import annotations

import logging

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import os
import tempfile
from functools import lru_cache, partial
from pathlib import Path
from shutil import which
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError
from urllib.request import urlopen

import networkx as nx
import pandas as pd
import wget
from biopandas.pdb import PandasPdb
from tqdm.contrib.concurrent import process_map

from .resi_atoms import BACKBONE_ATOMS, RESI_THREE_TO_1

log = logging.getLogger(__name__)


class ProteinGraphConfigurationError(Exception):
    """
    Exception when an invalid Graph configuration if provided to a downstream
    function or method.
    """

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


@lru_cache()
def get_obsolete_mapping() -> Dict[str, str]:
    """Returns a dictionary mapping obsolete PDB codes to their replacement.

    :return: Dictionary mapping obsolete PDB codes to their replacement.
    :rtype: Dictionary[str, str]
    """
    obs_dict: Dict[str, str] = {}

    response = urlopen("ftp://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat")
    for line in response:
        entry = line.split()
        if len(entry) == 4:
            obs_dict[entry[2].lower().decode("utf-8")] = (
                entry[3].lower().decode("utf-8")
            )
    return obs_dict


def download_pdb_multiprocessing(
    pdb_codes: List[str],
    out_dir: Union[str, Path],
    overwrite: bool = False,
    strict: bool = False,
    max_workers: int = 16,
    chunksize: int = 32,
) -> List[Path]:
    """Downloads PDB structures in parallel.

    :param pdb_codes: List of PDB codes to download.
    :type pdb_codes: List[str]
    :param out_dir: Path to directory to download PDB structures to.
    :type out_dir: Union[str, Path]
    :param overwrite: Whether to overwrite existing files, defaults to
        ``False``.
    :type overwrite: bool
    :param strict: Whether to check for successful download of each file,
        defaults to ``False``.
    :type strict: bool
    :param max_workers: Number of workers to uses, defaults to 16
    :type max_workers: int
    :param chunksize: Chunk to split list into for each worker, defaults to 32
    :type chunksize: int
    :return: List of Paths to downloaded PDB files.
    :rtype: List[Path]
    """
    out_dir: Path = Path(out_dir)
    func = partial(
        download_pdb, out_dir=out_dir, overwrite=overwrite, strict=strict
    )
    return process_map(
        func, pdb_codes, max_workers=max_workers, chunksize=chunksize
    )


def download_pdb(
    pdb_code: str,
    out_dir: Optional[Union[str, Path]] = None,
    check_obsolete: bool = False,
    overwrite: bool = False,
    strict: bool = True,
) -> Path:
    """
    Download PDB structure from PDB.

    If no structure is found, we perform a lookup against the record of
    obsolete PDB codes (ftp://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat)

    :param pdb_code: 4 character PDB accession code.
    :type pdb_code: str
    :param out_dir: Path to directory to download PDB structure to. If ``None``,
        will download to a temporary directory.
    :type out_dir: Optional[Union[str, Path]]
    :param check_obsolete: Whether to check for obsolete PDB codes,
        defaults to ``False``. If an obsolete PDB code is found, the updated PDB
        is downloaded.
    :type check_obsolete: bool
    :param overwrite: If ``True``, will overwrite existing files.
    :type overwrite: bool
    :param strict: If ``True``, will raise an exception if the PDB file is not
        found.
    :type strict: bool
    :return: returns filepath to downloaded structure.
    :rtype: Path
    """
    pdb_code = pdb_code.lower()

    # Make output directory if it doesn't exist or set it to tempdir if None
    if out_dir is not None:
        out_dir = Path(out_dir)
    else:
        out_dir = Path(tempfile.TemporaryDirectory().name)

    os.makedirs(Path(out_dir), exist_ok=True)

    if check_obsolete:
        obs_map = get_obsolete_mapping()
        try:
            new_pdb = obs_map[pdb_code.lower()].lower()
            log.info(
                "{pdb_code} is deprecated. Downloading {new_pdb} instead."
            )
            return download_pdb(new_pdb, out_dir, overwrite=overwrite)
        except KeyError:
            log.warning(
                f"PDB {pdb_code} not found. Possibly too large; large structures are only provided as mmCIF files."
            )
            return

    # Check if PDB already exists
    if os.path.exists(out_dir / f"{pdb_code}.pdb") and not overwrite:
        log.info(f"{pdb_code} already exists: {out_dir / f'{pdb_code}.pdb'}")
        return out_dir / f"{pdb_code}.pdb"

    # Download
    try:
        wget.download(
            f"https://files.rcsb.org/download/{pdb_code}.pdb",
            out=str(out_dir / f"{pdb_code}.pdb"),
        )
    except HTTPError:
        log.info(f"PDB {pdb_code} not found.")

    # Check file exists
    if strict:
        assert os.path.exists(
            out_dir / f"{pdb_code}.pdb"
        ), "{pdb_code} download failed. Not found in {out_dir}"
    log.info("{pdb_code} downloaded to {out_dir}")
    return out_dir / f"{pdb_code}.pdb"


def get_protein_name_from_filename(pdb_path: str) -> str:
    """
    Extracts a filename from a ``pdb_path``

    :param pdb_path: Path to extract filename from.
    :type pdb_path: str
    :return: file name.
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
    Filter function for DataFrame.

    Filters the DataFrame such that the ``by_column`` values have to be
    in the ``list_of_values`` list if ``boolean == True``, or not in the list
    if ``boolean == False``.

    :param dataframe: pd.DataFrame to filter.
    :type dataframe: pd.DataFrame
    :param by_column: str denoting column of DataFrame to filter.
    :type by_column: str
    :param list_of_values: List of values to filter with.
    :type list_of_values: List[Any]
    :param boolean: indicates whether to keep or exclude matching
        ``list_of_values``. ``True`` -> in list, ``False`` -> not in list.
    :type boolean: bool
    :returns: Filtered DataFrame.
    :rtype: pd.DataFrame
    """
    df = dataframe.copy()
    df = df[df[by_column].isin(list_of_values) == boolean]
    df.reset_index(inplace=True, drop=True)

    return df


def compute_rgroup_dataframe(pdb_df: pd.DataFrame) -> pd.DataFrame:
    """Return the atoms that are in R-groups and not the backbone chain.

    :param pdb_df: DataFrame to compute R group dataframe from.
    :type pdb_df: pd.DataFrame
    :returns: DataFrame containing R-groups only (backbone atoms removed).
    :rtype: pd.DataFrame
    """
    return filter_dataframe(pdb_df, "atom_name", BACKBONE_ATOMS, False)


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

    if not mmcif and not pdb:
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


def three_to_one_with_mods(res: str) -> str:
    """
    Converts three letter AA codes into 1 letter. Allows for modified residues.

    See: :const:`~graphein.protein.resi_atoms.RESI_THREE_TO_1`.

    :param res: Three letter residue code string.
    :type res: str
    :return: 1-letter residue code.
    :rtype: str
    """
    return RESI_THREE_TO_1[res]


def save_graph_to_pdb(
    g: nx.Graph,
    path: str,
    gz: bool = False,
    atoms: bool = True,
    hetatms: bool = True,
):
    """Saves processed ``pdb_df`` (``g.graph["pdb_df"]``) dataframe to a PDB
    file.

    N.B. PDBs do not contain connectivity information. This only captures the
    nodes in the graph. Connectivity is filled in according to standard rules by
    visualisation programs.

    :param g: Protein graph to save dataframe from.
    :type g: nx.Graph
    :param path: Path to save PDB file to.
    :type path: str
    :param gz: Whether to gzip the file. Defaults to ``False``.
    :type gz: bool
    """
    ppd = PandasPdb()
    atom_df = filter_dataframe(
        g.graph["pdb_df"], "record_name", ["ATOM"], boolean=True
    )
    hetatm_df = filter_dataframe(
        g.graph["pdb_df"], "record_name", ["HETATM"], boolean=True
    )
    if atoms:
        ppd.df["ATOM"] = atom_df
    if hetatms:
        ppd.df["HETATM"] = hetatm_df
    ppd.to_pdb(path=path, records=None, gz=gz, append_newline=True)
    log.info(f"Successfully saved graph to {path}")


def save_pdb_df_to_pdb(
    df: pd.DataFrame,
    path: str,
    gz: bool = False,
    atoms: bool = True,
    hetatms: bool = True,
):
    """Saves pdb dataframe to a PDB file.

    :param g: DataFrame to save as PDB
    :type g: pd.DataFrame
    :param path: Path to save PDB file to.
    :type path: str
    :param gz: Whether to gzip the file. Defaults to ``False``.
    :type gz: bool
    """
    atom_df = filter_dataframe(df, "record_name", ["ATOM"], boolean=True)
    hetatm_df = filter_dataframe(df, "record_name", ["HETATM"], boolean=True)
    ppd = PandasPdb()
    if atoms:
        ppd.df["ATOM"] = atom_df
    if hetatms:
        ppd.df["HETATM"] = hetatm_df
    ppd.to_pdb(path=path, records=None, gz=gz, append_newline=True)
    log.info(f"Successfully saved PDB dataframe to {path}")


def save_rgroup_df_to_pdb(
    g: nx.Graph,
    path: str,
    gz: bool = False,
    atoms: bool = True,
    hetatms: bool = True,
):
    """Saves R-group (``g.graph["rgroup_df"]``) dataframe to a PDB file.

    N.B. PDBs do not contain connectivity information.
    This only captures the atoms in the r groups.
    Connectivity is filled in according to standard rules by visualisation
    programs.

    :param g: Protein graph to save R group dataframe from.
    :type g: nx.Graph
    :param path: Path to save PDB file to.
    :type path: str
    :param gz: Whether to gzip the file. Defaults to ``False``.
    :type gz: bool
    """
    ppd = PandasPdb()
    atom_df = filter_dataframe(
        g.graph["rgroup_df"], "record_name", ["ATOM"], boolean=True
    )
    hetatm_df = filter_dataframe(
        g.graph["rgroup_df"], "record_name", ["HETATM"], boolean=True
    )
    if atoms:
        ppd.df["ATOM"] = atom_df
    if hetatms:
        ppd.df["HETATM"] = hetatm_df
    ppd.to_pdb(path=path, records=None, gz=gz, append_newline=True)
    log.info(f"Successfully saved rgroup data to {path}")


def is_tool(name: str) -> bool:
    """Checks whether ``name`` is on ``PATH`` and is marked as an executable.

    Source: https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script

    :param name: Name of program to check for execution ability.
    :type name: str
    :return: Whether ``name`` is on PATH and is marked as an executable.
    :rtype: bool
    """
    return which(name) is not None
