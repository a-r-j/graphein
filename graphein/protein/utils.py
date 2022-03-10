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
from biopandas.pdb import PandasPdb

from .resi_atoms import BACKBONE_ATOMS, RESI_THREE_TO_1

log = logging.getLogger(__name__)


class ProteinGraphConfigurationError(Exception):
    """Exception when an invalid Graph configuration if provided to a downstream function or method."""

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


def download_pdb(config, pdb_code: str) -> Path:
    """
    Download PDB structure from PDB.

    :param pdb_code: 4 character PDB accession code.
    :type pdb_code: str
    :return: returns filepath to downloaded structure.
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
        config.pdb_dir / f"pdb{pdb_code}.ent",
        config.pdb_dir / f"{pdb_code}.pdb",
    )

    # Assert file has been downloaded
    assert any(pdb_code in s for s in os.listdir(config.pdb_dir))
    log.info(f"Downloaded PDB file for: {pdb_code}")
    return config.pdb_dir / f"{pdb_code}.pdb"


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
    Filter function for dataframe.

    Filters the dataframe such that the ``by_column`` values have to be
    in the ``list_of_values`` list if ``boolean == True``, or not in the list
    if ``boolean == False``.

    :param dataframe: pd.DataFrame to filter.
    :type dataframe: pd.DataFrame
    :param by_column: str denoting column of dataframe to filter.
    :type by_column: str
    :param list_of_values: List of values to filter with.
    :type list_of_values: List[Any]
    :param boolean: indicates whether to keep or exclude matching ``list_of_values``. ``True`` -> in list, ``False`` -> not in list.
    :type boolean: bool
    :returns: Filtered dataframe.
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
    :returns: Dataframe containing R-groups only (backbone atoms removed).
    :rtype: pd.DataFrame
    """
    return filter_dataframe(pdb_df, "atom_name", BACKBONE_ATOMS, False)


def download_alphafold_structure(
    uniprot_id: str,
    out_dir: str = ".",
    pdb: bool = True,
    mmcif: bool = False,
    aligned_score: bool = True,
) -> Union[str, Tuple[str, str]]:
    BASE_URL = "https://alphafold.ebi.ac.uk/files/"
    """
    Downloads a structure from the Alphafold EBI database (https://alphafold.ebi.ac.uk/files/").

    :param uniprot_id: UniProt ID of desired protein.
    :type uniprot_id: str
    :param out_dir: String specifying desired output location. Default is pwd.
    :type out_dir: str
    :param mmcif: Bool specifying whether to download ``MMCiF`` or ``PDB``. Default is ``False`` (downloads pdb).
    :type mmcif: bool
    :param retrieve_aligned_score: Bool specifying whether or not to download score alignment json.
    :type retrieve_aligned_score: bool
    :return: path to output. Tuple if several outputs specified.
    :rtype: Union[str, Tuple[str, str]]
    """
    if not mmcif and not pdb:
        raise ValueError("Must specify either mmcif or pdb.")
    if mmcif:
        query_url = f"{BASE_URL}AF-{uniprot_id}F1-model_v1.cif"
    if pdb:
        query_url = f"{BASE_URL}AF-{uniprot_id}-F1-model_v1.pdb"

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

    See: :const:`~graphein.protein.resi_atoms.RESI_THREE_TO_1`.

    :param res: Three letter residue code string.
    :type res: str
    :return: 1-letter residue code.
    :rtype: str
    """
    return RESI_THREE_TO_1[res]


def save_graph_to_pdb(g: nx.Graph, path: str, gz: bool = False):
    """Saves processed ``pdb_df`` (``g.graph["pdb_df"]``) dataframe to a PDB file.

    N.B. PDBs do not contain connectivity information.
    This only captures the nodes in the graph.
    Connectivity is filled in according to standard rules by visualisation programs.

    :param g: Protein graph to save dataframe from.
    :type g: nx.Graph
    :param path: Path to save PDB file to.
    :type path: str
    :param gz: Whether to gzip the file. Defaults to ``False``.
    :type gz: bool
    """
    ppd = PandasPdb()
    ppd.df["ATOM"] = g.graph["pdb_df"]
    ppd.to_pdb(path=path, records=None, gz=gz, append_newline=True)
    log.info(f"Successfully saved graph to {path}")


def save_pdb_df_to_pdb(df: pd.DataFrame, path: str, gz: bool = False):
    """Saves pdb dataframe to a PDB file.

    :param g: Dataframe to save as PDB
    :type g: pd.DataFrame
    :param path: Path to save PDB file to.
    :type path: str
    :param gz: Whether to gzip the file. Defaults to ``False``.
    :type gz: bool
    """
    ppd = PandasPdb()
    ppd.df["ATOM"] = df
    ppd.to_pdb(path=path, records=None, gz=False, append_newline=True)
    log.info(f"Successfully saved PDB dataframe to {path}")


def save_rgroup_df_to_pdb(g: nx.Graph, path: str, gz: bool = False):
    """Saves R-group (``g.graph["rgroup_df"]``) dataframe to a PDB file.

    N.B. PDBs do not contain connectivity information.
    This only captures the atoms in the r groups.
    Connectivity is filled in according to standard rules by visualisation programs.

    :param g: Protein graph to save R group dataframe from.
    :type g: nx.Graph
    :param path: Path to save PDB file to.
    :type path: str
    :param gz: Whether to gzip the file. Defaults to ``False``.
    :type gz: bool
    """
    ppd = PandasPdb()
    ppd.df["ATOM"] = g.graph["rgroup_df"]
    ppd.to_pdb(path=path, records=None, gz=gz, append_newline=True)
    log.info(f"Successfully saved rgroup data to {path}")
