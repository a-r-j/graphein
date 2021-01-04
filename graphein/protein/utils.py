"""Provides utility functions for use across Graphein"""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import os
from pathlib import Path

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
