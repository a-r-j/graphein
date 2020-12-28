"""Provides utility functions for use across Graphein"""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import os
from Bio.PDB import PDBList


def download_pdb(config, pdb_code: str) -> str:
    """
    Download PDB structure from PDB

    :param pdb_code: 4 character PDB accession code
    :type pdb_code: str
    :return: # todo impl return
    """
    if not config.pdb_dir:
        config.pdb_dir = "/tmp/"

    # Initialise class and download pdb file
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(
        pdb_code, pdir=config.pdb_dir, overwrite=True, file_format="pdb"
    )
    # Rename file to .pdb from .ent
    os.rename(
        config.pdb_dir + "pdb" + pdb_code + ".ent",
        config.pdb_dir + pdb_code + ".pdb",
    )
    # Assert file has been downloaded
    assert any(pdb_code in s for s in os.listdir(config.pdb_dir))
    print(f"Downloaded PDB file for: {pdb_code}")
    return config.pdb_dir + pdb_code + ".pdb"
