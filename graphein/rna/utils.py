"""Utility Functions for working with RNA Secondary Structure Graphs."""

import os
import shutil
from typing import Tuple

import wget

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
# This submodule is heavily inspired by: https://github.com/emalgorithm/rna-design/blob/aec77a18abe4850958d6736ec185a6f8cbfdf20c/src/util.py#L9
from loguru import logger as log

BP_RNA_1M_DB_URL: str = (
    "http://bprna.cgrb.oregonstate.edu/bpRNA_1m/dbnFiles.zip"
)
BP_RNA_1M_90_DB_URL: str = "http://bprna.cgrb.oregonstate.edu/bpRNA_1m_90.zip"


def download_bp_rna_1m(out_path: str = "/tmp/"):
    """Downloads BP RNA Dotbrackets.

    :param out_path: _description_, defaults to "/tmp/"
    :type out_path: str, optional
    """
    if os.path.isdir(f"{out_path}/bp_rna_1m"):
        log.info(
            f"Skipping download. bp_rna_1m already downloaded to {out_path}/bp_rna_1m."
        )
    else:
        log.info(f"Downloading bp_rna_1m to {out_path}/bp_rna_1m...")
        fname = f"{out_path}/bp_rna_1m"
        wget.download(BP_RNA_1M_DB_URL, out=f"{fname}.zip")
        log.info("Unzipping bp_rna_1m...")
        shutil.unpack_archive(
            f"{fname}.zip", extract_dir=f"{out_path}/bp_rna_1m"
        )


def read_dbn_file(path: str) -> Tuple[str, str]:
    """
    Read a dotbracket file to sequence and dotbracket strings.

    :param path: Path to the dotbracket file in ``.dbn`` format.
    :type path: str
    :return: Tuple of sequence and dotbracket strings.
    :rtype: Tuple[str, str]
    """
    with open(path, "r") as f:
        lines = f.readlines()
    print(lines)
    sequence = lines[-2]
    dotbracket = lines[-1]
    return sequence, dotbracket


if __name__ == "__main__":
    # download_bp_rna_1m(".")
    seq, db = read_dbn_file("./bp_rna_1m/dbnFiles/bpRNA_CRW_23156.dbn")
    print(seq)
    print(db)
