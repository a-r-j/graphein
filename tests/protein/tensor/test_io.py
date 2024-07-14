"""Tests for graphein.protein.tensor.io"""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import os
from pathlib import Path

import pytest
from biopandas.pdb import PandasPdb
from pandas.testing import assert_frame_equal

from graphein.protein.tensor import Protein
from graphein.protein.tensor.io import (
    protein_df_to_chain_tensor,
    protein_df_to_tensor,
    protein_to_pyg,
    to_dataframe,
    to_pdb,
)
from graphein.protein.tensor.sequence import get_residue_id

try:
    import torch

    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False

PDB_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "test_data" / "4hhb.pdb"
)
CIF_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "test_data" / "4hhb.cif"
)


def get_example_df():
    p = PandasPdb().fetch_pdb("3EIY")
    return p.df["ATOM"]


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_protein_df_to_chain_tensor():
    df = get_example_df()
    num_chains = len(df.chain_id.unique())

    num_residues = len(get_residue_id(df))

    chains = protein_df_to_chain_tensor(df, one_hot=True)
    print(chains)
    assert (
        chains.shape[0] == num_residues
    ), "Number of residues and chain IDs do not match."
    assert (
        chains.shape[1] == num_chains
    ), "Number of chains do not match dimension"
    assert chains.max() == 1, "Chain IDs are not one hot."
    if num_chains > 1:
        assert chains.min() == 0, "Chain IDs are not one hot."

    chain = protein_df_to_chain_tensor(df, one_hot=False)
    assert (
        chain.shape[0] == num_residues
    ), "Number of residues and chain IDs do not match."
    assert chain.max() == num_chains - 1, "Chain IDs are not zero-indexed."
    assert chain.min() == 0, "Chain IDs are not zero-indexed."


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_protein_df_to_tensor():  # sourcery skip: extract-duplicate-method
    df = get_example_df()

    num_residues = len(get_residue_id(df))
    print(get_residue_id(df))
    assert num_residues == 174, "Incorrect number of residues."

    positions = protein_df_to_tensor(df)
    assert positions.shape[0] == num_residues, "Incorrect number of residues."
    assert positions.shape[1] == 37, "Incorrect number of atoms."
    assert positions.shape[2] == 3, "Incorrect number of coordinates."

    # Backbone only
    atoms_to_keep = ["N", "CA", "C", "O"]
    positions = protein_df_to_tensor(df, atoms_to_keep=atoms_to_keep)
    assert positions.shape[0] == num_residues, "Incorrect number of residues."
    assert positions.shape[1] == 4, "Incorrect number of atoms."
    assert positions.shape[2] == 3, "Incorrect number of coordinates."


def test_to_pdb():
    protein = Protein().from_pdb_code("4hhb")
    to_pdb(protein.coords, "test.pdb")
    assert os.path.exists("test.pdb"), "File does not exist"

    ppdb1 = PandasPdb().read_pdb("test.pdb")
    ppdb2 = PandasPdb().fetch_pdb("4hhb")

    assert_frame_equal(
        ppdb1.df["ATOM"][["x_coord", "y_coord", "z_coord"]][:50],
        ppdb2.df["ATOM"][["x_coord", "y_coord", "z_coord"]][:50],
    )

    assert_frame_equal(
        ppdb1.df["ATOM"][["atom_name", "residue_name", "element_symbol"]][:50],
        ppdb2.df["ATOM"][["atom_name", "residue_name", "element_symbol"]][:50],
    )


def test_pdb_to_pyg():
    pyg_object = protein_to_pyg(PDB_DATA_PATH)


def test_cif_to_pyg():
    pyg_object = protein_to_pyg(CIF_DATA_PATH)


def test_pdb_and_cif_parsing():
    pdb_pyg = protein_to_pyg(PDB_DATA_PATH)
    cif_pyg = protein_to_pyg(CIF_DATA_PATH)
    assert pdb_pyg.coords.shape == cif_pyg.coords.shape
