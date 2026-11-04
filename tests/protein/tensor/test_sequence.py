"""Tests for graphein.protein.tensor.sequence."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import pandas as pd
import pytest
from biopandas.pdb import PandasPdb

from graphein.protein.resi_atoms import RESI_THREE_TO_1
from graphein.protein.tensor.io import protein_df_to_tensor
from graphein.protein.tensor.sequence import (
    get_atom_indices,
    get_residue_id,
    get_sequence,
    infer_residue_types,
    residue_ids,
    residue_type_tensor,
)

try:
    import torch  # noqa: F401

    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_infer_sequence():
    # Get Protein
    p = PandasPdb().fetch_pdb("4hhb")
    df = p.df["ATOM"]
    df.head()

    sequence = get_residue_id(df)
    sequence = "".join(
        [RESI_THREE_TO_1[res.split(":")[1]] for res in sequence]
    )

    coords = protein_df_to_tensor(df)
    inferred_sequence = infer_residue_types(coords)

    assert sequence == inferred_sequence, "Sequences do not match."


def _df():
    return pd.DataFrame(
        {
            "chain_id": ["A", "A", "A", "B"],
            "residue_name": ["ALA", "ALA", "GLY", "SER"],
            "residue_number": [1, 1, 1, 7],
            "insertion": ["", "", "A", ""],
        }
    )


def test_residue_ids():
    df = _df()
    ids = residue_ids(df)
    assert ids.tolist() == ["A:ALA:1:", "A:ALA:1:", "A:GLY:1:A", "B:SER:7:"]
    # The IDs are stored and reused
    assert df.residue_id.tolist() == ids.tolist()
    assert residue_ids(df, insertions=False).tolist() == ids.tolist()

    assert residue_ids(_df(), insertions=False).tolist() == [
        "A:ALA:1",
        "A:ALA:1",
        "A:GLY:1",
        "B:SER:7",
    ]
    df = _df()
    df["insertion"] = [1, 1, 2, 3]  # Non-string insertion codes
    assert residue_ids(df).tolist()[-1] == "B:SER:7:3"

    assert get_residue_id(_df()) == ["A:ALA:1:", "A:GLY:1:A", "B:SER:7:"]
    assert get_sequence(_df()) == "AGS"
    assert get_sequence(_df(), per_atom=True, list_of_three=True) == [
        "ALA",
        "ALA",
        "GLY",
        "SER",
    ]


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_residue_type_tensor():
    vocabulary = ["G", "A", "X"]
    # SER is not in the vocabulary and so maps to "X"
    assert residue_type_tensor(_df(), vocabulary=vocabulary).tolist() == [
        1,
        0,
        2,
    ]
    one_hot = residue_type_tensor(_df(), vocabulary=vocabulary, one_hot=True)
    assert one_hot.shape == (3, 3)


def test_get_atom_indices_cache():
    get_atom_indices.cache_clear()
    for _ in range(3):
        indices = get_atom_indices()
        inverted = get_atom_indices(invert=True)
    assert get_atom_indices.cache_info().misses == 2
    assert inverted[indices["ALA"]] == "ALA"


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_infer_residue_types_ignores_oxt_and_unknown():
    from graphein.protein.resi_atoms import ATOM_NUMBERING_MODIFIED

    x = torch.full((2, 37, 3), 1e-5)
    for i in get_atom_indices()["ALA"]:
        x[:, i] = torch.rand(3) + 1
    x[0, ATOM_NUMBERING_MODIFIED["OXT"]] = 1.0
    x[1, ATOM_NUMBERING_MODIFIED["NZ"]] = 1.0  # ALA + NZ is no residue
    assert infer_residue_types(x, return_sequence=False) == ["ALA", "UNK"]
