"""Tests for graphein.protein.features.nodes.amino_acids
"""
from pandas.testing import assert_series_equal

from graphein.protein.features.nodes.amino_acid import (
    expasy_protein_scale,
    load_expasy_scales,
)


def test_load_expasy_scale():
    """Example-based test for `load_expasy_scales`."""
    scales = load_expasy_scales()
    scale = expasy_protein_scale(n="A13LEU", d={"residue_name": "LEU"})

    assert_series_equal(scale, scales["LEU"])


def test_load_meilier_embeddings():
    """Example-based test for `load_meiler_embeddings`."""
    # The test implemented here should test that something about the meiler_embeddings csv file is true.


# An execution test is one that simply tests that the function executes.
# In other words, the _only_ thing we are guaranteeing here
# is that the function will execute without erroring out.
# We are not guaranteeing the correctness of the output.
# This can be modified.
def test_expasy_protein_scale():
    """Execution test for `expasy_protein_scale` function."""
    d = {"residue_name": "LEU"}
    n = "DUMMY"
    expasy_protein_scale(n, d)


# def test_aaindex_1_feat():
#     """Execution test for `aaindex_1_feat`."""
#     d = {"residue_name": "LEU"}
#     n = "DUMMY"
#     aaindex_1_feat(n, d, feature_name="KRIW790103")
