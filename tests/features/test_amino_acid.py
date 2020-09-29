from pandas.testing import assert_series_equal

from graphein.features.amino_acid import (
    expasy_protein_scale,
    load_expasy_scales,
)


def test_expasy_protein_scale():
    scales = load_expasy_scales()
    scale = expasy_protein_scale(n="A13LEU", d={"residue_name": "LEU"})

    assert_series_equal(scale, scales["LEU"])
