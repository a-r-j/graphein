"""Tests for graphein.molecule.rdkit_utils"""

from pathlib import Path

import numpy as np
import rdkit

import graphein.molecule as gm
import graphein.molecule.utils as u

TEST_GRAPH = gm.construct_graph(
    smiles="CC1=C(C2=C(CCC(O2)(C)COC3=CC=C(C=C3)CC4C(=O)NC(=O)S4)C(=C1O)C)C",
    generate_conformer=True,
)
TEST_MOL_GRAPH = gm.construct_graph(
    path=str(Path(__file__).parent / "test_data/short_test.mol2")
)


def test_get_center():
    center = u.get_center(TEST_MOL_GRAPH)
    assert isinstance(
        center, np.ndarray
    ), f"Center is not a numpy array ({type(center)}"
    assert center.shape == (3,), f"Center has wrong shape ({center.shape})"
    np.testing.assert_allclose(
        np.array([-0.21303333, 0.06743333, 0.0818]), center
    )


def test_get_shape_moments():
    moments = u.get_shape_moments(TEST_MOL_GRAPH)
    assert isinstance(
        moments, tuple
    ), f"Moment is not a tuple ({type(moments)})"
    assert isinstance(
        moments[0], float
    ), f"Moment is not a float ({type(moments[0])})"
    assert isinstance(
        moments[1], float
    ), f"Moment is not a float ({type(moments[1])})"
    assert moments == (
        0.12940962096031391,
        0.8705903790396854,
    ), f"Moments are not correct ({moments})"


def test_count_fragment():
    count = u.count_fragments(TEST_GRAPH)
    assert isinstance(count, int), f"Count is not an int ({type(count)})"
    assert count == 1, f"Count is not 1 ({count})"


def test_get_max_ring_size():
    size = u.get_max_ring_size(TEST_GRAPH)
    assert isinstance(size, int), f"Max ring size is not an int ({type(size)})"
    assert size == 6, f"Max ring size is not 6 ({size})"


def test_get_morgan_fp():
    fp = u.get_morgan_fp(TEST_GRAPH)
    assert isinstance(
        fp, rdkit.DataStructs.cDataStructs.ExplicitBitVect
    ), f"fp is not a BitVect ({type(fp)})"


def test_get_morgan_fp_np():
    fp = u.get_morgan_fp_np(TEST_GRAPH)
    assert isinstance(fp, np.ndarray), f"fp is not a numpy array ({type(fp)})"


def test_get_qed_score():
    qed = u.get_qed_score(TEST_GRAPH)
    assert isinstance(qed, float), f"QED is not a float ({type(qed)})"
    np.testing.assert_almost_equal(qed, 0.7166041254699328)
