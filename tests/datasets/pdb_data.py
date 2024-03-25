import copy
import functools
import math

import pandas as pd
import pytest

from graphein.datasets import PDBManager
from graphein.protein.utils import read_fasta


@functools.lru_cache()
def init_class() -> PDBManager:
    return PDBManager("tmp/")


def assert_frame_not_equal(*args, **kwargs):
    try:
        pd.testing.assert_frame_equal(*args, **kwargs)
    except AssertionError:
        # frames are not equal
        pass
    else:
        # frames are equal
        raise AssertionError


def test_initialisation():
    manager = init_class()
    assert len(manager.df) == manager.get_num_chains()
    assert manager.get_longest_chain() > 16180
    assert manager.get_shortest_chain() < 2
    assert manager.get_num_unique_pdbs() > 201762
    assert len(manager.get_unique_pdbs()) == manager.get_num_unique_pdbs()
    assert set(manager.df.columns) == {
        "id",
        "pdb",
        "chain",
        "length",
        "molecule_type",
        "name",
        "sequence",
        "ligands",
        "source",
        "resolution",
        "experiment_type",
        "deposition_date",
        "split",
    }
    assert set(manager.get_experiment_types()) == {
        "diffraction",
        "NMR",
        "EM",
        "other",
    }


def test_sample():
    manager = copy.deepcopy(init_class())
    manager.sample(n=10, update=True)
    assert len(manager.df) == 10

    manager = copy.deepcopy(init_class())
    num_examples = len(manager.df)
    manager.sample(frac=0.01, update=True)
    assert len(manager.df) == math.ceil(num_examples * 0.01)


def test_molecule_type():
    manager = init_class()
    protein = manager.molecule_type("protein")
    assert protein.molecule_type.unique() == ["protein"]

    dna = manager.molecule_type("na")
    assert dna.molecule_type.unique() == ["na"]


def test_to_csv():
    manager = copy.deepcopy(init_class())
    manager.sample(n=10, update=True)
    manager.to_csv("tmp/out.csv")
    csv = pd.read_csv("tmp/out.csv", keep_default_na=False, na_values=["N/A"])
    pd.testing.assert_frame_equal(
        csv.reset_index(drop=True), manager.df.reset_index(drop=True)
    )


def test_reset():
    manager = copy.deepcopy(init_class())
    manager.sample(n=10, update=True)
    assert_frame_not_equal(manager.df, manager.source)
    manager.reset()
    pd.testing.assert_frame_equal(manager.df, manager.source)


def test_to_fasta():
    manager = copy.deepcopy(init_class())
    manager.sample(n=10, update=True)
    manager.to_fasta("tmp/out.fasta")

    fasta = read_fasta("tmp/out.fasta")
    for k, v in fasta.items():
        assert v == manager.df[manager.df.id == k].sequence.values[0]
    assert len(fasta) == len(manager.df)


def test_experiment_type():
    manager = init_class()
    diffraction = manager.experiment_type("diffraction")
    print(diffraction)
    assert diffraction.experiment_type.unique() == ["diffraction"]
    nmr = manager.experiment_type("NMR")
    assert nmr.experiment_type.unique() == ["NMR"]
    em = manager.experiment_type("EM")
    assert em.experiment_type.unique() == ["EM"]
