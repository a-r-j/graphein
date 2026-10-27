import numpy as np
import pytest
from rdkit.Chem.rdchem import BondType

from graphein.molecule.config import MoleculeGraphConfig
from graphein.molecule.features.edges.bonds import add_bond_type_one_hot
from graphein.molecule.graphs import construct_graph

pytest.importorskip("rdkit")




def _graph():
    return construct_graph(config=MoleculeGraphConfig(), smiles="c1ccccc1C=O")


def test_add_bond_type_one_hot_array():
    g = _graph()
    seen = set()
    for u, v, d in g.edges(data=True):
        feat = add_bond_type_one_hot(u, v, d)
        assert isinstance(feat, np.ndarray)
        assert feat.sum() == 1
        assert d["bond_type_one_hot"] is feat
        seen.add(d["bond"].GetBondType())
    assert BondType.AROMATIC in seen and BondType.DOUBLE in seen


def test_add_bond_type_one_hot_series_and_custom_set():
    g = _graph()
    u, v, d = next(iter(g.edges(data=True)))
    s = add_bond_type_one_hot(u, v, d, return_array=False)
    assert list(s.index) == [
        BondType.SINGLE,
        BondType.DOUBLE,
        BondType.TRIPLE,
        BondType.AROMATIC,
    ]
    out = add_bond_type_one_hot(u, v, d, allowable_set=[BondType.TRIPLE])
    assert out.tolist() == [0]


def test_add_bond_type_one_hot_missing_bond():
    d = {}
    assert add_bond_type_one_hot("a", "b", d) is None
    assert d["bond_type_one_hot"] is None
