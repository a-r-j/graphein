"""Tests for molecular graph construction"""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from graphein.molecule.config import MoleculeGraphConfig
from graphein.molecule.graphs import construct_graph
from graphein.utils.dependencies import import_message

try:
    import rdkit
except ImportError:
    import_message(
        "graphein.molecule.graphs", "rdkit", "rdkit", True, extras=True
    )

config = MoleculeGraphConfig()
add_hs_config = MoleculeGraphConfig(add_hs=True)
CONFIGS = [config, add_hs_config]


LONG_SDF = str(
    (Path(__file__).parent / "test_data" / "long_test.sdf").resolve()
)
SHORT_MOL2 = str(
    (Path(__file__).parent / "test_data" / "short_test.mol2").resolve()
)
SMILES = "N[C@H](CCCc1ccc(N(CCCl)CCCl)cc1)C(=O)O"
LONG_PDB = str(
    (Path(__file__).parent / "test_data" / "long_test.pdb").resolve()
)


@pytest.mark.parametrize("config", CONFIGS)
def test_generate_graph_sdf(config):
    """Tests graph construction from an SDF file."""
    g = construct_graph(config=config, path=LONG_SDF)
    assert isinstance(g, nx.Graph), f"{g} is not a graph"
    assert g.graph["name"] == "long_test"
    check_nodes(g)


@pytest.mark.parametrize("config", CONFIGS)
def test_generate_graph_smiles(config):
    """Tests graph construction from a SMILES string."""
    g = construct_graph(config=config, smiles=SMILES)
    assert g.graph["name"] == SMILES
    check_nodes(g)
    check_edges(g)


@pytest.mark.parametrize("config", CONFIGS)
def test_generate_graph_mol2(config):
    """Tests graph construction from a Mol2 file."""
    g = construct_graph(config=config, path=SHORT_MOL2)
    assert g.graph["name"] == "short_test"
    check_nodes(g)
    check_edges(g)


@pytest.mark.parametrize("config", CONFIGS)
def test_generate_graph_pdb(config):
    """Tests graph construction from a PDB file."""
    g = construct_graph(config=config, path=LONG_PDB)
    assert g.graph["name"] == "long_test"
    check_nodes(g)
    check_edges(g)


def check_nodes(g):
    # Check nodes
    for n, d in g.nodes(data=True):
        assert isinstance(
            d["atomic_num"], int
        ), f"{n} atomic_num is not an int"
        assert isinstance(d["element"], str), f"{n} element is not a string"
        assert isinstance(
            d["rdmol_atom"], rdkit.Chem.rdchem.Atom
        ), f"{n} rdmol_atom is not an rdmol.Atom"
        assert isinstance(
            d["atom_type_one_hot"], np.ndarray
        ), f"{n} atom_type_one_hot is not a numpy array"
        assert (
            sum(d["atom_type_one_hot"]) == 1
        ), f"{n} atom_type_one_hot is not a one-hot vector"


def check_edges(g):
    for u, v, d in g.edges(data=True):
        assert d["kind"] == {"bond"}, f"{u}-{v} kind is not bond"
        assert isinstance(
            d["bond"], rdkit.Chem.rdchem.Bond
        ), f"edge {u}-{v} does not have an RDKit bond associated with it."
