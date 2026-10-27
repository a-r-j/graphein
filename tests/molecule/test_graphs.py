"""Tests for molecular graph construction"""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from graphein.molecule.config import MoleculeGraphConfig
from graphein.molecule.features.nodes.atom_type import num_heavy_atoms
from graphein.molecule.graphs import construct_graph
from graphein.utils.dependencies import import_message

try:
    import rdkit
    from rdkit import Chem
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


def test_generate_graph_path_preserves_full_file_stem(tmp_path):
    """Tests file-derived graph names preserve periods before the extension."""
    path = tmp_path / "long.test.sdf"
    path.write_bytes(Path(LONG_SDF).read_bytes())

    g = construct_graph(path=str(path))

    assert g.graph["name"] == "long.test"


@pytest.mark.parametrize("config", CONFIGS)
def test_generate_graph_smiles(config):
    """Tests graph construction from a SMILES string."""
    g = construct_graph(config=config, smiles=SMILES)
    assert g.graph["name"] == SMILES
    check_nodes(g)
    check_edges(g)


def test_readme_molecule_graph_example():
    """Tests the molecule construction shown in the README."""
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    config = MoleculeGraphConfig()
    g = construct_graph(smiles=smiles, config=config)

    assert g.graph["name"] == smiles
    assert g.number_of_nodes() > 0
    assert g.number_of_edges() > 0
    check_nodes(g)
    check_edges(g)


@pytest.mark.parametrize(
    "add_hs", [False, True], ids=["implicit_hs", "explicit_hs"]
)
def test_num_heavy_atoms(add_hs):
    """Tests heavy-atom counts with implicit and explicit hydrogens."""
    Chem = pytest.importorskip("rdkit.Chem")
    mol = Chem.MolFromSmiles("[He].CCC")
    if add_hs:
        mol = Chem.AddHs(mol)

    isolated_atom = mol.GetAtomWithIdx(0)
    central_atom = mol.GetAtomWithIdx(2)
    if add_hs:
        assert any(
            neighbor.GetSymbol() == "H"
            for neighbor in central_atom.GetNeighbors()
        )
    else:
        assert central_atom.GetNumImplicitHs() > 0

    for atom, expected_count in ((isolated_atom, 0), (central_atom, 2)):
        node_data = {"rdmol_atom": atom}
        assert num_heavy_atoms(atom.GetIdx(), node_data) == expected_count
        assert node_data["num_heavy_atoms"] == expected_count


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


@pytest.mark.parametrize("add_hs", [False, True])
def test_num_heavy_atoms(add_hs):
    """Tests heavy atom counts with implicit and explicit hydrogens."""
    mol = Chem.MolFromSmiles("C.CCC")
    if add_hs:
        mol = Chem.AddHs(mol)

    for atom_idx, expected in [(0, 0), (2, 2)]:
        node_data = {"rdmol_atom": mol.GetAtomWithIdx(atom_idx)}
        assert num_heavy_atoms(str(atom_idx), node_data) == expected
        assert node_data["num_heavy_atoms"] == expected


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
