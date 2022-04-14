from pathlib import Path

from graphein.molecule.config import MoleculeGraphConfig
from graphein.molecule.graphs import construct_graph

config = MoleculeGraphConfig()


def test_generate_graph_sdf():
    g = construct_graph(
        config=config,
        sdf_path=str(
            (Path(__file__).parent / "test_data" / "long_test.sdf").resolve()
        ),
    )
    return g


def test_generate_graph_smiles():
    g = construct_graph(
        config=config, smiles="N[C@H](CCCc1ccc(N(CCCl)CCCl)cc1)C(=O)O"
    )
    return g


def test_generate_graph_mol2():
    g = construct_graph(
        config=config,
        mol2_path=str(
            (Path(__file__).parent / "test_data" / "short_test.mol2").resolve()
        ),
    )
    return g


def test_generate_graph_pdb():
    g = construct_graph(
        config=config,
        pdb_path=str(
            (Path(__file__).parent / "test_data" / "long_test.pdb").resolve()
        ),
    )
    return g
