import sys
sys.path.append('/data/yuanqi/graphein')

from graphein.molecule.graphs import construct_graph
from graphein.molecule.config import MoleculeGraphConfig

config = MoleculeGraphConfig()

def test_generate_graph_sdf():
    g = construct_graph(config=config, sdf_path="/data/yuanqi/graphein/tests/molecule/test_data/long_test.sdf")
    return g 

def test_generate_graph_smiles():
    g = construct_graph(config=config, smiles="N[C@H](CCCc1ccc(N(CCCl)CCCl)cc1)C(=O)O")
    return g 

def test_generate_graph_mol2():
    g = construct_graph(config=config, mol2_path="/data/yuanqi/graphein/tests/molecule/test_data/short_test.mol2")
    return g 

def test_generate_graph_pdb():
    g = construct_graph(config=config, pdb_path="/data/yuanqi/graphein/tests/molecule/test_data/long_test.pdb")
    return g 

test_generate_graph_sdf()
test_generate_graph_smiles()
test_generate_graph_mol2()
test_generate_graph_pdb()
