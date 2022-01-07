"""Tests for graphein.protein.graphs"""

from functools import partial
from pathlib import Path

import networkx as nx
import pytest

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import (
    add_aromatic_interactions,
    add_aromatic_sulphur_interactions,
    add_cation_pi_interactions,
    add_delaunay_triangulation,
    add_distance_threshold,
    add_disulfide_interactions,
    add_hydrogen_bond_interactions,
    add_hydrophobic_interactions,
    add_ionic_interactions,
    add_k_nn_edges,
    add_peptide_bonds,
)
from graphein.protein.features.nodes.aaindex import aaindex1
from graphein.protein.features.nodes.amino_acid import (
    expasy_protein_scale,
    meiler_embedding,
)
from graphein.protein.features.nodes.dssp import (
    asa,
    phi,
    psi,
    rsa,
    secondary_structure,
)
from graphein.protein.features.sequence.embeddings import (
    biovec_sequence_embedding,
    esm_residue_embedding,
    esm_sequence_embedding,
)
from graphein.protein.features.sequence.sequence import molecular_weight
from graphein.protein.graphs import construct_graph, read_pdb_to_dataframe
from graphein.protein.utils import (
    extract_subgraph_from_atom_types,
    extract_subgraph_from_node_list,
    extract_subgraph_from_residue_types,
)

DATA_PATH = Path(__file__).resolve().parent / "test_data" / "4hhb.pdb"


def generate_graph():
    """Generate PDB network.
    This is a helper function.
    """
    return construct_graph(pdb_path=str(DATA_PATH))


@pytest.fixture(scope="module")
def net():
    """Generate proteingraph from 2VUI.pdb."""
    return generate_graph()


@pytest.fixture()
def pdb_df():
    """Generate pdb_df from 2VIU.pdb."""
    return read_pdb_to_dataframe(DATA_PATH)


def test_nodes_are_strings(net):
    """
    Checks to make sure that the nodes are a string.
    For expediency, checks only 1/4 of the nodes.
    """
    for n in net.nodes():
        assert isinstance(n, str)


# Example-based Graph Construction test
def test_construct_graph():
    """Example-based test that graph construction works correctly.

    Uses 4hhb PDB file as an example test case.
    """
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    G = construct_graph(pdb_path=str(file_path))
    assert isinstance(G, nx.Graph)
    assert len(G) == 574

    # Check number of peptide bonds
    peptide_bond_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d["kind"] == {"peptide_bond"}
    ]
    assert len(peptide_bond_edges) == 570


def test_chain_selection():
    """Example-based test that chain selection works correctly.

    Uses 4hhb PDB file as an example test case.
    """
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    G = construct_graph(pdb_path=str(file_path))

    # Check default construction contains all chains
    assert G.graph["chain_ids"] == ["A", "B", "C", "D"]
    # Check nodes contain residues from chains
    for n, d in G.nodes(data=True):
        assert d["chain_id"] in ["A", "B", "C", "D"]

    # Check graph contains only chain selection
    G = construct_graph(pdb_path=str(file_path), chain_selection="AD")
    assert G.graph["chain_ids"] == ["A", "D"]
    # Check nodes only contain residues from chain selection
    for n, d in G.nodes(data=True):
        assert d["chain_id"] in ["A", "D"]


# Edge construction tests
# Removed - testing with GetContacts as a dependency is not a priority right now
"""
def test_intramolecular_edges():
    Example-based test that intramolecular edge construction using GetContacts works correctly.

    Uses 4hhb PDB file as an example test case.

    file_path = Path(__file__).parent / "test_data/4hhb.pdb"

    edge_functions = {
        "edge_construction_functions": [
            hydrogen_bond,
            hydrophobic,
            peptide_bonds,
            pi_cation,
            pi_stacking,
            salt_bridge,
            t_stacking,
            van_der_waals,
        ]
    }
    config = ProteinGraphConfig(**edge_functions)
    G = construct_graph(pdb_path=str(file_path), config=config)
    # Todo complete
"""


def test_distance_edges():
    """Example-based test that distance-based edge construction works correctly

    Uses 4hhb PDB file as an example test case.
    """
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"

    edge_functions = {
        "edge_construction_functions": [
            partial(add_k_nn_edges, k=5, long_interaction_threshold=10),
            add_hydrophobic_interactions,
            add_aromatic_interactions,  # Todo removed for now as ring centroids require precomputing
            add_aromatic_sulphur_interactions,
            add_delaunay_triangulation,
            add_cation_pi_interactions,
            add_peptide_bonds,
            add_hydrogen_bond_interactions,
            add_disulfide_interactions,
            add_ionic_interactions,
            partial(
                add_distance_threshold,
                threshold=12,
                long_interaction_threshold=10,
            ),
        ]
    }
    config = ProteinGraphConfig(**edge_functions)
    G = construct_graph(pdb_path=str(file_path), config=config)
    assert G is not None


# Featurisation tests
def test_node_features():
    # Todo this test requires attention
    # Tests node featurisers for a residue graph:
    # Amino acid features, ESM embedding, DSSP features, aaindex features

    file_path = Path(__file__).parent / "test_data/4hhb.pdb"

    node_feature_functions = {
        "node_metadata_functions": [
            expasy_protein_scale,  # Todo we need to refactor node data assingment flow
            meiler_embedding,
            # rsa,
            # asa,
            # phi,
            # psi,
            # secondary_structure,
            # partial(aaindex1, accession="FAUJ880111"),
        ]
    }
    config = ProteinGraphConfig(**node_feature_functions)
    G = construct_graph(pdb_path=str(file_path), config=config)

    # Check for existence of features
    for n, d in G.nodes(data=True):
        # assert "meiler_embedding" in d # Todo these functions return pd.Series, rather than adding to the node
        # assert expasy_protein_scale in d
        # assert "rsa" in d
        # assert "asa" in d
        # assert "phi" in d
        # assert "psi" in d
        # assert "secondary_structure" in d
        continue


@pytest.mark.skip(reason="Pretrained model download is large.")
def test_sequence_features():
    # Tests sequence featurisers for a residue graph:
    # ESM and BioVec embeddings, propy and sequence descriptors
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"

    sequence_feature_functions = {
        "graph_metadata_functions": [
            # esm_sequence_embedding,
            # esm_residue_embedding,
            biovec_sequence_embedding,
            molecular_weight,
        ]
    }
    config = ProteinGraphConfig(**sequence_feature_functions)
    G = construct_graph(pdb_path=str(file_path), config=config)

    # Check for existence on sequence-based features as node-level features
    # for n, d in G.nodes(data=True):
    # Todo this can probably be improved.
    # This only checks for the existence and shape of the esm_embedding for each node
    # assert "esm_embedding" in d
    # assert len(d["esm_embedding"]) == 1280

    # Check for existence of sequence-based features as Graph-level features
    for chain in G.graph["chain_ids"]:
        assert f"sequence_{chain}" in G.graph
        # assert f"esm_embedding_{chain}" in G.graph
        assert f"biovec_embedding_{chain}" in G.graph
        assert f"molecular_weight_{chain}" in G.graph


def test_node_list_subgraphing():
    """Tests subgraph extraction from a list of nodes."""
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    NODE_LIST = ["C:ALA:28", "C:ARG:31", "D:LEU:75", "A:THR:38"]

    G = construct_graph(pdb_path=str(file_path))

    g = extract_subgraph_from_node_list(G, NODE_LIST, filter_dataframe=True)

    # Check we get back a graph and it contains the correct nodes
    assert isinstance(g, nx.Graph)
    assert len(g) == len(NODE_LIST)
    for n in g.nodes():
        assert n in NODE_LIST
    assert (
        g.graph["pdb_df"]["node_id"]
        .str.contains("|".join(NODE_LIST), case=True)
        .all()
    )

    # Check the list of nodes is the same as the list of nodes in the original graph
    returned_node_list = extract_subgraph_from_node_list(
        G, NODE_LIST, return_node_list=True
    )
    assert all(elem in NODE_LIST for elem in returned_node_list)

    # Check there is no overlap when we inverse the selection
    g = extract_subgraph_from_node_list(
        G, NODE_LIST, inverse=True, filter_dataframe=True
    )
    assert len(g) == len(G) - len(NODE_LIST)
    for n in g.nodes():
        assert n not in NODE_LIST

    assert not (
        g.graph["pdb_df"]["node_id"]
        .str.contains("|".join(NODE_LIST), case=True)
        .any()
    )

    returned_node_list = extract_subgraph_from_node_list(
        G, NODE_LIST, inverse=True, return_node_list=True
    )

    assert all(elem not in NODE_LIST for elem in returned_node_list)


def test_extract_subgraph_from_atom_types():
    """Tests subgraph extraction from a list of allowed atom types"""
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    ATOM_TYPES = ["C"]

    G = construct_graph(pdb_path=str(file_path))

    g = extract_subgraph_from_atom_types(G, ATOM_TYPES, filter_dataframe=True)
    assert isinstance(g, nx.Graph)
    assert len(g) == len(G)
    assert g == G

    # Test there are no N atoms
    ATOM_TYPES = ["N"]
    returned_node_list = extract_subgraph_from_atom_types(
        G, ATOM_TYPES, filter_dataframe=True, return_node_list=True
    )
    assert len(returned_node_list) == 0


def test_extract_subgraph_from_residue_types():
    """Tests subgraph extraction from a list of nodes."""
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    RESIDUE_TYPES = ["ALA", "SER", "GLY"]
    ALANINES = 72
    SERINES = 32
    GLYCINES = 40

    G = construct_graph(pdb_path=str(file_path))

    g = extract_subgraph_from_residue_types(
        G, RESIDUE_TYPES, filter_dataframe=True
    )

    # Check we get back a graph and it contains the correct nodes
    assert isinstance(g, nx.Graph)
    assert len(g) == ALANINES + SERINES + GLYCINES
    for n, d in g.nodes(data=True):
        assert d["residue_name"] in RESIDUE_TYPES
    assert (
        g.graph["pdb_df"]["residue_name"]
        .str.contains("|".join(RESIDUE_TYPES), case=True)
        .all()
    )

    assert (
        len([n for n, d in g.nodes(data=True) if d["residue_name"] == "ALA"])
        == ALANINES
    )
    assert (
        len([n for n, d in g.nodes(data=True) if d["residue_name"] == "GLY"])
        == GLYCINES
    )
    assert (
        len([n for n, d in g.nodes(data=True) if d["residue_name"] == "SER"])
        == SERINES
    )

    # Check the list of nodes is the same as the list of nodes in the original graph
    returned_node_list = extract_subgraph_from_node_list(
        G, RESIDUE_TYPES, return_node_list=True
    )
    assert all(elem in RESIDUE_TYPES for elem in returned_node_list)

    # Check there is no overlap when we inverse the selection
    g = extract_subgraph_from_residue_types(
        G, RESIDUE_TYPES, inverse=True, filter_dataframe=True
    )

    assert len(g) == len(G) - GLYCINES - ALANINES - SERINES
    for n in g.nodes():
        assert n not in RESIDUE_TYPES

    assert not (
        g.graph["pdb_df"]["residue_name"]
        .str.contains("|".join(RESIDUE_TYPES), case=True)
        .any()
    )

    returned_node_list = extract_subgraph_from_residue_types(
        G, RESIDUE_TYPES, inverse=True, return_node_list=True
    )

    assert all(elem not in RESIDUE_TYPES for elem in returned_node_list)


if __name__ == "__main__":
    test_extract_subgraph_from_residue_types()
