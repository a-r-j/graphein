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


def test_insertion_handling():
    configs = {
        "granularity": "CA",
        "keep_hets": False,
        "insertions": False,
        "verbose": False,
        "node_metadata_functions": [meiler_embedding, expasy_protein_scale],
        "edge_construction_functions": [
            add_peptide_bonds,
            add_hydrogen_bond_interactions,
            add_ionic_interactions,
            add_aromatic_sulphur_interactions,
            add_hydrophobic_interactions,
            add_cation_pi_interactions,
        ],
    }

    config = ProteinGraphConfig(**configs)

    # This is a nasty PDB with a lot of insertions and altlocs
    g = construct_graph(config=config, pdb_code="6OGE")

    assert len(g.graph["sequence_A"]) + len(g.graph["sequence_B"]) + len(
        g.graph["sequence_C"]
    ) + len(g.graph["sequence_D"]) + len(g.graph["sequence_E"]) == len(g)
    assert g.graph["coords"].shape[0] == len(g)


def test_edges_do_not_add_nodes_for_chain_subset():
    new_funcs = {
        "edge_construction_functions": [
            add_peptide_bonds,
            add_hydrogen_bond_interactions,
            add_disulfide_interactions,
            add_ionic_interactions,
            add_aromatic_interactions,
            add_aromatic_sulphur_interactions,
            add_cation_pi_interactions,
        ],
    }
    config = ProteinGraphConfig(**new_funcs)
    g = construct_graph(config=config, pdb_code="2vvi", chain_selection="A")
    assert len(g) == 217
    g = construct_graph(config=config, pdb_code="2vvi", chain_selection="B")
    assert len(g) == 219
    g = construct_graph(config=config, pdb_code="2vvi", chain_selection="C")
    assert len(g) == 222
    g = construct_graph(config=config, pdb_code="2vvi", chain_selection="D")
    assert len(g) == 219
