"""Tests for graphein.protein.graphs"""

from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from graphein.protein.config import ProteinGraphConfig, ProteinMeshConfig
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
    compute_esm_embedding,
)
from graphein.protein.features.sequence.sequence import molecular_weight
from graphein.protein.graphs import (
    add_nodes_to_graph,
    compute_chain_graph,
    compute_secondary_structure_graph,
    construct_graph,
    construct_graphs_mp,
    read_pdb_to_dataframe,
    remove_alt_locs,
)
from graphein.protein.meshes import parse_pymol_commands

PDB_DATA_PATH = Path(__file__).resolve().parent / "test_data" / "4hhb.pdb"
CIF_DATA_PATH = Path(__file__).resolve().parent / "test_data" / "4hhb.cif"


def generate_graph():
    """Generate PDB network.
    This is a helper function.
    """
    return construct_graph(path=str(PDB_DATA_PATH))


@pytest.fixture(scope="module")
def net():
    """Generate proteingraph from 4HHB.pdb."""
    return generate_graph()


@pytest.fixture()
def pdb_df():
    """Generate pdb_df from 4HHB.pdb."""
    return read_pdb_to_dataframe(PDB_DATA_PATH)


@pytest.fixture()
def cif_to_pdb_df():
    """Generate pdb_df from 4HHB.cif."""
    return read_pdb_to_dataframe(CIF_DATA_PATH)


def test_nodes_are_strings(net):
    """
    Checks to make sure that the nodes are a string.
    For expediency, checks only 1/4 of the nodes.
    """
    for n in net.nodes():
        assert isinstance(n, str)


def test_pdb_vs_cif_file_parsing():
    """Generate graph from cif and pdb file and compare them"""
    G_pdb = construct_graph(path=str(PDB_DATA_PATH))
    G_cif = construct_graph(path=str(CIF_DATA_PATH))
    assert len(G_cif.nodes()) == len(G_pdb.nodes())


# Example-based Graph Construction test
def test_construct_graph():
    """Example-based test that graph construction works correctly.

    Uses 4hhb PDB file as an example test case.
    """
    G = construct_graph(path=str(PDB_DATA_PATH))
    assert isinstance(G, nx.Graph)
    assert len(G) == 574

    # Check number of peptide bonds
    peptide_bond_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d["kind"] == {"peptide_bond"}
    ]
    assert len(peptide_bond_edges) == 570


def test_construct_graph_with_dssp():
    """Makes sure protein graphs can be constructed with dssp

    Uses uses both a pdb code (6YC3) and a local pdb file to do so.
    """
    dssp_config_functions = {
        "edge_construction_functions": [
            add_peptide_bonds,
            add_aromatic_interactions,
            add_hydrogen_bond_interactions,
            add_disulfide_interactions,
            add_ionic_interactions,
            add_aromatic_sulphur_interactions,
            add_cation_pi_interactions,
        ],
        "graph_metadata_functions": [asa, rsa],
        "node_metadata_functions": [
            meiler_embedding,
            partial(expasy_protein_scale, add_separate=True),
        ],
    }

    dssp_prot_config = ProteinGraphConfig(**dssp_config_functions)

    g_pdb = construct_graph(
        config=dssp_prot_config, pdb_code="6yc3"
    )  # should download 6yc3.pdb to pdb_dir

    assert g_pdb.graph["pdb_code"] == "6yc3"
    assert g_pdb.graph["path"] is None
    assert g_pdb.graph["name"] == g_pdb.graph["pdb_code"]
    assert len(g_pdb.graph["dssp_df"]) == 1365

    file_path = str(
        Path(__file__).parent / "test_data" / "alphafold_structure.pdb"
    )
    g_local = construct_graph(config=dssp_prot_config, path=file_path)

    assert g_local.graph["pdb_code"] is None
    assert g_local.graph["path"] == file_path
    assert g_local.graph["name"] == "alphafold_structure"
    assert len(g_local.graph["dssp_df"]) == 382


def test_construct_graphs_mp():
    graph_list = [
        "2olg",
        "1bjq",
        "1omr",
        "1a4g",
        "2je9",
        "3vm5",
        "1el1",
        "3fzo",
        "1mn1",
        "1ff5",
        "1fic",
        "3a47",
        "1bir",
    ] * 5

    g = construct_graphs_mp(
        pdb_code_it=graph_list, config=ProteinGraphConfig(), return_dict=True
    )
    assert isinstance(g, dict)
    assert len(g.keys()) == len(graph_list) / 5
    for v in g.values():
        assert isinstance(v, (nx.Graph, None))
    g = construct_graphs_mp(
        pdb_code_it=graph_list, config=ProteinGraphConfig(), return_dict=False
    )
    assert isinstance(g, list)
    assert len(g) == len(graph_list)


def test_chain_selection():
    """Example-based test that chain selection works correctly.

    Uses 4hhb PDB file as an example test case.
    """
    file_path = Path(__file__).parent / "test_data" / "4hhb.pdb"
    G = construct_graph(path=str(file_path))

    # Check default construction contains all chains
    assert G.graph["chain_ids"] == ["A", "B", "C", "D"]
    # Check nodes contain residues from chains
    for n, d in G.nodes(data=True):
        assert d["chain_id"] in ["A", "B", "C", "D"]

    # Check graph contains only chain selection
    G = construct_graph(path=str(file_path), chain_selection=["A", "D"])
    assert G.graph["chain_ids"] == ["A", "D"]
    # Check nodes only contain residues from chain selection
    for n, d in G.nodes(data=True):
        assert d["chain_id"] in ["A", "D"]


# Edge construction tests
# Removed - testing with GetContacts as a dependency is not a priority right now
"""
def test_intramolecular_edges():
    Example-based test that intramolecular edge construction using GetContacts
    works correctly.

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
    G = construct_graph(path=str(file_path), config=config)
    # Todo complete
"""


def test_distance_edges():
    """Example-based test that distance-based edge construction works correctly,
    and that edges fulfill the distance threshold condition.

    Uses 4hhb and P04629 PDB file as an example test case.
    """
    file_path = Path(__file__).parent / "test_data" / "4hhb.pdb"

    edge_functions = {
        "edge_construction_functions": [
            partial(add_k_nn_edges, k=5, long_interaction_threshold=10),
            add_hydrophobic_interactions,
            # Todo removed for now as ring centroids require precomputing
            add_aromatic_interactions,
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
    G1 = construct_graph(path=str(file_path), config=config)
    assert G1 is not None

    """Use AlphaFold structure."""
    file_path = Path(__file__).parent / "test_data" / "P04629.pdb"
    threshold_distance = 10.0
    new_edge_funcs = {
        "edge_construction_functions": [
            partial(
                add_distance_threshold,
                long_interaction_threshold=-1,
                threshold=threshold_distance,
            )
        ]
    }
    config = ProteinGraphConfig(**new_edge_funcs)
    G2 = construct_graph(
        path=file_path,
        config=config,
    )
    assert G2 is not None
    # Check that the distance values of each edge are less than the threshold
    for u, v, d in G2.edges(data=True):
        assert d["distance"] < threshold_distance


# Featurisation tests
def test_node_features():
    # Todo this test requires attention
    # Tests node featurisers for a residue graph:
    # Amino acid features, ESM embedding, DSSP features, aaindex features

    file_path = Path(__file__).parent / "test_data" / "4hhb.pdb"

    config_params = {
        "node_metadata_functions": [
            expasy_protein_scale,  # Todo we need to refactor node data
            # assignment flow
            meiler_embedding,
        ],
        "graph_metadata_functions": [
            rsa,
            asa,
            phi,
            psi,
            secondary_structure,
            # partial(aaindex1, accession="FAUJ880111"),
        ],
    }
    config = ProteinGraphConfig(**config_params)
    G = construct_graph(path=str(file_path), config=config)

    # Check for existence of features
    for _, d in G.nodes(data=True):
        # Todo these functions return pd.Series, rather than adding to the node
        assert "meiler" in d
        assert "expasy" in d
        assert "rsa" in d
        assert "asa" in d
        assert "phi" in d
        assert "psi" in d
        assert "ss" in d
        continue


@pytest.mark.skip(reason="Pretrained model download is large.")
def test_sequence_features():
    # Tests sequence featurisers for a residue graph:
    # ESM and BioVec embeddings, propy and sequence descriptors
    file_path = Path(__file__).parent / "test_data" / "4hhb.pdb"

    sequence_feature_functions = {
        "graph_metadata_functions": [
            # esm_sequence_embedding,
            # esm_residue_embedding,
            biovec_sequence_embedding,
            molecular_weight,
        ]
    }
    config = ProteinGraphConfig(**sequence_feature_functions)
    G = construct_graph(path=str(file_path), config=config)

    # Check for existence on sequence-based features as node-level features
    # for n, d in G.nodes(data=True):
    # Todo this can probably be improved.
    # This only checks for the existence and shape of the esm_embedding for each
    # node
    # assert "esm_embedding" in d
    # assert len(d["esm_embedding"]) == 1280

    # Check for existence of sequence-based features as Graph-level features
    for chain in G.graph["chain_ids"]:
        assert f"sequence_{chain}" in G.graph
        # assert f"esm_embedding_{chain}" in G.graph
        assert f"biovec_embedding_{chain}" in G.graph
        assert f"molecular_weight_{chain}" in G.graph


# Checks that the sequence is extracted correctly from PDB file
def test_graph_sequence_feature():
    pdb_dir = Path(__file__).parent / "test_data"
    pdb = "4hhb"

    g_atom = construct_graph(
        pdb_code=pdb,
        config=ProteinGraphConfig(
            pdb_dir=pdb_dir,
            granularity="atom",  # atomistic
        ),
    )
    g_res = construct_graph(
        pdb_code=pdb,
        config=ProteinGraphConfig(
            pdb_dir=pdb_dir,
            granularity="CA",  # residue
        ),
    )

    for c in g_atom.graph["chain_ids"]:
        # assert sequences are equal
        assert g_atom.graph[f"sequence_{c}"] == g_res.graph[f"sequence_{c}"]


def test_insertion_and_alt_loc_handling():
    configs = {
        "granularity": "CA",
        "keep_hets": [],
        "insertions": False,
        "alt_locs": "max_occupancy",
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


def test_alt_loc_exclusion():
    configs = {
        "granularity": "CA",
        "keep_hets": [],
        "insertions": True,
        "alt_locs": "max_occupancy",
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

    # This is a PDB with three altlocs
    g = construct_graph(config=config, pdb_code="2VVI")

    # Test altlocs are dropped
    assert len(set(g.nodes())) == len(g.nodes())

    # Test the correct one is left
    for opt, expected_coords, node_id in (
        ("max_occupancy", [5.850, -9.326, -42.884], "A:CYS:195:A"),
        ("min_occupancy", [5.864, -9.355, -42.943], "A:CYS:195:B"),
        ("first", [5.850, -9.326, -42.884], "A:CYS:195:A"),
        ("last", [5.864, -9.355, -42.943], "A:CYS:195:B"),
    ):
        config.alt_locs = opt
        g = construct_graph(config=config, pdb_code="2VVI")
        assert np.array_equal(
            g.nodes[node_id]["coords"],
            np.array(expected_coords, dtype=np.float32),
        )


def test_alt_loc_inclusion():
    configs = {
        "granularity": "CA",
        "keep_hets": [],
        "insertions": False,
        "alt_locs": True,
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

    # This is a PDB with an altloc leading to different residues
    g = construct_graph(config=config, pdb_code="1ALX")

    # Test both are present
    assert "A:TYR:11:A" in g.nodes() and "A:TRP:11:B" in g.nodes()

    # TODO Test on other PDBs where altlocs are of the same residues


def _alt_loc_df(occupancy_a: float, occupancy_b: float, n_atoms: int = 20):
    """One residue with a blank-``alt_loc`` backbone and a side chain in two
    conformers, ``A`` and ``B``, listed atom by atom as in PDB files."""
    rows = [(atom, "") for atom in ("N", "CA", "C", "O")]
    rows += [(f"X{i}", alt) for i in range(n_atoms) for alt in "AB"]
    occupancy = {"": 1.0, "A": occupancy_a, "B": occupancy_b}
    return pd.DataFrame(
        {
            "record_name": "ATOM",
            "atom_number": np.arange(1, len(rows) + 1),
            "atom_name": [atom for atom, _ in rows],
            "alt_loc": [alt for _, alt in rows],
            "residue_name": "LYS",
            "chain_id": "A",
            "residue_number": 1,
            "insertion": "",
            "x_coord": 0.0,
            "y_coord": 0.0,
            "z_coord": 0.0,
            "occupancy": [occupancy[alt] for _, alt in rows],
        }
    )


@pytest.mark.parametrize(
    "keep,occupancies,expected",
    [
        ("max_occupancy", (0.5, 0.5), "A"),
        ("min_occupancy", (0.5, 0.5), "A"),
        ("max_occupancy", (0.4, 0.6), "B"),
        ("min_occupancy", (0.4, 0.6), "A"),
        ("first", (0.5, 0.5), "A"),
        ("last", (0.5, 0.5), "B"),
    ],
)
def test_remove_alt_locs_keeps_one_conformer(keep, occupancies, expected):
    """Atoms are not picked from different conformers, even when the
    conformers' occupancies are tied, and file order is kept."""
    df = _alt_loc_df(*occupancies)
    out = remove_alt_locs(df, keep=keep)
    assert set(out["alt_loc"]) == {"", expected}
    assert len(out) == 4 + 20
    assert out["atom_number"].is_monotonic_increasing


@pytest.mark.parametrize(
    "keep,expected",
    [
        ("max_occupancy", "A"),
        ("min_occupancy", "B"),
        ("first", "A"),
        ("last", "B"),
    ],
)
def test_remove_alt_locs_conformer_not_atom(keep, expected):
    """The conformer is chosen per residue, not per atom: here conformer
    ``B`` has the higher occupancy for one atom and is a different residue
    with an atom ``A`` lacks (microheterogeneity, e.g. Pro/Ser)."""
    df = _alt_loc_df(0.6, 0.4, n_atoms=4)
    df.loc[(df["atom_name"] == "X0") & (df["alt_loc"] == "B"), "occupancy"] = (
        0.7
    )
    extra = df.iloc[[-1]].assign(
        atom_name="OG", alt_loc="B", residue_name="SER", atom_number=99
    )
    df = pd.concat([df, extra], ignore_index=True)
    out = remove_alt_locs(df, keep=keep)
    assert set(out["alt_loc"]) == {"", expected}
    assert ("OG" in set(out["atom_name"])) == (expected == "B")


def test_remove_alt_locs_exclude():
    """``"exclude"`` drops every atom with an ``alt_loc``, including atoms
    found in only one conformer."""
    df = _alt_loc_df(0.5, 0.5, n_atoms=4)
    df = pd.concat(
        [df, df.iloc[[-1]].assign(atom_name="OG", atom_number=99)],
        ignore_index=True,
    )
    out = remove_alt_locs(df, keep="exclude")
    assert out["atom_name"].tolist() == ["N", "CA", "C", "O"]


@pytest.mark.parametrize("pdb_code", ["1AKE", "3NIR"])
def test_alt_locs_not_mixed_within_residue(pdb_code):
    """1AKE has side chains in two conformers at 0.5 occupancy each (e.g.
    Arg167). 3NIR has per-atom occupancies and Pro/Ser microheterogeneity
    (A22). Each residue must keep atoms from one conformer only."""
    config = ProteinGraphConfig(
        granularity="atom", edge_construction_functions=[]
    )
    g = construct_graph(config=config, pdb_code=pdb_code)
    df = g.graph["pdb_df"]
    conformers = (
        df.loc[df["alt_loc"] != ""]
        .groupby(["chain_id", "residue_number", "insertion"])["alt_loc"]
        .nunique()
    )
    assert not conformers.empty
    assert (conformers == 1).all(), conformers[conformers > 1]


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
    g = construct_graph(config=config, pdb_code="2vvi", chain_selection=["A"])
    assert len(g) == 217
    g = construct_graph(config=config, pdb_code="2vvi", chain_selection=["B"])
    assert len(g) == 219
    g = construct_graph(config=config, pdb_code="2vvi", chain_selection=["C"])
    assert len(g) == 222
    g = construct_graph(config=config, pdb_code="2vvi", chain_selection=["D"])
    assert len(g) == 219


def test_secondary_structure_graphs():
    file_path = Path(__file__).parent / "test_data" / "4hhb.pdb"
    config = ProteinGraphConfig(
        edge_construction_functions=[
            add_hydrophobic_interactions,
            add_aromatic_interactions,
            add_disulfide_interactions,
            add_peptide_bonds,
            add_hydrogen_bond_interactions,
        ],
        graph_metadata_functions=[secondary_structure],
    )
    g = construct_graph(path=str(file_path), config=config)

    h = compute_secondary_structure_graph(g, remove_non_ss=False)
    # Check number of residues preserved
    res_counts = sum(d["residue_counts"] for _, d in h.nodes(data=True))
    assert res_counts == len(g), (
        "Residue counts in SS graph should match number of residues in original \
        graph"
    )
    assert nx.is_connected(h), (
        "SS graph should be connected in this configuration"
    )

    h = compute_secondary_structure_graph(
        g,
        remove_non_ss=False,
        remove_self_loops=False,
        return_weighted_graph=False,
    )
    assert len(g.edges) == len(h.edges), (
        "Multigraph should have same number of edges."
    )


def test_chain_graph():
    file_path = Path(__file__).parent / "test_data" / "4hhb.pdb"
    config = ProteinGraphConfig(
        edge_construction_functions=[
            add_hydrophobic_interactions,
            add_aromatic_interactions,
            add_disulfide_interactions,
            add_peptide_bonds,
            add_hydrogen_bond_interactions,
        ],
        graph_metadata_functions=[secondary_structure],
    )
    g = construct_graph(path=str(file_path), config=config)
    h = compute_chain_graph(g)
    assert len(h.edges) == len(g.edges), "Number of edges do not match"

    h = compute_chain_graph(g, return_weighted_graph=True)
    node_sum = sum(d["num_residues"] for _, d in h.nodes(data=True))
    assert node_sum == len(g), "Number of residues do not match"


def test_df_processing():
    def return_even_df(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[df["residue_number"] % 2 == 0]

    def remove_hetatms(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[df["record_name"] == "ATOM"]

    params_to_change = {
        "protein_df_processing_functions": [return_even_df, remove_hetatms],
        "granularity": "atom",
    }

    config = ProteinGraphConfig(**params_to_change)
    config.dict()

    config2 = ProteinGraphConfig(granularity="atom", deprotonate=True)

    g1 = construct_graph(config=config, pdb_code="3eiy")
    g2 = construct_graph(config=config2, pdb_code="3eiy")
    g3 = construct_graph(config=config2, pdb_code="4cvi")

    for n, d in g1.nodes(data=True):
        assert int(d["residue_number"]) % 2 == 0, (
            "Only even residues should be present"
        )

    assert len(g1) != len(g2), "Graphs should not be equal"
    for n, d in g3.nodes(data=True):
        assert d["element_symbol"] not in [
            "H",
            "D",
            "T",
        ], "No hydrogen isotopes should be present"

    config3 = ProteinGraphConfig(granularity="atom", deprotonate=False)
    g4 = construct_graph(config=config3, pdb_code="4cvi")
    has_H = []
    for n, d in g4.nodes(data=True):
        if d["element_symbol"] in {"H", "D", "T"}:
            has_H.append(n)
    assert len(has_H) > 0, "No hydrogen isotopes are present"


def test_weighted_chain_graph_does_not_modify_input():
    config = ProteinGraphConfig(
        edge_construction_functions=[
            partial(add_k_nn_edges, k=5),
            add_ionic_interactions,
        ]
    )
    g = construct_graph(config=config, path=str(PDB_DATA_PATH))
    before = {(u, v): set(d["kind"]) for u, v, d in g.edges(data=True)}
    compute_chain_graph(g, return_weighted_graph=True)
    assert {(u, v): d["kind"] for u, v, d in g.edges(data=True)} == before


@pytest.mark.parametrize(
    "path",
    [
        PDB_DATA_PATH,  # chains A-D in order
        # chains L, H, A (not in alphabetical order) and insertion codes
        Path(__file__).resolve().parent / "test_data" / "input_pdb_cryst1.pdb",
    ],
)
def test_centroids_match_their_residue(path):
    """Each node's centroid is the mean of its own residue's atoms."""
    config = ProteinGraphConfig(granularity="centroids")
    g = construct_graph(config=config, path=str(path))
    atoms = read_pdb_to_dataframe(path=str(path))
    atoms = atoms.loc[atoms["record_name"] == "ATOM"]
    keys = ["chain_id", "residue_number", "insertion"]
    expected = atoms.groupby(keys)[["x_coord", "y_coord", "z_coord"]].mean()
    nodes = g.graph["pdb_df"].set_index(keys)
    np.testing.assert_allclose(
        np.stack([g.nodes[n]["coords"] for n in nodes["node_id"]]),
        expected.loc[nodes.index].to_numpy(),
    )


def test_add_nodes_to_graph_verbose():
    g = construct_graph(path=str(PDB_DATA_PATH))
    h = nx.Graph(**g.graph)
    add_nodes_to_graph(h, verbose=True)
    assert set(h) == set(g)


def test_parse_pymol_commands_none():
    assert parse_pymol_commands(ProteinMeshConfig(pymol_commands=None)) == []


def test_compute_esm_embedding_invalid_representation():
    with pytest.raises(ValueError):
        compute_esm_embedding("MKV", representation="token")


def test_construct_graph_does_not_mutate_config():
    def mark(G):
        G.graph["marked"] = True
        return G

    config = ProteinGraphConfig()
    g = construct_graph(
        config=config, path=str(PDB_DATA_PATH), graph_annotation_funcs=[mark]
    )
    assert g.graph["marked"]
    assert config.graph_metadata_functions is None


def test_insertions_false_keeps_parent_residue():
    # H 52 (TRP) and 52A (TYR): remove_alt_locs left the frame sorted by
    # occupancy, so remove_insertions could drop 52 as 52A's duplicate.
    path = Path(__file__).parent / "test_data" / "input_pdb_cryst1.pdb"
    g = construct_graph(
        config=ProteinGraphConfig(insertions=False), path=path, verbose=False
    )
    assert "H:TRP:52" in g
    assert not any(n.startswith("H:TYR:52") for n in g)


def test_remove_alt_locs_keeps_order_and_first_of_ties():
    df = pd.DataFrame(
        {
            "chain_id": "A",
            "residue_number": [1, 1, 2, 2, 3],
            "atom_name": "CA",
            "insertion": "",
            "alt_loc": ["", "", "A", "B", ""],
            "occupancy": [1.0, 1.0, 0.5, 0.5, 0.2],
        }
    )
    df.loc[0, "insertion"] = "A"  # 1A, then 1
    out = remove_alt_locs(df)
    assert out["residue_number"].tolist() == [1, 1, 2, 3]
    assert out["alt_loc"].tolist() == ["", "", "A", ""]
    out = remove_alt_locs(df, keep="min_occupancy")
    assert out["alt_loc"].tolist() == ["", "", "A", ""]


def test_read_mmcif_keeps_file_order(tmp_path):
    # HETATM records (here MET A 32 as MSE) stay where they are in the
    # file, as for PDB files, rather than after every ATOM record.
    lines = []
    for line in CIF_DATA_PATH.read_text().splitlines():
        f = line.split()
        if f[:1] == ["ATOM"] and f[5:7] == ["MET", "A"] and f[8] == "32":
            line = line.replace("ATOM  ", "HETATM", 1).replace(
                " MET ", " MSE "
            )
        lines.append(line)
    cif = tmp_path / "4hhb_mse.cif"
    cif.write_text("\n".join(lines) + "\n")
    df = read_pdb_to_dataframe(path=str(cif))
    assert df["atom_number"].is_monotonic_increasing
    assert df.index.equals(pd.RangeIndex(len(df)))
    chain_a = df[df["chain_id"] == "A"]["residue_number"].drop_duplicates()
    assert chain_a.tolist()[30:33] == [31, 32, 33]
