"""Functions for working with Protein Structure Graphs. Based on tests written by Eric Ma in PIN Library"""

import copy
from functools import partial

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma, Charlie Harris
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from pathlib import Path
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import graphein.protein as gp
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import (
    INFINITE_DIST,
    add_aromatic_sulphur_interactions,
    add_cation_pi_interactions,
    add_hydrogen_bond_interactions,
    add_k_nn_edges,
    add_peptide_bonds,
    add_salt_bridges,
    add_sequence_distance_edges,
    add_vdw_clashes,
    add_vdw_interactions,
    filter_distmat,
    get_edges_by_bond_type,
    get_ring_atoms,
    get_ring_centroids,
)
from graphein.protein.edges.intramolecular import peptide_bonds
from graphein.protein.graphs import construct_graph, read_pdb_to_dataframe
from graphein.protein.resi_atoms import (
    AROMATIC_RESIS,
    BOND_TYPES,
    CATION_RESIS,
    HYDROGEN_BOND_ACCEPTORS,
    HYDROGEN_BOND_DONORS,
    HYDROPHOBIC_RESIS,
    NEG_AA,
    PI_RESIS,
    POS_AA,
    RESI_NAMES,
    RESI_THREE_TO_1,
    SALT_BRIDGE_ANIONS,
    SALT_BRIDGE_CATIONS,
    SULPHUR_RESIS,
    VDW_RADII,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "test_data" / "4hhb.pdb"


def generate_graph():
    """Generate PDB network.
    This is a helper function.
    """
    return construct_graph(path=str(DATA_PATH))


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


def test_add_hydrophobic_interactions(net):
    """Test the function add_hydrophobic_interactions_."""
    resis = get_edges_by_bond_type(net, "hydrophobic")
    for r1, r2 in resis:
        assert net.nodes[r1]["residue_name"] in HYDROPHOBIC_RESIS
        assert net.nodes[r2]["residue_name"] in HYDROPHOBIC_RESIS


def test_add_disulfide_interactions(net):
    """Test the function add_disulfide_interactions_."""
    resis = get_edges_by_bond_type(net, "disulfide")

    for r1, r2 in resis:
        assert net.nodes[r1]["residue_name"] == "CYS"
        assert net.nodes[r2]["residue_name"] == "CYS"


@pytest.mark.skip(reason="Not yet implemented.")
def test_delaunay_triangulation(net):
    """
    Test delaunay triangulation.
    I am including this test here that always passes because I don't know how
    best to test it. The code in pin.py uses scipy's delaunay triangulation.
    """
    pass


def _hbond_graph(
    granularity: str, df: Optional[pd.DataFrame] = None, **kwargs
) -> nx.Graph:
    config = ProteinGraphConfig(
        granularity=granularity,
        edge_construction_functions=[
            partial(add_hydrogen_bond_interactions, **kwargs)
        ],
    )
    if df is not None:
        return construct_graph(config=config, df=df, name="4hhb")
    return construct_graph(config=config, path=str(DATA_PATH))


def _is_donor(node: dict) -> bool:
    res, atom = node["residue_name"], node["atom_type"]
    if atom == "N":
        return RESI_THREE_TO_1.get(res) != "P"
    return atom in HYDROGEN_BOND_DONORS.get(res, {})


def _is_acceptor(node: dict) -> bool:
    res, atom = node["residue_name"], node["atom_type"]
    return atom in {"O", "OXT"} or atom in HYDROGEN_BOND_ACCEPTORS.get(res, {})


def _residue(node_id: str) -> str:
    """Residue ID of an atom node ID (i.e. drop the trailing atom name)."""
    return node_id.rsplit(":", 1)[0]


def _peptide_bonded(g: nx.Graph, res_1: str, res_2: str) -> bool:
    """Whether the C of one residue is bonded (<= 2A) to the N of the other."""
    for c_res, n_res in ((res_1, res_2), (res_2, res_1)):
        c, n = f"{c_res}:C", f"{n_res}:N"
        if c in g and n in g:
            if (
                np.linalg.norm(g.nodes[c]["coords"] - g.nodes[n]["coords"])
                <= 2
            ):
                return True
    return False


def test_add_hydrogen_bond_interactions_atom_graph():
    """Each atomic hbond edge links a donor and acceptor in different residues.

    Regression test for #437: previously only exact matches on truncated atom
    names were used and backbone atoms were never considered.
    """
    g = _hbond_graph("atom")
    edges = get_edges_by_bond_type(g, "hbond")
    assert len(edges) > 0

    for u, v in edges:
        a, b = g.nodes[u], g.nodes[v]
        assert (_is_donor(a) and _is_acceptor(b)) or (
            _is_donor(b) and _is_acceptor(a)
        )
        assert _residue(u) != _residue(v)
        assert np.linalg.norm(a["coords"] - b["coords"]) <= 3.5
        if {a["atom_type"], b["atom_type"]} <= {"N", "O", "OXT"}:
            assert not _peptide_bonded(g, _residue(u), _residue(v))

    # Backbone-backbone hydrogen bonds should be found (e.g. alpha helices).
    assert any(
        g.nodes[u]["atom_type"] in {"N", "O"}
        and g.nodes[v]["atom_type"] in {"N", "O"}
        for u, v in edges
    )


def test_add_hydrogen_bond_interactions_residue_graph():
    """Residue-level hbonds are the atomic hbonds mapped onto residues."""
    atom_edges = {
        frozenset((_residue(u), _residue(v)))
        for u, v in get_edges_by_bond_type(_hbond_graph("atom"), "hbond")
    }
    residue_edges = {
        frozenset(e)
        for e in get_edges_by_bond_type(_hbond_graph("CA"), "hbond")
    }
    assert len(residue_edges) > 0
    assert residue_edges == atom_edges


@pytest.mark.parametrize(
    "hbond_types", [["bb-bb"], ["bb-sc"], ["sc-sc"], ["bb-sc", "sc-sc"]]
)
def test_add_hydrogen_bond_interactions_types(hbond_types):
    """Only the requested backbone/side chain hydrogen bond types are added."""
    g = _hbond_graph("atom", hbond_types=hbond_types)
    edges = get_edges_by_bond_type(g, "hbond")
    assert len(edges) > 0
    for u, v in edges:
        n_backbone = sum(
            g.nodes[n]["atom_type"] in {"N", "O", "OXT"} for n in (u, v)
        )
        assert ["sc-sc", "bb-sc", "bb-bb"][n_backbone] in hbond_types


def test_add_hydrogen_bond_interactions_invalid_type():
    with pytest.raises(ValueError):
        _hbond_graph("atom", hbond_types=["bb-xx"])


def _helical_backbone_hbond(g: nx.Graph) -> Tuple[str, str]:
    """An ``O(i)``-``N(i+4)`` hbond edge in chain A as ``(O node, N node)``."""
    for u, v in get_edges_by_bond_type(g, "hbond"):
        if g.nodes[u]["atom_type"] == "N":
            u, v = v, u
        o, n = g.nodes[u], g.nodes[v]
        if (
            o["atom_type"] == "O"
            and n["atom_type"] == "N"
            and o["chain_id"] == n["chain_id"] == "A"
            and n["residue_number"] - o["residue_number"] == 4
        ):
            return u, v
    raise AssertionError("No O(i)-N(i+4) hydrogen bond found in chain A")


def test_add_hydrogen_bond_interactions_insertion_codes():
    """Residues numbered ``i, iA, iB, ...`` are not treated as adjacent.

    Renumbers residues ``i..i+4`` of a helical ``O(i)``-``N(i+4)`` hbond to
    ``i, iA, iB, iC, iD``; the hbond must still be found.
    """
    o, n = _helical_backbone_hbond(_hbond_graph("atom"))
    start = int(o.split(":")[2])

    df = read_pdb_to_dataframe(DATA_PATH)
    for offset, insertion in enumerate(["", "A", "B", "C", "D"]):
        mask = (df["chain_id"] == "A") & (
            df["residue_number"] == start + offset
        )
        df.loc[mask, "residue_number"] = start
        df.loc[mask, "insertion"] = insertion

    g = _hbond_graph("atom", df=df)
    # Residue i keeps its number and has no insertion code, so ``o`` is
    # unchanged.
    n_new = n.replace(f":{start + 4}:", f":{start}:D:")
    assert o in g and n_new in g
    assert g.has_edge(o, n_new)
    assert "hbond" in g.edges[o, n_new]["kind"]


@pytest.mark.parametrize("variant", ["HYP", "DPR"])
def test_add_hydrogen_bond_interactions_proline_variants(variant):
    """Backbone N of proline variants is not a hydrogen bond donor."""
    _, n = _helical_backbone_hbond(_hbond_graph("atom"))
    chain, res_name, res_num, _ = n.split(":")

    df = read_pdb_to_dataframe(DATA_PATH)
    mask = (df["chain_id"] == chain) & (df["residue_number"] == int(res_num))
    df.loc[mask, "residue_name"] = variant

    g = _hbond_graph("atom", df=df)
    n_new = f"{chain}:{variant}:{res_num}:N"
    assert n_new in g
    assert not any(
        "hbond" in g.edges[n_new, nbr]["kind"] for nbr in g.neighbors(n_new)
    )


def test_add_aromatic_interactions(net):
    """
    Tests the function add_aromatic_interactions_.
    The test checks that each residue in an aromatic interaction
    is one of the aromatic residues.
    """
    resis = get_edges_by_bond_type(net, "aromatic")
    for n1, n2 in resis:
        assert net.nodes[n1]["residue_name"] in AROMATIC_RESIS
        assert net.nodes[n2]["residue_name"] in AROMATIC_RESIS


def test_add_aromatic_sulphur_interactions():
    """Tests the function add_aromatic_sulphur_interactions_."""
    config = ProteinGraphConfig(
        edge_construction_functions=[add_aromatic_sulphur_interactions]
    )
    net = construct_graph(config=config, path=str(DATA_PATH))
    resis = get_edges_by_bond_type(net, "aromatic_sulphur")
    assert resis
    for n1, n2 in resis:
        condition1 = (
            net.nodes[n1]["residue_name"] in SULPHUR_RESIS
            and net.nodes[n2]["residue_name"] in AROMATIC_RESIS
        )

        condition2 = (
            net.nodes[n2]["residue_name"] in SULPHUR_RESIS
            and net.nodes[n1]["residue_name"] in AROMATIC_RESIS
        )

        assert condition1 or condition2


def test_add_ionic_interactions(net):
    """
    Tests the function add_ionic_interactions_.
    This test checks that residues involved in ionic interactions
    are indeed oppositely-charged.
    Another test is needed to make sure that ionic interactions
    are not missed.
    """
    resis = get_edges_by_bond_type(net, "ionic")
    for n1, n2 in resis:
        resi1 = net.nodes[n1]["residue_name"]
        resi2 = net.nodes[n2]["residue_name"]

        condition1 = resi1 in POS_AA and resi2 in NEG_AA
        condition2 = resi2 in POS_AA and resi1 in NEG_AA

        assert condition1 or condition2


@pytest.mark.skip(reason="Not yet implemented.")
def test_get_ring_centroids(pdb_df):
    """Test the function get_ring_centroids."""
    print(pdb_df)
    ring_atom_TYR = get_ring_atoms(pdb_df, "TYR")
    assert len(ring_atom_TYR) == 32
    centroid_TYR = get_ring_centroids(ring_atom_TYR)
    assert len(centroid_TYR) == 16

    ring_atom_PHE = get_ring_atoms(pdb_df, "PHE")
    assert len(ring_atom_PHE) == 36
    centroid_PHE = get_ring_centroids(ring_atom_PHE)
    assert len(centroid_PHE) == 18


def test_add_cation_pi_interactions():
    """Tests the function add_cation_pi_interactions."""
    config = ProteinGraphConfig(
        edge_construction_functions=[add_cation_pi_interactions]
    )
    net = construct_graph(config=config, path=str(DATA_PATH))
    resis = get_edges_by_bond_type(net, "cation_pi")
    assert resis
    for n1, n2 in resis:
        resi1 = net.nodes[n1]["residue_name"]
        resi2 = net.nodes[n2]["residue_name"]

        condition1 = resi1 in CATION_RESIS and resi2 in PI_RESIS
        condition2 = resi2 in CATION_RESIS and resi1 in PI_RESIS

        assert condition1 or condition2


def test_add_peptide_bonds():
    file_path = Path(__file__).parent.parent / "test_data/4hhb.pdb"
    # Peptide bonds are default
    G = construct_graph(path=str(file_path))

    for u, v in G.edges():
        assert abs(int(u.split(":")[2]) - int(v.split(":")[2])) == 1


def test_add_sequence_distance_edges():
    D = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    file_path = Path(__file__).parent.parent / "test_data/4hhb.pdb"

    for d in D:
        config = ProteinGraphConfig(
            edge_construction_functions=[
                partial(add_sequence_distance_edges, d=d)
            ]
        )
        G = construct_graph(path=str(file_path), config=config)
        for u, v in G.edges():
            assert abs(int(u.split(":")[2]) - int(v.split(":")[2])) == d


def test_salt_bridge_interactions():
    """
    Test to ensure salt bridges are only between anionic and cationic residues.
    Also checks distances are correct on atom graphs.
    """
    g = construct_graph(
        pdb_code="216L",
        config=ProteinGraphConfig(
            edge_construction_functions=[add_salt_bridges]
        ),
    )
    for u, v, d in g.edges(data=True):
        assert "salt_bridge" in d["kind"]
        res1 = g.nodes[u]["residue_name"]
        res2 = g.nodes[v]["residue_name"]

        assert (
            res1 in SALT_BRIDGE_CATIONS and res2 in SALT_BRIDGE_ANIONS
        ) or (res1 in SALT_BRIDGE_ANIONS and res2 in SALT_BRIDGE_CATIONS), (
            "Salt bridge not between cations and anions"
        )

    # Test distances on atom graphs
    g = construct_graph(
        pdb_code="216L",
        config=ProteinGraphConfig(
            edge_construction_functions=[add_salt_bridges], granularity="atom"
        ),
    )
    for u, v, d in g.edges(data=True):
        assert "salt_bridge" in d["kind"]
        res1 = g.nodes[u]["residue_name"]
        res2 = g.nodes[v]["residue_name"]

        assert (
            res1 in SALT_BRIDGE_CATIONS and res2 in SALT_BRIDGE_ANIONS
        ) or (res1 in SALT_BRIDGE_ANIONS and res2 in SALT_BRIDGE_CATIONS), (
            "Salt bridge not between cations and anions"
        )
        assert d["distance"] < 4.0, "Salt bridge distance is too long"


def test_vdw_interactions():
    """
    Test to ensure vdw interactions are identified and are of correct length in
    atomic graphs.
    """
    g = construct_graph(
        pdb_code="216L",
        config=ProteinGraphConfig(
            edge_construction_functions=[add_vdw_interactions]
        ),
    )
    for u, v, d in g.edges(data=True):
        assert "vdw" in d["kind"], "Edge not vdw"

    # Test distances on atom graphs
    g = construct_graph(
        pdb_code="216L",
        config=ProteinGraphConfig(
            edge_construction_functions=[add_vdw_interactions],
            granularity="atom",
        ),
    )
    for u, v, d in g.edges(data=True):
        assert "vdw" in d["kind"]
        res1 = g.nodes[u]["element_symbol"]
        res2 = g.nodes[v]["element_symbol"]
        rad1 = VDW_RADII[res1]
        rad2 = VDW_RADII[res2]

        assert res1 != "H" and res2 != "H", "Hydrogen in vdw interaction"
        assert d["distance"] < 0.5 + rad1 + rad2, (
            f"Vdw distance is too long: {d['distance']}"
        )


def test_vdw_clashes():
    """
    Test to ensure vdw clashes are identified and are of correct length in
    atomic graphs.
    """
    g = construct_graph(
        pdb_code="216L",
        config=ProteinGraphConfig(
            edge_construction_functions=[add_vdw_clashes]
        ),
    )
    for u, v, d in g.edges(data=True):
        assert "vdw_clash" in d["kind"], "Edge not vdw_clash"

    # Test distances on atom graphs
    g = gp.construct_graph(
        pdb_code="216L",
        config=gp.ProteinGraphConfig(
            edge_construction_functions=[add_vdw_clashes],
            granularity="atom",
        ),
    )
    for u, v, d in g.edges(data=True):
        assert "vdw_clash" in d["kind"]
        res1 = g.nodes[u]["element_symbol"]
        res2 = g.nodes[v]["element_symbol"]
        rad1 = VDW_RADII[res1]
        rad2 = VDW_RADII[res2]

        assert res1 != "H" and res2 != "H", "Hydrogen in vdw clash interaction"
        assert d["distance"] < rad1 + rad2, (
            f"Vdw clash distance is too long: {d['distance']}"
        )


def test_filter_distmat():
    pdb_df = pd.DataFrame(
        {
            "node_id": ["A:HIS:1", "A:TYR:2", "B:ALA:3"],
            "chain_id": ["A", "A", "B"],
        }
    )

    distmat = pd.DataFrame([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 2.0, 0.0]])

    exclude_edges_vals = [["intra"], ["inter"]]
    expected_distmats = [
        pd.DataFrame(
            [
                [0.0, INFINITE_DIST, 2.0],
                [INFINITE_DIST, 0.0, 2.0],
                [2.0, 2.0, 0.0],
            ]
        ),
        pd.DataFrame(
            [
                [0.0, 1.0, INFINITE_DIST],
                [1.0, 0.0, INFINITE_DIST],
                [INFINITE_DIST, INFINITE_DIST, 0.0],
            ]
        ),
    ]

    for vals, distmat_expected in zip(exclude_edges_vals, expected_distmats):
        distmat_out = filter_distmat(pdb_df, distmat, vals, False)
        pd.testing.assert_frame_equal(distmat_expected, distmat_out)


def test_add_k_nn_edges():
    pdb_df = pd.DataFrame(
        {
            "residue_number": [1, 2, 3, 4],
            "node_id": ["A:HIS:1", "A:TYR:2", "B:ALA:3", "B:ALA:4"],
            "chain_id": ["A", "A", "B", "B"],
            "x_coord": [1.0, 2.0, 4.0, 8.0],
            "y_coord": [0.0, 0.0, 0.0, 0.0],
            "z_coord": [0.0, 0.0, 0.0, 0.0],
        }
    )
    g_empty = nx.empty_graph(pdb_df["node_id"])
    g_empty.graph["pdb_df"] = pdb_df
    args_edges_pairs = [
        (
            dict(k=1),
            [
                ("A:HIS:1", "A:TYR:2"),
                ("A:TYR:2", "B:ALA:3"),
                ("B:ALA:3", "B:ALA:4"),
            ],
        ),
        (
            dict(k=2),
            [
                ("A:HIS:1", "A:TYR:2"),
                ("A:HIS:1", "B:ALA:3"),
                ("A:TYR:2", "B:ALA:3"),
                ("A:TYR:2", "B:ALA:4"),
                ("B:ALA:3", "B:ALA:4"),
            ],
        ),
        (
            dict(k=1, exclude_edges=["intra"]),
            [
                ("A:HIS:1", "B:ALA:3"),
                ("A:TYR:2", "B:ALA:3"),
                ("A:TYR:2", "B:ALA:4"),
            ],
        ),
        (
            dict(k=1, exclude_edges=["inter"]),
            [
                ("A:HIS:1", "A:TYR:2"),
                ("B:ALA:3", "B:ALA:4"),
            ],
        ),
        (
            dict(k=2, exclude_edges=["intra"], exclude_self_loops=False),
            [
                ("A:HIS:1", "B:ALA:3"),
                ("A:TYR:2", "B:ALA:3"),
                ("A:TYR:2", "B:ALA:4"),
                ("A:HIS:1", "A:HIS:1"),
                ("A:TYR:2", "A:TYR:2"),
                ("B:ALA:3", "B:ALA:3"),
                ("B:ALA:4", "B:ALA:4"),
            ],
        ),
        (
            dict(k=1, exclude_edges=["intra"], exclude_self_loops=False),
            [
                ("A:HIS:1", "A:HIS:1"),
                ("A:TYR:2", "A:TYR:2"),
                ("B:ALA:3", "B:ALA:3"),
                ("B:ALA:4", "B:ALA:4"),
            ],
        ),
    ]

    for args, edges_expected in args_edges_pairs:
        g = copy.deepcopy(g_empty)
        add_k_nn_edges(g, **args)
        edges_real = list(g.edges())
        assert set(edges_real) == set(edges_expected)


def test_insertion_codes_in_add_k_nn_edges():
    pdb_df = pd.DataFrame(
        {
            "residue_number": [1, 2, 3],
            "node_id": ["A:HIS:1", "A:TYR:2", "B:ALA:4:altB"],
            "chain_id": ["A", "A", "B"],
            "x_coord": [1.0, 2.0, 3.0],
            "y_coord": [4.0, 5.0, 6.0],
            "z_coord": [7.0, 8.0, 9.0],
        },
        index=[0, 1, 3],  # simulating dropped "B:ALA:4:altA"
    )
    g = nx.empty_graph(pdb_df["node_id"])
    g.graph["pdb_df"] = pdb_df
    add_k_nn_edges(g, k=3)

    edges_expected = [
        ("A:HIS:1", "A:TYR:2"),
        ("A:HIS:1", "B:ALA:4:altB"),
        ("A:TYR:2", "B:ALA:4:altB"),
    ]

    assert set(g.edges()) == set(edges_expected)


@pytest.mark.parametrize("peptide_fn", [add_peptide_bonds, peptide_bonds])
def test_peptide_bonds_keep_existing_edge_kinds(peptide_fn):
    """Adding peptide bonds must not overwrite kinds of existing edges."""
    knn = partial(add_k_nn_edges, k=3)
    kinds = []
    for funcs in ([knn, peptide_fn], [peptide_fn, knn]):
        config = ProteinGraphConfig(edge_construction_functions=funcs)
        g = construct_graph(config=config, path=str(DATA_PATH))
        kinds.append(
            {frozenset((u, v)): d["kind"] for u, v, d in g.edges(data=True)}
        )
    assert kinds[0] == kinds[1]
    assert any({"knn", "peptide_bond"} <= k for k in kinds[0].values())
