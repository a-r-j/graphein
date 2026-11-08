# basic
import functools
import pathlib

# test
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from Bio.PDB.DSSP import residue_max_acc

from graphein.protein import dssp_core
from graphein.protein.config import DSSPConfig, ProteinGraphConfig
from graphein.protein.edges import distance as D
from graphein.protein.features.nodes import rsa, secondary_structure
from graphein.protein.features.nodes.dssp import (
    DSSP_COLS,
    compute_dssp_df,
    phi,
)
from graphein.protein.graphs import (
    construct_graph,
    read_pdb_to_dataframe,
    sort_dataframe,
)
from graphein.protein.subgraphs import extract_surface_subgraph
from graphein.protein.utils import ProteinGraphConfigurationError

# ---------- input ----------
pdb_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "test_data"
    / "input_pdb_cryst1.pdb"
)
RSA_THRESHOLD = 0.2


@pytest.fixture
def dssp_graph():
    # ---------- graph config ----------
    params_to_change = {
        "granularity": "centroids",  # "atom", "CA", "centroids"
        "insertions": True,
        "edge_construction_functions": [
            # graphein.protein.edges.distance.add_peptide_bonds,
            D.add_distance_to_edges,
            D.add_hydrogen_bond_interactions,
            D.add_ionic_interactions,
            D.add_backbone_carbonyl_carbonyl_interactions,
            D.add_salt_bridges,
            # distance
            functools.partial(
                D.add_distance_threshold,
                long_interaction_threshold=4,
                threshold=4.5,
            ),
        ],
        "graph_metadata_functions": [rsa],
    }
    config = ProteinGraphConfig(**params_to_change)
    # ---------- construct graph ----------
    return construct_graph(config=config, path=pdb_path, verbose=False)


# ---------- test: dssp DataFrame ----------
def test_assert_nonempty_dssp_df(dssp_graph):
    """if not provided dssp version to dssp.add_dssp_df, will output an empty DataFrame"""
    if dssp_graph.graph["dssp_df"].empty:
        pytest.fail("DSSP dataframe is empty")


# ---------- test: surface subgraph nodes with insertion code ----------
def test_extract_surface_subgraph_insertion_node(dssp_graph):
    """if not added insertion codes, will raise ProteinGraphConfigurationError"""
    try:
        # without the modification, the following line will raise
        # ProteinGraphConfigurationError RSA not defined for all nodes (H:TYR:52:A).
        extract_surface_subgraph(dssp_graph, RSA_THRESHOLD)
    except ProteinGraphConfigurationError as e:
        pytest.fail(
            f"extract_surface_subgraph raised ProteinGraphConfigurationError:\n{e}"
        )


# ---------- test: native DSSP matches mkdssp ----------
# Rows of mkdssp 4.6.1's classic output for 1lds, as parsed by Biopython
# (all columns but "aa"): a terminus, two strand residues, a turn, C-terminus.
MKDSSP_1LDS_ROWS = [
    ("A", 0, " ", "-", 203, 360.0, 160.8, 1, 0, 0.0, 2, -0.3, 0, 0.0, 0, 0.0),
    (
        "A",
        8,
        " ",
        "E",
        77,
        -129.2,
        131.2,
        9,
        18,
        -2.1,
        18,
        -3.0,
        -2,
        -0.5,
        2,
        -0.5,
    ),
    (
        "A",
        22,
        " ",
        "E",
        39,
        -122.3,
        134.7,
        23,
        -10,
        -2.4,
        -10,
        -1.9,
        -2,
        -0.4,
        2,
        -0.5,
    ),
    (
        "A",
        60,
        " ",
        "T",
        131,
        -111.6,
        26.6,
        61,
        1,
        -0.1,
        2,
        -0.4,
        -29,
        -0.1,
        -1,
        -0.3,
    ),
    (
        "A",
        96,
        " ",
        "-",
        182,
        -95.6,
        360.0,
        97,
        -87,
        -0.6,
        -85,
        -0.1,
        -2,
        -0.5,
        -1,
        -0.1,
    ),
]


def test_compute_dssp_df_matches_mkdssp():
    path = pdb_path.parent / "1lds.pdb"
    df = compute_dssp_df(read_pdb_to_dataframe(path=str(path)))
    assert list(df.columns) == DSSP_COLS
    assert len(df) == 97
    rows = df.drop(columns="aa").iloc[[0, 8, 22, 60, 96]]
    assert [tuple(r) for r in rows.itertuples(index=False)] == MKDSSP_1LDS_ROWS


def test_deprecated_dssp_config_still_works():
    config = ProteinGraphConfig(
        dssp_config=DSSPConfig(),
        graph_metadata_functions=[secondary_structure],
    )
    with pytest.warns(FutureWarning):
        g = construct_graph(
            config=config, path=str(pdb_path.parent / "1lds.pdb")
        )
    assert all("ss" in d for _, d in g.nodes(data=True))


def _4hhb_df():
    return read_pdb_to_dataframe(path=str(pdb_path.parent / "4hhb.pdb"))


def test_compute_dssp_df_ignores_atom_serials():
    # Serials wrap past 99,999 atoms or restart per chain in merged files.
    df = _4hhb_df()
    ref = compute_dssp_df(df)
    wrapped = compute_dssp_df(df.assign(atom_number=df["atom_number"] % 1000))
    assert wrapped.equals(ref)


def test_rsa_of_modified_residue_uses_parent_max_acc():
    df = _4hhb_df()
    met = (df["chain_id"] == "A") & (df["residue_number"] == 32)
    df.loc[met, "residue_name"] = "MSE"
    df.loc[met, "record_name"] = "HETATM"
    G = nx.Graph(dssp_df=compute_dssp_df(df), config=ProteinGraphConfig())
    G = rsa(G)
    row = G.graph["dssp_df"].query("chain == 'A' and resnum == 32").iloc[0]
    assert row["aa"] == "MSE"
    assert row["rsa"] == row["asa"] / residue_max_acc["Sander"]["MET"]


def test_compute_dssp_df_without_backbone_raises():
    df = _4hhb_df()
    with pytest.raises(ValueError, match="complete backbone"):
        compute_dssp_df(df[df["atom_name"] == "CA"])


def test_dssp_core_degenerate_inputs():
    x = np.random.default_rng(0).random((1, 4, 3)).astype(np.float32)
    assert dssp_core.assign_dssp_np(x).shape == (1, 3)
    acc = dssp_core._accessibility(np.zeros((2, 5, 3)), np.zeros((2, 5), bool))
    assert acc.tolist() == [0, 0]


def _1lds_df():
    return read_pdb_to_dataframe(path=str(pdb_path.parent / "1lds.pdb"))


def test_dssp_features_reach_nodes_with_alt_loc_labels():
    # Node ids carry the kept atom's alt_loc label (A:MET:32:A).
    df = _4hhb_df()
    met = (df["chain_id"] == "A") & (df["residue_number"] == 32)
    df.loc[met, "alt_loc"] = "A"
    config = ProteinGraphConfig(
        graph_metadata_functions=[rsa, secondary_structure]
    )
    g = construct_graph(config=config, df=df, name="4hhb")
    assert "A:MET:32:A" in g
    assert all("ss" in d and "rsa" in d for _, d in g.nodes(data=True))


def test_compute_dssp_df_restores_file_order():
    # construct_graph sorts raw_pdb_df by chain and residue number, which
    # reorders chains L, H, A to A, H, L.
    df = read_pdb_to_dataframe(path=str(pdb_path))
    ref = compute_dssp_df(df)
    assert ref["chain"].unique().tolist() == ["L", "H", "A"]
    assert compute_dssp_df(sort_dataframe(df)).equals(ref)


def test_compute_dssp_df_non_monotonic_numbering():
    # e.g. a T4 lysozyme insertion numbered 1001+ inside a GPCR loop.
    ref = compute_dssp_df(_1lds_df())
    df = _1lds_df()
    df.loc[df["residue_number"].between(40, 49), "residue_number"] += 1000
    out = compute_dssp_df(sort_dataframe(df))
    for col in ("ss", "dssp_index", "phi", "psi", "NH_O_1_relidx"):
        assert out[col].tolist() == ref[col].tolist(), col


def test_compute_dssp_df_keeps_modified_residues_from_mmcif(tmp_path):
    # mmCIF HETATM records (here an MSE) must stay in place in the chain.
    cif = pdb_path.parent / "4hhb.cif"
    lines = []
    for line in cif.read_text().splitlines():
        f = line.split()
        if f[:1] == ["ATOM"] and f[5:7] == ["MET", "A"] and f[8] == "32":
            line = line.replace("ATOM  ", "HETATM", 1).replace(
                " MET ", " MSE "
            )
        lines.append(line)
    mse = tmp_path / "4hhb_mse.cif"
    mse.write_text("\n".join(lines) + "\n")
    ref = compute_dssp_df(read_pdb_to_dataframe(path=str(cif)))
    out = compute_dssp_df(read_pdb_to_dataframe(path=str(mse)))
    assert out.query("chain == 'A' and resnum == 32")["aa"].item() == "MSE"
    assert out.drop(columns="aa").equals(ref.drop(columns="aa"))


def test_chain_id_change_is_a_chain_break():
    df = _1lds_df()
    df.loc[df["residue_number"] >= 50, "chain_id"] = "B"
    out = compute_dssp_df(df)
    i = out.index[out["chain"] == "B"][0]
    assert out.at[i, "dssp_index"] == out.at[i - 1, "dssp_index"] + 2
    assert out.at[i, "phi"] == 360.0
    assert out.at[i - 1, "psi"] == 360.0


def test_compute_dssp_df_ignores_other_residues_with_same_number():
    # A water numbered like a protein residue must not join it, even when
    # its O has a higher occupancy than the carbonyl O.
    df = _4hhb_df()
    o = (
        (df["chain_id"] == "A")
        & (df["residue_number"] == 32)
        & (df["atom_name"] == "O")
    )
    df.loc[o, "occupancy"] = 0.5
    ref = compute_dssp_df(df)
    water = df[o].assign(
        residue_name="HOH",
        record_name="HETATM",
        occupancy=1.0,
        x_coord=100.0,
        y_coord=100.0,
        z_coord=100.0,
    )
    out = compute_dssp_df(pd.concat([df, water], ignore_index=True))
    assert out.equals(ref)


def test_insertions_false_keeps_parent_residue_features():
    # H 52 (TRP) and 52A (TYR); without insertions only 52 is a node.
    config = ProteinGraphConfig(
        insertions=False, graph_metadata_functions=[phi]
    )
    g = construct_graph(config=config, path=pdb_path)
    ref = compute_dssp_df(read_pdb_to_dataframe(path=str(pdb_path)))
    h52 = ref.query("chain == 'H' and resnum == 52").set_index("icode")
    assert h52.at[" ", "phi"] != h52.at["A", "phi"]
    assert g.nodes["H:TRP:52"]["phi"] == h52.at[" ", "phi"]


def test_compute_dssp_df_ignores_hydrogens_without_element():
    # e.g. MD output with a blank element column
    df = _4hhb_df()
    ref = compute_dssp_df(df)
    cb = df[df["atom_name"] == "CB"]
    hs = [
        cb.assign(atom_name="HB2", element_symbol="", x_coord=cb.x_coord + 1),
        cb.assign(
            atom_name="1HB", element_symbol=np.nan, y_coord=cb.y_coord + 1
        ),
    ]
    out = compute_dssp_df(pd.concat([df, *hs], ignore_index=True))
    assert out.equals(ref)


def test_dssp_energies_avoid_dense_distance_matrix(monkeypatch):
    # An L x L matrix runs out of memory on large assemblies.
    def dense(*args):
        raise AssertionError("dense pairwise distances")

    monkeypatch.setattr(dssp_core, "_pdist2", dense)
    ref = compute_dssp_df(_1lds_df())
    assert (
        ref.drop(columns="aa")
        .iloc[[0, 8, 22, 60, 96]]
        .apply(tuple, axis=1)
        .tolist()
        == MKDSSP_1LDS_ROWS
    )
