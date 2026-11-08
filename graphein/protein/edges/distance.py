"""Functions for computing biochemical edges of graphs."""

# Graphein
# Author: Eric Ma, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import itertools
from collections.abc import Iterable
from itertools import combinations, product

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger as log
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

from graphein.protein.resi_atoms import (
    AA_RING_ATOMS,
    AROMATIC_RESIS,
    BOND_TYPES,
    CATION_PI_RESIS,
    CATION_RESIS,
    DISULFIDE_ATOMS,
    DISULFIDE_RESIS,
    HYDROGEN_BOND_ACCEPTORS,
    HYDROGEN_BOND_DONORS,
    HYDROPHOBIC_RESIS,
    IONIC_RESIS,
    NEG_AA,
    NON_STANDARD_RESIS_PARENT,
    PI_RESIS,
    POS_AA,
    RESI_THREE_TO_1,
    RING_NORMAL_ATOMS,
    SALT_BRIDGE_ANIONS,
    SALT_BRIDGE_ATOMS,
    SALT_BRIDGE_CATIONS,
    SALT_BRIDGE_RESIDUES,
    SULPHUR_RESIS,
    VDW_RADII,
)
from graphein.protein.utils import filter_dataframe, sequence_positions

INFINITE_DIST = 10_000.0  # np.inf leads to errors in some cases


def compute_distmat(pdb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Euclidean distances between every atom.

    Design choice: passed in a ``pd.DataFrame`` to enable easier testing on
    dummy data.

    :param pdb_df: Dataframe containing protein structure. Must contain columns
        ``["x_coord", "y_coord", "z_coord"]``.
    :type pdb_df: pd.DataFrames
    :raises: ValueError if ``pdb_df`` does not contain the required columns.
    :return: pd.Dataframe of Euclidean distance matrix.
    :rtype: pd.DataFrame
    """
    if (
        not pd.Series(["x_coord", "y_coord", "z_coord"])
        .isin(pdb_df.columns)
        .all()
    ):
        raise ValueError(
            "Dataframe must contain columns ['x_coord', 'y_coord', 'z_coord']"
        )
    eucl_dists = pdist(
        pdb_df[["x_coord", "y_coord", "z_coord"]], metric="euclidean"
    )
    eucl_dists = pd.DataFrame(squareform(eucl_dists))
    eucl_dists.index = pdb_df.index
    eucl_dists.columns = pdb_df.index

    return eucl_dists


def filter_distmat(
    pdb_df: pd.DataFrame,
    distmat: pd.DataFrame,
    exclude_edges: Iterable[str],
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Filter distance matrix in place based on edge types to exclude.

    :param pdb_df: Data frame representing a PDB graph.
    :type pdb_df: pd.DataFrame
    :param distmat: Pairwise-distance matrix between all nodes
    :type pdb_df: pd.DataFrame
    :param exclude_edges: Supported values: `inter`, `intra`
        - `inter` removes inter-connections between nodes of the same chain.
        - `intra` removes intra-connections between nodes of different chains.
    :type exclude_edges: Iterable[str]
    :param inplace: False to create a deep copy.
    :type inplace: bool
    :return: Modified pairwise-distance matrix between all nodes.
    :rtype: pd.DataFrame
    """
    # Process input argument values
    supported_exclude_edges_vals = ["inter", "intra"]
    for val in exclude_edges:
        if val not in supported_exclude_edges_vals:
            raise ValueError(f"Unknown `exclude_edges` value '{val}'.")
    if not inplace:
        distmat = distmat.copy(deep=True)

    # Prepare
    chain_to_nodes = (
        pdb_df.groupby("chain_id")["node_id"].apply(list).to_dict()
    )
    node_id_to_int = dict(zip(pdb_df["node_id"], pdb_df.index))
    chain_to_nodes = {
        ch: [node_id_to_int[n] for n in nodes]
        for ch, nodes in chain_to_nodes.items()
    }

    # Construct indices of edges to exclude
    edges_to_excl = []
    if "intra" in exclude_edges:
        for nodes in chain_to_nodes.values():
            edges_to_excl.extend(list(combinations(nodes, 2)))
    if "inter" in exclude_edges:
        for nodes0, nodes1 in combinations(chain_to_nodes.values(), 2):
            edges_to_excl.extend(list(product(nodes0, nodes1)))

    # Filter distance matrix based on indices of edges to exclude
    if len(edges_to_excl):
        row_idx_to_excl, col_idx_to_excl = zip(*edges_to_excl)
        distmat.iloc[row_idx_to_excl, col_idx_to_excl] = INFINITE_DIST
        distmat.iloc[col_idx_to_excl, row_idx_to_excl] = INFINITE_DIST

    return distmat


def add_edge(G, n1, n2, kind_name):
    if G.has_edge(n1, n2):
        G.edges[n1, n2]["kind"].add(kind_name)
    else:
        G.add_edge(n1, n2, kind={kind_name})


def add_distance_to_edges(G: nx.Graph) -> nx.Graph:
    """Adds Euclidean distance between nodes in an edge as an edge attribute.

    :param G: Graph to add distances to.
    :type G: nx.Graph
    :return: Graph with added distances.
    :rtype: nx.Graph
    """
    if "atomic_dist_mat" in G.graph:
        dist_mat = G.graph["atomic_dist_mat"]
    elif "dist_mat" in G.graph:
        dist_mat = G.graph["dist_mat"]
    else:
        dist_mat = compute_distmat(G.graph["pdb_df"])
        G.graph["dist_mat"] = dist_mat

    mat = np.where(nx.to_numpy_array(G), dist_mat, 0)
    node_map = {n: i for i, n in enumerate(G.nodes)}
    for u, v, d in G.edges(data=True):
        d["distance"] = mat[node_map[u], node_map[v]]
    return G


def add_sequence_distance_edges(
    G: nx.Graph, d: int, name: str = "sequence_edge"
) -> nx.Graph:
    """
    Adds edges based on sequence distance to residues in each chain.

    Eg. if ``d=6`` then we join: nodes ``(1,7), (2,8), (3,9)..``
    based on their position in sequence (see
    :func:`~graphein.protein.utils.sequence_positions`).

    :param G: Networkx protein graph.
    :type G: nx.Graph
    :param d: Sequence separation to add edges on.
    :param name: Name of the edge type. Defaults to ``"sequence_edge"``.
    :type name: str
    :return G: Networkx protein graph with added peptide bonds.
    :rtype: nx.Graph
    """
    pdb_df = G.graph["pdb_df"]
    positions = dict(zip(pdb_df["node_id"], sequence_positions(pdb_df)))
    # Iterate over every chain
    for chain_id in G.graph["chain_ids"]:
        # Find chain residues
        chain_residues = [
            (n, v) for n, v in G.nodes(data=True) if v["chain_id"] == chain_id
        ]

        # Subset to only N and C atoms in the case of full-atom
        # peptide bond addition
        try:
            if (
                G.graph["config"].granularity == "atom"
                and name == "peptide_bond"
            ):
                chain_residues = [
                    (n, v)
                    for n, v in chain_residues
                    if v["atom_type"] in {"N", "C"}
                ]
        # If we don't don't find a config, assume it's a residue graph
        # This is brittle
        except KeyError:
            continue

        # Iterate over every residue in chain
        for i, residue in enumerate(chain_residues):
            try:
                # Checks not at chain terminus - is this versatile enough?
                if i == len(chain_residues) - d:
                    continue
                # Asserts residues are on the same chain
                cond_1 = (
                    residue[1]["chain_id"]
                    == chain_residues[i + d][1]["chain_id"]
                )
                # Asserts residues are d apart in sequence
                cond_2 = (
                    abs(
                        positions[residue[0]]
                        - positions[chain_residues[i + d][0]]
                    )
                    == d
                )

                # If this checks out, we add a peptide bond
                if (cond_1) and (cond_2):
                    # Adds "peptide bond" between current residue and the next
                    add_edge(G, residue[0], chain_residues[i + d][0], name)
            except IndexError:
                continue
    return G


def add_peptide_bonds(G: nx.Graph) -> nx.Graph:
    """
    Adds peptide backbone as edges to residues in each chain.

    :param G: Networkx protein graph.
    :type G: nx.Graph
    :return G: Networkx protein graph with added peptide bonds.
    :rtype: nx.Graph
    """
    return add_sequence_distance_edges(G, d=1, name="peptide_bond")


def assign_atoms_to_nodes(G: nx.Graph, df: pd.DataFrame) -> pd.DataFrame:
    """Assign the atoms in an atomic dataframe to the nodes of a graph.

    Atoms are matched to nodes by chain, residue name and number and insertion
    code (and atom name and alt_loc for atom graphs) rather than by
    ``node_id``, which is ignored if present: residue node IDs carry the CA's
    alt_loc, which the residue's other atoms may not share, and without
    insertions inserted residues share their parent's node ID.

    On residue graphs, a node whose CA has an alt_loc (e.g. ``A:LYS:16:A``)
    gets the residue's atoms of that conformer and those without an alt_loc.
    A node without an alt_loc gets one conformer of each atom, chosen by the
    graph's ``config.alt_locs`` (see
    :func:`~graphein.protein.graphs.remove_alt_locs`), or every conformer if
    it is ``"include"``.

    Missing ``insertion`` and ``alt_loc`` columns are treated as blank, and a
    missing ``occupancy`` column as equal occupancies.

    :param G: Protein graph.
    :type G: nx.Graph
    :param df: Atomic dataframe, e.g. ``G.graph["raw_pdb_df"]``.
    :type df: pd.DataFrame
    :return: The rows of ``df`` that belong to a node of ``G``, in their
        original order, with the node's ID in ``node_id``. Atoms shared by
        several nodes appear once per node.
    :rtype: pd.DataFrame
    """
    nodes = G.graph["pdb_df"]
    nodes = nodes.loc[nodes["node_id"].isin(list(G.nodes()))]
    nodes = nodes.fillna({"insertion": "", "alt_loc": ""})
    df = df.drop(columns="node_id", errors="ignore").reset_index(drop=True)
    for col in ("insertion", "alt_loc"):
        df[col] = df[col].fillna("") if col in df else ""
    if "occupancy" not in df:
        df["occupancy"] = 1.0

    key = ["chain_id", "residue_name", "residue_number", "insertion"]
    if G.graph.get("node_type") == "atom":
        key += ["atom_name", "alt_loc"]
        return df.merge(nodes[key + ["node_id"]], on=key)

    nodes = nodes[key + ["alt_loc", "node_id"]].rename(
        columns={"alt_loc": "node_alt_loc"}
    )
    df = df.merge(nodes, on=key)
    has_conformer = df["node_alt_loc"] != ""
    keep = has_conformer & (
        (df["alt_loc"] == "") | (df["alt_loc"] == df["node_alt_loc"])
    )
    alt_locs = getattr(G.graph.get("config"), "alt_locs", "max_occupancy")
    if alt_locs == "include":
        keep |= ~has_conformer
    else:
        # Avoid a circular import (graphs imports this module)
        from graphein.protein.graphs import remove_alt_locs

        shared = df.loc[~has_conformer]
        shared = remove_alt_locs(shared.assign(row=shared.index), alt_locs)
        keep[shared["row"].to_numpy()] = True
    return df.loc[keep].drop(columns="node_alt_loc").reset_index(drop=True)


def add_hydrophobic_interactions(
    G: nx.Graph, rgroup_df: pd.DataFrame | None = None
):
    """
    Find all hydrophobic interactions.

    Performs searches between the following residues:
    ``[ALA, VAL, LEU, ILE, MET, PHE, TRP, PRO, TYR]``
    (:const:`~graphein.protein.resi_atoms.HYDROPHOBIC_RESIS`).

    Criteria: R-group residues are within 5A distance.

    :param G: nx.Graph to add hydrophobic interactions to.
    :type G: nx.Graph
    :param rgroup_df: Optional dataframe of R-group atoms.
    :type rgroup_df: pd.DataFrame, optional
    """
    if rgroup_df is None:
        rgroup_df = G.graph["rgroup_df"]
    hydrophobics_df = filter_dataframe(
        rgroup_df, "residue_name", HYDROPHOBIC_RESIS, True
    )
    hydrophobics_df = assign_atoms_to_nodes(G, hydrophobics_df)
    if hydrophobics_df.shape[0] > 0:
        interacting_atoms = get_interacting_atom_pairs(5, hydrophobics_df)
        add_interacting_resis(
            G, interacting_atoms, hydrophobics_df, ["hydrophobic"]
        )


def add_disulfide_interactions(
    G: nx.Graph, rgroup_df: pd.DataFrame | None = None
):
    """
    Find all disulfide interactions between CYS residues
    (:const:`~graphein.protein.resi_atoms.DISULFIDE_RESIS`,
    :const:`~graphein.protein.resi_atoms.DISULFIDE_ATOMS`).

    Criteria: sulfur atom pairs are within 2.2A of each other.

    :param G: networkx protein graph
    :type G: nx.Graph
    :param rgroup_df: pd.DataFrame containing rgroup data, defaults to ``None``,
        which retrieves the df from the provided nx graph.
    :type rgroup_df: pd.DataFrame, optional
    """
    # Check for existence of at least two Cysteine residues
    residues = [d["residue_name"] for _, d in G.nodes(data=True)]
    if residues.count("CYS") < 2:
        log.debug(
            f"{residues.count('CYS')} CYS residues found. Cannot add disulfide \
                interactions with fewer than two CYS residues."
        )
        return

    if rgroup_df is None:
        rgroup_df = G.graph["rgroup_df"]
    disulfide_df = filter_dataframe(
        rgroup_df, "residue_name", DISULFIDE_RESIS, True
    )
    disulfide_df = filter_dataframe(
        disulfide_df, "atom_name", DISULFIDE_ATOMS, True
    )
    # Ensure only residues in the graph are kept
    disulfide_df = assign_atoms_to_nodes(G, disulfide_df)
    if disulfide_df.shape[0] > 0:
        interacting_atoms = get_interacting_atom_pairs(2.2, disulfide_df)
        add_interacting_resis(
            G, interacting_atoms, disulfide_df, ["disulfide"]
        )


HBOND_TYPE_BACKBONE_ATOMS: dict[str, int] = {
    "sc-sc": 0,
    "bb-sc": 1,
    "bb-bb": 2,
}
"""Number of backbone atoms in the donor-acceptor pair of each hydrogen bond
type."""

HBOND_TYPES: list[str] = list(HBOND_TYPE_BACKBONE_ATOMS)
"""Hydrogen bond types by the location (backbone or side chain) of the donor
and acceptor atoms."""

NO_AMIDE_HYDROGEN_RESIS: list[str] = ["3HP", "4HP", "BMT", "MLE", "MVA", "SAR"]
"""Residues whose backbone ``N`` has no hydrogen, so is not a hydrogen bond
donor, besides those :const:`~graphein.protein.resi_atoms.RESI_THREE_TO_1`
maps to proline: hydroxyprolines mapped to ``X`` and N-methylated residues."""

MODIFIED_SIDE_CHAIN_RESIS: list[str] = [
    "5HP",
    "BMT",
    "CGU",
    "KCX",
    "LLP",
    "PCA",
    "PTR",
    "PYL",
    "SEP",
    "STY",
    "TPO",
    "TYS",
]
"""Residues in :const:`~graphein.protein.resi_atoms.NON_STANDARD_RESIS_PARENT`
whose side chain hydrogen bond donors and acceptors differ from those of their
parent residue (e.g. phosphorylated ``SEP``, ``TPO`` and ``PTR``)."""

HBOND_RESIDUE_VARIANTS: dict[str, str] = {
    res: parent
    for res, parent in NON_STANDARD_RESIS_PARENT.items()
    if parent in HYDROGEN_BOND_DONORS.keys() | HYDROGEN_BOND_ACCEPTORS.keys()
    and res not in MODIFIED_SIDE_CHAIN_RESIS
}
"""Residue variants (D-amino acids and Amber/CHARMM protonation states) with
the same side chain hydrogen bond donors and acceptors as their parent
residue: the entries of
:const:`~graphein.protein.resi_atoms.NON_STANDARD_RESIS_PARENT` with a parent
that has side chain donors or acceptors, except
:const:`MODIFIED_SIDE_CHAIN_RESIS`. Protonation states are approximated by the
parent residue (e.g. the protonated ``OD2`` of ``ASH`` is an acceptor, not a
donor)."""


def add_hydrogen_bond_interactions(
    G: nx.Graph,
    rgroup_df: pd.DataFrame | None = None,
    distance_threshold: float = 3.5,
    hbond_types: Iterable[str] = HBOND_TYPES,
):
    """
    Add hydrogen bond interactions between donor and acceptor atoms.

    An edge is added between two nodes if a hydrogen bond donor atom in one is
    within ``distance_threshold`` Angstroms of a hydrogen bond acceptor atom in
    the other. Side chain donors and acceptors are defined in
    :const:`~graphein.protein.resi_atoms.HYDROGEN_BOND_DONORS` and
    :const:`~graphein.protein.resi_atoms.HYDROGEN_BOND_ACCEPTORS`. Backbone
    amide nitrogens (``N``, except in proline variants and N-methylated
    residues; :const:`NO_AMIDE_HYDROGEN_RESIS`) are donors and backbone
    oxygens (``O`` and ``OXT``) are acceptors.

    Side chain atoms of the residue variants in :const:`HBOND_RESIDUE_VARIANTS`
    (e.g. ``HIE``, ``DSN``) are matched to those of their parent residue.
    Side chain atoms of other residues (e.g. ligands, or modified residues
    such as ``SEP``) are not considered.

    Atoms in the same residue are never paired. Covalent geometry alone puts
    the backbone ``N`` and ``O`` atoms either side of a peptide bond ~2.3
    Angstroms apart, and a Ser ``OG`` or Thr ``OG1`` ~3 Angstroms from the next
    residue's ``N``, so these pairs are ignored between residues joined by a
    peptide bond. Other side chain hydrogen bonds to a neighbouring residue's
    backbone (e.g. Asx motifs) are kept. Peptide bonds are identified from
    the backbone ``C`` and ``N`` coordinates, so insertion codes and gaps in
    residue numbering are handled.

    Hydrogen positions are not used, so this is a geometric approximation.

    :param G: Protein graph to add hydrogen bond edges to.
    :type G: nx.Graph
    :param rgroup_df: Atomic dataframe to find interactions in. If ``None``
        (default), ``G.graph["raw_pdb_df"]`` is used. Backbone hydrogen bonds
        are only found if this dataframe contains backbone atoms (unlike
        ``G.graph["rgroup_df"]``), and a warning is logged if they are
        requested but there are none. Atoms are assigned to nodes with
        :func:`assign_atoms_to_nodes`, and a warning is logged if none match.
    :type rgroup_df: Optional[pd.DataFrame]
    :param distance_threshold: Maximum donor-acceptor distance in Angstroms.
        Defaults to ``3.5``.
    :type distance_threshold: float
    :param hbond_types: Hydrogen bonds to add, by the location of the donor and
        acceptor atoms: ``"bb-bb"`` (backbone-backbone), ``"bb-sc"``
        (backbone-side chain) and/or ``"sc-sc"`` (side chain-side chain).
        A single type can be given as a string. Defaults to all three
        (:const:`HBOND_TYPES`).
    :type hbond_types: Iterable[str]
    :raises ValueError: If ``hbond_types`` contains an unknown type.
    """
    if isinstance(hbond_types, str):
        hbond_types = [hbond_types]
    hbond_types = set(hbond_types)
    if not hbond_types:
        raise ValueError(f"hbond_types is empty. Must be in {HBOND_TYPES}.")
    if not hbond_types.issubset(HBOND_TYPES):
        raise ValueError(
            f"Unknown hbond_types: {hbond_types - set(HBOND_TYPES)}. "
            f"Must be in {HBOND_TYPES}."
        )
    if rgroup_df is None:
        rgroup_df = G.graph["raw_pdb_df"]

    df = assign_atoms_to_nodes(G, rgroup_df)
    if df.empty:
        log.warning(
            "No atoms in rgroup_df match a graph node, so no hydrogen bonds "
            "were added."
        )
        return

    # Map residue variants (e.g. HIE, DSN) to their parent residue
    parent = df["residue_name"].replace(HBOND_RESIDUE_VARIANTS)
    res_atom = parent + ":" + df["atom_name"]
    sc_donors = [
        f"{res}:{atom}"
        for res, atoms in HYDROGEN_BOND_DONORS.items()
        for atom in atoms
    ]
    sc_acceptors = [
        f"{res}:{atom}"
        for res, atoms in HYDROGEN_BOND_ACCEPTORS.items()
        for atom in atoms
    ]
    is_amino_acid = df["residue_name"].isin(list(RESI_THREE_TO_1))
    # Proline variants (e.g. HYP, DPR) and N-methylated residues also lack a
    # backbone amide hydrogen
    no_amide_h = (df["residue_name"].map(RESI_THREE_TO_1) == "P") | df[
        "residue_name"
    ].isin(NO_AMIDE_HYDROGEN_RESIS)
    bb_donor = is_amino_acid & (df["atom_name"] == "N") & ~no_amide_h
    bb_acceptor = is_amino_acid & df["atom_name"].isin(["O", "OXT"])
    donor = (bb_donor | res_atom.isin(sc_donors)).to_numpy()
    acceptor = (bb_acceptor | res_atom.isin(sc_acceptors)).to_numpy()
    backbone = (bb_donor | bb_acceptor).to_numpy()
    # e.g. G.graph["rgroup_df"], which keeps only the terminal OXT
    if not bb_donor.any() and hbond_types - {"sc-sc"}:
        log.warning(
            "No backbone N atoms to find backbone hydrogen bonds with. Pass "
            "a dataframe with backbone atoms (e.g. G.graph['raw_pdb_df'], the "
            "default) as rgroup_df, or hbond_types=['sc-sc']."
        )

    all_residues = (
        df["chain_id"].astype(str)
        + ":"
        + df["residue_number"].astype(str)
        + ":"
        + df["insertion"]
    )
    merged = df
    keep = donor | acceptor
    df = df.loc[keep].reset_index(drop=True)
    residue = all_residues[keep].to_numpy()
    donor, acceptor, backbone = donor[keep], acceptor[keep], backbone[keep]

    coord_cols = ["x_coord", "y_coord", "z_coord"]
    i, j = (
        cKDTree(df[coord_cols].to_numpy())
        .query_pairs(distance_threshold, output_type="ndarray")
        .T
    )

    n_backbone = backbone[i].astype(int) + backbone[j].astype(int)
    allowed = [HBOND_TYPE_BACKBONE_ATOMS[t] for t in hbond_types]
    is_hbond = (
        ((donor[i] & acceptor[j]) | (donor[j] & acceptor[i]))
        & (residue[i] != residue[j])
        & np.isin(n_backbone, allowed)
    )

    # Drop bb-bb pairs, and Ser OG / Thr OG1 (i) to N (i + 1), between
    # residues joined by a peptide bond (C-N ~1.33A, in any conformer).
    atom = df["atom_name"].to_numpy()

    def _gamma_to_n(x, y):
        return np.isin(atom[x], ["OG", "OG1"]) & (atom[y] == "N") & backbone[y]

    gamma_ij, gamma_ji = _gamma_to_n(i, j), _gamma_to_n(j, i)
    bb = np.flatnonzero(is_hbond & ((n_backbone == 2) | gamma_ij | gamma_ji))
    if len(bb) > 0:
        c = is_amino_acid & (merged["atom_name"] == "C")
        n = is_amino_acid & (merged["atom_name"] == "N")
        c_res, n_res = all_residues[c].to_numpy(), all_residues[n].to_numpy()
        bonded = {
            (c_res[a], n_res[b])
            for a, b in cKDTree(merged.loc[c, coord_cols].to_numpy())
            .sparse_distance_matrix(
                cKDTree(merged.loc[n, coord_cols].to_numpy()),
                2.0,
                output_type="ndarray",
            )[["i", "j"]]
            .tolist()
        }
        res_i, res_j = residue[i[bb]], residue[j[bb]]

        def _peptide_bonded(c_res, n_res):
            return np.array([pair in bonded for pair in zip(c_res, n_res)])

        bonded_ij = _peptide_bonded(res_i, res_j)
        bonded_ji = _peptide_bonded(res_j, res_i)
        bb_bb = n_backbone[bb] == 2
        drop = (
            (bb_bb & (bonded_ij | bonded_ji))
            | (gamma_ij[bb] & bonded_ij)
            | (gamma_ji[bb] & bonded_ji)
        )
        is_hbond[bb[drop]] = False
    add_interacting_resis(G, (i[is_hbond], j[is_hbond]), df, ["hbond"])


def add_ionic_interactions(G: nx.Graph, rgroup_df: pd.DataFrame | None = None):
    """
    Find all ionic interactions.

    Criteria: ``[ARG, LYS, HIS, ASP, and GLU]``
    (:const:`~graphein.protein.resi_atoms.IONIC_RESIS`) residues are within 6A.

    We also check for opposing charges
    (:const:`~graphein.protein.resi_atoms.POS_AA`,
    :const:`~graphein.protein.resi_atoms.NEG_AA`).

    :param G: nx.Graph to add ionic interactions to.
    :type G: nx.Graph
    :param rgroup_df: Optional dataframe of R-group atoms. Default is ``None``.
    :type rgroup_df: Optional[pd.DataFrame]
    """
    if rgroup_df is None:
        rgroup_df = G.graph["rgroup_df"]
    ionic_df = filter_dataframe(rgroup_df, "residue_name", IONIC_RESIS, True)
    ionic_df = assign_atoms_to_nodes(G, ionic_df)
    if ionic_df.shape[0] > 0:
        interacting_atoms = get_interacting_atom_pairs(6, ionic_df)
        add_interacting_resis(G, interacting_atoms, ionic_df, ["ionic"])
        # Check that the interacting residues are of opposite charges
        for r1, r2 in get_edges_by_bond_type(G, "ionic"):
            condition1 = (
                G.nodes[r1]["residue_name"] in POS_AA
                and G.nodes[r2]["residue_name"] in NEG_AA
            )
            condition2 = (
                G.nodes[r2]["residue_name"] in POS_AA
                and G.nodes[r1]["residue_name"] in NEG_AA
            )
            is_ionic = condition1 or condition2
            if not is_ionic:
                G.edges[r1, r2]["kind"].remove("ionic")
                if len(G.edges[r1, r2]["kind"]) == 0:
                    G.remove_edge(r1, r2)


def add_aromatic_interactions(G: nx.Graph, pdb_df: pd.DataFrame | None = None):
    """
    Find all aromatic-aromatic interaction.

    Criteria: phenyl ring centroids separated between 4.5A to 7A.
    Phenyl rings are present on ``PHE, TRP, HIS, TYR``
    (:const:`~graphein.protein.resi_atoms.AROMATIC_RESIS`).
    Phenyl ring atoms on these amino acids are defined by the following
    atoms:
    - PHE: CG, CD, CE, CZ
    - TRP: CD, CE, CH, CZ
    - HIS: CG, CD, ND, NE, CE
    - TYR: CG, CD, CE, CZ
    Centroids of these atoms are taken by taking:
        (mean x), (mean y), (mean z)
    for each of the ring atoms.
    Notes for future self/developers:
    - Because of the requirement to pre-compute ring centroids, we do not
        use the functions written above (filter_dataframe, compute_distmat,
        get_interacting_atoms), as they do not return centroid atom
        euclidean coordinates.
    """
    if pdb_df is None:
        pdb_df = G.graph["raw_pdb_df"]
    dfs = []
    for resi in AROMATIC_RESIS:
        resi_rings_df = get_ring_atoms(pdb_df, resi)
        resi_rings_df = assign_atoms_to_nodes(G, resi_rings_df)
        resi_centroid_df = get_ring_centroids(resi_rings_df)
        dfs.append(resi_centroid_df)
    aromatic_df = (
        pd.concat(dfs).sort_values(by="node_id").reset_index(drop=True)
    )
    if aromatic_df.shape[0] > 0:
        distmat = compute_distmat(aromatic_df)
        distmat.set_index(aromatic_df["node_id"], inplace=True)
        distmat.columns = aromatic_df["node_id"]
        distmat = distmat[(distmat >= 4.5) & (distmat <= 7)].fillna(0)
        indices = np.where(distmat > 0)

        interacting_resis = [
            (distmat.index[r], distmat.index[c])
            for r, c in zip(indices[0], indices[1])
        ]
        log.info(
            f"Found: {len(interacting_resis)} aromatic-aromatic interactions"
        )
        for n1, n2 in interacting_resis:
            assert G.nodes[n1]["residue_name"] in AROMATIC_RESIS
            assert G.nodes[n2]["residue_name"] in AROMATIC_RESIS
            add_edge(G, n1, n2, "aromatic")


def add_aromatic_sulphur_interactions(
    G: nx.Graph, rgroup_df: pd.DataFrame | None = None
):
    """Find all aromatic-sulphur interactions.

    Criteria: Sulphur containing residue () within 5.3 Angstroms of an aromatic
    residue ().

    :param G: The graph to add the aromatic-sulphur interactions to.
    :type G: nx.Graph
    :param rgroup_df: The rgroup dataframe. If ``None`` (default), the graph's
        rgroup dataframe is used.
    :type rgroup_df: Optional[pd.DataFrame].
    """
    if rgroup_df is None:
        rgroup_df = G.graph["rgroup_df"]
    RESIDUES = SULPHUR_RESIS + PI_RESIS

    aromatic_sulphur_df = filter_dataframe(
        rgroup_df, "residue_name", RESIDUES, True
    )
    aromatic_sulphur_df = assign_atoms_to_nodes(G, aromatic_sulphur_df)

    if aromatic_sulphur_df.shape[0] > 0:
        interacting_atoms = get_interacting_atom_pairs(
            5.3, aromatic_sulphur_df
        )
        interacting_atoms = list(
            zip(interacting_atoms[0], interacting_atoms[1])
        )

        node_ids = aromatic_sulphur_df["node_id"].to_numpy()
        residue_names = aromatic_sulphur_df["residue_name"].to_numpy()
        for a1, a2 in interacting_atoms:
            resi1, resi2 = node_ids[a1], node_ids[a2]
            name1, name2 = residue_names[a1], residue_names[a2]

            condition1 = name1 in SULPHUR_RESIS and name2 in PI_RESIS
            condition2 = name1 in PI_RESIS and name2 in SULPHUR_RESIS

            if (condition1 or condition2) and resi1 != resi2:
                add_edge(G, resi1, resi2, "aromatic_sulphur")


def add_cation_pi_interactions(
    G: nx.Graph, rgroup_df: pd.DataFrame | None = None
):
    """Add cation-pi interactions.

    Criteria:
        # Todo

    :param G: Graph to add cation-pi interactions to.
    :type G: nx.Graph
    :param rgroup_df: Dataframe containing rgroup information. Defaults to
        ``None``.
    :type rgroup_df: Optional[pd.DataFrame].
    """
    if rgroup_df is None:
        rgroup_df = G.graph["rgroup_df"]
    cation_pi_df = filter_dataframe(
        rgroup_df, "residue_name", CATION_PI_RESIS, True
    )
    cation_pi_df = assign_atoms_to_nodes(G, cation_pi_df)

    if cation_pi_df.shape[0] > 0:
        interacting_atoms = get_interacting_atom_pairs(6, cation_pi_df)
        interacting_atoms = list(
            zip(interacting_atoms[0], interacting_atoms[1])
        )

        node_ids = cation_pi_df["node_id"].to_numpy()
        residue_names = cation_pi_df["residue_name"].to_numpy()
        for a1, a2 in interacting_atoms:
            resi1, resi2 = node_ids[a1], node_ids[a2]
            name1, name2 = residue_names[a1], residue_names[a2]

            condition1 = name1 in CATION_RESIS and name2 in PI_RESIS
            condition2 = name1 in PI_RESIS and name2 in CATION_RESIS

            if (condition1 or condition2) and resi1 != resi2:
                add_edge(G, resi1, resi2, "cation_pi")


def add_vdw_interactions(
    g: nx.Graph,
    threshold: float = 0.5,
    remove_intraresidue: bool = False,
    name: str = "vdw",
):
    """Criterion: Any non-H atoms within the sum of
    their VdW Radii (:const:`~graphein.protein.resi_atoms.VDW_RADII`) +
    threshold (default: ``0.5``) Angstroms of each other.

    :param g: Graph to add van der Waals interactions to.
    :type g: nx.Graph
    :param threshold: Threshold distance for van der Waals interactions.
        Default: ``0.5`` Angstroms.
    :type threshold: float
    :param remove_intraresidue: Whether to remove intra-residue interactions.
    :type remove_intraresidue: bool
    """
    df = g.graph["raw_pdb_df"]
    df = filter_dataframe(df, "atom_name", ["H"], boolean=False)
    df = assign_atoms_to_nodes(g, df)
    radii = df["element_symbol"].map(VDW_RADII).to_numpy()
    coords = df[["x_coord", "y_coord", "z_coord"]].to_numpy()

    # No pair can be further apart than the two largest radii + threshold
    i, j = get_interacting_atom_pairs(
        threshold + 2 * np.nanmax(radii, initial=0), df
    )
    dists = np.linalg.norm(coords[i] - coords[j], axis=1)
    keep = dists - (radii[i] + radii[j]) <= threshold
    interacting_atoms = (i[keep], j[keep])
    add_interacting_resis(g, interacting_atoms, df, [name])

    if remove_intraresidue:
        for u, v in get_edges_by_bond_type(g, name):
            u_id = "".join(u.split(":")[:-1])
            v_id = "".join(v.split(":")[:-1])
            if u_id == v_id:
                g.edges[u, v]["kind"].remove(name)
                if len(g.edges[u, v]["kind"]) == 0:
                    g.remove_edge(u, v)


def add_vdw_clashes(
    g: nx.Graph, threshold: float = 0.0, remove_intraresidue: bool = False
):
    """Adds van der Waals clashes to graph.

    These are atoms that are within the sum of their VdW Radii
    (:const:`~graphein.protein.resi_atoms.VDW_RADII`).

    :param g: Graph to add van der Waals clashes to.
    :type g: nx.Graph
    :param threshold: Threshold, defaults to ``0.0``.
    :type threshold: float
    :param remove_intraresidue: Whether to remove clashes within a residue,
        defaults to ``False``.
    :type remove_intraresidue: bool
    """
    add_vdw_interactions(
        g,
        threshold=threshold,
        remove_intraresidue=remove_intraresidue,
        name="vdw_clash",
    )


def add_pi_stacking_interactions(
    G: nx.Graph,
    pdb_df: pd.DataFrame | None = None,
    centroid_distance: float = 7.0,
):
    """Adds Pi-stacking interactions to graph.

    Criteria:
        - aromatic ring centroids within 7.0 (default) Angstroms. (|A1A2| < 7.0)
        - Angle between ring normal vectors < 30° (∠(n1, n2) < 30°)
        - Angle between ring normal vectors and centroid vector < 45°
            (∠(n1, A1A2) < 45°), (∠(n2, A1A2) < 45°)

    :param G: _description_
    :type G: nx.Graph
    :param pdb_df: _description_, defaults to None
    :type pdb_df: Optional[pd.DataFrame], optional
    """
    if pdb_df is None:
        pdb_df = G.graph["raw_pdb_df"]
    dfs = []
    # Compute centroids and normal for each ring
    for resi in PI_RESIS:
        resi_rings_df = get_ring_atoms(pdb_df, resi)
        resi_rings_df = assign_atoms_to_nodes(G, resi_rings_df)
        resi_centroid_df = get_ring_centroids(resi_rings_df)
        resi_normals = get_ring_normals(resi_rings_df)
        resi_centroid_df = pd.merge(
            resi_centroid_df, resi_normals, on="node_id"
        )
        dfs.append(resi_centroid_df)
    aromatic_df = (
        pd.concat(dfs).sort_values(by="node_id").reset_index(drop=True)
    )

    if aromatic_df.shape[0] > 0:
        distmat = compute_distmat(aromatic_df)
        distmat.set_index(aromatic_df["node_id"], inplace=True)
        distmat.columns = aromatic_df["node_id"]
        distmat = distmat[distmat <= centroid_distance].fillna(0)
        indices = np.where(distmat > 0)
        interacting_resis = [
            (distmat.index[r], distmat.index[c])
            for r, c in zip(indices[0], indices[1])
        ]
        # log.info(f"Found: {len(interacting_resis)} aromatic-aromatic interactions")
        centroids = dict(
            zip(
                aromatic_df["node_id"],
                aromatic_df[["x_coord", "y_coord", "z_coord"]].to_numpy(),
            )
        )
        normals = dict(zip(aromatic_df["node_id"], aromatic_df[0]))
        for n1, n2 in interacting_resis:
            assert G.nodes[n1]["residue_name"] in PI_RESIS
            assert G.nodes[n2]["residue_name"] in PI_RESIS
            n1_centroid = centroids[n1]
            n2_centroid = centroids[n2]

            n1_normal = normals[n1]
            n2_normal = normals[n2]

            centroid_vector = n2_centroid - n1_centroid

            norm_angle = compute_angle(n1_normal, n2_normal)
            n1_centroid_angle = compute_angle(n1_normal, centroid_vector)
            n2_centroid_angle = compute_angle(n2_normal, centroid_vector)

            if (
                norm_angle >= 30
                or n1_centroid_angle >= 45
                or n2_centroid_angle >= 45
            ):
                continue
            add_edge(G, n1, n2, "pi_stacking")


def add_t_stacking(G: nx.Graph, pdb_df: pd.DataFrame | None = None):
    if pdb_df is None:
        pdb_df = G.graph["raw_pdb_df"]
    dfs = []
    # Compute centroids and normal for each ring
    for resi in PI_RESIS:
        resi_rings_df = get_ring_atoms(pdb_df, resi)
        resi_rings_df = assign_atoms_to_nodes(G, resi_rings_df)
        resi_centroid_df = get_ring_centroids(resi_rings_df)
        resi_normals = get_ring_normals(resi_rings_df)
        resi_centroid_df = pd.merge(
            resi_centroid_df, resi_normals, on="node_id"
        )
        dfs.append(resi_centroid_df)
    aromatic_df = (
        pd.concat(dfs).sort_values(by="node_id").reset_index(drop=True)
    )

    if aromatic_df.shape[0] > 0:
        distmat = compute_distmat(aromatic_df)
        distmat.set_index(aromatic_df["node_id"], inplace=True)
        distmat.columns = aromatic_df["node_id"]
        distmat = distmat[distmat <= 7].fillna(0)
        indices = np.where(distmat > 0)
        interacting_resis = [
            (distmat.index[r], distmat.index[c])
            for r, c in zip(indices[0], indices[1])
        ]
        # log.info(f"Found: {len(interacting_resis)} aromatic-aromatic interactions")
        centroids = dict(
            zip(
                aromatic_df["node_id"],
                aromatic_df[["x_coord", "y_coord", "z_coord"]].to_numpy(),
            )
        )
        normals = dict(zip(aromatic_df["node_id"], aromatic_df[0]))
        for n1, n2 in interacting_resis:
            assert G.nodes[n1]["residue_name"] in PI_RESIS
            assert G.nodes[n2]["residue_name"] in PI_RESIS
            n1_centroid = centroids[n1]
            n2_centroid = centroids[n2]

            n1_normal = normals[n1]
            n2_normal = normals[n2]

            centroid_vector = n2_centroid - n1_centroid

            norm_angle = compute_angle(n1_normal, n2_normal)
            n1_centroid_angle = compute_angle(n1_normal, centroid_vector)
            n2_centroid_angle = compute_angle(n2_normal, centroid_vector)

            if (
                norm_angle >= 90
                or norm_angle <= 60
                or n1_centroid_angle >= 45
                or n2_centroid_angle >= 45
            ):
                continue
            add_edge(G, n1, n2, "t_stacking")


def add_backbone_carbonyl_carbonyl_interactions(
    G: nx.Graph, threshold: float = 3.2
):
    """Adds backbone-carbonyl-carbonyl interactions.

    Default is to consider C═O···C═O interactions below 3.2 Angstroms
    (sum of O+C vdw radii).

    Source:
    > Rahim, A., Saha, P., Jha, K.K. et al. Reciprocal carbonyl–carbonyl
    > interactions in small molecules and proteins. Nat Commun 8, 78 (2017).
    > https://doi.org/10.1038/s41467-017-00081-x

    :param G: Protein graph to add edges to.
    :type G: nx.Graph
    :param threshold: Threshold below which to consider an interaction,
        defaults to 3.2 Angstroms.
    :type threshold: float, optional
    """
    df = G.graph["raw_pdb_df"]
    df = filter_dataframe(df, "atom_name", ["C", "O"], boolean=True)
    df = assign_atoms_to_nodes(G, df)
    interacting_atoms = get_interacting_atom_pairs(threshold, df)

    # Filter out O-O and C-C edges
    atom_1 = df.iloc[interacting_atoms[0]]["atom_name"]
    atom_2 = df.iloc[interacting_atoms[1]]["atom_name"]
    diff_atoms = atom_1.values != atom_2.values
    interacting_atoms = (
        interacting_atoms[0][diff_atoms],
        interacting_atoms[1][diff_atoms],
    )

    # Filter out O-C edges on the same residue (in any conformer)
    residue = (
        df["chain_id"].astype(str)
        + ":"
        + df["residue_number"].astype(str)
        + ":"
        + df["insertion"]
    ).to_numpy()
    diff_atoms = residue[interacting_atoms[0]] != residue[interacting_atoms[1]]
    interacting_atoms = (
        interacting_atoms[0][diff_atoms],
        interacting_atoms[1][diff_atoms],
    )

    add_interacting_resis(G, interacting_atoms, df, ["bb_carbonyl_carbonyl"])


def add_salt_bridges(
    G: nx.Graph,
    rgroup_df: pd.DataFrame | None = None,
    threshold: float = 4.0,
):
    """Compute salt bridge interactions.

    Criterion: Anion-Cation residue atom pairs within threshold (``4.0``)
    Angstroms of each other.

    Anions: ASP/OD1+OD2, GLU/OE1+OE2
    Cations: LYS/NZ, ARG/NH1+NH2

    :param G: Graph to add salt bridge interactions to.
    :type G: nx.Graph
    :param rgroup_df: R group dataframe, defaults to ``None``.
    :type rgroup_df: Optional[pd.DataFrame]
    :param threshold: Distance threshold, defaults to ``4.0`` Angstroms.
    :type threshold: float, optional
    """
    if rgroup_df is None:
        rgroup_df = G.graph["rgroup_df"]
    salt_bridge_df = filter_dataframe(
        rgroup_df, "residue_name", SALT_BRIDGE_RESIDUES, boolean=True
    )
    salt_bridge_df = filter_dataframe(
        salt_bridge_df, "atom_name", SALT_BRIDGE_ATOMS, boolean=True
    )
    salt_bridge_df = assign_atoms_to_nodes(G, salt_bridge_df)
    if salt_bridge_df.shape[0] > 0:
        interacting_atoms = get_interacting_atom_pairs(
            threshold, salt_bridge_df
        )
        add_interacting_resis(
            G, interacting_atoms, salt_bridge_df, ["salt_bridge"]
        )

        for r1, r2 in get_edges_by_bond_type(G, "salt_bridge"):
            condition1 = (
                G.nodes[r1]["residue_name"] in SALT_BRIDGE_ANIONS
                and G.nodes[r2]["residue_name"] in SALT_BRIDGE_CATIONS
            )
            condition2 = (
                G.nodes[r2]["residue_name"] in SALT_BRIDGE_ANIONS
                and G.nodes[r1]["residue_name"] in SALT_BRIDGE_CATIONS
            )
            is_ionic = condition1 or condition2
            if not is_ionic:
                G.edges[r1, r2]["kind"].remove("salt_bridge")
                if len(G.edges[r1, r2]["kind"]) == 0:
                    G.remove_edge(r1, r2)


def get_interacting_atoms(
    angstroms: float, distmat: pd.DataFrame
) -> np.ndarray:
    """Find the atoms that are within a particular radius of one another.

    :param angstroms: The radius in angstroms.
    :type angstroms: float
    :param distmat: The distance matrix.
    :type distmat: pd.DataFrame
    """
    return np.where(distmat <= angstroms)


def get_interacting_atom_pairs(
    angstroms: float, df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Find the pairs of atoms that are within a radius of one another.

    Unlike :func:`get_interacting_atoms`, this uses a KD-tree rather than a
    dense distance matrix, and gives each pair once (``i < j``).

    :param angstroms: The radius in angstroms.
    :type angstroms: float
    :param df: Dataframe with columns ``["x_coord", "y_coord", "z_coord"]``.
    :type df: pd.DataFrame
    :return: Positional indices of the first and second atom of each pair.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    coords = df[["x_coord", "y_coord", "z_coord"]].to_numpy()
    pairs = cKDTree(coords).query_pairs(angstroms, output_type="ndarray")
    return pairs[:, 0], pairs[:, 1]


def add_delaunay_triangulation(
    G: nx.Graph, allowable_nodes: list[str] | None = None
):
    """
    Compute the Delaunay triangulation of the protein structure.

    This has been used in prior work. References:

        Harrison, R. W., Yu, X. & Weber, I. T. Using triangulation to include
        target structure improves drug resistance prediction accuracy. in 1–1
        (IEEE, 2013). doi:10.1109/ICCABS.2013.6629236

        Yu, X., Weber, I. T. & Harrison, R. W. Prediction of HIV drug
        resistance from genotype with encoded three-dimensional protein
        structure. BMC Genomics 15 Suppl 5, S1 (2014).

    Notes:
    1. We do not use the add_interacting_resis function, because this
        interaction is computed on the ``CA`` atoms. Therefore, there is code
        duplication. For now, I have chosen to leave this code duplication
        in.

    :param G: The networkx graph to add the triangulation to.
    :type G: nx.Graph
    :param allowable_nodes: The nodes to include in the triangulation.
        If ``None`` (default), no filtering is done. This parameter is used to
        filter out nodes that are not desired in the triangulation. Eg if you
        wanted to construct a delaunay triangulation of the CA atoms of an
        atomic graph.
    :type allowable_nodes: List[str], optional
    """
    if allowable_nodes is None:
        coords = np.array([d["coords"] for _, d in G.nodes(data=True)])
        node_map: dict[int, str] = dict(enumerate(G.nodes()))
    else:
        coords = np.array(
            [
                d["coords"]
                for _, d in G.nodes(data=True)
                if d["atom_type"] in allowable_nodes
            ]
        )
        node_map: dict[int, str] = {
            i: n
            for i, (n, d) in enumerate(G.nodes(data=True))
            if d["atom_type"] in allowable_nodes
        }
        node_map: dict[int, str] = dict(enumerate(node_map.values()))

    tri = Delaunay(coords)  # this is the triangulation
    log.debug(
        f"Detected {len(tri.simplices)} simplices in the Delaunay Triangulation."
    )
    for simplex in tri.simplices:
        nodes = [node_map[s] for s in simplex]
        for n1, n2 in combinations(nodes, 2):
            if n1 not in G.nodes or n2 not in G.nodes:
                continue
            add_edge(G, n1, n2, "delaunay")


def add_distance_threshold(
    G: nx.Graph, long_interaction_threshold: int, threshold: float = 5.0
):
    """
    Adds edges to any nodes within a given distance of each other.
    Long interaction threshold is used to specify minimum separation in sequence
    to add an edge between networkx nodes within the distance threshold

    :param G: Protein Structure graph to add distance edges to
    :type G: nx.Graph
    :param long_interaction_threshold: minimum distance in sequence for two
        nodes to be connected. See
        :func:`~graphein.protein.utils.sequence_positions`.
    :type long_interaction_threshold: int
    :param threshold: Distance in angstroms, below which two nodes are connected
    :type threshold: float
    :return: Graph with distance-based edges added
    """
    pdb_df = filter_dataframe(
        G.graph["pdb_df"], "node_id", list(G.nodes()), True
    )
    interacting_nodes = get_interacting_atom_pairs(threshold, pdb_df)
    interacting_nodes = list(zip(interacting_nodes[0], interacting_nodes[1]))

    log.info(f"Found: {len(interacting_nodes)} distance edges")
    node_ids = pdb_df["node_id"].to_numpy()
    chains = pdb_df["chain_id"].to_numpy()
    positions = sequence_positions(pdb_df)
    count = 0
    for a1, a2 in interacting_nodes:
        condition_1 = chains[a1] == chains[a2]
        condition_2 = (
            abs(positions[a1] - positions[a2]) < long_interaction_threshold
        )

        if not (condition_1 and condition_2):
            count += 1
            add_edge(G, node_ids[a1], node_ids[a2], "distance_threshold")

    log.info(
        f"Added {count} distance edges. ({len(list(interacting_nodes)) - count}\
            removed by LIN)"
    )


def add_distance_window(
    G: nx.Graph, min: float, max: float, long_interaction_threshold: int = -1
):
    """
    Adds edges to any nodes within a given window of distances of each other.
    Long interaction threshold is used
    to specify minimum separation in sequence to add an edge between networkx
    nodes within the distance threshold

    :param G: Protein Structure graph to add distance edges to
    :type G: nx.Graph
    :param min: Minimum distance in angstroms required for an edge.
    :type min: float
    :param max: Maximum distance in angstroms allowed for an edge.
    :param long_interaction_threshold: minimum distance in sequence for two
        nodes to be connected. See
        :func:`~graphein.protein.utils.sequence_positions`.
    :type long_interaction_threshold: int
    :return: Graph with distance-based edges added
    """
    pdb_df = filter_dataframe(
        G.graph["pdb_df"], "node_id", list(G.nodes()), True
    )
    # Nodes less than the minimum distance
    less_than_min = get_interacting_atom_pairs(min, pdb_df)
    less_than_min = list(zip(less_than_min[0], less_than_min[1]))

    interacting_nodes = get_interacting_atom_pairs(max, pdb_df)
    interacting_nodes = list(zip(interacting_nodes[0], interacting_nodes[1]))
    less_than_min = set(less_than_min)
    interacting_nodes = [
        i for i in interacting_nodes if i not in less_than_min
    ]

    log.info(f"Found: {len(interacting_nodes)} distance edges")
    node_ids = pdb_df["node_id"].to_numpy()
    chains = pdb_df["chain_id"].to_numpy()
    positions = sequence_positions(pdb_df)
    count = 0
    for a1, a2 in interacting_nodes:
        condition_1 = chains[a1] == chains[a2]
        condition_2 = (
            abs(positions[a1] - positions[a2]) < long_interaction_threshold
        )

        if not (condition_1 and condition_2):
            count += 1
            add_edge(
                G, node_ids[a1], node_ids[a2], f"distance_window_{min}_{max}"
            )
    log.info(
        f"Added {count} distance edges. ({len(list(interacting_nodes)) - count}\
            removed by LIN)"
    )


def add_fully_connected_edges(G: nx.Graph):
    """
    Adds fully connected edges to nodes.

    :param G: Protein structure graph to add fully connected edges to.
    :type G: nx.Graph
    """
    for n1, n2 in itertools.product(G.nodes(), G.nodes()):
        add_edge(G, n1, n2, "fully_connected")


# TODO Support for directed edges
def add_k_nn_edges(
    G: nx.Graph,
    long_interaction_threshold: int = 0,
    k: int = 5,
    exclude_edges: Iterable[str] = (),
    exclude_self_loops: bool = True,
    kind_name: str = "knn",
):
    """
    Adds edges to nodes based on K nearest neighbours. Long interaction
    threshold is used to specify minimum separation in sequence to add an edge
    between networkx nodes within the distance threshold

    :param G: Protein Structure graph to add distance edges to
    :type G: nx.Graph
    :param long_interaction_threshold: minimum distance in sequence for two
        nodes to be connected. See
        :func:`~graphein.protein.utils.sequence_positions`.
    :type long_interaction_threshold: int
    :param k: Number of neighbors for each sample.
    :type k: int
    :param exclude_edges: Types of edges to exclude. Supported values are
        `inter` and `intra`.
        - `inter` removes inter-connections between nodes of the same chain.
        - `intra` removes intra-connections between nodes of different chains.
    :type exclude_edges: Iterable[str].
    :param exclude_self_loops: Whether or not to mark each sample as the first
        nearest neighbor to itself.
    :type exclude_self_loops: Union[bool, str]
    :param kind_name: Name for kind of edges in networkx graph.
    :type kind_name: str
    :return: Graph with knn-based edges added
    :rtype: nx.Graph
    """
    # Prepare dataframe
    pdb_df = filter_dataframe(
        G.graph["pdb_df"], "node_id", list(G.nodes()), True
    )
    if (
        pdb_df["x_coord"].isna().sum()
        or pdb_df["y_coord"].isna().sum()
        or pdb_df["z_coord"].isna().sum()
    ):
        raise ValueError("Coordinates contain a NaN value.")
    pdb_df = pdb_df.reset_index(drop=True)

    # Construct distance matrix
    dist_mat = compute_distmat(pdb_df)

    # Filter edges
    dist_mat = filter_distmat(pdb_df, dist_mat, exclude_edges)

    # Add self-loops if specified
    if not exclude_self_loops:
        k -= 1
        for n1, n2 in zip(G.nodes(), G.nodes()):
            add_edge(G, n1, n2, kind_name)

    # Reduce k if number of nodes is less (to avoid sklearn error)
    # Note: - 1 because self-loops are not included
    if G.number_of_nodes() - 1 < k:
        k = G.number_of_nodes() - 1

    if k == 0:
        return

    # Run k-NN search
    neigh = NearestNeighbors(n_neighbors=k, metric="precomputed")
    neigh.fit(dist_mat)
    nn = neigh.kneighbors_graph()

    # Create iterable of node indices
    outgoing = np.repeat(np.arange(len(pdb_df)), k)
    incoming = nn.indices
    interacting_nodes = list(zip(outgoing, incoming))
    log.info(f"Found: {len(interacting_nodes)} KNN edges")
    dists = dist_mat.to_numpy()
    node_ids = pdb_df["node_id"].to_numpy()
    chains = pdb_df["chain_id"].to_numpy()
    positions = sequence_positions(pdb_df)
    for a1, a2 in interacting_nodes:
        if dists[a1, a2] == INFINITE_DIST:
            continue

        # Check residues are not on same chain
        condition_1 = chains[a1] != chains[a2]
        # Check residues are separated by long_interaction_threshold
        condition_2 = (
            abs(positions[a1] - positions[a2]) > long_interaction_threshold
        )

        # If not on same chain add edge or
        # If on same chain and separation is sufficient add edge
        if condition_1 or condition_2:
            add_edge(G, node_ids[a1], node_ids[a2], kind_name)


def get_ring_atoms(dataframe: pd.DataFrame, aa: str) -> pd.DataFrame:
    """
    Return ring atoms from a dataframe.

    A helper function for add_aromatic_interactions.

    Gets the ring atoms from the particular aromatic amino acid.

    Parameters:
    ===========
    - dataframe: the dataframe containing the atom records.
    - aa: the amino acid of interest, passed in as 3-letter string.

    Returns:
    ========
    - dataframe: a filtered dataframe containing just those atoms from the
                    particular amino acid selected. e.g. equivalent to
                    selecting just the ring atoms from a particular amino
                    acid.
    """
    ring_atom_df = filter_dataframe(dataframe, "residue_name", [aa], True)

    return filter_dataframe(ring_atom_df, "atom_name", AA_RING_ATOMS[aa], True)


def get_ring_centroids(ring_atom_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return aromatic ring centroids.

    A helper function for add_aromatic_interactions.

    Computes the ring centroids for each a particular amino acid's ring
    atoms.

    Ring centroids are computed by taking the mean of the x, y, and z
    coordinates.

    Parameters:
    ===========
    - ring_atom_df: a dataframe computed using get_ring_atoms.
    - aa: the amino acid under study
    Returns:
    ========
    - centroid_df: a dataframe containing just the centroid coordinates of
                    the ring atoms of each residue.
    """
    return (
        ring_atom_df.groupby("node_id")
        .mean(numeric_only=True)[["x_coord", "y_coord", "z_coord"]]
        .reset_index()
    )


def compute_ring_normal(ring_df: pd.DataFrame) -> np.ndarray:
    """Compute the normal vector of a ring.

    :param ring_df: Dataframe of atoms in the ring.
    :type ring_df: pd.DataFrame
    :return: Normal vector of the ring.
    :rtype: np.ndarray
    """
    res_name = ring_df["residue_name"].iloc[0]
    atoms = RING_NORMAL_ATOMS[res_name]
    coords = ring_df.loc[ring_df["atom_name"].isin(atoms)][
        ["x_coord", "y_coord", "z_coord"]
    ].values
    pos1, pos2, pos3 = coords[0], coords[1], coords[2]
    return np.cross(pos2 - pos1, pos3 - pos1)


def get_ring_normals(ring_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the normal vector of each ring.

    :param ring_df: Dataframe of atoms in the rings.
    :type ring_df: pd.DataFrame
    :return: Normal vector of the rings.
    :rtype: pd.DataFrame
    """
    return ring_df.groupby("node_id").apply(compute_ring_normal).reset_index()


def compute_angle(
    v1: np.ndarray, v2: np.ndarray, return_degrees: bool = True
) -> float:
    """Computes angle between two vectors.

    :param v1: First vector
    :type v1: np.ndarray
    :param v2: Second vector
    :type v2: np.ndarray
    :param return_degrees: Whether to return angle in degrees or radians
    :type return_degrees: bool
    """
    angle = np.arccos(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )
    return 180 * angle / np.pi if return_degrees else angle


def get_edges_by_bond_type(
    G: nx.Graph, bond_type: str
) -> list[tuple[str, str]]:
    """
    Return edges of a particular bond type.

    Parameters:
    ===========
    - bond_type: (str) one of the elements in the variable BOND_TYPES

    Returns:
    ========
    - resis: (list) a list of tuples, where each tuple is an edge.
    """
    return [
        (n1, n2) for n1, n2, d in G.edges(data=True) if bond_type in d["kind"]
    ]


def node_coords(G: nx.Graph, n: str) -> tuple[float, float, float]:
    """
    Return the ``x, y, z`` coordinates of a node.
    This is a helper function. Simplifies the code.

    :param G: nx.Graph protein structure graph to extract coordinates from
    :type G: nx.Graph
    :param n: str node ID in graph to extract coordinates from
    :type n: str
    :return: Tuple of coordinates ``(x, y, z)``
    :rtype: Tuple[float, float, float]
    """
    x, y, z = tuple(G.nodes[n]["coords"])
    return x, y, z


def add_interacting_resis(
    G: nx.Graph,
    interacting_atoms: np.ndarray,
    dataframe: pd.DataFrame,
    kind: list[str],
):
    """
    Add interacting residues to graph.

    Returns a list of 2-tuples indicating the interacting residues based
    on the interacting atoms. This is most typically called after the
    get_interacting_atoms function above.

    Also filters out the list such that the residues have to be at least
    two apart.

    ### Parameters

    - interacting_atoms:    (numpy array) result from ``get_interacting_atoms``.
    - dataframe:            (pandas dataframe) a pandas dataframe that
                            houses the euclidean locations of each atom.
    - kind:                 (list) the kind of interaction. Contains one
                            of :
                            - hydrophobic
                            - disulfide
                            - hbond
                            - ionic
                            - aromatic
                            - aromatic_sulphur
                            - cation_pi
                            - delaunay

    Returns:
    ========
    - filtered_interacting_resis: (set of tuples) the residues that are in
        an interaction, with the interaction kind specified
    """
    # This assertion/check is present for defensive programming!
    for k in kind:
        assert k in BOND_TYPES

    resi1 = dataframe.loc[interacting_atoms[0], "node_id"].values
    resi2 = dataframe.loc[interacting_atoms[1], "node_id"].values

    interacting_resis = set(zip(resi1, resi2))
    log.info(f"Found {len(interacting_resis)} {kind} interactions.")
    for i1, i2 in interacting_resis:
        if i1 != i2:
            if G.has_edge(i1, i2):
                for k in kind:
                    G.edges[i1, i2]["kind"].add(k)
            else:
                G.add_edge(i1, i2, kind=set(kind))
