"""Functions for computing biochemical edges of graphs."""

# Graphein
# Author: Eric Ma, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import itertools
from itertools import combinations, product
from typing import Dict, Iterable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger as log
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from graphein.protein.resi_atoms import (
    AA_RING_ATOMS,
    AROMATIC_RESIS,
    BACKBONE_ATOMS,
    BOND_TYPES,
    CATION_PI_RESIS,
    CATION_RESIS,
    DISULFIDE_ATOMS,
    DISULFIDE_RESIS,
    HYDROPHOBIC_RESIS,
    IONIC_RESIS,
    NEG_AA,
    PI_RESIS,
    POS_AA,
    RING_NORMAL_ATOMS,
    SALT_BRIDGE_ANIONS,
    SALT_BRIDGE_ATOMS,
    SALT_BRIDGE_CATIONS,
    SALT_BRIDGE_RESIDUES,
    SULPHUR_RESIS,
    VDW_RADII,
)
from graphein.protein.utils import filter_dataframe

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
    if "atomic_dist_mat" in G.graph.keys():
        dist_mat = G.graph["atomic_dist_mat"]
    elif "dist_mat" in G.graph.keys():
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
    based on their sequence number.

    :param G: Networkx protein graph.
    :type G: nx.Graph
    :param d: Sequence separation to add edges on.
    :param name: Name of the edge type. Defaults to ``"sequence_edge"``.
    :type name: str
    :return G: Networkx protein graph with added peptide bonds.
    :rtype: nx.Graph
    """
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
                # Asserts residue numbers are adjacent
                cond_2 = (
                    abs(
                        residue[1]["residue_number"]
                        - chain_residues[i + d][1]["residue_number"]
                    )
                    == d
                )

                # If this checks out, we add a peptide bond
                if (cond_1) and (cond_2):
                    # Adds "peptide bond" between current residue and the next
                    if G.has_edge(i, i + d):
                        G.edges[i, i + d]["kind"].add(name)
                    else:
                        G.add_edge(
                            residue[0],
                            chain_residues[i + d][0],
                            kind={name},
                        )
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


def add_hydrophobic_interactions(
    G: nx.Graph, rgroup_df: Optional[pd.DataFrame] = None
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
    hydrophobics_df = filter_dataframe(
        hydrophobics_df, "node_id", list(G.nodes()), True
    )
    if hydrophobics_df.shape[0] > 0:
        distmat = compute_distmat(hydrophobics_df)
        interacting_atoms = get_interacting_atoms(5, distmat)
        add_interacting_resis(
            G, interacting_atoms, hydrophobics_df, ["hydrophobic"]
        )


def add_disulfide_interactions(
    G: nx.Graph, rgroup_df: Optional[pd.DataFrame] = None
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
    disulfide_df = filter_dataframe(
        disulfide_df, "node_id", list(G.nodes), True
    )
    if disulfide_df.shape[0] > 0:
        distmat = compute_distmat(disulfide_df)
        interacting_atoms = get_interacting_atoms(2.2, distmat)
        add_interacting_resis(
            G, interacting_atoms, disulfide_df, ["disulfide"]
        )


def add_hydrogen_bond_interactions(
    G: nx.Graph, rgroup_df: Optional[pd.DataFrame] = None
):
    """Add all hydrogen-bond interactions."""
    # For these atoms, find those that are within 3.5A of one another.
    if rgroup_df is None:
        rgroup_df = G.graph["rgroup_df"]
    rgroup_df = filter_dataframe(rgroup_df, "node_id", list(G.nodes()), True)
    HBOND_ATOMS = [
        "ND",  # histidine and asparagine
        "NE",  # glutamate, tryptophan, arginine, histidine
        "NH",  # arginine
        "NZ",  # lysine
        "OD1",
        "OD2",
        "OE",
        "OG",
        "OH",
        "SD",  # cysteine
        "SG",  # methionine
        "N",
        "O",
    ]
    hbond_df = filter_dataframe(rgroup_df, "atom_name", HBOND_ATOMS, True)
    if len(hbond_df.index) > 0:
        distmat = compute_distmat(hbond_df)
        interacting_atoms = get_interacting_atoms(3.5, distmat)
        add_interacting_resis(G, interacting_atoms, hbond_df, ["hbond"])

    # For these atoms, find those that are within 4.0A of one another.
    HBOND_ATOMS_SULPHUR = ["SD", "SG"]
    hbond_df = filter_dataframe(
        rgroup_df, "atom_name", HBOND_ATOMS_SULPHUR, True
    )
    if len(hbond_df.index) > 0:
        distmat = compute_distmat(hbond_df)
        interacting_atoms = get_interacting_atoms(4.0, distmat)
        add_interacting_resis(G, interacting_atoms, hbond_df, ["hbond"])


def add_ionic_interactions(
    G: nx.Graph, rgroup_df: Optional[pd.DataFrame] = None
):
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
    ionic_df = filter_dataframe(rgroup_df, "node_id", list(G.nodes()), True)
    if ionic_df.shape[0] > 0:
        distmat = compute_distmat(ionic_df)
        interacting_atoms = get_interacting_atoms(6, distmat)
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


def add_aromatic_interactions(
    G: nx.Graph, pdb_df: Optional[pd.DataFrame] = None
):
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
        resi_rings_df = filter_dataframe(
            resi_rings_df, "node_id", list(G.nodes()), True
        )
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
    G: nx.Graph, rgroup_df: Optional[pd.DataFrame] = None
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
    aromatic_sulphur_df = filter_dataframe(
        aromatic_sulphur_df, "node_id", list(G.nodes()), True
    )

    if aromatic_sulphur_df.shape[0] > 0:
        distmat = compute_distmat(aromatic_sulphur_df)
        interacting_atoms = get_interacting_atoms(5.3, distmat)
        interacting_atoms = list(
            zip(interacting_atoms[0], interacting_atoms[1])
        )

        for a1, a2 in interacting_atoms:
            resi1 = aromatic_sulphur_df.loc[a1, "node_id"]
            resi2 = aromatic_sulphur_df.loc[a2, "node_id"]

            condition1 = resi1 in SULPHUR_RESIS and resi2 in PI_RESIS
            condition2 = resi1 in PI_RESIS and resi2 in SULPHUR_RESIS

            if (condition1 or condition2) and resi1 != resi2:
                add_edge(G, resi1, resi2, "aromatic_sulphur")


def add_cation_pi_interactions(
    G: nx.Graph, rgroup_df: Optional[pd.DataFrame] = None
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
    cation_pi_df = filter_dataframe(
        cation_pi_df, "node_id", list(G.nodes()), True
    )

    if cation_pi_df.shape[0] > 0:
        distmat = compute_distmat(cation_pi_df)
        interacting_atoms = get_interacting_atoms(6, distmat)
        interacting_atoms = list(
            zip(interacting_atoms[0], interacting_atoms[1])
        )

        for a1, a2 in interacting_atoms:
            resi1 = cation_pi_df.loc[a1, "node_id"]
            resi2 = cation_pi_df.loc[a2, "node_id"]

            condition1 = resi1 in CATION_RESIS and resi2 in PI_RESIS
            condition2 = resi1 in PI_RESIS and resi2 in CATION_RESIS

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
    df = filter_dataframe(df, "node_id", list(g.nodes()), boolean=True)
    dist_mat = compute_distmat(df)

    radii = df["element_symbol"].map(VDW_RADII).values
    radii = np.expand_dims(radii, axis=1)
    radii = radii + radii.T

    dist_mat = dist_mat - radii
    interacting_atoms = get_interacting_atoms(threshold, dist_mat)
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
    pdb_df: Optional[pd.DataFrame] = None,
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
        resi_rings_df = filter_dataframe(
            resi_rings_df, "node_id", list(G.nodes()), True
        )
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
        for n1, n2 in interacting_resis:
            assert G.nodes[n1]["residue_name"] in PI_RESIS
            assert G.nodes[n2]["residue_name"] in PI_RESIS
            n1_centroid = aromatic_df.loc[aromatic_df["node_id"] == n1][
                ["x_coord", "y_coord", "z_coord"]
            ].values[0]
            n2_centroid = aromatic_df.loc[aromatic_df["node_id"] == n2][
                ["x_coord", "y_coord", "z_coord"]
            ].values[0]

            n1_normal = aromatic_df.loc[aromatic_df["node_id"] == n1][
                0
            ].values[0]
            n2_normal = aromatic_df.loc[aromatic_df["node_id"] == n2][
                0
            ].values[0]

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


def add_t_stacking(G: nx.Graph, pdb_df: Optional[pd.DataFrame] = None):
    if pdb_df is None:
        pdb_df = G.graph["raw_pdb_df"]
    dfs = []
    # Compute centroids and normal for each ring
    for resi in PI_RESIS:
        resi_rings_df = get_ring_atoms(pdb_df, resi)
        resi_rings_df = filter_dataframe(
            resi_rings_df, "node_id", list(G.nodes()), True
        )
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
        for n1, n2 in interacting_resis:
            assert G.nodes[n1]["residue_name"] in PI_RESIS
            assert G.nodes[n2]["residue_name"] in PI_RESIS
            n1_centroid = aromatic_df.loc[aromatic_df["node_id"] == n1][
                ["x_coord", "y_coord", "z_coord"]
            ].values[0]
            n2_centroid = aromatic_df.loc[aromatic_df["node_id"] == n2][
                ["x_coord", "y_coord", "z_coord"]
            ].values[0]

            n1_normal = aromatic_df.loc[aromatic_df["node_id"] == n1][
                0
            ].values[0]
            n2_normal = aromatic_df.loc[aromatic_df["node_id"] == n2][
                0
            ].values[0]

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
    df = filter_dataframe(df, "node_id", list(G.nodes()), boolean=True)
    df = filter_dataframe(df, "atom_name", ["C", "O"], boolean=True)
    distmat = compute_distmat(df)
    interacting_atoms = get_interacting_atoms(threshold, distmat)

    # Filter out O-O and C-C edges
    atom_1 = df.iloc[interacting_atoms[0]]["atom_name"]
    atom_2 = df.iloc[interacting_atoms[1]]["atom_name"]
    diff_atoms = atom_1.values != atom_2.values
    interacting_atoms = (
        interacting_atoms[0][diff_atoms],
        interacting_atoms[1][diff_atoms],
    )

    # Filter out O-C edges on the same residue
    atom_1 = df.iloc[interacting_atoms[0]]["residue_id"]
    atom_2 = df.iloc[interacting_atoms[1]]["residue_id"]
    diff_atoms = atom_1.values != atom_2.values
    interacting_atoms = (
        interacting_atoms[0][diff_atoms],
        interacting_atoms[1][diff_atoms],
    )

    add_interacting_resis(G, interacting_atoms, df, ["bb_carbonyl_carbonyl"])


def add_salt_bridges(
    G: nx.Graph,
    rgroup_df: Optional[pd.DataFrame] = None,
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
    if salt_bridge_df.shape[0] > 0:
        distmat = compute_distmat(salt_bridge_df)
        interacting_atoms = get_interacting_atoms(threshold, distmat)
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


def add_delaunay_triangulation(
    G: nx.Graph, allowable_nodes: Optional[List[str]] = None
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
        node_map: Dict[int, str] = dict(enumerate(G.nodes()))
    else:
        coords = np.array(
            [
                d["coords"]
                for _, d in G.nodes(data=True)
                if d["atom_type"] in allowable_nodes
            ]
        )
        node_map: Dict[int, str] = {
            i: n
            for i, (n, d) in enumerate(G.nodes(data=True))
            if d["atom_type"] in allowable_nodes
        }
        node_map: Dict[int, str] = dict(enumerate(node_map.values()))

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
        nodes to be connected
    :type long_interaction_threshold: int
    :param threshold: Distance in angstroms, below which two nodes are connected
    :type threshold: float
    :return: Graph with distance-based edges added
    """
    pdb_df = filter_dataframe(
        G.graph["pdb_df"], "node_id", list(G.nodes()), True
    )
    dist_mat = compute_distmat(pdb_df)
    interacting_nodes = get_interacting_atoms(threshold, distmat=dist_mat)
    interacting_nodes = list(zip(interacting_nodes[0], interacting_nodes[1]))

    log.info(f"Found: {len(interacting_nodes)} distance edges")
    count = 0
    for a1, a2 in interacting_nodes:
        n1 = G.graph["pdb_df"].loc[a1, "node_id"]
        n2 = G.graph["pdb_df"].loc[a2, "node_id"]
        n1_chain = G.graph["pdb_df"].loc[a1, "chain_id"]
        n2_chain = G.graph["pdb_df"].loc[a2, "chain_id"]
        n1_position = G.graph["pdb_df"].loc[a1, "residue_number"]
        n2_position = G.graph["pdb_df"].loc[a2, "residue_number"]

        condition_1 = n1_chain == n2_chain
        condition_2 = (
            abs(n1_position - n2_position) < long_interaction_threshold
        )

        if not (condition_1 and condition_2):
            count += 1
            add_edge(G, n1, n2, "distance_threshold")

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
        nodes to be connected
    :type long_interaction_threshold: int
    :return: Graph with distance-based edges added
    """
    pdb_df = filter_dataframe(
        G.graph["pdb_df"], "node_id", list(G.nodes()), True
    )
    dist_mat = compute_distmat(pdb_df)
    # Nodes less than the minimum distance
    less_than_min = get_interacting_atoms(min, distmat=dist_mat)
    less_than_min = list(zip(less_than_min[0], less_than_min[1]))

    interacting_nodes = get_interacting_atoms(max, distmat=dist_mat)
    interacting_nodes = list(zip(interacting_nodes[0], interacting_nodes[1]))
    interacting_nodes = [
        i for i in interacting_nodes if i not in less_than_min
    ]

    log.info(f"Found: {len(interacting_nodes)} distance edges")
    count = 0
    for a1, a2 in interacting_nodes:
        n1 = G.graph["pdb_df"].loc[a1, "node_id"]
        n2 = G.graph["pdb_df"].loc[a2, "node_id"]
        n1_chain = G.graph["pdb_df"].loc[a1, "chain_id"]
        n2_chain = G.graph["pdb_df"].loc[a2, "chain_id"]
        n1_position = G.graph["pdb_df"].loc[a1, "residue_number"]
        n2_position = G.graph["pdb_df"].loc[a2, "residue_number"]

        condition_1 = n1_chain == n2_chain
        condition_2 = (
            abs(n1_position - n2_position) < long_interaction_threshold
        )

        if not (condition_1 and condition_2):
            count += 1
            add_edge(G, n1, n2, f"distance_window_{min}_{max}")
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
        add_edge(G, n1, n2, f"fully_connected")


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
        nodes to be connected
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
    outgoing = np.repeat(np.array(range(len(pdb_df))), k)
    incoming = nn.indices
    interacting_nodes = list(zip(outgoing, incoming))
    log.info(f"Found: {len(interacting_nodes)} KNN edges")
    for a1, a2 in interacting_nodes:
        if dist_mat.loc[a1, a2] == INFINITE_DIST:
            continue

        # Get nodes IDs from indices
        n1 = pdb_df.loc[a1, "node_id"]
        n2 = pdb_df.loc[a2, "node_id"]

        # Get chains
        n1_chain = pdb_df.loc[a1, "chain_id"]
        n2_chain = pdb_df.loc[a2, "chain_id"]

        # Get sequence position
        n1_position = pdb_df.loc[a1, "residue_number"]
        n2_position = pdb_df.loc[a2, "residue_number"]

        # Check residues are not on same chain
        condition_1 = n1_chain != n2_chain
        # Check residues are separated by long_interaction_threshold
        condition_2 = (
            abs(n1_position - n2_position) > long_interaction_threshold
        )

        # If not on same chain add edge or
        # If on same chain and separation is sufficient add edge
        if condition_1 or condition_2:
            add_edge(G, n1, n2, kind_name)


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

    ring_atom_df = filter_dataframe(
        ring_atom_df, "atom_name", AA_RING_ATOMS[aa], True
    )
    return ring_atom_df


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
) -> List[Tuple[str, str]]:
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


def node_coords(G: nx.Graph, n: str) -> Tuple[float, float, float]:
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
    kind: List[str],
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

    resi1 = dataframe.loc[interacting_atoms[0]]["node_id"].values
    resi2 = dataframe.loc[interacting_atoms[1]]["node_id"].values

    interacting_resis = set(list(zip(resi1, resi2)))
    log.info(f"Found {len(interacting_resis)} {k} interactions.")
    for i1, i2 in interacting_resis:
        if i1 != i2:
            if G.has_edge(i1, i2):
                for k in kind:
                    G.edges[i1, i2]["kind"].add(k)
            else:
                G.add_edge(i1, i2, kind=set(kind))
