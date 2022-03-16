"""Functions for computing biochemical edges of graphs."""
# Graphein
# Author: Eric Ma, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean, pdist, rogerstanimoto, squareform
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

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
)
from graphein.protein.utils import filter_dataframe

log = logging.getLogger(__name__)


def compute_distmat(pdb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise euclidean distances between every atom.

    Design choice: passed in a DataFrame to enable easier testing on
    dummy data.

    :param pdb_df: pd.Dataframe containing protein structure. Must contain columns ["x_coord", "y_coord", "z_coord"]
    :type pdb_df: pd.DataFrame
    :return: pd.Dataframe of euclidean distance matrix
    :rtype: pd.DataFrame
    """
    eucl_dists = pdist(
        pdb_df[["x_coord", "y_coord", "z_coord"]], metric="euclidean"
    )
    eucl_dists = pd.DataFrame(squareform(eucl_dists))
    eucl_dists.index = pdb_df.index
    eucl_dists.columns = pdb_df.index

    return eucl_dists


def add_peptide_bonds(G: nx.Graph) -> nx.Graph:
    """
    Adds peptide backbone as edges to residues in each chain.

    :param G: networkx protein graph.
    :type G: nx.Graph
    :return G: networkx protein graph with added peptide bonds.
    :rtype: nx.Graph
    """
    # Iterate over every chain
    for chain_id in G.graph["chain_ids"]:

        # Find chain residues
        chain_residues = [
            (n, v) for n, v in G.nodes(data=True) if v["chain_id"] == chain_id
        ]

        # Iterate over every residue in chain
        for i, residue in enumerate(chain_residues):
            # Checks not at chain terminus - is this versatile enough?
            if i == len(chain_residues) - 1:
                continue
            # Asserts residues are on the same chain
            cond_1 = (
                residue[1]["chain_id"] == chain_residues[i + 1][1]["chain_id"]
            )
            # Asserts residue numbers are adjacent
            cond_2 = (
                abs(
                    residue[1]["residue_number"]
                    - chain_residues[i + 1][1]["residue_number"]
                )
                == 1
            )

            # If this checks out, we add a peptide bond
            if (cond_1) and (cond_2):
                # Adds "peptide bond" between current residue and the next
                if G.has_edge(i, i + 1):
                    G.edges[i, i + 1]["kind"].add("peptide_bond")
                else:
                    G.add_edge(
                        residue[0],
                        chain_residues[i + 1][0],
                        kind={"peptide_bond"},
                    )
            else:
                continue
    return G


def add_hydrophobic_interactions(
    G: nx.Graph, rgroup_df: Optional[pd.DataFrame] = None
):
    """
    Find all hydrophobic interactions.

    Performs searches between the following residues:
    ``[ALA, VAL, LEU, ILE, MET, PHE, TRP, PRO, TYR]`` (:const:`~graphein.protein.resi_atoms.HYDROPHOBIC_RESIS`).

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
    distmat = compute_distmat(hydrophobics_df)
    interacting_atoms = get_interacting_atoms(5, distmat)
    add_interacting_resis(
        G, interacting_atoms, hydrophobics_df, ["hydrophobic"]
    )


def add_disulfide_interactions(
    G: nx.Graph, rgroup_df: Optional[pd.DataFrame] = None
):
    """
    Find all disulfide interactions between CYS residues (:const:`~graphein.protein.resi_atoms.DISULFIDE_RESIS`, :const:`~graphein.protein.resi_atoms.DISULFIDE_ATOMS`).

    Criteria: sulfur atom pairs are within 2.2A of each other.

    :param G: networkx protein graph
    :type G: nx.Graph
    :param rgroup_df: pd.DataFrame containing rgroup data, defaults to None, which retrieves the df from the provided nx graph.
    :type rgroup_df: pd.DataFrame, optional
    """
    # Check for existence of at least two Cysteine residues
    residues = [d["residue_name"] for _, d in G.nodes(data=True)]
    if residues.count("CYS") < 2:
        log.debug(
            f"{residues.count('CYS')} CYS residues found. Cannot add disulfide interactions with fewer than two CYS residues."
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
    distmat = compute_distmat(disulfide_df)
    interacting_atoms = get_interacting_atoms(2.2, distmat)
    add_interacting_resis(G, interacting_atoms, disulfide_df, ["disulfide"])


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

    Criteria: ``[ARG, LYS, HIS, ASP, and GLU]`` (:const:`~graphein.protein.resi_atoms.IONIC_RESIS`) residues are within 6A.
    We also check for opposing charges (:const:`~graphein.protein.resi_atoms.POS_AA`, :const:`~graphein.protein.resi_atoms.NEG_AA`)
    """
    if rgroup_df is None:
        rgroup_df = G.graph["rgroup_df"]
    ionic_df = filter_dataframe(rgroup_df, "residue_name", IONIC_RESIS, True)
    ionic_df = filter_dataframe(rgroup_df, "node_id", list(G.nodes()), True)
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
    Phenyl rings are present on ``PHE, TRP, HIS, TYR`` (:const:`~graphein.protein.resi_atoms.AROMATIC_RESIS`).
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
    distmat = compute_distmat(aromatic_df)
    distmat.set_index(aromatic_df["node_id"], inplace=True)
    distmat.columns = aromatic_df["node_id"]
    distmat = distmat[(distmat >= 4.5) & (distmat <= 7)].fillna(0)
    indices = np.where(distmat > 0)

    interacting_resis = [
        (distmat.index[r], distmat.index[c])
        for r, c in zip(indices[0], indices[1])
    ]
    log.info(f"Found: {len(interacting_resis)} aromatic-aromatic interactions")
    for n1, n2 in interacting_resis:
        assert G.nodes[n1]["residue_name"] in AROMATIC_RESIS
        assert G.nodes[n2]["residue_name"] in AROMATIC_RESIS
        if G.has_edge(n1, n2):
            G.edges[n1, n2]["kind"].add("aromatic")
        else:
            G.add_edge(n1, n2, kind={"aromatic"})


def add_aromatic_sulphur_interactions(
    G: nx.Graph, rgroup_df: Optional[pd.DataFrame] = None
):
    """Find all aromatic-sulphur interactions."""
    if rgroup_df is None:
        rgroup_df = G.graph["rgroup_df"]
    RESIDUES = ["MET", "CYS", "PHE", "TYR", "TRP"]
    SULPHUR_RESIS = ["MET", "CYS"]
    AROMATIC_RESIS = ["PHE", "TYR", "TRP"]

    aromatic_sulphur_df = filter_dataframe(
        rgroup_df, "residue_name", RESIDUES, True
    )
    aromatic_sulphur_df = filter_dataframe(
        aromatic_sulphur_df, "node_id", list(G.nodes()), True
    )
    distmat = compute_distmat(aromatic_sulphur_df)
    interacting_atoms = get_interacting_atoms(5.3, distmat)
    interacting_atoms = list(zip(interacting_atoms[0], interacting_atoms[1]))

    for (a1, a2) in interacting_atoms:
        resi1 = aromatic_sulphur_df.loc[a1, "node_id"]
        resi2 = aromatic_sulphur_df.loc[a2, "node_id"]

        condition1 = resi1 in SULPHUR_RESIS and resi2 in AROMATIC_RESIS
        condition2 = resi1 in AROMATIC_RESIS and resi2 in SULPHUR_RESIS

        if (condition1 or condition2) and resi1 != resi2:
            if G.has_edge(resi1, resi2):
                G.edges[resi1, resi2]["kind"].add("aromatic_sulphur")
            else:
                G.add_edge(resi1, resi2, kind={"aromatic_sulphur"})


def add_cation_pi_interactions(
    G: nx.Graph, rgroup_df: Optional[pd.DataFrame] = None
):
    """Add cation-pi interactions."""
    if rgroup_df is None:
        rgroup_df = G.graph["rgroup_df"]
    cation_pi_df = filter_dataframe(
        rgroup_df, "residue_name", CATION_PI_RESIS, True
    )
    cation_pi_df = filter_dataframe(
        cation_pi_df, "node_id", list(G.nodes()), True
    )
    distmat = compute_distmat(cation_pi_df)
    interacting_atoms = get_interacting_atoms(6, distmat)
    interacting_atoms = list(zip(interacting_atoms[0], interacting_atoms[1]))

    for (a1, a2) in interacting_atoms:
        resi1 = cation_pi_df.loc[a1, "node_id"]
        resi2 = cation_pi_df.loc[a2, "node_id"]

        condition1 = resi1 in CATION_RESIS and resi2 in PI_RESIS
        condition2 = resi1 in PI_RESIS and resi2 in CATION_RESIS

        if (condition1 or condition2) and resi1 != resi2:
            if G.has_edge(resi1, resi2):
                G.edges[resi1, resi2]["kind"].add("cation_pi")
            else:
                G.add_edge(resi1, resi2, kind={"cation_pi"})


def get_interacting_atoms(angstroms: float, distmat: pd.DataFrame):
    """Find the atoms that are within a particular radius of one another."""
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
    :param allowable_nodes: The nodes to include in the triangulation. If ``None`` (default), no filtering is done.
        This parameter is used to filter out nodes that are not desired in the triangulation.
        Eg if you wanted to construct a delaunay triangulation of the CA atoms of an atomic graph.
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
            if G.has_edge(n1, n2):
                G.edges[n1, n2]["kind"].add("delaunay")
            else:
                G.add_edge(n1, n2, kind={"delaunay"})


def add_distance_threshold(
    G: nx.Graph, long_interaction_threshold: int, threshold: float = 5.0
):
    """
    Adds edges to any nodes within a given distance of each other. Long interaction threshold is used
    to specify minimum separation in sequence to add an edge between networkx nodes within the distance threshold

    :param G: Protein Structure graph to add distance edges to
    :type G: nx.Graph
    :param long_interaction_threshold: minimum distance in sequence for two nodes to be connected
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
    interacting_nodes = zip(interacting_nodes[0], interacting_nodes[1])

    log.info(f"Found: {len(list(interacting_nodes))} distance edges")
    for a1, a2 in interacting_nodes:
        n1 = G.graph["pdb_df"].loc[a1, "node_id"]
        n2 = G.graph["pdb_df"].loc[a2, "node_id"]
        n1_chain = G.graph["pdb_df"].loc[a1, "chain_id"]
        n2_chain = G.graph["pdb_df"].loc[a2, "chain_id"]
        n1_position = G.graph["pdb_df"].loc[a1, "residue_number"]
        n2_position = G.graph["pdb_df"].loc[a2, "residue_number"]

        condition_1 = n1_chain != n2_chain
        condition_2 = (
            abs(n1_position - n2_position) > long_interaction_threshold
        )

        if condition_1 or condition_2:
            if G.has_edge(n1, n2):
                G.edges[n1, n2]["kind"].add("distance_threshold")
            else:
                G.add_edge(n1, n2, kind={"distance_threshold"})


def add_k_nn_edges(
    G: nx.Graph,
    long_interaction_threshold: int,
    k: int = 5,
    mode: str = "connectivity",
    metric: str = "minkowski",
    p: int = 2,
    include_self: Union[bool, str] = False,
):
    """
    Adds edges to nodes based on K nearest neighbours. Long interaction threshold is used
    to specify minimum separation in sequence to add an edge between networkx nodes within the distance threshold

    :param G: Protein Structure graph to add distance edges to
    :type G: nx.Graph
    :param long_interaction_threshold: minimum distance in sequence for two nodes to be connected
    :type long_interaction_threshold: int
    :param k: Number of neighbors for each sample.
    :type k: int
    :param mode: Type of returned matrix: ``"connectivity"`` will return the connectivity matrix with ones and zeros,
        and ``"distance"`` will return the distances between neighbors according to the given metric.
    :type mode: str
    :param metric: The distance metric used to calculate the k-Neighbors for each sample point.
        The DistanceMetric class gives a list of available metrics.
        The default distance is ``"euclidean"`` (``"minkowski"`` metric with the ``p`` param equal to ``2``).
    :type metric: str
    :param p: Power parameter for the Minkowski metric. When ``p = 1``, this is equivalent to using ``manhattan_distance`` (l1),
        and ``euclidean_distance`` (l2) for ``p = 2``. For arbitrary ``p``, ``minkowski_distance`` (l_p) is used. Default is ``2`` (euclidean).
    :type p: int
    :param include_self: Whether or not to mark each sample as the first nearest neighbor to itself.
        If ``"auto"``, then ``True`` is used for ``mode="connectivity"`` and ``False`` for ``mode="distance"``. Default is ``False``.
    :type include_self: Union[bool, str]
    :return: Graph with knn-based edges added
    :rtype: nx.Graph
    """
    pdb_df = filter_dataframe(
        G.graph["pdb_df"], "node_id", list(G.nodes()), True
    )
    dist_mat = compute_distmat(pdb_df)

    nn = kneighbors_graph(
        X=dist_mat,
        n_neighbors=k,
        mode=mode,
        metric=metric,
        p=p,
        include_self=include_self,
    )

    # Create iterable of node indices
    outgoing = np.repeat(np.array(range(len(G.graph["pdb_df"]))), k)
    incoming = nn.indices
    interacting_nodes = list(zip(outgoing, incoming))
    log.info(f"Found: {len(interacting_nodes)} KNN edges")
    for a1, a2 in interacting_nodes:
        # Get nodes IDs from indices
        n1 = G.graph["pdb_df"].loc[a1, "node_id"]
        n2 = G.graph["pdb_df"].loc[a2, "node_id"]

        # Get chains
        n1_chain = G.graph["pdb_df"].loc[a1, "chain_id"]
        n2_chain = G.graph["pdb_df"].loc[a2, "chain_id"]

        # Get sequence position
        n1_position = G.graph["pdb_df"].loc[a1, "residue_number"]
        n2_position = G.graph["pdb_df"].loc[a2, "residue_number"]

        # Check residues are not on same chain
        condition_1 = n1_chain != n2_chain
        # Check residues are separated by long_interaction_threshold
        condition_2 = (
            abs(n1_position - n2_position) > long_interaction_threshold
        )

        # If not on same chain add edge or
        # If on same chain and separation is sufficient add edge
        if condition_1 or condition_2:
            if G.has_edge(n1, n2):
                G.edges[n1, n2]["kind"].add("k_nn")
            else:
                G.add_edge(n1, n2, kind={"k_nn"})


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
    Return aromatic ring centrods.

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
        .mean()[["x_coord", "y_coord", "z_coord"]]
        .reset_index()
    )


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
    x = G.nodes[n]["x_coord"]
    y = G.nodes[n]["y_coord"]
    z = G.nodes[n]["z_coord"]

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

    - interacting_atoms:    (numpy array) result from get_interacting_atoms function.
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
