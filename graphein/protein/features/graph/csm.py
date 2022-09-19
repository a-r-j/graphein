"""Provides CSM feature computation for structure graphs.

See: Pires DE, de Melo-Minardi RC, dos Santos MA, da Silveira CH, Santoro MM,
Meira W Jr. Cutoff Scanning Matrix (CSM): structural classification and
function prediction by protein inter-residue distance patterns. BMC Genomics.
2011 Dec 22;12 Suppl 4(Suppl 4):S12. doi: 10.1186/1471-2164-12-S4-S12. Epub
2011 Dec 22. PMID: 22369665; PMCID: PMC3287581.
"""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import List

import networkx as nx
import numpy as np

from graphein.protein.edges.distance import compute_distmat


def compute_csm(
    g: nx.Graph,
    distance_min: float = 0.0,
    distance_max: float = 30.0,
    distance_step: float = 0.2,
) -> np.ndarray:
    """Computes the Cutoff Scanning Matrix.

    We use the Euclidean distance between all pairs of nodes in the graph (the
    original paper uses Cα) and define a range of distances (cutoffs) to be
    considered and a distance step. We scan through these distances, computing
    the frequency of pairs of residues, each represented by its Cα , that are
    close according to this distance threshold.

    See: Pires DE, de Melo-Minardi RC, dos Santos MA, da Silveira CH, Santoro
    MM, Meira W Jr. Cutoff Scanning Matrix (CSM): structural classification and
    function prediction by protein inter-residue distance patterns. BMC
    Genomics. 2011 Dec 22;12 Suppl 4(Suppl 4):S12. doi:
    10.1186/1471-2164-12-S4-S12. Epub 2011 Dec 22. PMID: 22369665; PMCID:
    PMC3287581.

    :param g: Graph to compute CSM for.
    :type g: nx.Graph
    :param distance_min: Minimum distance to consider. Defaults to ``0.0``.
    :type distance_min: float
    :param distance_max: Maximum distance to consider (angstroms). Defaults to ``30.0``.
    :type distance_max: float
    :param distance_step: Distance step size (angstroms). Defaults to ``0.2``.
    :type distance_step: float
    :return: CSM Array
    :rtype: np.ndarray
    """
    if "dist_mat" in g.graph.keys():
        dist_mat = g.graph["dist_mat"]
    else:
        dist_mat = compute_distmat(g.graph["pdb_df"])

    # Getter upper triangle of distmat and remove 0 counts
    lengths = np.triu(dist_mat).flatten().tolist()
    lengths = [i for i in lengths if i != 0.0]

    # Initialise CSM Array
    CSM: List[int] = [
        0 for _ in np.arange(distance_min, distance_max, distance_step)
    ]

    i = 0
    d = distance_min
    while d <= distance_max - distance_step:
        for dist in lengths:
            if dist >= d and dist <= d + distance_step:
                CSM[i] += 1
        d += distance_step
        i += 1
    return np.array(CSM)


def compute_csm_pdf(
    g: nx.Graph,
    distance_min: float = 0.0,
    distance_max: float = 30.0,
    distance_step: float = 0.2,
) -> np.ndarray:
    """Computes the probability density function of the Cutoff Scanning Matrix.

    :param g: NetworkX graph to compute CSM PDF for.
    :type g: nx.Graph
    :param distance_min: Minimum distance to consider (Angstroms). Defaults to ``0.0``.
    :type distance_min: float
    :param distance_max: Maximum distance to consider (Angstroms). Defaults to ``30.0``.
    :type distance_max: float
    :param distance_step: Distance step size (Angstroms). Defaults to ``0.2``.
    :type distance_step: float
    :return: CSM PDF
    :rtype: np.ndarray
    """
    csm = compute_csm(g, distance_min, distance_max, distance_step)
    return csm / np.sum(csm)


def compute_csm_cdf(
    g: nx.Graph,
    distance_min: float = 0.0,
    distance_max: float = 30.0,
    distance_step: float = 0.2,
) -> np.ndarray:
    """Computes the cumulative density function of the Cutoff Scanning Matrix.

    :param g: NetworkX graph to compute CSM CDF for.
    :type g: nx.Graph
    :param distance_min: Minimum distance to consider (Angstroms). Defaults to ``0.0``.
    :type distance_min: float
    :param distance_max: Maximum distance to consider (Angstroms). Defaults to ``30.0``.
    :type distance_max: float
    :param distance_step: Distance step size (Angstroms). Defaults to ``0.2``.
    :type distance_step: float
    :return: CSM PDF
    :rtype: np.ndarray
    """
    csm_pdf = compute_csm_pdf(g, distance_min, distance_max, distance_step)
    return np.cumsum(csm_pdf)


def add_csm(
    g: nx.Graph,
    distance_min: float = 0.0,
    distance_max: float = 30.0,
    distance_step: float = 0.2,
) -> nx.Graph:
    """Adds CSM features to a graph.

    :param g: Graph to add CSM features to.
    :type g: nx.Graph
    :param distance_min: Minimum distance to consider (Angstroms). Defaults to ``0.0``.
    :type distance_min: float
    :param distance_max: Maximum distance to consider (Angstroms). Defaults to ``30.0``.
    :type distance_max: float
    :param distance_step: Distance step size (Angstroms). Defaults to ``0.2``.
    :type distance_step: float
    :return: Graph with CSM features added.
    :rtype: nx.Graph
    """
    g.graph["CSM"] = compute_csm(g, distance_min, distance_max, distance_step)
    return g


def add_csm_cdf(
    g: nx.Graph,
    distance_min: float = 0.0,
    distance_max: float = 30.0,
    distance_step: float = 0.2,
) -> nx.Graph:
    """Adds CSM CDF features to a graph.

    :param g: Graph to add CSM CDF features to.
    :type g: nx.Graph
    :param distance_min: Minimum distance to consider (Angstroms). Defaults to ``0.0``.
    :type distance_min: float
    :param distance_max: Maximum distance to consider (Angstroms). Defaults to ``30.0``.
    :type distance_max: float
    :param distance_step: Distance step size (Angstroms). Defaults to ``0.2``.
    :type distance_step: float
    :return: Graph with CSM CDF features added.
    :rtype: nx.Graph
    """
    g.graph["CSM_cdf"] = compute_csm_cdf(
        g, distance_min, distance_max, distance_step
    )
    return g


def add_csm_pdf(
    g: nx.Graph,
    distance_min: float = 0.0,
    distance_max: float = 30.0,
    distance_step: float = 0.2,
) -> nx.Graph:
    """Adds CSM PDF features to a graph.

    :param g: Graph to add CSM PDF features to.
    :type g: nx.Graph
    :param distance_min: Minimum distance to consider (Angstroms). Defaults to ``0.0``.
    :type distance_min: float
    :param distance_max: Maximum distance to consider (Angstroms). Defaults to ``30.0``.
    :type distance_max: float
    :param distance_step: Distance step size (Angstroms). Defaults to ``0.2``.
    :type distance_step: float
    :return: Graph with CSM PDF features added.
    :rtype: nx.Graph
    """
    g.graph["CSM_pdf"] = compute_csm_pdf(
        g, distance_min, distance_max, distance_step
    )
    return g
