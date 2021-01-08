"""Functions for creating Chromatin Structure Graphs from HiC Data"""
# %%
# Graphein
# Author: Dominic Hall, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import logging
from typing import Dict, Optional

import cooler
import networkx as nx
import numpy as np

from graphein.chromatin.parse_cooler import (
    fetch_bins_from_cooler,
    get_unique_bins,
)

log = logging.getLogger(__name__)


def initialise_graph_with_metadata(cooler_file, region1, region2) -> nx.Graph:
    return nx.Graph(cooler_file=cooler_file, region1=region1, region2=region2)


def compute_HiC_graph_from_regions(
    contacts: cooler,
    regions: Dict[str, np.ndarray],
    balance: Optional[bool] = False,
    cis: Optional[bool] = True,
    trans: Optional[bool] = True,
) -> nx.Graph:
    """
    Computes a HiC Graph from a cooler file
    :param contacts: cooler file generated from a Hi-C experiment
    :param regions: Dictionary specifying chromosomes and regions to collect data over. Dictionary should contain chromosomes as keys and 2D integer numpy arrays as values.
    :params balance: Optional boolean to determine whether returned weights should be balanced or not.
    :param graph_annotation_funcs: List of functions functools annotate graph metadata
    :param node_annotation_funcs: List of functions to annotate node metadata
    :param edge_annotation_funcs: List of function to annotate edge metadata
    :return: nx.Graph of Hi-C Contacts
    """

    c = cooler.Cooler(contacts)

    # Fetch relevant bin_ids from the cooler file
    b_ids = fetch_bins_from_cooler(cooler=c, regions=regions)
    # Identify unique bin_ids and isolate disjoint regions
    slices = get_unique_bins(b_ids=b_ids)

    # Initialise Graph List
    Glist = []
    for idx, s1 in enumerate(slices):
        for s2 in slices[idx:]:
            if s1[0] == s2[0] and not cis:
                continue
            if s1[0] != s2[0] and not trans:
                continue

            # Chromosome, start, end, bins and node names for region 1
            c1 = c.bins()[s1[0]]["chrom"].values[0]
            st1 = c.bins()[s1[0]]["start"].values[0]
            e1 = c.bins()[s1[-1] + 1]["end"].values[0]

            s1_id = f"{c1}:{st1}-{e1}"

            b1 = c.bins()[s1[0] : s1[-1] + 1]
            n1 = b1.apply(lambda x: f"{x[0]}:{x[1]}-{x[2]}", axis=1).values

            # Chromosome, start, end, bins and node names for region 2
            c2 = c.bins()[s2[0]]["chrom"].values[0]
            st2 = c.bins()[s2[0]]["start"].values[0]
            e2 = c.bins()[s2[-1] + 1]["end"].values[0]

            s2_id = f"{c2}:{st2}-{e2}"

            b2 = c.bins()[s2[0] : s2[-1] + 1]
            n2 = b1.apply(lambda x: f"{x[0]}:{x[1]}-{x[2]}", axis=1).values

            # Create graph and add unique bins as nodes
            G = nx.Graph(cooler_file=contacts, region1=s1_id, region2=s2_id)

            if s1_id == s2_id:
                unique_bins = b1.index.values
                unique_nodes = n1
            else:
                unique_bins = np.append(b1.index.values, b2.index.values)
                unique_nodes = np.append(n1, n2)

            G.add_nodes_from(unique_bins)
            log.debug(
                f"Added {len(unique_bins)} nodes to {s1_id}-{s2_id} graph"
            )

            nx.set_node_attributes(
                G, dict(zip(unique_bins, unique_nodes)), "bin_regions"
            )

            mat = c.matrix(balance=balance, sparse=True)[
                s1[0] : s1[-1] + 1, s2[0] : s2[-1] + 1
            ]

            edge_data = np.concatenate(
                [
                    s1[mat.row][:, None],
                    s2[mat.col][:, None],
                    mat.data[:, None],
                ],
                axis=1,
            )
            G.add_weighted_edges_from(
                [(row[0], row[1], row[2]) for row in edge_data]
            )

            if s1_id == s2_id and backbone:
                bbone_edges = [(b_id, b_id + 1) for b_id in s1]
                not_included = []
                for edge in bbone_edges:
                    if edge in G.edges:
                        G[edge[0]][edge[1]]["backbone"] = True
                    else:
                        G.add_edge(edge[0], edge[1], weight=0, backbone=True)

            Glist.append(G)

    return Glist


if __name__ == "__main__":
    regions = {
        "chr1": np.array([[1, 10000], [11000, 20000]]),
        "chr2": np.array([[1, 10000], [11000, 20000]]),
        "chr3": np.array([[1, 10000], [11000, 20000]]),
    }

    compute_HiC_graph_from_regions(
        "Dixon2012-H1hESC-HindIII-allreps-filtered.1000kb.cool",
        regions=regions,
    )
