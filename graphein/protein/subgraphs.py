"""Provides functions for extracting subgraphs from protein graphs."""
import logging

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from graphein.protein.edges.distance import compute_distmat

log = logging.getLogger(__name__)


def extract_subgraph_from_node_list(
    g,
    node_list: Optional[List[str]],
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    """Extracts a subgraph from a graph based on a list of nodes.

    :param g: The graph to extract the subgraph from.
    :type g: nx.Graph
    :param node_list: The list of nodes to extract.
    :type node_list: List[str]
    :param filter_dataframe: Whether to filter the pdb_df dataframe of the graph. Defaults to True.
    :type filter_dataframe: bool
    :param update_coords: Whether to update the coordinates of the graph. Defaults to True.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the graph. Defaults to False.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to False.
    :type inverse: bool
    :return: The subgraph or node list if return_node_list is True.
    :rtype: Union[nx.Graph, List[str]]
    """
    if node_list:
        # Get all nodes not in nodelist if inversing the selection
        if inverse:
            node_list = [n for n in g.nodes() if n not in node_list]

        # If we are just returning the node list, return it here before subgraphing.
        if return_node_list:
            return node_list

        log.debug(f"Creating subgraph from nodes: {node_list}.")
        # Create a subgraph from the node list.
        g = g.subgraph(node_list)
        # Filter the PDB DF accordingly
        if filter_dataframe:
            g.graph["pdb_df"] = g.graph["pdb_df"].loc[
                g.graph["pdb_df"]["node_id"].isin(node_list)
            ]
        if update_coords:
            g.graph["coords"] = np.array(
                d["coords"] for d in g.nodes(data=True)
            )
        if recompute_distmat:
            if not filter_dataframe:
                log.warning("Recomputing distmat without filtering dataframe.")
            g.graph["distmat"] = compute_distmat(g.graph["pdb_df"])
    if return_node_list:
        return node_list

    return g


def extract_subgraph_from_point(
    g: nx.Graph,
    centre_point: Union[np.ndarray, Tuple[float, float, float]],
    radius: float,
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    """Extracts a subgraph from a graph based on a centre point and radius.

    :param g: The graph to extract the subgraph from.
    :type g: nx.Graph
    :param centre_point: The centre point of the subgraph.
    :type centre_point: Tuple[float, float, float]
    :param radius: The radius of the subgraph.
    :type radius: float
    :param filter_dataframe: Whether to filter the pdb_df dataframe of the graph. Defaults to True.
    :type filter_dataframe: bool
    :param update_coords: Whether to update the coordinates of the graph. Defaults to True.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the graph. Defaults to False.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to False.
    :type inverse: bool
    :return: The subgraph or node list if return_node_list is True.
    :rtype: Union[nx.Graph, List[str]]
    """
    node_list: List = []

    for n, d in g.nodes(data=True):
        coords = d["coords"]
        dist = np.linalg.norm(coords - centre_point)
        if dist < radius:
            node_list.append(n)

    node_list = list(set(node_list))
    log.debug(
        f"Found {len(node_list)} nodes in the spatial point-radius subgraph."
    )

    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        recompute_distmat=recompute_distmat,
        update_coords=update_coords,
    )


def extract_subgraph_from_atom_types(
    g: nx.Graph,
    atom_types: List[str],
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    """Extracts a subgraph from a graph based on a list of atom types.

    :param g: The graph to extract the subgraph from.
    :type g: nx.Graph
    :param atom_types: The list of atom types to extract.
    :type atom_types: List[str]
    :param filter_dataframe: Whether to filter the pdb_df dataframe of the graph. Defaults to True.
    :type filter_dataframe: bool
    :param update_coords: Whether to update the coordinates of the graph. Defaults to True.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the graph. Defaults to False.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to False.
    :type inverse: bool
    :return: The subgraph or node list if return_node_list is True.
    :rtype: Union[nx.Graph, List[str]]
    """
    node_list: List = [
        n for n, d in g.nodes(data=True) if d["atom_type"] in atom_types
    ]

    node_list = list(set(node_list))
    log.debug(f"Found {len(node_list)} nodes in the atom type subgraph.")

    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        update_coords=update_coords,
        recompute_distmat=recompute_distmat,
    )


def extract_subgraph_from_residue_types(
    g: nx.Graph,
    residue_types: List[str],
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    """Extracts a subgraph from a graph based on a list of allowable residue types.

    :param g: The graph to extract the subgraph from.
    :type g: nx.Graph
    :param residue_types: List of allowable residue types (3 letter residue names).
    :type residue_types: List[str]
    :param filter_dataframe: Whether to filer the pdb_df of the graph, defaults to True
    :type filter_dataframe: bool, optional
    :param update_coords: Whether to update the coordinates of the graph. Defaults to True.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the graph. Defaults to False.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to False.
    :type inverse: bool
    :return: The subgraph or node list if return_node_list is True.
    :rtype: Union[nx.Graph, List[str]]
    """
    node_list: List = [
        n for n, d in g.nodes(data=True) if d["residue_name"] in residue_types
    ]

    node_list = list(set(node_list))
    log.debug(f"Found {len(node_list)} nodes in the residue type subgraph.")

    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        update_coords=update_coords,
        recompute_distmat=recompute_distmat,
    )


def extract_subgraph_from_chains(
    g: nx.Graph,
    chains: List[str],
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    """Extracts a subgraph from a graph based on a chain.

    :param g: The graph to extract the subgraph from.
    :type g: nx.Graph
    :param chain: The chain(s) to extract.
    :type chain: List[str]
    :param filter_dataframe: Whether to filter the pdb_df dataframe of the graph. Defaults to True.
    :type filter_dataframe: bool
    :param update_coords: Whether to update the coordinates of the graph. Defaults to True.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the graph. Defaults to False.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to False.
    :type inverse: bool
    :return: The subgraph or node list if return_node_list is True.
    :rtype: Union[nx.Graph, List[str]]
    """
    node_list: List = [
        n for n, d in g.nodes(data=True) if d["chain_id"] in chains
    ]

    node_list = list(set(node_list))
    log.debug(f"Found {len(node_list)} nodes in the chain subgraph.")
    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        update_coords=update_coords,
        recompute_distmat=recompute_distmat,
        inverse=inverse,
        return_node_list=return_node_list,
    )


def extract_subgraph_by_sequence_position(
    g: nx.Graph,
    sequence_positions: List[int],
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    """Extracts a subgraph from a graph based on a chain.

    :param g: The graph to extract the subgraph from.
    :type g: nx.Graph
    :param chain: The sequence positions to extract.
    :type chain: List[int]
    :param filter_dataframe: Whether to filter the pdb_df dataframe of the graph. Defaults to True.
    :type filter_dataframe: bool
    :param update_coords: Whether to update the coordinates of the graph. Defaults to True.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the graph. Defaults to False.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to False.
    :type inverse: bool
    :return: The subgraph or node list if return_node_list is True.
    :rtype: Union[nx.Graph, List[str]]
    """
    node_list: List = [
        n
        for n, d in g.nodes(data=True)
        if d["residue_number"] in sequence_positions
    ]

    node_list = list(set(node_list))
    log.debug(
        f"Found {len(node_list)} nodes in the sequence position subgraph."
    )
    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        recompute_distmat=recompute_distmat,
        update_coords=update_coords,
    )


def extract_subgraph_by_bond_type(
    g: nx.Graph,
    bond_types: List[str],
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    """Extracts a subgraph from a graph based on a list of allowable bond types.

    :param g: The graph to extract the subgraph from.
    :type g: nx.Graph
    :param bond_types: List of allowable bond types.
    :type bond_types: List[str]
    :param filter_dataframe: Whether to filter the pdb_df of the graph, defaults to True
    :type filter_dataframe: bool, optional
    :param update_coords: Whether to update the coordinates of the graph. Defaults to True.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the graph. Defaults to False.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection, defaults to False
    :type inverse: bool, optional
    :param return_node_list: Whether to return the node list, defaults to False
    :type return_node_list: bool, optional
    :return: The subgraph or node list if return_node_list is True.
    :rtype: Union[nx.Graph, List[str]]
    """

    node_list: List = []

    for u, v, d in g.edges(data=True):
        for bond_type in list(d["kind"]):
            if bond_type in bond_types:
                node_list.append(u)
                node_list.append(v)
    node_list = list(set(node_list))
    log.debug(f"Found {len(node_list)} nodes in the bond type subgraph.")

    # Remove bond annotations
    for u, v, d in g.edges(data=True):
        for bond in list(d["kind"]):
            if not inverse:
                if bond not in bond_types:
                    d["kind"].discard(bond)
            elif inverse:
                if bond in bond_types:
                    d["kind"].discard(bond)

    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        recompute_distmat=recompute_distmat,
        update_coords=update_coords,
    )


def extract_k_hop_subgraph(
    g: nx.Graph,
    central_node: str,
    k: int,
    k_only: bool = False,
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    """Extracts a k-hop subgraph.

    :param g: The graph to extract the subgraph from.
    :type g: nx.Graph
    :param central_node: The central node to extract the subgraph from.
    :type central_node: str
    :param k: The number of hops to extract.
    :type k: int
    :param k_only: Whether to only extract the exact k-hop subgraph (e.g. include 2-hop neighbours in 5-hop graph). Defaults to False.
    :type k_only: bool
    :param filter_dataframe: Whether to filter the pdb_df of the graph, defaults to True
    :type filter_dataframe: bool, optional
    :param update_coords: Whether to update the coordinates of the graph. Defaults to True.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the graph. Defaults to False.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection, defaults to False
    :type inverse: bool, optional
    :param return_node_list: Whether to return the node list. Defaults to False.
    :type return_node_list: bool
    :return: The subgraph or node list if return_node_list is True.
    :rtype: Union[nx.Graph, List[str]]
    """
    neighbours: Dict[int, Union[List[str], set]] = {0: [central_node]}

    for i in range(1, k + 1):
        neighbours[i] = set()
        for node in neighbours[i - 1]:
            neighbours[i].update(g.neighbors(node))
        neighbours[i] = list(set(neighbours[i]))

    if k_only:
        node_list = neighbours[k]
    else:
        node_list = list(
            set([value for values in neighbours.values() for value in values])
        )

    log.debug(f"Found {len(node_list)} nodes in the k-hop subgraph.")

    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        recompute_distmat=recompute_distmat,
        update_coords=update_coords,
    )


def extract_subgraph(
    g: nx.Graph,
    node_list: Optional[List[str]] = None,
    sequence_positions: Optional[List[str]] = None,
    chains: Optional[List[str]] = None,
    residue_types: Optional[List[str]] = None,
    atom_types: Optional[List[str]] = None,
    bond_types: Optional[List[str]] = None,
    centre_point: Optional[
        Union[np.ndarray, Tuple[float, float, float]]
    ] = None,
    radius: Optional[float] = None,
    k_hop_central_node: Optional[str] = None,
    k_hops: Optional[int] = None,
    k_only: Optional[bool] = None,
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    """Extracts a subgraph from a graph based on a list of nodes, sequence positions, chains, residue types, atom types, centre point and radius.

    :param g: The graph to extract the subgraph from.
    :type g: nx.Graph
    :param node_list: List of nodes to extract specified by their node_id. Defaults to None.
    :type node_list: List[str], optional
    :param sequence_positions: The sequence positions to extract. Defaults to None.
    :type sequence_positions: List[str], optional
    :param chains: The chain(s) to extract. Defaults to None.
    :type chains: List[str], optional
    :param residue_types: List of allowable residue types (3 letter residue names). Defaults to None.
    :type residue_types: List[str], optional
    :param atom_types: List of allowable atom types. Defaults to None.
    :type atom_types: List[str], optional
    :param centre_point: The centre point to extract the subgraph from. Defaults to None.
    :type centre_point: Union[np.ndarray, Tuple[float, float, float]], optional
    :param radius: The radius to extract the subgraph from. Defaults to None.
    :type radius: float, optional
    :param filter_dataframe: Whether to filter the pdb_df dataframe of the graph. Defaults to True. Defaults to None.
    :type filter_dataframe: bool, optional
    :param update_coords: Whether to update the coordinates of the graph. Defaults to True.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the graph. Defaults to False.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to False.
    :type inverse: bool, optional
    :return: The subgraph or node list if return_node_list is True.
    :rtype: Union[nx.Graph, List[str]]
    """
    if node_list is None:
        node_list = []

    if sequence_positions is not None:
        node_list += extract_subgraph_by_sequence_position(
            g, sequence_positions, return_node_list=True
        )

    if chains is not None:
        node_list += extract_subgraph_from_chains(
            g, chains, return_node_list=True
        )

    if residue_types is not None:
        node_list += extract_subgraph_from_residue_types(
            g, residue_types, return_node_list=True
        )

    if atom_types is not None:
        node_list += extract_subgraph_from_atom_types(
            g, atom_types, return_node_list=True
        )

    if bond_types is not None:
        node_list += extract_subgraph_by_bond_type(
            g, bond_types, return_node_list=True
        )

    if centre_point is not None and radius is not None:
        node_list += extract_subgraph_from_point(
            g, centre_point, radius, return_node_list=True
        )

    if k_hop_central_node is not None and k_hops and k_only is not None:
        node_list += extract_k_hop_subgraph(
            g, k_hop_central_node, k_hops, k_only, return_node_list=True
        )

    node_list = list(set(node_list))

    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        recompute_distmat=recompute_distmat,
        update_coords=update_coords,
    )
