"""Provides functions for extracting subgraphs from RNA graphs."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from loguru import logger as log

import graphein.protein.subgraphs as protein
from graphein.rna.constants import (
    PHOSPHORIC_ACID_ATOMS,
    RIBOSE_ATOMS,
    RNA_BACKBONE_ATOMS,
)


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
    :type node_list: List[str], optional
    :param filter_dataframe: Whether to filter the ``pdb_df`` DataFrame of the
        graph. Defaults to ``True``.
    :type filter_dataframe: bool
    :param update_coords: Whether to update the coordinates of the graph.
        Defaults to ``True``.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the
        graph. Defaults to ``False``.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to ``False``.
    :type inverse: bool
    :return: The subgraph or node list if ``return_node_list=True``.
    :rtype: Union[nx.Graph, List[str]]
    """
    return protein.extract_subgraph_from_node_list(
        g,
        node_list=node_list,
        filter_dataframe=filter_dataframe,
        update_coords=update_coords,
        recompute_distmat=recompute_distmat,
        inverse=inverse,
        return_node_list=return_node_list,
    )


def extract_subgraph_from_backbone(
    g: nx.Graph,
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = True,
    inverse: bool = False,
    return_node_list: bool = False,
):
    """Extracts a subgraph from an RNA structure retaining only backbone atoms.

    Backbone atoms are defined in
    :ref:`graphein.rna.constants.RNA_BACKBONE_ATOMS`.

    :param g: RNA Structure Graph to extract subgraph from.
    :type g: nx.Graph
    :param filter_dataframe: Whether to filter the ``pdb_df`` dataframe of the
        graph. Defaults to ``True``.
    :type filter_dataframe: bool
    :param update_coords: Whether to update the coordinates of the graph.
        Defaults to ``True``.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the
        graph. Defaults to ``False``.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to ``False``.
    :type inverse: bool
    :return: The subgraph or node list if ``return_node_list=True``.
    :rtype: Union[nx.Graph, List[str]]
    """
    return protein.extract_subgraph_from_atom_types(
        g,
        atom_types=RNA_BACKBONE_ATOMS,
        filter_dataframe=filter_dataframe,
        update_coords=update_coords,
        recompute_distmat=recompute_distmat,
        inverse=inverse,
        return_node_list=return_node_list,
    )


def extract_subgraph_from_bases(
    g: nx.Graph,
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = True,
    inverse: bool = False,
    return_node_list: bool = False,
):
    """Extracts a subgraph from an RNA structure retaining only base atoms.

    Backbone atoms are defined in
    :ref:`graphein.rna.constancts.RNA_BACKBONE_ATOMS`. We exclude these to
    perform the selection.

    :param g: RNA Structure Graph to extract subgraph from.
    :type g: nx.Graph
    :param filter_dataframe: Whether to filter the ``pdb_df`` dataframe of the
        graph. Defaults to ``True``.
    :type filter_dataframe: bool
    :param update_coords: Whether to update the coordinates of the graph.
        Defaults to ``True``.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the
        graph. Defaults to ``False``.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to ``False``.
    :type inverse: bool
    :return: The subgraph or node list if ``return_node_list=True``.
    :rtype: Union[nx.Graph, List[str]]
    """
    inverse = not inverse
    return protein.extract_subgraph_from_atom_types(
        g,
        atom_types=RNA_BACKBONE_ATOMS,
        filter_dataframe=filter_dataframe,
        update_coords=update_coords,
        recompute_distmat=recompute_distmat,
        inverse=inverse,
        return_node_list=return_node_list,
    )


def extract_subgraph_from_phosphoric_acid(
    g: nx.Graph,
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = True,
    inverse: bool = False,
    return_node_list: bool = False,
):
    """Extracts a subgraph from an RNA structure retaining only base atoms.

    Phosphoric acid atoms are defined in
    :ref:`graphein.rna.constancts.PHOSPHORIC_ACID_ATOMS`.

    :param g: RNA Structure Graph to extract subgraph from.
    :type g: nx.Graph
    :param filter_dataframe: Whether to filter the ``pdb_df`` dataframe of the
        graph. Defaults to ``True``.
    :type filter_dataframe: bool
    :param update_coords: Whether to update the coordinates of the graph.
        Defaults to ``True``.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the
        graph. Defaults to ``False``.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to ``False``.
    :type inverse: bool
    :return: The subgraph or node list if ``return_node_list=True``.
    :rtype: Union[nx.Graph, List[str]]
    """
    return protein.extract_subgraph_from_atom_types(
        g,
        atom_types=PHOSPHORIC_ACID_ATOMS,
        filter_dataframe=filter_dataframe,
        update_coords=update_coords,
        recompute_distmat=recompute_distmat,
        inverse=inverse,
        return_node_list=return_node_list,
    )


def extract_subgraph_from_ribose(
    g: nx.Graph,
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = True,
    inverse: bool = False,
    return_node_list: bool = False,
):
    """Extracts a subgraph from an RNA structure retaining only base atoms.

    Ribose atoms are defined in :ref:`graphein.rna.constants.RIBOSE_ATOMS`.

    :param g: RNA Structure Graph to extract subgraph from.
    :type g: nx.Graph
    :param filter_dataframe: Whether to filter the ``pdb_df`` dataframe of the
        graph. Defaults to ``True``.
    :type filter_dataframe: bool
    :param update_coords: Whether to update the coordinates of the graph.
        Defaults to ``True``.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the
        graph. Defaults to ``False``.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to ``False``.
    :type inverse: bool
    :return: The subgraph or node list if ``return_node_list=True``.
    :rtype: Union[nx.Graph, List[str]]
    """
    return protein.extract_subgraph_from_atom_types(
        g,
        atom_types=RIBOSE_ATOMS,
        filter_dataframe=filter_dataframe,
        update_coords=update_coords,
        recompute_distmat=recompute_distmat,
        inverse=inverse,
        return_node_list=return_node_list,
    )


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
    :param filter_dataframe: Whether to filter the ``pdb_df`` dataframe of the
        graph. Defaults to ``True``.
    :type filter_dataframe: bool
    :param update_coords: Whether to update the coordinates of the graph.
        Defaults to ``True``.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the
    graph. Defaults to ``False``.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to ``False``.
    :type inverse: bool
    :return: The subgraph or node list if ``return_node_list=True``.
    :rtype: Union[nx.Graph, List[str]]
    """
    return protein.extract_subgraph_from_point(
        g=g,
        centre_point=centre_point,
        radius=radius,
        filter_dataframe=filter_dataframe,
        update_coords=update_coords,
        recompute_distmat=recompute_distmat,
        inverse=inverse,
        return_node_list=return_node_list,
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
    """Extracts a subgraph from an RNA graph based on a list of atom types.

    :param g: The graph to extract the subgraph from.
    :type g: nx.Graph
    :param atom_types: The list of atom types to extract.
    :type atom_types: List[str]
    :param filter_dataframe: Whether to filter the ``pdb_df`` dataframe of the
        graph. Defaults to ``True``.
    :type filter_dataframe: bool
    :param update_coords: Whether to update the coordinates of the graph.
        Defaults to ``True``.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the
        graph. Defaults to ``False``.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to ``False``.
    :type inverse: bool
    :return: The subgraph or node list if ``return_node_list=True``.
    :rtype: Union[nx.Graph, List[str]]
    """
    return protein.extract_subgraph_from_atom_types(
        g,
        atom_types=atom_types,
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
    """Extracts a subgraph from a graph based on a list of allowable
    (nucleotide) residue types.

    :param g: The graph to extract the subgraph from.
    :type g: nx.Graph
    :param residue_types: List of allowable residue types (1-letter residue
        names). E.g. ``["A", "G"]``.
    :type residue_types: List[str]
    :param filter_dataframe: Whether to filer the ``pdb_df`` of the graph,
        defaults to ``True``.
    :type filter_dataframe: bool, optional
    :param update_coords: Whether to update the coordinates of the graph.
        Defaults to ``True``.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the
        graph. Defaults to ``False``.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to ``False``.
    :type inverse: bool
    :return: The subgraph or node list if ``return_node_list=True``.
    :rtype: Union[nx.Graph, List[str]]
    """
    return protein.extract_subgraph_from_residue_types(
        g,
        residue_types,
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
    :param chains: The chain(s) to extract. E.g. ``["A", "C"]``.
    :type chains: List[str]
    :param filter_dataframe: Whether to filter the ``pdb_df`` dataframe of the
        graph. Defaults to ``True``.
    :type filter_dataframe: bool
    :param update_coords: Whether to update the coordinates of the graph.
        Defaults to ``True``.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the
        graph. Defaults to ``False``.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to ``False``.
    :type inverse: bool
    :return: The subgraph or node list if ``return_node_list=True``.
    :rtype: Union[nx.Graph, List[str]]
    """
    return protein.extract_subgraph_from_chains(
        g,
        chains,
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
    :param chain: The sequence positions to extract. E.g. ``[1, 3, 5, 7]``.
    :type chain: List[int]
    :param filter_dataframe: Whether to filter the ``pdb_df`` dataframe of the
        graph. Defaults to ``True``.
    :type filter_dataframe: bool
    :param update_coords: Whether to update the coordinates of the graph.
        Defaults to ``True``.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the
        graph. Defaults to ``False``.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection. Defaults to ``False``.
    :type inverse: bool
    :return: The subgraph or node list if ``return_node_list=True``.
    :rtype: Union[nx.Graph, List[str]]
    """
    return protein.extract_subgraph_by_sequence_position(
        g,
        sequence_positions=sequence_positions,
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
    :param filter_dataframe: Whether to filter the ``pdb_df`` of the graph,
        defaults to ``True``.
    :type filter_dataframe: bool, optional
    :param update_coords: Whether to update the coordinates of the graph.
        Defaults to ``True``.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the
        graph. Defaults to ``False``.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection, defaults to ``False``.
    :type inverse: bool, optional
    :param return_node_list: Whether to return the node list, defaults to
        ``False``.
    :type return_node_list: bool, optional
    :return: The subgraph or node list if ``return_node_list=True``.
    :rtype: Union[nx.Graph, List[str]]
    """
    return protein.extract_subgraph_by_bond_type(
        g,
        bond_types=bond_types,
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
    :param k_only: Whether to only extract the exact k-hop subgraph (e.g.
        include 2-hop neighbours in 5-hop graph). Defaults to ``False``.
    :type k_only: bool
    :param filter_dataframe: Whether to filter the ``pdb_df`` of the graph,
        defaults to ``True``.
    :type filter_dataframe: bool, optional
    :param update_coords: Whether to update the coordinates of the graph.
        Defaults to ``True``.
    :type update_coords: bool
    :param recompute_distmat: Whether to recompute the distance matrix of the
        graph. Defaults to ``False``.
    :type recompute_distmat: bool
    :param inverse: Whether to inverse the selection, defaults to ``False``.
    :type inverse: bool, optional
    :param return_node_list: Whether to return the node list. Defaults to
        ``False``.
    :type return_node_list: bool
    :return: The subgraph or node list if ``return_node_list=True``.
    :rtype: Union[nx.Graph, List[str]]
    """
    return protein.extract_k_hop_subgraph(
        g,
        central_node=central_node,
        k=k,
        k_only=k_only,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        recompute_distmat=recompute_distmat,
        update_coords=update_coords,
    )
