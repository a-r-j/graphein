"""Functions for working with Small Molecule Graphs."""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma, Charlie Harris
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import logging
import multiprocessing
import traceback
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import rdkit
from rdkit import Chem
import networkx as nx
import numpy as np
import pandas as pd

from graphein.molecule.edges.atomic import add_atom_bonds

# from graphein.molecule.config import (
#     MoleculeGraphConfig,
# )

from graphein.utils.utils import (
    annotate_edge_metadata,
    annotate_graph_metadata,
    annotate_node_metadata,
    compute_edges,
)

from .config import (
    MoleculeGraphConfig,
)

def initialise_graph_with_metadata(
    rdmol: rdkit.Mol,
    coords: np.array,
) -> nx.Graph:
    """
    Initializes the nx Graph object with initial metadata.

    :param rdmol: Processed Dataframe of molecule structure.
    :type rdmol: rdkit.Mol
    :return: Returns initial molecule structure graph with metadata.
    :rtype: nx.Graph
    """
    G = nx.Graph(
        rdmol=rdmol,
        coords=coords,
    )

    return G


def add_nodes_to_graph(
    G: nx.Graph,
    verbose: bool = False,
) -> nx.Graph:
    """Add nodes into molecule graph.

    :param G: ``nx.Graph`` with metadata to populate with nodes.
    :type G: nx.Graphptional
    :param verbose: Controls verbosity of this step.
    :type verbose: bool
    :returns: nx.Graph with nodes added.
    :rtype: nx.Graph
    """
    for atom in G.graph["rdmol"].GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum())

    # TODO: include charge, line_idx for traceability?
    if verbose:
        print(nx.info(G))
        print(G.nodes())

    return G

def construct_graph(
    config: Optional[MoleculeGraphConfig] = None,
    sdf_path: Optional[str] = None,
    smiles: Optional[str] = None,
    mol2_path: Optional[str] = None,
    pdb_path: Optional[str] = None,
    edge_construction_funcs: Optional[str] = None,
    edge_annotation_funcs: Optional[List[Callable]] = None,
    node_annotation_funcs: Optional[List[Callable]] = None,
    graph_annotation_funcs: Optional[List[Callable]] = None,
) -> nx.Graph:
    """
    Constructs protein structure graph from a ``sdf_path`` or ``smiles``.

    Users can provide a :class:`~graphein.molecule.config.MoleculeGraphConfig`
    object to specify construction parameters.

    However, config parameters can be overridden by passing arguments directly to the function.

    :param config: :class:`~graphein.molecule.config.MoleculeGraphConfig` object. If None, defaults to config in ``graphein.molecule.config``.
    :type config: graphein.molecule.config.MoleculeGraphConfig, optional
    :param sdf_path: Path to ``sdf_file`` to build graph from. Default is ``None``.
    :type sdf_path: str, optional
    :param smiles: smiles string to build graph from. Default is ``None``.
    :type smiles: str, optional
    :param mol2_path: Path to ``mol2_file`` to build graph from. Default is ``None``.
    :type mol2_path: str, optional
    :param pdb_path: Path to ``pdb_file`` to build graph from. Default is ``None``.
    :type pdb_path: str, optional
    :param edge_construction_funcs: List of edge construction functions. Default is ``None``.
    :type edge_construction_funcs: List[Callable], optional
    :param edge_annotation_funcs: List of edge annotation functions. Default is ``None``.
    :type edge_annotation_funcs: List[Callable], optional
    :param node_annotation_funcs: List of node annotation functions. Default is ``None``.
    :type node_annotation_funcs: List[Callable], optional
    :param graph_annotation_funcs: List of graph annotation function. Default is ``None``.
    :type graph_annotation_funcs: List[Callable]
    :return: Molecule Structure Graph
    :type: nx.Graph
    """

    # If no config is provided, use default
    if config is None:
        config = MoleculeGraphConfig()

    config.node_metadata_functions = (
        node_annotation_funcs
        if config.node_metadata_functions is None
        else config.node_metadata_functions
    )
    config.graph_metadata_functions = (
        graph_annotation_funcs
        if config.graph_metadata_functions is None
        else config.graph_metadata_functions
    )
    config.edge_metadata_functions = (
        edge_annotation_funcs
        if config.edge_metadata_functions is None
        else config.edge_metadata_functions
    )
    config.edge_construction_functions = (
        edge_construction_funcs
        if config.edge_construction_functions is None
        else config.edge_construction_functions
    )

    coords = None
    if smiles is not None:
        rdmol = Chem.MolFromSmiles(smiles)
    
    if sdf_path is not None:
        rdmol = Chem.SDMolSupplier(sdf_path)[0]
        coords = []
        for idx in range(rdmol.GetNumAtoms()):
            coords.append(list(rdmol.GetConformer(0).GetAtomPosition(idx)))

    if mol2_path is not None:
        rdmol = Chem.MolFromMol2File(mol2_path)
        coords = []
        for idx in range(rdmol.GetNumAtoms()):
            coords.append(list(rdmol.GetConformer(0).GetAtomPosition(idx)))

    if pdb_path is not None:
        rdmol = Chem.MolFromPDBFile(pdb_path)
        coords = []
        for idx in range(rdmol.GetNumAtoms()):
            coords.append(list(rdmol.GetConformer(0).GetAtomPosition(idx)))
    
    if coords is None:
        # If no coords are provided, add edges by bonds
        config.edge_construction_functions = [add_atom_bonds]
    else:
        # If config params are provided, overwrite them
        config.edge_construction_functions = (
            edge_construction_funcs
            if config.edge_construction_functions is None
            else config.edge_construction_functions
            )   

    # Initialise graph with metadata
    if coords is None:
        g = initialise_graph_with_metadata(
            rdmol=rdmol,
            coords=None,
        )
    else:
        g = initialise_graph_with_metadata(
            rdmol=rdmol,
            coords=np.asarray(coords),
        )

    # Add nodes to graph
    g = add_nodes_to_graph(g)

    # Add config to graph
    g.graph["config"] = config

    # Annotate additional node metadata
    if config.node_metadata_functions is not None:
        g = annotate_node_metadata(g, config.node_metadata_functions)

    # Compute graph edges
    g = compute_edges(
        g,
        funcs=config.edge_construction_functions,
    )

    # Annotate additional graph metadata
    if config.graph_metadata_functions is not None:
        g = annotate_graph_metadata(g, config.graph_metadata_functions)

    # Annotate additional edge metadata
    if config.edge_metadata_functions is not None:
        g = annotate_edge_metadata(g, config.edge_metadata_functions)

    return g