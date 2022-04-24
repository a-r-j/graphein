"""Functions for working with Small Molecule Graphs."""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma, Charlie Harris
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from typing import Callable, List, Optional

import networkx as nx
import numpy as np

from graphein.molecule.edges.atomic import add_atom_bonds
from graphein.utils.utils import (
    annotate_edge_metadata,
    annotate_graph_metadata,
    annotate_node_metadata,
    compute_edges,
    import_message,
)
from graphein.utils.junction_tree.jt_utils import (
    get_mol,
    get_smiles,
    tree_decomp, 
    get_clique_mol,
)

from .config import MoleculeGraphConfig

try:
    import rdkit
    from rdkit import Chem
except ImportError:
    import_message("graphein.molecule.graphs", "rdkit", "rdkit", True)


def initialise_graph_with_metadata(
    name: str,
    rdmol: rdkit.Mol,
    coords: np.ndarray,
) -> nx.Graph:
    """
    Initializes the nx Graph object with initial metadata.

    :param name: Name of the molecule. Either the smiles or filename depending on how the graph was created.
    :type name: str
    :param rdmol: Processed Dataframe of molecule structure.
    :type rdmol: rdkit.Mol
    :return: Returns initial molecule structure graph with metadata.
    :rtype: nx.Graph
    """
    return nx.Graph(
        name=name, rdmol=rdmol, coords=coords, smiles=Chem.MolToSmiles(rdmol)
    )


def add_nodes_to_graph(
    G: nx.Graph,
    verbose: bool = False,
) -> nx.Graph:
    """Add nodes into molecule graph.

    :param G: ``nx.Graph`` with metadata to populate with nodes.
    :type G: nx.Graph
    :param verbose: Controls verbosity of this step.
    :type verbose: bool
    :returns: nx.Graph with nodes added.
    :rtype: nx.Graph
    """
    for i, atom in enumerate(G.graph["rdmol"].GetAtoms()):
        coords = (
            G.graph["coords"][i] if G.graph["coords"] is not None else None
        )
        G.add_node(
            f"{atom.GetSymbol()}:{str(atom.GetIdx())}",
            atomic_num=atom.GetAtomicNum(),
            element=atom.GetSymbol(),
            rdmol_atom=atom,
            coords=coords,
        )

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
    Constructs molecule structure graph from a ``sdf_path``, ``mol2_path``  or ``smiles``.

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
        name = smiles
        rdmol = Chem.MolFromSmiles(smiles)

    if sdf_path is not None:
        name = sdf_path.split("/")[-1].split(".")[0]
        rdmol = Chem.SDMolSupplier(sdf_path)[0]
        coords = [
            list(rdmol.GetConformer(0).GetAtomPosition(idx))
            for idx in range(rdmol.GetNumAtoms())
        ]

    if mol2_path is not None:
        name = mol2_path.split("/")[-1].split(".")[0]
        rdmol = Chem.MolFromMol2File(mol2_path)
        coords = [
            list(rdmol.GetConformer(0).GetAtomPosition(idx))
            for idx in range(rdmol.GetNumAtoms())
        ]

    if pdb_path is not None:
        name = pdb_path.split("/")[-1].split(".")[0]
        rdmol = Chem.MolFromPDBFile(pdb_path)
        coords = [
            list(rdmol.GetConformer(0).GetAtomPosition(idx))
            for idx in range(rdmol.GetNumAtoms())
        ]

    if config.add_hs:
        rdmol = Chem.AddHs(rdmol)

    if coords is None:
        # If no coords are provided, add edges by bonds
        config.edge_construction_functions = [add_atom_bonds]
        g = initialise_graph_with_metadata(
            name=name,
            rdmol=rdmol,
            coords=None,
        )
    else:
        # If config params are provided, overwrite them
        config.edge_construction_functions = (
            edge_construction_funcs
            if config.edge_construction_functions is None
            else config.edge_construction_functions
        )

        g = initialise_graph_with_metadata(
            name=name,
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

def construct_junction_tree(
    smiles: Optional[str] = None,
) -> nx.Graph:
    """
    Constructs molecule structure junction tree graph from a ``smiles``.

    :param smiles: smiles string to build graph from. Default is ``None``.
    :type smiles: str, optional
    :return: Molecule Structure Junction Tree Graph
    :type: nx.Graph
    """

    mol = get_mol(smiles)

    g = nx.Graph(
        name=smiles, smiles=smiles
    )

    cliques, edges = tree_decomp(mol)
        
    for i, c in enumerate(cliques):
        cmol = get_clique_mol(mol, c)
        g.add_node(
            f"{get_smiles(cmol)}:{str(i)}",
        )

    for n1, n2 in edges:
        if g.has_edge(n1, n2):
            g.edges[n1, n2]["kind"].add("junction_tree")
        else:
            g.add_edge(n1, n2, kind={"junction_tree"})

    return g