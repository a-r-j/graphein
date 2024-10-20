"""Functions for working with Small Molecule Graphs."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma, Charlie Harris
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import traceback
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import networkx as nx
import numpy as np
from loguru import logger as log
from tqdm.contrib.concurrent import process_map, thread_map

from graphein.utils.dependencies import import_message, requires_python_libs
from graphein.utils.utils import (
    annotate_edge_metadata,
    annotate_graph_metadata,
    annotate_node_metadata,
    compute_edges,
)

from .chembl import get_smiles_from_chembl
from .config import MoleculeGraphConfig
from .edges.atomic import add_atom_bonds
from .utils import compute_fragments
from .zinc import get_smiles_from_zinc

try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    msg = import_message("graphein.molecule.graphs", "rdkit", "rdkit", True)
    log.warning(msg)


@requires_python_libs("rdkit")
def initialise_graph_with_metadata(
    name: str,
    rdmol: rdkit.Mol,
    coords: np.ndarray,
) -> nx.Graph:
    """
    Initializes the nx Graph object with initial metadata.

    :param name: Name of the molecule. Either the smiles or filename depending
        on how the graph was created.
    :type name: str
    :param rdmol: Processed DataFrame of molecule structure.
    :type rdmol: rdkit.Mol
    :return: Returns initial molecule structure graph with metadata.
    :rtype: nx.Graph
    """
    return nx.Graph(
        name=name, rdmol=rdmol, coords=coords, smiles=Chem.MolToSmiles(rdmol)
    )


@requires_python_libs("rdkit")
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


@requires_python_libs("rdkit")
def generate_3d(
    mol: Union[nx.Graph, Chem.Mol], recompute_graph: bool = False
) -> Union[nx.Graph, rdkit.Chem.rdchem.Mol]:
    """
    Generate a 3D structure for a RDKit molecule.

    Steps:
    1. Adds Hydrogens
    2. Embeds molecule with ``AllChem.ETKDGv3``
    (using ``useSmallRingTorsions=True``)
    3. Optimizes molecule with ``AllChem.MMFFOptimizeMolecule``
    4. Removes Hydrogens
    5. Returns molecule OR (optionally) recomputes the molecular graph
    (if ``recompute_graph=True``) using the new coordinates.

    :param mol: input molecule
    :type mol: Union[nx.Graph, Chem.Mol]
    :param recompute_graph: whether to recompute the graph based on the
        generated conformer.
    :type recompute_graph: bool
    :return: molecule with 3D coordinates or recomputed molecular graph.
    :rtype: Union[nx.Graph, Chem.Mol]
    """
    rdmol = mol.graph["rdmol"] if isinstance(mol, nx.Graph) else mol
    rdmol = Chem.AddHs(rdmol)
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True
    Chem.AddHs(rdmol)
    AllChem.EmbedMolecule(rdmol, params=params)
    AllChem.MMFFOptimizeMolecule(rdmol)
    rdmol = Chem.RemoveHs(rdmol)

    if recompute_graph:
        return construct_graph(config=mol.graph["config"], mol=rdmol)

    return rdmol


@requires_python_libs("rdkit")
def construct_graph(
    config: Optional[MoleculeGraphConfig] = None,
    mol: Optional[rdkit.Mol] = None,
    path: Optional[str] = None,
    smiles: Optional[str] = None,
    zinc_id: Optional[str] = None,
    chembl_id: Optional[str] = None,
    generate_conformer: Optional[bool] = False,
    edge_construction_funcs: Optional[str] = None,
    edge_annotation_funcs: Optional[List[Callable]] = None,
    node_annotation_funcs: Optional[List[Callable]] = None,
    graph_annotation_funcs: Optional[List[Callable]] = None,
) -> nx.Graph:
    """
    Constructs molecular structure graph from a ``sdf_path``, ``mol2_path``,
    ``smiles`` or RDKit Mol.

    Users can provide a :class:`~graphein.molecule.config.MoleculeGraphConfig`
    object to specify construction parameters.

    However, config parameters can be overridden by passing arguments directly
    to the function.

    :param config: :class:`~graphein.molecule.config.MoleculeGraphConfig`
        object. If None, defaults to config in ``graphein.molecule.config``.
    :type config: graphein.molecule.config.MoleculeGraphConfig, optional
    :param mol: rdkit.Mol object to build graph from. Defaults to ``None``.
    :type mol: rdkit.Mol, optional
    :param path: Path to either a ``.sdf``, ``.mol2``, ``.smi`` or ``pdb`` file.
        Defaults to ``None``.
    :type path: str
    :param smiles: smiles string to build graph from. Default is ``None``.
    :type smiles: str, optional
    :param zinc_id: Zinc ID to build graph from. Default is ``None``.
    :type zinc_id: str, optional
    :param chembl_id: ChEMBL ID to build graph from. Default is ``None``.
    :type chembl_id: str, optional
    :param generate_conformer: Whether to generate a conformer for the molecule.
        Defaults to ``False``.
    :type generate_conformer: bool, optional
    :param edge_construction_funcs: List of edge construction functions.
        Default is ``None``.
    :type edge_construction_funcs: List[Callable], optional
    :param edge_annotation_funcs: List of edge annotation functions.
        Default is ``None``.
    :type edge_annotation_funcs: List[Callable], optional
    :param node_annotation_funcs: List of node annotation functions.
        Default is ``None``.
    :type node_annotation_funcs: List[Callable], optional
    :param graph_annotation_funcs: List of graph annotation function.
        Default is ``None``.
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
        if edge_construction_funcs is not None
        else config.edge_construction_functions
    )

    if zinc_id is not None:
        smiles = get_smiles_from_zinc(zinc_id)

    if chembl_id is not None:
        smiles = get_smiles_from_chembl(chembl_id)

    coords = None
    if smiles is not None:
        name = smiles
        rdmol = Chem.MolFromSmiles(smiles)
        if config.generate_conformer or generate_conformer:
            rdmol = generate_3d(mol=rdmol, recompute_graph=False)
            coords = [
                list(rdmol.GetConformer(0).GetAtomPosition(idx))
                for idx in range(rdmol.GetNumAtoms())
            ]

    if path is not None:
        name = path.split("/")[-1].split(".")[0]
        if path.lower().endswith(".sdf"):
            rdmol = Chem.SDMolSupplier(path)[0]
            coords = [
                list(rdmol.GetConformer(0).GetAtomPosition(idx))
                for idx in range(rdmol.GetNumAtoms())
            ]
        elif path.lower().endswith(".mol2"):
            rdmol = Chem.MolFromMol2File(path)
            coords = [
                list(rdmol.GetConformer(0).GetAtomPosition(idx))
                for idx in range(rdmol.GetNumAtoms())
            ]
        elif path.lower().endswith(".pdb"):
            name = path.split("/")[-1].split(".")[0]
            rdmol = Chem.MolFromPDBFile(path)
            coords = [
                list(rdmol.GetConformer(0).GetAtomPosition(idx))
                for idx in range(rdmol.GetNumAtoms())
            ]
        elif path.lower().endswith(".smi"):
            with open(path) as f:
                smiles = f.readlines()[0]
            name = smiles
            rdmol = Chem.MolFromSmiles(smiles)
            if config.generate_conformer or generate_conformer:
                rdmol = generate_3d(mol=rdmol, recompute_graph=False)
                coords = [
                    list(rdmol.GetConformer(0).GetAtomPosition(idx))
                    for idx in range(rdmol.GetNumAtoms())
                ]

    elif mol is not None:
        name = Chem.MolToSmiles(mol)
        rdmol = mol
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


def compute_fragment_graphs(
    g: Union[nx.Graph, Chem.Mol], config: Optional[MoleculeGraphConfig] = None
) -> List[nx.Graph]:
    """Computes graphs for each fragment in a molecule.

    :param g: Input graph or molecule to fragment.
    :type g: Union[nx.Graph, Chem.Mol]
    :param config: Molecular graph construction config.
        See: :class:`graphein.molecule.config`. Defaults to ``None``.
    :type config: Optional[MoleculeGraphConfig], optional
    :return: List of fragment graphs.
    :rtype: List[nx.Graph]
    """
    if config is None:
        try:
            config = g.graph["config"]
        except KeyError:
            config = MoleculeGraphConfig()
    fragments = compute_fragments(g)
    return [construct_graph(mol=m, config=config) for m in fragments]


def _mp_graph_constructor(
    input: Union[str, Chem.Mol],
    config: MoleculeGraphConfig,
    use_mol: bool = False,
    use_file: bool = False,
    use_smiles: bool = False,
) -> nx.Graph:
    """
    Molecule graph constructor for use in multiprocessing several molecular
    graphs.

    :param args: Tuple of pdb code/path and the chain selection for that PDB
    :type args: Tuple[str, str]
    :param use_mol: Whether or not we are using RDKit Mols
    :type use_mol: bool
    :param use_file: Whether or not we are using file paths
    :type use_file: bool
    :param use_smiles: Whether or not we are using SMILES strings
    :type use_smiles: bool
    :param config: Molecule structure graph construction config
    :type config: MoleculeGraphConfig
    :return: Molecule structure graph
    :rtype: nx.Graph
    """
    func = partial(construct_graph, config=config)
    try:
        if use_mol:
            return func(mol=input)
        if use_file:
            return func(path=input)
        if use_smiles:
            return func(smiles=input)
    except Exception as e:
        log.info(
            f"Graph construction error for ({input})! {traceback.format_exc()}"
        )
        log.info(e)
        return None


def construct_graphs_mp(
    path_it: Optional[List[str]] = None,
    smiles_it: Optional[List[str]] = None,
    mol_it: Optional[List[Chem.Mol]] = None,
    config: MoleculeGraphConfig = MoleculeGraphConfig(),
    num_cores: int = 16,
    return_dict: bool = True,
) -> Union[List[nx.Graph], Dict[str, nx.Graph]]:
    """
    Constructs molecular graphs for a list of smiles or paths using
    multiprocessing.

    :param path_it: List of paths to use for molecule graph construction.
    :type path_it: Optional[List[str]], defaults to ``None``.
    :param smiles_it: List of smiles to use for molecule graph construction.
    :type smiles_it: Optional[List[str]], defaults to ``None``.
    :param mol_it: List of rdkit Mols to use for molecule graph construction.
    :type mol_it: Optional[List[Chem.Mol]], defaults to ``None``.
    :param config: MoleculeGraphConfig to use.
    :type config: graphein.molecule.config.MoleculeGraphConfig,
        defaults to default config params
    :param num_cores: Number of cores to use for multiprocessing.
        The more the merrier
    :type num_cores: int, defaults to 16
    :param return_dict: Whether or not to return a dictionary
        (indexed by smiles/paths) or a list of graphs.
    :type return_dict: bool, default to True
    :return: Iterable of molecule graphs. None values indicate there was a
        problem in constructing the graph for this particular molecule.
    :rtype: Union[List[nx.Graph], Dict[str, nx.Graph]]
    """
    assert (
        path_it is not None or smiles_it is not None or mol_it is not None
    ), "Iterable of paths, smiles strings OR RDKit mols is required."

    use_file, use_smiles, use_mol = False, False, False

    if path_it is not None:
        inputs = path_it
        use_file = True

    if smiles_it is not None:
        inputs = smiles_it
        use_smiles = True

    if mol_it is not None:
        inputs = mol_it
        use_mol = True

    constructor = partial(
        _mp_graph_constructor,
        use_file=use_file,
        use_mol=use_mol,
        use_smiles=use_smiles,
        config=config,
    )

    graphs = list(thread_map(constructor, list(inputs), max_workers=num_cores))

    if return_dict:
        graphs = {molecule: graphs[i] for i, molecule in enumerate(inputs)}

    return graphs
