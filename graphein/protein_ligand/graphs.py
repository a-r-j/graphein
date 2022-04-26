"""Functions for working with Protein-ligand Structure Graphs."""
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
import requests
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from Bio.PDB.Polypeptide import three_to_one
from biopandas.pdb import PandasPdb

from graphein.protein.graphs import (
    remove_insertions,
    process_dataframe,
)

from graphein.protein_ligand.config import (
    ProteinLigandGraphConfig,
)
from graphein.protein_ligand.utils import (
    ProteinLigandGraphConfigurationError,
)
from graphein.protein.utils import (
    compute_rgroup_dataframe,
    filter_dataframe,
    get_protein_name_from_filename,
    three_to_one_with_mods,
)
from graphein.utils.utils import (
    annotate_edge_metadata,
    annotate_graph_metadata,
    annotate_node_metadata,
)

try:
    import rdkit
    import rdkit.Chem as Chem
    import rdkit.Chem.AllChem as AllChem
except ImportError:
    import_message("graphein.protein_ligand.graphs", "rdkit", "rdkit", True)

try:
    from prody import parsePDB, writePDBStream
except ImportError:
    import_message("graphein.protein_ligand.graphs", "prody", "prody", True)


logging.basicConfig(level="DEBUG")
log = logging.getLogger(__name__)


def read_pdb_to_dataframe(
    pdb_path: Optional[str] = None,
    pdb_code: Optional[str] = None,
    verbose: bool = False,
    granularity: str = "CA",
) -> pd.DataFrame:
    """
    Reads PDB file to ``PandasPDB`` object.

    Returns ``atomic_df``, which is a dataframe enumerating all atoms and their cartesian coordinates in 3D space. Also
    contains associated metadata from the PDB file.

    :param pdb_path: path to PDB file. Defaults to None.
    :type pdb_path: str, optional
    :param pdb_code: 4-character PDB accession. Defaults to None.
    :type pdb_code: str, optional
    :param verbose: print dataframe?
    :type verbose: bool
    :param granularity: Specifies granularity of dataframe. See :class:`~graphein.protein.config.ProteinLigandGraphConfig` for further
        details.
    :type granularity: str
    :returns: ``pd.DataFrame`` containing protein structure
    :rtype: pd.DataFrame
    """
    if pdb_code is None and pdb_path is None:
        raise NameError("One of pdb_code or pdb_path must be specified!")

    atomic_df = (
        PandasPdb().read_pdb(pdb_path)
        if pdb_path is not None
        else PandasPdb().fetch_pdb(pdb_code)
    )

    # Assign Node IDs to dataframes
    atomic_df.df["ATOM"]["node_id"] = (
        atomic_df.df["ATOM"]["chain_id"].apply(str)
        + ":"
        + atomic_df.df["ATOM"]["residue_name"]
        + ":"
        + atomic_df.df["ATOM"]["residue_number"].apply(str)
    )
    atomic_df.df["HETATM"]["node_id"] = (
        atomic_df.df["HETATM"]["residue_name"]
        + ":"
        + atomic_df.df["HETATM"]["element_symbol"]
        + ":"
        + atomic_df.df["HETATM"]["residue_number"].apply(str)
        + ":"
        + atomic_df.df["ATOM"]["atom_number"].apply(str)
    )
    if granularity == "atom":
        atomic_df.df["ATOM"]["node_id"] = (
            atomic_df.df["ATOM"]["node_id"]
            + ":"
            + atomic_df.df["ATOM"]["atom_name"]
        )
    if verbose:
        print(atomic_df)
    return atomic_df

def initialise_graph_with_metadata(
    protein_df: pd.DataFrame,
    ligands_df: List[pd.DataFrame],
    raw_pdb_df: Dict[pd.DataFrame],
    pdb_id: str,
    granularity: str,
) -> nx.Graph:
    """
    Initializes the nx Graph object with initial metadata.

    :param protein_df: Processed Dataframe of protein structure.
    :type protein_df: pd.DataFrame
    :param ligands_df: Processed Dataframe of ligands structures.
    :type ligands_df: List[pd.DataFrame]
    :param raw_pdb_df: Unprocessed dataframe of protein structure for comparison and traceability downstream.
    :type raw_pdb_df: pd.DataFrame
    :param pdb_id: PDB Accession code.
    :type pdb_id: str
    :param granularity: Granularity of the graph (eg ``"atom"``, ``"CA"``, ``"CB"`` etc or ``"centroid"``).
        See: :const:`~graphein.protein.config.GRAPH_ATOMS` and :const:`~graphein.protein.config.GRANULARITY_OPTS`.
    :type granularity: str
    :return: Returns initial protein structure graph with metadata.
    :rtype: nx.Graph
    """
    G = nx.Graph(
        name=pdb_id,
        pdb_id=pdb_id,
        chain_ids=list(protein_df["chain_id"].unique()),
        protein_df=protein_df,
        ligands_df=ligands_df,
        raw_pdb_df=raw_pdb_df,
        rgroup_df=compute_rgroup_dataframe(remove_insertions(raw_pdb_df.df["ATOM"])),
        protein_coords=np.asarray(protein_df[["x_coord", "y_coord", "z_coord"]]),
        ligands_coords=np.asarray([ligand_df[["x_coord", "y_coord", "z_coord"]] for ligand_df in ligands_df]),
    )

    # Create graph and assign intrinsic graph-level metadata
    G.graph["node_type"] = granularity

    # Add Sequences to graph metadata
    for c in G.graph["chain_ids"]:
        G.graph[f"sequence_{c}"] = (
            protein_df.loc[protein_df["chain_id"] == c]["residue_name"]
            .apply(three_to_one_with_mods)
            .str.cat()
        )
    return G


def add_nodes_to_graph(
    G: nx.Graph,
    protein_df: Optional[pd.DataFrame] = None,
    ligands_df: List[pd.DataFrame] = None,
    verbose: bool = False,
) -> nx.Graph:
    """Add nodes into protein graph.

    :param G: ``nx.Graph`` with metadata to populate with nodes.
    :type G: nx.Graph
    :protein_df: DataFrame of protein structure containing nodes & initial node metadata to add to the graph.
    :type protein_df: pd.DataFrame, optional
    :ligands_df: DataFrame of ligands structures containing nodes & initial node metadata to add to the graph.
    :type ligands_df: List[pd.DataFrame], optional
    :param verbose: Controls verbosity of this step.
    :type verbose: bool
    :returns: nx.Graph with nodes added.
    :rtype: nx.Graph
    """

    # If no protein dataframe is supplied, use the one stored in the Graph object
    if protein_df is None:
        protein_df = G.graph["protein_df"]
        ligands_df = G.graph["ligands_df"]
    # Assign intrinsic node attributes
    chain_id = protein_df["chain_id"].apply(str)
    residue_name = protein_df["residue_name"]
    residue_number = protein_df["residue_number"]  # .apply(str)
    coords = np.asarray(protein_df[["x_coord", "y_coord", "z_coord"]])
    b_factor = protein_df["b_factor"]
    atom_type = protein_df["atom_name"]
    nodes = protein_df["node_id"]
    element_symbol = protein_df["element_symbol"]
    G.add_nodes_from(nodes)

    # Set intrinsic node attributes
    nx.set_node_attributes(G, dict(zip(nodes, chain_id)), "chain_id")
    nx.set_node_attributes(G, dict(zip(nodes, residue_name)), "residue_name")
    nx.set_node_attributes(
        G, dict(zip(nodes, residue_number)), "residue_number"
    )
    nx.set_node_attributes(G, dict(zip(nodes, atom_type)), "atom_type")
    nx.set_node_attributes(
        G, dict(zip(nodes, element_symbol)), "element_symbol"
    )
    nx.set_node_attributes(G, dict(zip(nodes, coords)), "coords")
    nx.set_node_attributes(G, dict(zip(nodes, b_factor)), "b_factor")
    nx.set_node_attributes(G, dict(zip(nodes, ["protein" for _ in range(len(nodes))])), "source")

    rdmols = []
    for i, ligand in enumerate(ligands_df):
        G.add_nodes_from(ligand["node_id"])
        nx.set_node_attributes(G, dict(zip(ligand["node_id"], ["ligand"])), "source")
        nx.set_node_attributes(G, dict(zip(ligand["node_id"], np.asarray(ligand[["x_coord", "y_coord", "z_coord"]]))), "coords")
        nx.set_node_attributes(G, dict(zip(ligand["node_id"], ligand["element_symbol"])), "element_symbol")
        # Convert mol to rdmol and save it
        ppdb = PandasPdb()
        ppdb.df['HETATM'] = ligands_df[i]
        ppdb.to_pdb(path="./tmp_saved_mol.pdb")
        rdmol = Chem.MolFromPDBFile("./tmp_saved_mol.pdb")
        rdmols.append(rdmol)
    G.graph["ligands_rdmol"] = rdmols

    # TODO: include charge, line_idx for traceability?
    if verbose:
        print(nx.info(G))
        print(G.nodes())

    return G


def compute_edges(
    G: nx.Graph,
    protein_funcs: List[Callable],
    ligand_funcs: List[Callable],
    protein_ligand_funcs: List[Callable],
    get_contacts_config: Optional[GetContactsConfig] = None,
) -> nx.Graph:
    """
    Computes edges for the protein-ligand structure graph. Will compute a pairwise
    distance matrix between nodes which is
    added to the graph metadata to facilitate some edge computations.

    :param G: nx.Graph with nodes to add edges to.
    :type G: nx.Graph
    :param protein_funcs: List of protein edge construction functions.
    :type funcs: List[Callable]
    :param ligand_funcs: List of ligands edge construction functions.
    :type funcs: List[Callable]
    :param protein_ligand_funcs: List of protein-ligand edge construction functions.
    :type funcs: List[Callable]
    :param get_contacts_config: Config object for ``GetContacts`` if
        intramolecular edges are being used.
    :type get_contacts_config: graphein.protein_ligand.config.GetContactsConfig
    :return: Graph with added edges.
    :rtype: nx.Graph
    """

    for func in protein_funcs:
        func(G)
    
    for func in ligand_funcs:
        func(G)

    for func in protein_ligand_funcs:
        func(G)

    return G


def construct_graph(
    config: Optional[ProteinLigandGraphConfig] = None,
    pdb_path: Optional[str] = None,
    pdb_code: Optional[str] = None,
    chain_selection: str = "all",
    df_processing_funcs: Optional[List[Callable]] = None,
    protein_edge_construction_funcs: Optional[List[Callable]] = None,
    protein_edge_annotation_funcs: Optional[List[Callable]] = None,
    protein_node_annotation_funcs: Optional[List[Callable]] = None,
    ligand_edge_construction_funcs: Optional[List[Callable]] = None,
    ligand_edge_annotation_funcs: Optional[List[Callable]] = None,
    ligand_node_annotation_funcs: Optional[List[Callable]] = None,
    protein_ligand_edge_construction_funcs: Optional[List[Callable]] = None,
    protein_ligand_edge_annotation_funcs: Optional[List[Callable]] = None,
    graph_annotation_funcs: Optional[List[Callable]] = None,
) -> nx.Graph:
    """
    Constructs protein structure graph from a ``pdb_code`` or ``pdb_path``.

    Users can provide a :class:`~graphein.protein.config.ProteinLigandGraphConfig`
    object to specify construction parameters.

    However, config parameters can be overridden by passing arguments directly to the function.

    :param config: :class:`~graphein.protein.config.ProteinLigandGraphConfig` object. If None, defaults to config in ``graphein.protein.config``.
    :type config: graphein.protein.config.ProteinLigandGraphConfig, optional
    :param pdb_path: Path to ``pdb_file`` to build graph from. Default is ``None``.
    :type pdb_path: str, optional
    :param pdb_code: 4-character PDB accession pdb_code to build graph from. Default is ``None``.
    :type pdb_code: str, optional
    :param chain_selection: String of polypeptide chains to include in graph. E.g ``"ABDF"`` or ``"all"``. Default is ``"all"``.
    :type chain_selection: str
    :param df_processing_funcs: List of dataframe processing functions. Default is ``None``.
    :type df_processing_funcs: List[Callable], optional
    :param protein_edge_construction_funcs: List of protein edge construction functions. Default is ``None``.
    :type edge_construction_funcs: List[Callable], optional
    :param protein_edge_annotation_funcs: List of protein edge annotation functions. Default is ``None``.
    :type edge_annotation_funcs: List[Callable], optional
    :param protein_node_annotation_funcs: List of protein node annotation functions. Default is ``None``.
    :type node_annotation_funcs: List[Callable], optional
    :param ligand_edge_construction_funcs: List of ligand edge construction functions. Default is ``None``.
    :type edge_construction_funcs: List[Callable], optional
    :param ligand_edge_annotation_funcs: List of ligand edge annotation functions. Default is ``None``.
    :type edge_annotation_funcs: List[Callable], optional
    :param ligand_node_annotation_funcs: List of ligand node annotation functions. Default is ``None``.
    :type node_annotation_funcs: List[Callable], optional
    :param protein_ligand_edge_construction_funcs: List of protein-ligand edge construction functions. Default is ``None``.
    :type edge_construction_funcs: List[Callable], optional
    :param protein_ligand_edge_annotation_funcs: List of protein-ligand edge annotation functions. Default is ``None``.
    :type edge_annotation_funcs: List[Callable], optional
    :param graph_annotation_funcs: List of graph annotation function. Default is ``None``.
    :type graph_annotation_funcs: List[Callable]
    :return: Protein Structure Graph
    :type: nx.Graph
    """

    # If no config is provided, use default
    if config is None:
        config = ProteinLigandGraphConfig()

    # Get name from pdb_file is no pdb_code is provided
    if pdb_path and (pdb_code is None):
        pdb_code = get_protein_name_from_filename(pdb_path)

    # If config params are provided, overwrite them
    config.protein_df_processing_functions = (
        df_processing_funcs
        if config.protein_df_processing_functions is None
        else config.protein_df_processing_functions
    )
    config.protein_edge_construction_functions = (
        protein_edge_construction_funcs
        if config.protein_edge_construction_functions is None
        else config.protein_edge_construction_functions
    )
    config.protein_node_metadata_functions = (
        protein_node_annotation_funcs
        if config.protein_node_metadata_functions is None
        else config.protein_node_metadata_functions
    )
    config.graph_metadata_functions = (
        graph_annotation_funcs
        if config.graph_metadata_functions is None
        else config.graph_metadata_functions
    )
    config.protein_edge_metadata_functions = (
        protein_edge_annotation_funcs
        if config.protein_edge_metadata_functions is None
        else config.protein_edge_metadata_functions
    )
    config.ligand_edge_construction_functions = (
        ligand_edge_construction_funcs
        if config.ligand_edge_construction_functions is None
        else config.ligand_edge_construction_functions
    )
    config.ligand_node_metadata_functions = (
        ligand_node_annotation_funcs
        if config.ligand_node_metadata_functions is None
        else config.ligand_node_metadata_functions
    )
    config.ligand_edge_metadata_functions = (
        ligand_edge_annotation_funcs
        if config.ligand_edge_metadata_functions is None
        else config.ligand_edge_metadata_functions
    )
    config.protein_ligand_edge_construction_functions = (
        protein_ligand_edge_construction_funcs
        if config.protein_ligand_edge_construction_functions is None
        else config.protein_ligand_edge_construction_functions
    )
    config.protein_ligand_edge_metadata_functions = (
        protein_ligand_edge_annotation_funcs
        if config.protein_ligand_edge_metadata_functions is None
        else config.protein_ligand_edge_metadata_functions
    )

    # Read raw df from pdb
    raw_df = read_pdb_to_dataframe(
        pdb_path,
        pdb_code,
        verbose=config.verbose,
        granularity=config.granularity,
    )

    # Read ligand from pdb
    df_dict = read_ligand_expo()
    pdb = parsePDB(pdb_code)
    ligand = pdb.select('not protein and not water')
    res_name_list = list(set(ligand.getResnames()))
    ligands = []
    ligands_df = []
    # TODO: filter out ions, co-factors, etc.
    for res in res_name_list:
        try:
            new_mol = process_ligand(ligand, res, df_dict)
            ligands.append(new_mol)
            ligands_df.append(raw_df.df['HETATM'].loc[raw_df.df['HETATM']["residue_name"] == res])
        except:
            continue

    protein_df = process_dataframe(
        raw_df, chain_selection=chain_selection, granularity=config.granularity
    )

    # Initialise graph with metadata
    g = initialise_graph_with_metadata(
        protein_df=protein_df,
        ligands_df=ligands_df,
        raw_pdb_df=raw_df,
        pdb_id=pdb_code,
        granularity=config.granularity,
    )
    
    # Add nodes to graph
    g = add_nodes_to_graph(g)

    # Add config to graph
    g.graph["config"] = config

    # Annotate additional node metadata
    if config.protein_node_metadata_functions is not None:
        g = annotate_node_metadata(g, config.protein_node_metadata_functions)
    if config.ligand_node_metadata_functions is not None:
        g = annotate_node_metadata(g, config.ligand_node_metadata_functions)

    # Compute graph edges
    g = compute_edges(
        g,
        protein_funcs=config.protein_edge_construction_functions,
        ligand_funcs=config.ligand_edge_construction_functions,
        protein_ligand_funcs=config.protein_ligand_edge_construction_functions,
        get_contacts_config=None,
    )

    # Annotate additional graph metadata
    if config.graph_metadata_functions is not None:
        g = annotate_graph_metadata(g, config.graph_metadata_functions)

    # Annotate additional edge metadata
    if config.protein_edge_metadata_functions is not None:
        g = annotate_edge_metadata(g, config.protein_edge_metadata_functions)
    if config.ligand_edge_metadata_functions is not None:
        g = annotate_edge_metadata(g, config.ligand_edge_metadata_functions)
    if config.protein_ligand_edge_metadata_functions is not None:
        g = annotate_edge_metadata(g, config.protein_ligand_edge_metadata_functions)

    return g


def _mp_graph_constructor(
    args: Tuple[str, str], use_pdb_code: bool, config: ProteinLigandGraphConfig
) -> nx.Graph:
    """
    Protein-ligand graph constructor for use in multiprocessing several protein structure graphs.

    :param args: Tuple of pdb code/path and the chain selection for that PDB
    :type args: Tuple[str, str]
    :param use_pdb_code: Whether or not we are using pdb codes or paths
    :type use_pdb_code: bool
    :param config: Protein structure graph construction config
    :type config: ProteinLigandGraphConfig
    :return: Protein structure graph
    :rtype: nx.Graph
    """
    log.info(f"Constructing graph for: {args[0]}. Chain selection: {args[1]}")
    func = partial(construct_graph, config=config)
    try:
        return (
            func(pdb_code=args[0], chain_selection=args[1])
            if use_pdb_code
            else func(pdb_path=args[0], chain_selection=args[1])
        )

    except Exception as ex:
        log.info(
            f"Graph construction error (PDB={args[0]})! {traceback.format_exc()}"
        )
        log.info(ex)
        return None

def process_ligand(ligand: prody, res_name: str, expo_dict: Dict):
    """
    Process ligand in the following steps:
        1. Select the ligand component with name "res_name"
        2. Get the corresponding SMILES from the Ligand Expo dictionary
        3. Create a template molecule from the SMILES in step 2
        4. Write the PDB file to a stream
        5. Read the stream into an RDKit molecule
        6. Assign the bond orders from the template from step 3

    :param ligand: ligand as generated by prody
    :type ligand: prody
    :param res_name: residue name of ligand to extract
    :type res_name: str
    :param expo_dict: dictionary with LigandExpo
    :type expo_dict: Dict
    :return: molecule with bond orders assigned
    :rtype: rdkit.Mol
    """
    output = StringIO()
    sub_mol = ligand.select(f"resname {res_name}")
    sub_smiles = expo_dict['SMILES'][res_name]
    template = AllChem.MolFromSmiles(sub_smiles)
    writePDBStream(output, sub_mol)
    pdb_string = output.getvalue()
    rd_mol = AllChem.MolFromPDBBlock(pdb_string)
    new_mol = AllChem.AssignBondOrdersFromTemplate(template, rd_mol)
    return new_mol

def read_ligand_expo():
    """
    Read Ligand Expo data, try to find a file called
    Components-smiles-stereo-oe.smi in the current directory.
    If you can't find the file, grab it from the RCSB

    :return: Ligand Expo as a dictionary with ligand id as the key
    :rtype: Dict
    """
    file_name = "Components-smiles-stereo-oe.smi"
    try:
        df = pd.read_csv(file_name, sep="\t",
                         header=None,
                         names=["SMILES", "ID", "Name"])
    except FileNotFoundError:
        url = f"http://ligand-expo.rcsb.org/dictionaries/{file_name}"
        print(url)
        r = requests.get(url, allow_redirects=True)
        open('Components-smiles-stereo-oe.smi', 'wb').write(r.content)
        df = pd.read_csv(file_name, sep="\t",
                         header=None,
                         names=["SMILES", "ID", "Name"])
    df.set_index("ID", inplace=True)
    return df.to_dict()