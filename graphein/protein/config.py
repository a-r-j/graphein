"""Base Config object for use with Protein Graph Construction"""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from deepdiff import DeepDiff
from deepdiff.operator import BaseOperator
from pydantic import BaseModel
from typing_extensions import Literal

from graphein.protein.edges.distance import add_peptide_bonds
from graphein.protein.features.nodes.amino_acid import meiler_embedding


def partial_functions_equal(func1, func2):
    if not (isinstance(func1, partial) and isinstance(func2, partial)):
        return False
    are_equal = all(
        [
            getattr(func1, attr) == getattr(func2, attr)
            for attr in ["func", "args", "keywords"]
        ]
    )
    return are_equal


class PartialMatchOperator(BaseOperator):
    def give_up_diffing(self, level, diff_instance):
        return partial_functions_equal(level.t1, level.t2)


class PathMatchOperator(BaseOperator):
    def give_up_diffing(self, level, diff_instance):
        return level.t1 == level.t2


class DSSPConfig(BaseModel):
    executable: str = "mkdssp"


class GetContactsConfig(BaseModel):
    """
    Config object for parameters relating to running GetContacts. GetContacts is an optional dependency from which intramolecular interactions can be computed and used as edges in the graph.

    :param get_contacts_path: Path to GetContacts installation
    :type get_contacts_path: pathlib.Path
    :param contacts_dir: Path to store output of GetContacts
    :type contacts_dir: pathlib.Path
    :param pdb_dir: Path to PDB files to be used to compute intramolecular interactions.
    :type pdb_dir: pathlib.Path
    :param granularity: Specifies the node types of the graph, defaults to "CA" for alpha-carbons as nodes. Other options are "CB" (beta-carbon), "atom" for all-atom graphs, and "centroid" for nodes positioned as residue centroids.
    :type granularity: str
    """

    get_contacts_path: Path = Path(
        "/Users/arianjamasb/github/getcontacts/"
    ).resolve()
    contacts_dir: Path = Path("../examples/contacts/").resolve()
    pdb_dir: Path = Path("../examples/pdbs/").resolve()
    granularity: str = "CA"


GraphAtoms = Literal[
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "OG",
    "CG",
    "CD1",
    "CD2",
    "CE1",
    "CE2",
    "CZ",
    "OD1",
    "ND2",
    "CG1",
    "CG2",
    "CD",
    "CE",
    "NZ",
    "OD2",
    "OE1",
    "NE2",
    "OE2",
    "OH",
    "NE",
    "NH1",
    "NH2",
    "OG1",
    "SD",
    "ND1",
    "SG",
    "NE1",
    "CE3",
    "CZ2",
    "CZ3",
    "CH2",
    "OXT",
]
GranularityOpts = Literal["atom", "centroids"]


class ProteinGraphConfig(BaseModel):
    """
    Config Object for Protein Structure Graph Construction

    :param granularity: Controls the granularity of the graph construction. "atom" builds an atomic-scale graph where
        nodes are constituent atoms. Residue-level graphs can be build by specifying which constituent atom should
        represent node positions (see graphein.protein.config.GraphAtoms). Additionally, "centroids" can be specified to
        compute the centre of gravity for a given atom. Defaults to "CA" (alpha-Carbon)
    :type granularity: str (Union[graphein.protein.config.GraphAtoms, graphein.protein.config.GranularityOpts])
    :param keep_hets: Controls whether or not heteroatoms are removed from the PDB file. These are typically modified
        residules, bound ligands, crystallographic adjuvants, ions or water molecules.
    :type keep_hets: bool
    :param insertions: Controls whether or not insertions are allowed.
    :type insertions: bool
    :param pdb_dir: Specifies path to download protein structures into.
    :type pdb_dir: pathlib.Path
    :param verbose: Specifies verbosity of graph creation process.
    :type verbose: bool
    :param exclude_waters: Specifies whether or not water molecules are excluded from the structure
    :type excluded_waters: bool
    :param deprotonate: Specifies whether or not to remove H atoms from the graph.
    :type deprotonate: bool
    :param protein_df_processing_functions: List of functions that take a pd.DataFrame and return a pd.DataFrame.
        This allows users to define their own series of processing functions for the protein structure DataFrame and
        override the default sequencing of processing steps provided by Graphein. We refer users to our low-level API
        tutorial for more details.
    :type protein_df_processing_functions: Optional[List[Callable]]

    :param edge_construction_functions: List of functions that take an nx.Graph and return an nx.Graph with desired
        edges added. Prepared edge constructions can be found in graphein.protein.edges
    :type edge_construction_functions: List[Callable]

    :param node_metadata_functions: List of functions that taken an nx.Graph
    :type node_metadata_functions: List[Callable], optional

    :param edge_metadata_functions: List of functions that take an
    :type edge_metadata_functions: List[Callable], optional

    :param graph_metadata_functions: List of functions that take an nx.Graph and return an nx.Graph with added
        graph-level features and metadata.
    :type graph_metadata_functions: List[Callable], optional

    :param get_contacts_config: Config object containing parameters for running GetContacts for computing intramolecular
        contact-based edges. Defaults to None.
    :type get_contacts_config: GetContactsConfig, optional
    :param dssp_config: Config Object containing reference to executable. Defaults to None.
    :type dssp_config: DSSPConfig, optional
    """

    granularity: Union[GraphAtoms, GranularityOpts] = "CA"
    keep_hets: bool = False
    insertions: bool = False
    pdb_dir: Path = Path(
        "../examples/pdbs/"
    )  # Also suggest to avoid hard-coding paths if possible!
    verbose: bool = False
    exclude_waters: bool = True
    deprotonate: bool = False

    # Graph construction functions
    protein_df_processing_functions: Optional[List[Callable]] = None
    edge_construction_functions: List[Union[Callable, str]] = [
        add_peptide_bonds
    ]
    node_metadata_functions: Optional[List[Union[Callable, str]]] = [
        meiler_embedding
    ]
    edge_metadata_functions: Optional[List[Union[Callable, str]]] = None
    graph_metadata_functions: Optional[List[Callable]] = None

    # External Dependency configs
    get_contacts_config: Optional[GetContactsConfig] = None
    dssp_config: Optional[DSSPConfig] = None

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ProteinGraphConfig):
            return (
                DeepDiff(
                    self,
                    other,
                    custom_operators=[
                        PartialMatchOperator(types=[partial]),
                        PathMatchOperator(types=[Path]),
                    ],
                )
                == {}
            )
        else:
            return self.dict() == other


class ProteinMeshConfig(BaseModel):
    """
    Config object for parameters relating to ProteinMeshConfig Mesh construction with PyMol

    :param pymol_command_line_options: List of CLI args for running PyMol.
        See: https://pymolwiki.org/index.php/Command_Line_Options. Defaults to "-cKq"
    :type pymol_command_line_options: str, optional
    :param pymol_commands: List of Commands passed to PyMol in surface construction.
    :type pymol_commands: List[str], optional
    """

    pymol_command_line_options: Optional[str] = "-cKq"
    pymol_commands: Optional[List[str]] = ["show surface"]
