"""Base Config object for use with Protein Graph Construction"""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Literal, Optional, Union

from pydantic import BaseModel

from graphein.protein.edges.distance import add_peptide_bonds
from graphein.protein.features.nodes.amino_acid import meiler_embedding


class DSSPConfig(BaseModel):
    executable: str = "mkdssp"


class GetContactsConfig(BaseModel):
    """Config object for parameters relating to running GetContacts"""

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
    """Config Object for Protein Structure Graph Construction"""

    granularity: Union[GraphAtoms, GranularityOpts] = "CA"
    keep_hets: bool = False
    insertions: bool = False
    pdb_dir: Path = Path(
        "../examples/pdbs/"
    )  # Also suggest to avoid hard-coding paths if possible!
    verbose: bool = True
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


class ProteinMeshConfig(BaseModel):
    """Config object for parameters relating to ProteinMeshConfig Mesh construction with PyMol"""

    pymol_command_line_options: Optional[str] = "-cKq"
    pymol_commands: Optional[List[str]] = ["show surface"]
