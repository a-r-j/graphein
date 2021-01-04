"""Base Config object for use with Protein Graph Construction"""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import Callable, List, Optional, Union

from pydantic import BaseModel

from graphein.features.amino_acid import meiler_embedding
from graphein.features.edges.intramolecular import peptide_bonds


class ProteinGraphConfig(BaseModel):
    granularity: str = "CA"
    keep_hets: bool = False
    insertions: bool = False
    get_contacts_path: str = "/Users/arianjamasb/github/getcontacts"
    pdb_dir: str = "../examples/pdbs/"
    contacts_dir: str = "../examples/contacts/"
    verbose: bool = True
    exclude_waters: bool = True
    covalent_bonds: bool = True
    include_ss: bool = True
    include_ligand: bool = False
    intramolecular_interactions: Optional[List[str]] = None  # Todo rm
    graph_constructor: Optional[str] = None
    edge_distance_cutoff: Optional[float] = None
    verbose: bool = True
    deprotonate: bool = False
    remove_string_labels: bool = False
    long_interaction_threshold: Optional[int] = None
    edge_functions: List[Union[Callable, str]] = [peptide_bonds]
    node_metadata_functions: List[Union[Callable, str]] = [meiler_embedding]
    edge_metadata_functions: Optional[List[Union[Callable, str]]] = None


class ProteinMeshConfig(BaseModel):
    pymol_command_line_options: Optional[str] = "-cKq"
    pymol_commands: Optional[List[str]] = ["show surface"]
