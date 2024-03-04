"""Tests for graphein.utils.config"""

from functools import partial
from pathlib import Path

from graphein.protein.config import (
    DSSPConfig,
    GetContactsConfig,
    ProteinGraphConfig,
)
from graphein.protein.edges.distance import (
    add_distance_threshold,
    add_peptide_bonds,
)
from graphein.protein.features.nodes.amino_acid import (
    expasy_protein_scale,
    meiler_embedding,
)
from graphein.utils.config_parser import parse_config

DATA_PATH = Path(__file__).resolve().parent

protein_graph_config = {
    "granularity": "CA",
    "keep_hets": [],
    "insertions": False,
    "verbose": False,
    "node_metadata_functions": [meiler_embedding, expasy_protein_scale],
    "edge_construction_functions": [
        add_peptide_bonds,
        partial(
            add_distance_threshold,
            long_interaction_threshold=5,
            threshold=10.0,
        ),
    ],
    "get_contacts_config": GetContactsConfig(
        contacts_dir="../examples/contacts/",
        pdb_dir="../examples/contacts/",
    ),
    "dssp_config": DSSPConfig(),
}


def test_protein_graph_config():
    """Test the protein graph config yaml parser."""
    config = ProteinGraphConfig(**protein_graph_config)
    yml_config = parse_config(DATA_PATH / "test_protein_graph_config.yml")
    assert config == yml_config
