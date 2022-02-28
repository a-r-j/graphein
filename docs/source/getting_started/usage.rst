Usage
========

Graphein has a simple command line interface to get started and convert PDB files into graphs.

.. code-block:: bash
    graphein -c config.yaml -p path/to/pdbs -o path/to/output

A .yaml config file can be specified to specify any of the config objects.

.. code-block:: yaml
    !ProteinGraphConfig
        granularity: "CA"
        keep_hets: False
        insertions: False
        verbose: False
        node_metadata_functions:
            - !func:graphein.protein.features.nodes.amino_acid.meiler_embedding
            - !func:graphein.protein.features.nodes.amino_acid.expasy_protein_scale
        edge_construction_functions:
            - !func:graphein.protein.edges.distance.add_peptide_bonds
            - !func:graphein.protein.edges.distance.add_distance_threshold
                long_interaction_threshold: 5
                threshold: 10.
        dssp_config: !DSSPConfig

.. code-block:: python
    from graphein.utils.config import parse_config
    yml_config = parse_config(PATH / "protein_graph_config.yml")

Reading the example .yaml file above with the `parse_config` function, would be the equivalent of specifying a Python dict of arguments and loading it into the ProteinGraphConfig.

.. code-block:: python
    protein_graph_config = {
        "granularity": "CA",
        "keep_hets": False,
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
        "dssp_config": DSSPConfig(),
    }
    config = ProteinGraphConfig(**protein_graph_config)