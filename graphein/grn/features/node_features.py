from graphein.utils.utils import import_message

try:
    from bioservices import HGNC, UniProt
except ImportError:
    import_message(
        submodule="graphein.grn.features.node_features",
        package="bioservices",
        conda_channel="bioconda",
        pip_install=True,
    )


def add_sequence_to_nodes(n, d):
    """
    Maps UniProt ACC to UniProt ID. Retrieves sequence from UniProt and adds it to the node

    :param n: Graph node.
    :param d: Graph attribute dictionary.
    """

    h = HGNC(verbose=False)
    u = UniProt(verbose=False)

    d["uniprot_ids"] = h.fetch("symbol", d["gene_id"])["response"]["docs"][0][
        "uniprot_ids"
    ]

    # Todo these API calls should probably be batched
    # Todo mapping with bioservices to support other protein IDs?

    for id in d["uniprot_ids"]:
        d[f"sequence_{id}"] = u.get_fasta_sequence(id)
