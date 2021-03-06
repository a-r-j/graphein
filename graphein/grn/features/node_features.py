from bioservices import HGNC, UniProt


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
