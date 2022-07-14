import logging
from typing import Any, Dict

from graphein.utils.utils import import_message

log = logging.getLogger(__name__)

try:
    from bioservices import HGNC, UniProt
except ImportError:
    message = import_message(
        submodule="graphein.grn.features.node_features",
        package="bioservices",
        conda_channel="bioconda",
        pip_install=True,
    )
    log.warning(message)


def add_sequence_to_nodes(n: str, d: Dict[str, Any]):
    """
    Maps UniProt ACC to UniProt ID. Retrieves sequence from UniProt and adds
    it to the node

    :param n: Graph node. Unused (retained for a consistent function signature).
    :type n: str
    :param d: Graph attribute dictionary.
    :type d: Dict[str, Any]
    """

    h = HGNC(verbose=False)
    u = UniProt(verbose=False)

    d["uniprot_ids"] = h.fetch("symbol", d["gene_id"])["response"]["docs"][0][
        "uniprot_ids"
    ]

    # Todo these API calls should probably be batched
    # Todo mapping with bioservices to support other protein IDs?

    for id in d["uniprot_ids"]:
        d[f"sequence_{id}"] = u.search(id, columns="sequence").split("\n")[1]
