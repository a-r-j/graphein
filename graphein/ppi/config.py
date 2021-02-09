# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Ramon Vinas
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class STRINGConfig(BaseModel):
    """
    Config for specifying parameters for API calls to STRINGdb
    """

    species: Optional[int] = 9606  # NCBI taxon identifiers
    required_score: Optional[
        int
    ] = 50  # threshold of significance to include a interaction, a number between 0 and 1000 (default depends on the network)
    network_type: Optional[
        str
    ] = "functional"  # network type: functional (default), physical
    add_nodes: Optional[
        int
    ] = 0  # adds a number of proteins to the network based on their confidence score, e.g., extends the interaction neighborhood of selected proteins to desired value
    show_query_node_labels: Optional[
        bool
    ] = 0  # when available use submitted names in the preferredName column when (0 or 1) (default:0)


class BioGridConfig(BaseModel):
    """
    Config for specifying parameters for API calls to BIOGRID
    """

    searchNames: Optional[
        bool
    ] = True  # If ‘true’, the interactor OFFICIAL_SYMBOL will be examined for a match with the geneList.
    max: Optional[int] = 10000  # Number of results to fetch
    interSpeciesExcluded: Optional[
        bool
    ] = True  # If ‘true’, interactions with interactors from different species will be excluded.
    selfInteractionsExcluded: Optional[
        bool
    ] = False  # If ‘true’, interactions with one interactor will be excluded.
    evidenceList: Optional[
        str
    ] = ""  # Any interaction evidence with its Experimental System in the list will be excluded from the results unless includeEvidence is set to true.
    includeEvidence: Optional[
        bool
    ] = False  # If set to true, any interaction evidence with its Experimental System in the evidenceList will be included in the result
    searchIds: Optional[
        bool
    ] = True  # If ‘true’, the interactor ENTREZ_GENE, ORDERED LOCUS and SYSTEMATIC_NAME (orf) will be examined for a match with the geneList.
    searchNames: Optional[
        bool
    ] = True  # If ‘true’, the interactor OFFICIAL_SYMBOL will be examined for a match with the geneList.
    searchSynonyms: Optional[
        bool
    ] = True  # If ‘true’, the interactor SYNONYMS will be examined for a match with the geneList.
    searchBiogridIds: Optional[
        bool
    ] = True  # If ‘true’, the entries in 'GENELIST' will be compared to BIOGRID internal IDS which are provided in all Tab2 formatted files.
    additionalIdentifierTypes: Optional[
        str
    ] = ""  # Identifier types on this list are examined for a match with the geneList.
    excludeGenes: Optional[
        bool
    ] = False  # If ‘true’, interactions containing genes in the geneList will be excluded from the results.
    includeInteractors: Optional[
        bool
    ] = True  # If ‘true’, in addition to interactions between genes on the geneList, interactions will also be fetched which have only one interactor on the geneList
    includeInteractorInteractions: Optional[
        bool
    ] = False  # If ‘true’ interactions between the geneList’s first order interactors will be included.
    pubmedList: Optional[
        str
    ] = ""  # Interactions will be fetched whose Pubmed Id is/ is not in this list, depending on the value of excludePubmeds.
    excludePubmeds: Optional[
        bool
    ] = False  # If ‘false’, interactions with Pubmed ID in pubmedList will be included in the results; if ‘true’ they will be excluded.
    htpThreshold: Optional[
        int
    ] = 20  # Interactions whose Pubmed ID has more than this number of interactions will be excluded from the results. Ignored if excludePubmeds is ‘false’.
    throughputTag: Optional[
        str
    ] = "any"  # If set to 'low or 'high', only interactions with 'Low throughput' or 'High throughput' in the 'throughput' field will be returned.


class PPIGraphConfig(BaseModel):
    """
    Config for specifying parameters for PPI Graph Construction
    """

    paginate: bool = True
    ncbi_taxon_id: int = 9606
    kwargs: Optional[Dict[str, Union[str, int, float]]] = {
        "STRING_escore": 0.2,  # Keeps STRING interactions with an experimental score >= 0.2
        "BIOGRID_throughputTag": "high",  # Keeps high throughput BIOGRID interactions
    }
    string_config: Optional[STRINGConfig] = None
    biogrid_config: Optional[BioGridConfig] = None

    class Config:
        arbitrary_types_allowed: bool = True
