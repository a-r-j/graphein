"""Functions for making and parsing API calls to BIOGRID."""

# %%
# Graphein
# Author: Ramon Vinas, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import Dict, List, Union

import pandas as pd
import requests
from loguru import logger as log


def params_BIOGRID(
    params: Dict[str, Union[str, int, List[str], List[int]]], **kwargs
) -> Dict[str, Union[str, int]]:
    """
    Updates default parameters with user parameters for the method
    "interactions" of the BIOGRID API REST.

    See also https://wiki.thebiogrid.org/doku.php/biogridrest
    :param params: Dictionary of default parameters
    :type params: Dict[str, Union[str, int, List[str], List[int]]]
    :param kwargs: User parameters for the method "network" of the BIOGRID API
        REST. The key must start with "BIOGRID"
    :type kwargs: Dict[str, Union[str, int, List[str], List[int]]]
    :return: Dictionary of parameters
    :rtype: Dict[str, Union[str, int]]
    """
    fields = [
        "searchNames",  # If ‘true’, the interactor OFFICIAL_SYMBOL will be
        # examined for a match with the geneList.
        "max",  # Number of results to fetch
        "interSpeciesExcluded",  # If ‘true’, interactions with interactors from
        # different species will be excluded.
        "selfInteractionsExcluded",  # If ‘true’, interactions with one
        # interactor will be excluded.
        "evidenceList",  # Any interaction evidence with its Experimental System
        # in the list will be excluded from the results unless includeEvidence
        # is set to true.
        "includeEvidence",  # If set to true, any interaction evidence with its
        # Experimental System in the evidenceList will be included in the result
        "searchIds",  # If ‘true’, the interactor ENTREZ_GENE, ORDERED LOCUS and
        # SYSTEMATIC_NAME (orf) will be examined for a match with the geneList.
        "searchNames",  # If ‘true’, the interactor OFFICIAL_SYMBOL will be
        # examined for a match with the geneList.
        "searchSynonyms",  # If ‘true’, the interactor SYNONYMS will be examined
        # for a match with the geneList.
        "searchBiogridIds",  # If ‘true’, the entries in 'GENELIST' will be
        # compared to BIOGRID internal IDS which are provided in all Tab2
        # formatted files.
        "additionalIdentifierTypes",  # Identifier types on this list are
        # examined for a match with the geneList.
        "excludeGenes",  # If ‘true’, interactions containing genes in the
        # geneList will be excluded from the results.
        "includeInteractors",  # If ‘true’, in addition to interactions between
        # genes on the geneList, interactions will also be fetched which have
        # only one interactor on the geneList
        "includeInteractorInteractions",  # If ‘true’ interactions between the
        # geneList’s first order interactors will be included.
        "pubmedList",  # Interactions will be fetched whose Pubmed Id is/ is
        # not in this list, depending on the value of excludePubmeds.
        "excludePubmeds",  # If ‘false’, interactions with Pubmed ID in
        # pubmedList will be included in the results; if ‘true’ they will be
        # excluded.
        "htpThreshold",  # Interactions whose Pubmed ID has more than this
        # number of interactions will be excluded from the results. Ignored if
        # excludePubmeds is ‘false’.
        "throughputTag",  # If set to 'low or 'high', only interactions with
        # 'Low throughput' or 'High throughput' in the 'throughput' field
        # will be returned.
    ]
    for p in fields:
        kwarg_name = "BIOGRID_" + p
        if kwarg_name in kwargs:
            value = kwargs[kwarg_name]
            if type(value) is list:
                value = "|".join(value)
            params[p] = value
    return params


def parse_BIOGRID(
    protein_list: List[str],
    ncbi_taxon_id: Union[int, str, List[int], List[str]],
    paginate: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Makes BIOGRID API call and returns a source specific Pandas dataframe.

    See also [1] BIOGRID: https://wiki.thebiogrid.org/doku.php/biogridrest
    :param protein_list: Proteins to include in the graph
    :type protein_list: List[str]
    :param ncbi_taxon_id: NCBI taxonomy identifiers for the organism.
        Default is ``9606`` (Homo Sapiens).
    :type ncbi_taxon_id: Union[int, str, List[int], List[str]]
    :param paginate: boolean indicating whether to paginate the calls (for
        BIOGRID, the maximum number of rows per call is ``10000).`` Defaults to
        ``True``.
    :type paginate: bool
    :param kwargs: Parameters of the "interactions" method of the BIOGRID API
        REST, used to select the results. The parameter names are of the form
        ``"BIOGRID_<param>"``, where ``<param>`` is the name of the parameter.
        Information about these parameters can be found at [1].
    :type kwargs: Dict[str, Union[str, int, List[str], List[int]]]
    :return: Source specific Pandas dataframe.
    :rtype: pd.DataFrame
    """
    # Prepare call to BIOGRID API
    biogrid_api_url = "https://webservice.thebiogrid.org"
    method = "interactions"
    request_url = "/".join([biogrid_api_url, method])
    if type(ncbi_taxon_id) is list:
        ncbi_taxon_id = "|".join(str(t) for t in ncbi_taxon_id)
    params = {  # Default parameters
        "geneList": "|".join(protein_list),
        "accesskey": "c4ab86373e0bb921a878bb6d15ee4fb4",
        "taxId": ncbi_taxon_id,  # 9606 is human
        "format": "json",
        "max": 10000,  # Number of results to fetch
        "searchNames": "true",
        "includeInteractors": "false",  # Set to true to get any interaction
        # involving EITHER gene, set to false to get interactions between genes
        "selfInteractionsExcluded": "true",  # If ‘true’, interactions with one
        # interactor will be excluded
    }
    params = params_BIOGRID(params, **kwargs)

    # Call BIOGRID
    def make_call(
        request_url: str,
        params: Dict[str, Union[str, int]],
        start: int = 0,
        max: int = 10000,
        paginate: bool = paginate,
    ) -> pd.DataFrame:
        """
        Makes call to BIOGRID API.

        :param request_url: BIOGRID URL to make request
        :type request_url: str
        :param params: BIOGRID API parameters to use
        :type params: Dict[str, Union[str, int]]
        :param start: index in gene list to start from
        :type start: int
        :param max: number of genes to use in API call. Results are limited to
            10k per call
        :type max: int
        :param paginate: bool indicating whether or not to paginate calls.
            Above 10k calls this is required
        :type paginate: bool
        :return: pd.DataFrame containing BIOGRID_df API call output
        :rtype: pd.DataFrame
        """
        params["start"] = start
        response = requests.post(request_url, data=params)
        df = pd.read_json(response.text.strip()).transpose()

        # Maximum number of results is limited to 10k. Paginate to
        # retrieve everything
        if paginate and df.shape[0] == max:
            next_df = make_call(request_url, params, start + max, max)
            df = pd.concat([df, next_df])

        return df

    return make_call(
        request_url=request_url, params=params, start=0, max=params["max"]
    )


def filter_BIOGRID(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Filters results of the BIOGRID API call according to user kwargs.

    :param df: Source specific Pandas DataFrame (BIOGRID) with results of the
        API call.
    :type df: pd.DataFrame
    :param kwargs: User thresholds used to filter the results. The parameter
        names are of the form ``"BIOGRID_<param>"``, where ``<param>`` is the
        name of the parameter. All the parameters are numerical values.
    :type kwargs: Dict[str, Union[str, int, List[str], List[int]]]
    :return: Source specific Pandas DataFrame with filtered results.
    :rtype: pd.DataFrame
    """
    # Note: To filter BIOGRID interactions, use parameters from
    # https://wiki.thebiogrid.org/doku.php/biogridrest
    # TODO: Make sure that user can filter results of API call via the parameters.
    #       Otherwise implement filtering here.
    # TODO: Perhaps can filter by EXPERIMENTAL_SYSTEM (e.g. Co-fractionation)
    #       and EXPERIMENTAL_SYSTEM_TYPE (e.g. physical)
    return df


def standardise_BIOGRID(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises BIOGRID DataFrame, e.g. puts everything into a common format.

    :param df: Source specific Pandas DataFrame.
    :type df: pd.DataFrame
    :return: Standardised DataFrame.
    :rtype: pd.DataFrame
    """
    if df.empty:
        return pd.DataFrame({"p1": [], "p2": [], "source": []})

    # Rename & delete columns
    df = df.rename(
        columns={"OFFICIAL_SYMBOL_A": "p1", "OFFICIAL_SYMBOL_B": "p2"}
    )
    df = df[["p1", "p2"]]

    # Add source column
    df["source"] = "BIOGRID"

    return df


def BIOGRID_df(
    protein_list: List[str],
    ncbi_taxon_id: Union[int, str, List[int], List[str]],
    **kwargs,
) -> pd.DataFrame:
    """
    Generates standardised DataFrame with BIOGRID protein-protein interactions,
    filtered according to user's input.

    :param protein_list: List of proteins (official symbol) that will be
        included in the PPI graph.
    :type protein_list: List[str]
    :param ncbi_taxon_id: NCBI taxonomy identifiers for the organism.
        ``9606`` corresponds to Homo Sapiens.
    :type ncbi_taxon_id: int
    :param kwargs:  Additional parameters to pass to BIOGRID API calls.
    :type kwargs: Union[int, str, List[int], List[str]]
    :return: Standardised DataFrame with BIOGRID interactions.
    :rtype: pd.DataFrame
    """
    df = parse_BIOGRID(
        protein_list=protein_list, ncbi_taxon_id=ncbi_taxon_id, **kwargs
    )
    df = filter_BIOGRID(df, **kwargs)
    df = standardise_BIOGRID(df)
    return df
