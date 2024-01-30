"""Functions for adding metadata to PPI Graphs from STRING and BIOGRID."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Ramon Vinas
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import Dict, Union

import networkx as nx
import pandas as pd
from loguru import logger as log

from graphein.ppi.parse_biogrid import BIOGRID_df
from graphein.ppi.parse_stringdb import STRING_df


def add_string_metadata(
    G: nx.Graph, kwargs: Dict[str, Union[str, int]]
) -> nx.Graph:
    """
    Adds interaction dataframe from STRING to graph.

    :param G: PPI Graph to add metadata to.
    :type G: nx.Graph
    :param kwargs:  Additional parameters for STRING API call.
    :type kwargs: Dict[str, Union[str, int]]
    :return: PPIGraph with added STRING ``interaction_df`` as metadata.
    :rtype: nx.Graph
    """
    G.graph["string_df"] = STRING_df(
        G.graph["protein_list"], G.graph["ncbi_taxon_id"], kwargs
    )
    log.debug("Added STRING interaction dataframe as graph metadata")
    return G


def add_biogrid_metadata(
    G: nx.Graph, kwargs: Dict[str, Union[str, int]]
) -> nx.Graph:
    """
    Adds interaction dataframe from BIOGRID to graph.

    :param G: PPI Graph to add metadata to
    :type G: nx.Graph
    :param kwargs:  Additional parameters for BIOGRID API call
    :type kwargs: Dict[str, Union[str, int]]
    :return: PPIGraph with added BIOGRID interaction_df as metadata
    :rtype: nx.Graph
    """
    G.graph["string_df"] = BIOGRID_df(
        G.graph["protein_list"], G.graph["ncbi_taxon_id"], kwargs
    )
    log.debug("Added BIOGRID interaction dataframe as graph metadata")
    return G


def add_string_biogrid_metadata(
    G: nx.Graph, kwargs: Dict[str, Union[str, int]]
) -> nx.Graph:
    """
    Adds interaction dataframe from STRING and BIOGRID to graph.

    :param G: PPIGraph to add metadata to.
    :type G: nx.Graph
    :param kwargs:  Additional parameters for STRING and BIOGRID API calls.
    :type kwargs: Dict[str, Union[str, int]]
    :return: PPIGraph with added STRING and BIOGRID interaction_df as metadata.
    :rtype: nx.Graph
    """
    G.graph["string_df"] = STRING_df(
        G.graph["protein_list"], G.graph["ncbi_taxon_id"], kwargs
    )
    G.graph["biogrid_df"] = BIOGRID_df(
        G.graph["protein_list"], G.graph["ncbi_taxon_id"], kwargs
    )
    G.graph["combined_interaction_df"] = pd.concat(
        [G.graph["string_df"], G.graph["biogrid_df"]]
    )
    log.debug(
        "Added combined STRING and BIOGRID interaction dataframe as graph metadata"
    )
    return G
