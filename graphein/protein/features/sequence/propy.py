"""Sequence featurisation functions wrapping ProPy."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from functools import partial
from typing import Any, Callable, Dict, List, Optional

import networkx as nx
from loguru import logger as log

from graphein.protein.features.sequence.utils import (
    aggregate_feature_over_chains,
    compute_feature_over_chains,
)
from graphein.protein.features.utils import (
    aggregate_graph_feature_over_chains,
    convert_graph_dict_feat_to_series,
)


def compute_propy_feature(
    G: nx.Graph,
    func: Callable,
    feature_name: str,
    aggregation_type: Optional[List[str]] = None,
) -> nx.Graph:
    """
    Computes Propy Descriptors over chains in a Protein Graph

    :param G: Protein Graph
    :type G: nx.Graph
    :param func: ProPy wrapper function to compute
    :type func: Callable
    :param feature_name: Name of feature to index it in the nx.Graph object
    :type feature_name: str
    :param aggregation_type: Type of aggregation to use when aggregating a
        feature over multiple chains. One of: ``["mean", "max", "sum"]``.
        Defaults to ``None``.
    :type aggregation_type: List[str], optional
    :return G: Returns protein Graph with features added. Features are
        accessible with ``G.graph[{feature_name}_{chain | aggegation_type}]``
    :rtype: nx.Graph
    """
    G = compute_feature_over_chains(G, func, feature_name=feature_name)

    # Convert to Series
    G = convert_graph_dict_feat_to_series(G, feature_name=feature_name)

    # Aggregate features
    if len(G.graph["chain_ids"]) > 1:
        if aggregation_type:
            G = aggregate_graph_feature_over_chains(
                G, feature_name=feature_name, aggregation_type=aggregation_type
            )
    else:
        log.debug("Aggregation not carried out on single-chain graph")
    return G


def amino_acid_composition(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Calculate the composition of Amino acids for a given protein sequence.

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use.
    :type aggregation_type: Optional[List[str]]
    :return: Protein Graph with ``amino_acid_composition`` feature added.
        ``G.graph["amino_acid_composition_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.AAComposition import CalculateAAComposition

    func = CalculateAAComposition
    feature_name = "amino_acid_composition"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def dipeptide_composition(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Calculate the composition of dipeptide for a given protein sequence.
    Contains composition of 400 dipeptides.

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``dipeptide_composition`` feature added.
        ``G.graph["dipeptide_composition_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.AAComposition import CalculateDipeptideComposition

    func = CalculateDipeptideComposition
    feature_name = "dipeptide_composition"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def aa_dipeptide_composition(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Calculate the composition of AADs, dipeptide and 3-mers for a given protein
    sequence. Contains all composition values of AADs, dipeptide and 3-mers
    (8420).

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``aa_dipeptide_composition`` feature added.
        ``G.graph["aa_dipeptide_composition_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.AAComposition import CalculateAADipeptideComposition

    func = CalculateAADipeptideComposition
    feature_name = "aa_dipeptide_composition"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def aa_spectrum(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Calculate the spectrum descriptors of 3-mers for a given protein. Contains
    the composition values of 8000 3-mers

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with aa_spectrum feature added.
        ``G.graph["aa_spectrum_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.AAComposition import GetSpectrumDict

    func = GetSpectrumDict
    feature_name = "aa_spectrum"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


######## CTD Funcs #######


def all_composition_descriptors(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Calculate all composition descriptors based on seven different properties of
    AADs.

    :param G: Protein Graph to featurise
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``composition_descriptors`` feature added.
        ``G.graph["composition_descriptors_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateC

    func = CalculateC
    feature_name = "composition_descriptors"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def all_ctd_descriptors(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Calculate all CTD descriptors based seven different properties of AADs.

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``ctd_descriptors`` feature added.
        ``G.graph["ctd_descriptors_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateCTD

    func = CalculateCTD
    feature_name = "ctd_descriptors"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def composition_descriptor(
    G: nx.Graph,
    AAProperty: Dict[Any, Any],
    AAPName: str,
    aggregation_type: Optional[List[str]] = None,
) -> nx.Graph:
    """
    Compute composition descriptors.

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param AAProperty: contains classification of amino acids such as
        ``_Polarizability.``
    :type AAProperty: Dict[Any, Any]
    :param AAPName: used for indicating a AAP name.
    :type AAPName: str
    :param aggregation_type: Aggregation types to use over chains
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``composition_{AAPName}`` feature added.
        ``G.graph["composition_{AAPName}_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateComposition

    func = partial(
        CalculateComposition, AAProperty=AAProperty, AAPName=AAPName
    )
    feature_name = f"composition_{AAPName}"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def composition_charge(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Calculate composition descriptors based on Charge of AADs.

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``composition_charge`` feature added.
        ``G.graph["composition_charge_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateCompositionCharge

    func = CalculateCompositionCharge
    feature_name = "composition_charge"

    G = compute_feature_over_chains(G, func, feature_name="composition_charge")

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def composition_hydrophobicity(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Calculate composition descriptors based on Hydrophobicity of AADs.

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``composition_hydrophobicity`` feature added.
        ``G.graph["composition_hydrophobicity_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateCompositionHydrophobicity

    func = CalculateCompositionHydrophobicity
    feature_name = "composition_hydrophobicity"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def composition_normalized_vdwv(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Calculate composition descriptors based on NormalizedVDWV of AADs.

    :param G: Protein Graph to featurise
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``composition_normalized_vdwv`` feature added.
        ``G.graph["composition_normalized_vdwv_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateCompositionNormalizedVDWV

    func = CalculateCompositionNormalizedVDWV
    feature_name = "composition_normalised_vdwv"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def composition_polarity(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate composition descriptors based on Polarity of AADs.

    :param G: Protein Graph to featurise
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``composition_polarity`` feature added.
        ``G.graph["composition_polarity_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateCompositionPolarity

    func = CalculateCompositionPolarity
    feature_name = "composition_polarity"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def composition_polarizability(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate composition descriptors based on Polarizability of AADs.

    :param G: Protein graph to featurise
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``composition_polarizability`` feature added.
        ``G.graph["composition_polarizability_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateCompositionPolarizability

    func = CalculateCompositionPolarizability
    feature_name = "composition_polarizability"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def composition_secondary_str(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate composition descriptors based on SecondaryStr of AADs.

    :param G: Protein graph to featurise
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``composition_secondary_str`` feature added.
        ``G.graph["composition_secondary_str_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateCompositionSecondaryStr

    func = CalculateCompositionSecondaryStr
    feature_name = "composition_secondary_str"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def composition_solvent_accessibility(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate composition descriptors based on SolventAccessibility of AADs.

    :param G: Protein graph to featurise
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with composition_solvent_accessibility feature added.
        G.graph["composition_solvent_accessibility_{chain | aggregation_type}"]
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateCompositionSolventAccessibility

    func = CalculateCompositionSolventAccessibility
    feature_name = "composition_solvent_accessibility"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def all_distribution_descriptors(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Calculate all distribution descriptors based on seven different properties
    of AADs.

    :param G: Protein Graph to featurise
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``distribution_descriptors`` feature added.
        ``G.graph["distribution_descriptors_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateC

    func = CalculateC
    feature_name = "distribution_descriptors"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def distribution_descriptor(
    G: nx.Graph,
    AAProperty: Dict[Any, Any],
    AAPName: str,
    aggregation_type: Optional[List[str]] = None,
) -> nx.Graph:
    """
    Compute distribution descriptors.

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param AAProperty: contains classifciation of amino acids such as _Polarizability.
    :type AAProperty: Dict[Any, Any]
    :param AAPName: used for indicating a AAP name.
    :type AAPName: str
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``distribution_{AAPName}`` feature added.
        ``G.graph["distribution_{AAPName}_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateDistribution

    func = partial(
        CalculateDistribution, AAProperty=AAProperty, AAPName=AAPName
    )
    feature_name = f"distribution_{AAPName}"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def distribution_charge(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Calculate distribution descriptors based on Charge of AADs.

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``distribution_charge`` feature added.
        ``G.graph["distribution_charge_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateDistributionCharge

    func = CalculateDistributionCharge
    feature_name = "distribution_charge"

    G = compute_feature_over_chains(
        G, func, feature_name="distribution_charge"
    )

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def distribution_hydrophobicity(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Calculate distribution descriptors based on Hydrophobicity of AADs.

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``distribution_hydrophobicity`` feature added.
        ``G.graph["distribution_hydrophobicity_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateDistributionHydrophobicity

    func = CalculateDistributionHydrophobicity
    feature_name = "distribution_hydrophobicity"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def distribution_normalized_vdwv(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate distribution descriptors based on NormalizedVDWV of AADs.

    :param G: Protein Graph to featurise
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``distribution_normalized_vdwv`` feature added.
        ``G.graph["distribution_normalized_vdwv_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateDistributionNormalizedVDWV

    func = CalculateDistributionNormalizedVDWV
    feature_name = "distribution_normalised_vdwv"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def distribution_polarity(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate distribution descriptors based on Polarity of AADs.

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``distribution_polarity`` feature added.
        ``G.graph["distribution_polarity_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateDistributionPolarity

    func = CalculateDistributionPolarity
    feature_name = "distribution_polarity"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def distribution_polarizability(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate distribution descriptors based on Polarizability of AADs.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``distribution_polarizability`` feature added.
        ``G.graph["distribution_polarizability_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateDistributionPolarizability

    func = CalculateDistributionPolarizability
    feature_name = "distribution_polarizability"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def distribution_secondary_str(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate distribution descriptors based on SecondaryStr of AADs.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``distribution_secondary_str`` feature added.
        ``G.graph["distribution_secondary_str_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateDistributionSecondaryStr

    func = CalculateDistributionSecondaryStr
    feature_name = "distribution_secondary_str"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def distribution_solvent_accessibility(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate distribution descriptors based on SolventAccessibility of AADs.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``distribution_solvent_accessibility`` feature
        added.
        ``G.graph["distribution_solvent_accessibility_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateDistributionSolventAccessibility

    func = CalculateDistributionSolventAccessibility
    feature_name = "distribution_solvent_accessibility"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def all_transition_descriptors(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Calculate all transition descriptors based on seven different properties of
    AADs.

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``transition_descriptors`` feature added.
        ``G.graph["transition_descriptors_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateC

    func = CalculateC
    feature_name = "transition_descriptors"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def transition_descriptor(
    G: nx.Graph,
    AAProperty: Dict[Any, Any],
    AAPName: str,
    aggregation_type: Optional[List[str]] = None,
) -> nx.Graph:
    """
    Compute transition descriptors.

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param AAProperty: contains classifciation of amino acids such as _Polarizability.
    :type AAProperty: Dict[Any, Any]
    :param AAPName: used for indicating a AAP name.
    :type AAPName: str
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``transition_{AAPName}`` feature added.
        ``G.graph["transition_{AAPName}_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateTransition

    func = partial(CalculateTransition, AAProperty=AAProperty, AAPName=AAPName)
    feature_name = f"transition_{AAPName}"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def transition_charge(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Calculate transition descriptors based on Charge of AADs.

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``transition_charge`` feature added.
        ``G.graph["transition_charge_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateTransitionCharge

    func = CalculateTransitionCharge
    feature_name = "transition_charge"

    G = compute_feature_over_chains(G, func, feature_name="transition_charge")

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def transition_hydrophobicity(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Calculate transition descriptors based on Hydrophobicity of AADs.

    :param G: Protein Graph to featurise
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``transition_hydrophobicity`` feature added.
        ``G.graph["transition_hydrophobicity_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateTransitionHydrophobicity

    func = CalculateTransitionHydrophobicity
    feature_name = "transition_hydrophobicity"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def transition_normalized_vdwv(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate transition descriptors based on NormalizedVDWV of AADs.

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``transition_normalized_vdwv`` feature added.
        ``G.graph["transition_normalized_vdwv_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateTransitionNormalizedVDWV

    func = CalculateTransitionNormalizedVDWV
    feature_name = "transition_normalised_vdwv"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def transition_polarity(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate transition descriptors based on Polarity of AADs.

    :param G: Protein Graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``transition_polarity`` feature added.
        ``G.graph["transition_polarity_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateTransitionPolarity

    func = CalculateTransitionPolarity
    feature_name = "transition_polarity"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def transition_polarizability(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate transition descriptors based on Polarizability of AADs.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``transition_polarizability`` feature added.
        ``G.graph["transition_polarizability_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateTransitionPolarizability

    func = CalculateTransitionPolarizability
    feature_name = "transition_polarizability"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def transition_secondary_str(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate transition descriptors based on SecondaryStr of AADs.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``transition_secondary_str`` feature added.
        ``G.graph["transition_secondary_str_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateTransitionSecondaryStr

    func = CalculateTransitionSecondaryStr
    feature_name = "transition_secondary_str"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def transition_solvent_accessibility(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate transition descriptors based on SolventAccessibility of AADs.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``transition_solvent_accessibility`` feature
        added.
        ``G.graph["transition_solvent_accessibility_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.CTD import CalculateTransitionSolventAccessibility

    func = CalculateTransitionSolventAccessibility
    feature_name = "transition_solvent_accessibility"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


####### Autocorrelation descriptors ########


def autocorrelation_total(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Compute all autocorrelation descriptors based on 8 properties of AADs.
    result contains 30*8*3=720 normalized Moreau Broto, Moran, and Geary
    autocorrelation descriptors.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_total`` feature added.
        ``G.graph["autocorrelation_total_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateAutoTotal

    func = CalculateAutoTotal
    feature_name = "autocorrelation_total"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_geary_all(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Compute Geary autocorrelation descriptors based on 8 properties of AADs.
    Result contains 30*8=240 Geary autocorrelation descriptors based on the
    given properties(i.e., _AAPropert).

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_geary_all`` feature added.
        ``G.graph["autocorrelation_geary_all_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateGearyAutoTotal

    func = CalculateGearyAutoTotal
    feature_name = "autocorrelation_geary_all"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_geary_av_flexibility(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Geary Autocorrelation descriptors based on AvFlexibility.
    contains 30 Geary Autocorrelation descriptors based on AvFlexibility.

    :param G: Protein graph to featurise
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_geary_av_flexibility``
        feature added.
        ``G.graph["autocorrelation_geary_av_flexibility_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateGearyAutoAvFlexibility

    func = CalculateGearyAutoAvFlexibility
    feature_name = "autocorrelation_geary_av_flexibility"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_geary_free_energy(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Geary Autocorrelation descriptors based on FreeEnergy.
    Result contains 30 Geary Autocorrelation descriptors based on FreeEnergy.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_geary_free_energy`` feature
        added.
        ``G.graph["autocorrelation_geary_av_free_energy_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateGearyAutoFreeEnergy

    func = CalculateGearyAutoFreeEnergy
    feature_name = "autocorrelation_geary_free_energy"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_geary_hydrophobicity(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Geary Autocorrelation descriptors based on hydrophobicity.
    result contains 30 Geary Autocorrelation descriptors based on hydrophobicity.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_geary_hydrophobicity`` feature
        added.
        ``G.graph["autocorrelation_geary_hydrophobicity_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateGearyAutoHydrophobicity

    func = CalculateGearyAutoHydrophobicity
    feature_name = "autocorrelation_geary_hydrophobicity"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_geary_mutability(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Geary Autocorrelation descriptors based on Mutability.
    Result contains 30 Geary Autocorrelation descriptors based on mutability.

    :param G: Protein graph to featurise
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_geary_mutability`` feature
        added.
        ``G.graph["autocorrelation_geary_mutability_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateGearyAutoMutability

    func = CalculateGearyAutoMutability
    feature_name = "autocorrelation_geary_mutability"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_geary_polarizability(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Geary Autocorrelation descriptors based on polarizability.
    Result contains 30 Geary Autocorrelation descriptors based on
    polarizability.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_geary_polarizability``
        feature added.
        ``G.graph["autocorrelation_geary_polarizability_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateGearyAutoPolarizability

    func = CalculateGearyAutoPolarizability
    feature_name = "autocorrelation_geary_polarizability"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_geary_residue_asa(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Geary Autocorrelation descriptors based on ResidueASA.
    Result contains 30 Geary Autocorrelation descriptors based on ResidueASA.

    :param G: Protein graph to featurise
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_geary_residue_asa`` feature added.
        ``G.graph["autocorrelation_geary_residue_asa_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateGearyAutoResidueASA

    func = CalculateGearyAutoResidueASA
    feature_name = "autocorrelation_geary_residue_asa"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_geary_residue_vol(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Geary Autocorrelation descriptors based on ResidueVol.
    Result contains 30 Geary Autocorrelation descriptors based on ResidueVol.

    :param G: Protein graph to featurise
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_geary_residue_vol`` feature
        added.
        ``G.graph["autocorrelation_geary_residue_vol_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateGearyAutoResidueVol

    func = CalculateGearyAutoResidueVol
    feature_name = "autocorrelation_geary_residue_vol"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_geary_steric(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Geary Autocorrelation descriptors based on Steric. Result
    contains 30 Geary Autocorrelation descriptors based on Steric

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_geary_steric`` feature added.
        ``G.graph["autocorrelation_geary_steric_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateGearyAutoSteric

    func = CalculateGearyAutoSteric()
    feature_name = "autocorrelation_geary_steric"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_moran_all(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Compute Moran autocorrelation descriptors based on 8 properties of AADs.
    Result contains 30*8=240 Moran autocorrelation descriptors based on the
    given properties(i.e., _AAPropert).

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_moran_all`` feature added.
        ``G.graph["autocorrelation_moran_all_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateMoranAutoTotal

    func = CalculateMoranAutoTotal
    feature_name = "autocorrelation_moran_all"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_moran_av_flexibility(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Moran Autocorrelation descriptors based on AvFlexibility.
    Contains 30 Moran Autocorrelation descriptors based on AvFlexibility.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_moran_av_flexibility``
        feature added.
        ``G.graph["autocorrelation_moran_av_flexibility_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateMoranAutoAvFlexibility

    func = CalculateMoranAutoAvFlexibility
    feature_name = "autocorrelation_moran_av_flexibility"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_moran_free_energy(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Moran Autocorrelation descriptors based on FreeEnergy.
    Result contains 30 Moran Autocorrelation descriptors based on FreeEnergy.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_moran_free_energy`` feature
        added.
        ``G.graph["autocorrelation_moran_av_free_energy_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateMoranAutoFreeEnergy

    func = CalculateMoranAutoFreeEnergy
    feature_name = "autocorrelation_moran_free_energy"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_moran_hydrophobicity(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Moran Autocorrelation descriptors based on hydrophobicity.
    Result contains 30 Moran Autocorrelation descriptors based on hydrophobicity.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_moran_hydrophobicity`` feature
        added.
        ``G.graph["autocorrelation_moran_hydrophobicity_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateMoranAutoHydrophobicity

    func = CalculateMoranAutoHydrophobicity
    feature_name = "autocorrelation_moran_hydrophobicity"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_moran_mutability(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Moran Autocorrelation descriptors based on Mutability.
    Result contains 30 Moran Autocorrelation descriptors based on mutability.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_moran_mutability`` feature
        added.
        ``G.graph["autocorrelation_moran_mutability_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateMoranAutoMutability

    func = CalculateMoranAutoMutability
    feature_name = "autocorrelation_moran_mutability"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_moran_polarizability(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Moran Autocorrelation descriptors based on polarizability.
    Result contains 30 Moran Autocorrelation descriptors based on polarizability.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_moran_polarizability`` feature
        added.
        ``G.graph["autocorrelation_moran_polarizability_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateMoranAutoPolarizability

    func = CalculateMoranAutoPolarizability
    feature_name = "autocorrelation_moran_polarizability"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_moran_residue_asa(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Moran Autocorrelation descriptors based on ResidueASA.
    Result contains 30 Moran Autocorrelation descriptors based on ResidueASA.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_moran_residue_asa`` feature added.
        ``G.graph["autocorrelation_moran_residue_asa_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateMoranAutoResidueASA

    func = CalculateMoranAutoResidueASA
    feature_name = "autocorrelation_moran_residue_asa"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_moran_residue_vol(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Moran Autocorrelation descriptors based on ResidueVol.
    Result contains 30 Moran Autocorrelation descriptors based on ResidueVol.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_moran_residue_vol`` feature
        added.
        ``G.graph["autocorrelation_moran_residue_vol_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateMoranAutoResidueVol

    func = CalculateMoranAutoResidueVol
    feature_name = "autocorrelation_moran_residue_vol"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_moran_steric(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the Moran Autocorrelation descriptors based on Steric. Result
    contains 30 Moran Autocorrelation descriptors based on Steric

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_moran_steric`` feature added.
        ``G.graph["autocorrelation_moran_steric_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateMoranAutoSteric

    func = CalculateMoranAutoSteric()
    feature_name = "autocorrelation_moran_steric"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_normalized_moreau_broto_all(
    G: nx.Graph, aggregation_type: Optional[List[str]] = None
) -> nx.Graph:
    """
    Compute NormalizedMoreauBroto autocorrelation descriptors based on 8
    properties of AADs. Result contains 30*8=240 NormalizedMoreauBroto
    autocorrelation descriptors based on the given properties(i.e., _AAPropert).

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with ``autocorrelation_normalized_moreau_broto_all``
        feature added.
        ``G.graph["autocorrelation_normalized_moreau_broto_all_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateNormalizedMoreauBrotoAutoTotal

    func = CalculateNormalizedMoreauBrotoAutoTotal
    feature_name = "autocorrelation_normalized_moreau_broto_all"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_normalized_moreau_broto_av_flexibility(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the NormalizedMoreauBroto Autocorrelation descriptors based on
    AvFlexibility. contains 30 NormalizedMoreauBroto Autocorrelation descriptors
    based on AvFlexibility.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with
        ``autocorrelation_normalized_moreau_broto_av_flexibility`` feature
        added.
        ``G.graph["autocorrelation_normalized_moreau_broto_av_flexibility_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import (
        CalculateNormalizedMoreauBrotoAutoAvFlexibility,
    )

    func = CalculateNormalizedMoreauBrotoAutoAvFlexibility
    feature_name = "autocorrelation_normalized_moreau_broto_av_flexibility"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_normalized_moreau_broto_free_energy(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the NormalizedMoreauBroto Autocorrelation descriptors based on
    FreeEnergy. Result contains 30 NormalizedMoreauBroto Autocorrelation
    descriptors based on FreeEnergy.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with
        ``autocorrelation_normalized_moreau_broto_free_energy`` feature added.
        ``G.graph["autocorrelation_normalized_moreau_broto_av_free_energy_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import (
        CalculateNormalizedMoreauBrotoAutoFreeEnergy,
    )

    func = CalculateNormalizedMoreauBrotoAutoFreeEnergy
    feature_name = "autocorrelation_normalized_moreau_broto_free_energy"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_normalized_moreau_broto_hydrophobicity(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the NormalizedMoreauBroto Autocorrelation descriptors based on
    hydrophobicity. Result contains 30 NormalizedMoreauBroto Autocorrelation
    descriptors based on hydrophobicity.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with
        ``autocorrelation_normalized_moreau_broto_hydrophobicity`` feature
        added.
        ``G.graph["autocorrelation_normalized_moreau_broto_hydrophobicity_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import (
        CalculateNormalizedMoreauBrotoAutoHydrophobicity,
    )

    func = CalculateNormalizedMoreauBrotoAutoHydrophobicity
    feature_name = "autocorrelation_normalized_moreau_broto_hydrophobicity"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_normalized_moreau_broto_mutability(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the NormalizedMoreauBroto Autocorrelation descriptors based on
    Mutability. Result contains 30 NormalizedMoreauBroto Autocorrelation
    descriptors based on mutability.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with
        ``autocorrelation_normalized_moreau_broto_mutability`` feature added.
        ``G.graph["autocorrelation_normalized_moreau_broto_mutability_{chain | aggregation_type}"]``
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import (
        CalculateNormalizedMoreauBrotoAutoMutability,
    )

    func = CalculateNormalizedMoreauBrotoAutoMutability
    feature_name = "autocorrelation_normalized_moreau_broto_mutability"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_normalized_moreau_broto_polarizability(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the NormalizedMoreauBroto Autocorrelation descriptors based on
    polarizability. Result contains 30 NormalizedMoreauBroto Autocorrelation
    descriptors based on polarizability.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with
        ``autocorrelation_normalized_moreau_broto_polarizability`` feature
        added.
        ``G.graph["autocorrelation_normalized_moreau_broto_polarizability_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import (
        CalculateNormalizedMoreauBrotoAutoPolarizability,
    )

    func = CalculateNormalizedMoreauBrotoAutoPolarizability
    feature_name = "autocorrelation_normalized_moreau_broto_polarizability"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_normalized_moreau_broto_residue_asa(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the NormalizedMoreauBroto Autocorrelation descriptors based on
    ResidueASA. Result contains 30 NormalizedMoreauBroto Autocorrelation
    descriptors based on ResidueASA.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with
        ``autocorrelation_normalized_moreau_broto_residue_asa`` feature added.
        ``G.graph["autocorrelation_normalized_moreau_broto_residue_asa_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import (
        CalculateNormalizedMoreauBrotoAutoResidueASA,
    )

    func = CalculateNormalizedMoreauBrotoAutoResidueASA
    feature_name = "autocorrelation_normalized_moreau_broto_residue_asa"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_normalized_moreau_broto_residue_vol(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the NormalizedMoreauBroto Autocorrelation descriptors based on
    ResidueVol. Result contains 30 NormalizedMoreauBroto Autocorrelation
    descriptors based on ResidueVol.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with
        ``autocorrelation_normalized_moreau_broto_residue_vol`` feature added.
        ``G.graph["autocorrelation_normalized_moreau_broto_residue_vol_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import (
        CalculateNormalizedMoreauBrotoAutoResidueVol,
    )

    func = CalculateNormalizedMoreauBrotoAutoResidueVol
    feature_name = "autocorrelation_normalized_moreau_broto_residue_vol"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


def autocorrelation_normalized_moreau_broto_steric(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    """
    Calculate the NormalizedMoreauBroto Autocorrelation descriptors based on
    Steric. Result contains 30 NormalizedMoreauBroto Autocorrelation descriptors
    based on Steric.

    :param G: Protein graph to featurise.
    :type G: nx.Graph
    :param aggregation_type: Aggregation types to use over chains.
    :type aggregation_type: List[Optional[str]]
    :return: Protein Graph with
        ``autocorrelation_normalized_moreau_broto_steric`` feature added.
        ``G.graph["autocorrelation_normalized_moreau_broto_steric_{chain | aggregation_type}"]``.
    :rtype: nx.Graph
    """
    from propy.Autocorrelation import CalculateNormalizedMoreauBrotoAutoSteric

    func = CalculateNormalizedMoreauBrotoAutoSteric()
    feature_name = "autocorrelation_normalized_moreau_broto_steric"

    return compute_propy_feature(
        G,
        func=func,
        feature_name=feature_name,
        aggregation_type=aggregation_type,
    )


####### Quasi Sequence Order Descriptors ######


def quasi_sequence_order_aa_composition(G: nx.Graph) -> nx.Graph:
    from propy.QuasiSequenceOrder import GetAAComposition

    func = GetAAComposition
    G = compute_feature_over_chains(
        G, func, feature_name="quasi_seq_order_aa_composition"
    )

    return G


def quasi_sequence_order(
    G: nx.Graph, maxlag: int = 30, weight: float = 0.1
) -> nx.Graph:
    """
    Compute quasi-sequence-order descriptors for a given protein.

    Kuo-Chen Chou. Prediction of Protein Subcellar Locations by Incorporating
    Quasi-Sequence-Order Effect. Biochemical and Biophysical Research
    Communications 2000, 278, 477-483.

    :param maxlag: The maximum lag and the
        length of the protein should be larger than maxlag.
    :type maxlag: int, defaults to ``30``.
    :param weight: Weight factor. Please see reference 1 for its choice.
    :type weight: float, defaults to ``0.1``.
    :returns: Protein Graph with ``quasi_sequence_order`` feature added.
    :rtype: nx.Graph
    """
    from propy.QuasiSequenceOrder import GetQuasiSequenceOrder

    func = partial(GetQuasiSequenceOrder, weight=weight, maxlag=maxlag)
    G = compute_feature_over_chains(
        G, func, feature_name="quasi_sequence_order"
    )
    return G


def sequence_order_coupling_number_total(
    G: nx.Graph, maxlag: int = 30
) -> nx.Graph:
    """
    Compute the sequence order coupling numbers from 1 to maxlag for a given
    protein sequence.

    :param G: Protein Graph.
    :type G: nx.Graph
    :param maxlag: The maximum lag and the length of the protein should be
        larger.
    :type maxlag: int, defaults to ``30``.
    :returns: Protein Graph with ``sequence_order_coupling_number_total``
        feature.
    :rtype: nx.Graph
    """
    from propy.QuasiSequenceOrder import GetSequenceOrderCouplingNumberTotal

    func = partial(GetSequenceOrderCouplingNumberTotal, maxlag=maxlag)
    G = compute_feature_over_chains(
        G, func, feature_name="sequence_order_coupling_number_total"
    )

    return G


# TODO feature aggregation
