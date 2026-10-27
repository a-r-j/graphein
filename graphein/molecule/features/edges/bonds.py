"""Functions for computing atomic features for molecules."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger as log

from graphein.utils.utils import onek_encoding_unk

try:
    import rdkit.Chem.rdchem
except ImportError:
    pass


def add_bond_type(
    u: str, v: str, d: Dict[str, Any]
) -> rdkit.Chem.rdchem.BondType:
    """Adds bond type as an edge feature to the graph.

    :param u: First node in the edge.
    :type u: str
    :param v: Second node in the edge.
    :type v: str
    :param d: Dictionary of edge metadata.
    :type d: Dict[str, Any]
    :return: Returns the bond type.
    :rtype: rdkit.Chem.rdchem.BondType
    """
    if "bond" not in d.keys():
        log.debug(f"No RDKit bond found on edge {u}-{v}")
        d["bond_type"] = None
        return None
    bond_type = d["bond"].GetBondType()
    d["bond_type"] = bond_type
    return bond_type


def bond_is_aromatic(u: str, v: str, d: Dict[str, Any]) -> bool:
    """Adds indicator of aromaticity of a bond to the graph as an edge feature.

    :param u: First node in the edge.
    :type u: str
    :param v: Second node in the edge.
    :type v: str
    :param d: Dictionary of edge metadata.
    :type d: Dict[str, Any]
    :return: Returns indicator of aromaticity of bond.
    :rtype: bool
    """
    if "bond" not in d.keys():
        log.debug(f"No RDKit bond found on edge {u}-{v}")
        d["aromatic"] = None
        return None
    bond_is_aromatic = d["bond"].GetIsAromatic()
    d["_aromatic"] = bond_is_aromatic
    return bond_is_aromatic


def bond_is_conjugated(u: str, v: str, d: Dict[str, Any]) -> bool:
    """Adds indicator of conjugated bond to the graph as an edge feature.

    :param u: First node in the edge.
    :type u: str
    :param v: Second node in the edge.
    :type v: str
    :param d: Dictionary of edge metadata.
    :type d: Dict[str, Any]
    :return: Returns indicator of conjugated bond.
    :rtype: bool
    """
    if "bond" not in d.keys():
        log.debug(f"No RDKit bond found on edge {u}-{v}")
        d["conjugated"] = None
        return None
    bond_is_conjugated = d["bond"].GetIsConjugated()
    d["conjugated"] = bond_is_conjugated
    return bond_is_conjugated


def bond_is_in_ring(u: str, v: str, d: Dict[str, Any]) -> bool:
    """Adds indicator of ring membership to the graph as an edge feature.

    :param u: First node in the edge.
    :type u: str
    :param v: Second node in the edge.
    :type v: str
    :param d: Dictionary of edge metadata.
    :type d: Dict[str, Any]
    :return: Returns indicator of ring membership of bond.
    :rtype: bool
    """
    if "bond" not in d.keys():
        log.debug(f"No RDKit bond found on edge {u}-{v}")
        d["ring"] = None
        return None
    bond_is_in_ring = d["bond"].IsInRing()
    d["ring"] = bond_is_in_ring
    return bond_is_in_ring


def bond_is_in_ring_size(
    u: str, v: str, d: Dict[str, Any], ring_size: int
) -> int:
    """Adds indicator of ring membership of size ``ring_size`` to the graph
    as an edge feature.

    :param u: First node in the edge.
    :type u: str
    :param v: Second node in the edge.
    :type v: str
    :param d: Dictionary of edge metadata.
    :type d: Dict[str, Any]
    :param ring_size: Size of the ring to look for
    :type ring_size: int
    :return: Returns ring size of bond.
    :rtype: int
    """
    if "bond" not in d.keys():
        log.debug(f"No RDKit bond found on edge {u}-{v}")
        d[f"ring_size_{ring_size}"] = None
        return None
    bond_is_in_ring = d["bond"].IsInRingSize(ring_size)
    d[f"ring_size_{ring_size}"] = bond_is_in_ring
    return bond_is_in_ring


def bond_stereo(
    u: str, v: str, d: Dict[str, Any]
) -> rdkit.Chem.rdchem.BondStereo:
    """Adds bond stereo configuration as an edge feature to the graph.

    :param u: First node in the edge.
    :type u: str
    :param v: Second node in the edge.
    :type v: str
    :param d: Dictionary of edge metadata.
    :type d: Dict[str, Any]
    :return: Returns the bond stereo.
    :rtype: rdkit.Chem.rdchem.BondStereo
    """
    if "bond" not in d.keys():
        log.debug(f"No RDKit bond found on edge {u}-{v}")
        d["bond_stereo"] = None
        return None
    bond_stereo = d["bond"].GetStereo()
    d["bond_stereo"] = bond_stereo
    return bond_stereo


def add_bond_type_one_hot(
    u: str,
    v: str,
    d: Dict[str, Any],
    return_array: bool = True,
    allowable_set: Optional[List[Any]] = None,
) -> np.ndarray:
    """Adds a one-hot encoding of the bond type as an edge feature.

    :param u: First node in the edge.
    :type u: str
    :param v: Second node in the edge.
    :type v: str
    :param d: Dictionary of edge metadata.
    :type d: Dict[str, Any]
    :param return_array: If ``True``, returns a ``np.ndarray``, otherwise a
        ``pd.Series`` indexed by the allowable set. Defaults to ``True``.
    :type return_array: bool
    :param allowable_set: Bond types to encode. Defaults to single, double,
        triple and aromatic.
    :type allowable_set: Optional[List[rdkit.Chem.rdchem.BondType]]
    :return: One-hot encoding of the bond type.
    :rtype: Union[np.ndarray, pd.Series]
    """
    if allowable_set is None:
        allowable_set = [
            rdkit.Chem.rdchem.BondType.SINGLE,
            rdkit.Chem.rdchem.BondType.DOUBLE,
            rdkit.Chem.rdchem.BondType.TRIPLE,
            rdkit.Chem.rdchem.BondType.AROMATIC,
        ]

    if "bond" not in d.keys():
        log.debug(f"No RDKit bond found on edge {u}-{v}")
        d["bond_type_one_hot"] = None
        return None

    features = onek_encoding_unk(d["bond"].GetBondType(), allowable_set)

    if return_array:
        features = np.array(features).astype(int)
    else:
        features = pd.Series(features).astype(int)
        features.index = allowable_set

    d["bond_type_one_hot"] = features
    return features
