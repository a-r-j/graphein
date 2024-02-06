"""Functions for featurising Small Molecule Graphs."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Yuanqi Du
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from graphein.molecule.atoms import BASE_ATOMS
from graphein.utils.utils import onek_encoding_unk


def atom_type_one_hot(
    n,
    d: Dict[str, Any],
    return_array: bool = True,
    allowable_set: Optional[List[str]] = None,
) -> np.ndarray:
    """Adds a one-hot encoding of atom types as a node attribute.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data.
    :type d: Dict[str, Any]
    :param return_array: If ``True``, returns a numpy ``np.ndarray`` of one-hot encoding, otherwise returns a ``pd.Series``. Default is ``True``.
    :type return_array: bool
    :param allowable_set: Specifies vocabulary of amino acids. Default is ``None`` (which uses `graphein.molecule.atoms.BASE_ATOMS`).
    :return: One-hot encoding of amino acid types.
    :rtype: Union[pd.Series, np.ndarray]
    """

    if allowable_set is None:
        allowable_set = BASE_ATOMS

    features = onek_encoding_unk(d["element"], allowable_set)

    if return_array:
        features = np.array(features).astype(int)
    else:
        features = pd.Series(features).astype(int)
        features.index = allowable_set

    d["atom_type_one_hot"] = features
    return features


def degree(n: str, d: Dict[str, Any]) -> int:
    """Adds the degree of the node to the node data.

    N.B. this is the degree as defined by RDKit rather than the 'true' degree of the node in the graph. For the latter, use nx.degree()

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :return: Degree of the atom.
    :rtype: int
    """
    degree = d["rdmol_atom"].GetDegree()
    d["degree"] = degree
    return degree


def degree_one_hot(n: str, d, allowable_set: List[int]):
    deg = degree(n, d)
    deg = onek_encoding_unk(deg, allowable_set)
    d["degree"] = deg
    return deg


def total_degree(n: str, d: Dict[str, Any]) -> int:
    """Adds the total degree of the atom to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data.
    :type d: Dict[str, Any]
    :return: Total degree of the atom.
    :rtype: int
    """
    total_degree = d["rdmol_atom"].GetTotalDegree()
    d["total_degree"] = total_degree
    return total_degree


def total_valence(n: str, d: Dict[str, Any]) -> int:
    """Adds the total valence of the atom to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data.
    :type d: Dict[str, Any]
    :return: Total valence of the atom.
    :rtype: int
    """
    total_valence = d["rdmol_atom"].GetTotalValence()
    d["total_valence"] = total_valence
    return total_valence


def explicit_valence(n: str, d: Dict[str, Any]) -> int:
    """Adds explicit valence of the atom to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :return: Explicit valence of the atom.
    :rtype: int
    """
    explicit_valence = d["rdmol_atom"].GetExplicitValence()
    d["explicit_valence"] = explicit_valence
    return explicit_valence


def implicit_valence(n: str, d: Dict[str, Any]) -> int:
    """Adds implicit valence of the atom to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :return: Implicit valence of the atom.
    :rtype: int
    """
    implicit_valence = d["rdmol_atom"].GetImplicitValence()
    d["implicit_valence"] = implicit_valence
    return implicit_valence


def num_implicit_h(n: str, d: Dict[str, Any]) -> int:
    """Adds the number of implicit Hydrogens of the atom to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :return: Number of implicit Hydrogens of the atom.
    :rtype: int
    """
    implicit_hs = d["rdmol_atom"].GetNumImplicitHs()
    d["num_implicit_h"] = implicit_hs
    return implicit_hs


def num_explicit_h(n: str, d: Dict[str, Any]) -> int:
    """Adds the number of explicit Hydrogens of the atom to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :return: Number of explicit Hydrogens of the atom.
    :rtype: int
    """
    num_explicit_h = d["rdmol_atom"].GetNumExplicitHs()
    d["num_explicit_h"] = num_explicit_h
    return num_explicit_h


def total_num_h(n: str, d: Dict[str, Any]) -> int:
    """Adds the total number of Hydrogens of the atom to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :return: Total number of Hydrogens of the atom.
    :rtype: int
    """
    total_num_h = d["rdmol_atom"].GetTotalNumHs()
    d["total_num_h"] = total_num_h
    return total_num_h


def num_radical_electrons(n: str, d: Dict[str, Any]) -> int:
    """Adds the number of radical electrons of the atom to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :return: Number of radical electrons of the atom.
    :rtype: int
    """
    num_radical_electrons = d["rdmol_atom"].GetNumRadicalElectrons()
    d["num_radical_electrons"] = num_radical_electrons
    return num_radical_electrons


def formal_charge(n: str, d: Dict[str, Any]) -> int:
    """Adds the formal charge of the atom to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :return: Formal charge of the atom.
    :rtype: int
    """
    formal_charge = d["rdmol_atom"].GetFormalCharge()
    d["formal_charge"] = formal_charge
    return formal_charge


def hybridization(
    n: str, d: Dict[str, Any]
) -> rdkit.Chem.rdchem.HybridizationType:
    """Adds the hybridization of the atom to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :return: Hybridization of the atom.
    :rtype: rdkit.Chem.rdchem.HybridizationType
    """
    hybridization = d["rdmol_atom"].GetHybridization()
    d["hybridization"] = hybridization
    return hybridization


def is_aromatic(n: str, d: Dict[str, Any]) -> bool:
    """Adds indicator of aromaticity of the atom to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :return: Indicator of aromaticity of the atom.
    :rtype: bool
    """
    aromatic = d["rdmol_atom"].GetIsAromatic()
    d["is_aromatic"] = aromatic
    return aromatic


def is_ring(n: str, d: Dict[str, Any]) -> bool:
    """Adds indicator of ring membership of the atom to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :return: Indicator of ring membership of the atom.
    :rtype: bool
    """
    ring = d["rdmol_atom"].IsInRing()
    d["is_ring"] = ring
    return ring


def is_ring_size(n: str, d: Dict[str, Any], ring_size: int) -> bool:
    """Adds indicator of ring membership of size ``ring_size`` of the atom to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data.
    :type d: Dict[str, Any]
    :param ring_size: The size of the ring to look for.
    :type ring_size: int
    :return: Indicator of ring membership of size ``ring_size`` of the atom.
    :rtype: bool
    """
    ring = d["rdmol_atom"].IsInRingSize(ring_size)
    d[f"is_ring_{ring_size}"] = ring
    return ring


def is_isotope(n: str, d: Dict[str, Any]) -> int:
    """Adds indicator of whether or not the atom is an isotope to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :return: Indicator of whether or not the atom is an isotope.
    :rtype: int
    """
    is_isotope = d["rdmol_atom"].GetIsotope()
    d["is_isotope"] = is_isotope
    return is_isotope


def atomic_mass(n: str, d: Dict[str, Any]) -> float:
    """Adds mass of the atom to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :return: Mass of the atom.
    :rtype: float
    """
    mass = d["rdmol_atom"].GetMass()
    d["mass"] = mass
    return mass


def chiral_tag(n: str, d: Dict[str, Any]) -> rdkit.Chem.rdchem.ChiralType:
    """Adds indicator of atom chirality to the node data.

    :param n: Node name, this is unused and only included for compatibility with the other functions.
    :type n: str
    :param d: Node data
    :type d: Dict[str, Any]
    :return: Indicator of atom chirality.
    :rtype: rdkit.Chem.rdchem.ChiralType
    """
    tag = d["rdmol_atom"].GetChiralTag()
    d["chiral_tag"] = tag
    return tag
