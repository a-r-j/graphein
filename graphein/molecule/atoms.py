"""
Author: Eric J. Ma, Arian Jamasb
Purpose: This is a set of utility variables and functions related to small molecules that can be used
across the Graphein project.

These include various collections of standard atom types used molecule-focussed ML
"""
# Graphein
# Author: Eric J. Ma, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from typing import Dict, List

BASE_ATOMS: List[str] = [
    "C",
    "H",
    "O",
    "N",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "B",
]
"""Vocabulary of 11 standard atom types."""

EXTENDED_ATOMS = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "Unknown",
]
"""Vocabulary of additional atom types."""

ALLOWED_DEGREES: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
"""Vocabulary of allowed atom degrees."""

ALLOWED_VALENCES: List[int] = [0, 1, 2, 3, 4, 5, 6]
"""Vocabulary of allowed atom valences."""

ALLOWED_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
"""Vocabulary of allowed hybridizations."""

ALLOWED_NUM_H: List[int] = [0, 1, 2, 3, 4]
"""Vocabulary of allowed number of Hydrogens."""

ALLOWED_BOND_TYPES: List[Chem.BondType] = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]
"""Vocabulary of allowed bondtypes."""

BONDTYPE_TO_CHANNEL: Dict[Chem.BondType, int] = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}
"""Mapping of bondtypes to integer values."""
