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

from graphein.utils.utils import import_message

try:
    import rdkit.Chem as Chem
except ImportError:
    import_message("graphein.molecule.atoms", "rdkit", "rdkit", True)


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

ALLOWED_HYBRIDIZATIONS: List[Chem.rdchem.HybridizationType] = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
"""Vocabulary of allowed hybridizations."""

ALLOWED_NUM_H: List[int] = [0, 1, 2, 3, 4]
"""Vocabulary of allowed number of Hydrogens."""

ALLOWED_BOND_TYPES: List[Chem.rdchem.BondType] = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
"""Vocabulary of allowed bondtypes."""

ALLOWED_BOND_TYPE_TO_CHANNEL: Dict[Chem.rdchem.BondType, int] = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
"""Mapping of bondtypes to integer values."""


ALL_BOND_TYPES: List[Chem.rdchem.BondType] = [
    Chem.rdchem.BondType.AROMATIC,
    Chem.rdchem.BondType.DATIVE,
    Chem.rdchem.BondType.DATIVEL,
    Chem.rdchem.BondType.DATIVER,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.FIVEANDAHALF,
    Chem.rdchem.BondType.FOURANDAHALF,
    Chem.rdchem.BondType.HEXTUPLE,
    Chem.rdchem.BondType.HYDROGEN,
    Chem.rdchem.BondType.IONIC,
    Chem.rdchem.BondType.ONEANDAHALF,
    Chem.rdchem.BondType.OTHER,
    Chem.rdchem.BondType.QUADRUPLE,
    Chem.rdchem.BondType.QUINTUPLE,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.THREEANDAHALF,
    Chem.rdchem.BondType.THREECENTER,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.TWOANDAHALF,
    Chem.rdchem.BondType.UNSPECIFIED,
    Chem.rdchem.BondType.ZERO,
]
"""Vocabulary of all RDkit BondTypes."""


ALL_BOND_TYPES_TO_CHANNEL: Dict[Chem.rdchem.BondType, int] = {
    Chem.rdchem.BondType.AROMATIC: 0,
    Chem.rdchem.BondType.DATIVE: 1,
    Chem.rdchem.BondType.DATIVEL: 2,
    Chem.rdchem.BondType.DATIVER: 3,
    Chem.rdchem.BondType.DOUBLE: 4,
    Chem.rdchem.BondType.FIVEANDAHALF: 5,
    Chem.rdchem.BondType.FOURANDAHALF: 6,
    Chem.rdchem.BondType.HEXTUPLE: 7,
    Chem.rdchem.BondType.HYDROGEN: 8,
    Chem.rdchem.BondType.IONIC: 9,
    Chem.rdchem.BondType.ONEANDAHALF: 10,
    Chem.rdchem.BondType.OTHER: 11,
    Chem.rdchem.BondType.QUADRUPLE: 12,
    Chem.rdchem.BondType.QUINTUPLE: 13,
    Chem.rdchem.BondType.SINGLE: 14,
    Chem.rdchem.BondType.THREEANDAHALF: 15,
    Chem.rdchem.BondType.THREECENTER: 16,
    Chem.rdchem.BondType.TRIPLE: 17,
    Chem.rdchem.BondType.TWOANDAHALF: 18,
    Chem.rdchem.BondType.UNSPECIFIED: 19,
    Chem.rdchem.BondType.ZERO: 20,
}
"""Vocabulary of all RDkit BondTypes mapped to integer values."""

ALL_STEREO_TYPES: List[Chem.rdchem.BondStereo] = [
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOTRANS,
    Chem.rdchem.BondStereo.STEREOZ,
]
"""Vocabulary of all RDKit bond stereo types."""

ALL_STEREO_TO_CHANNEL: Dict[Chem.rdchem.BondStereo, int] = {
    Chem.rdchem.BondStereo.STEREOANY: 0,
    Chem.rdchem.BondStereo.STEREOCIS: 1,
    Chem.rdchem.BondStereo.STEREOE: 2,
    Chem.rdchem.BondStereo.STEREONONE: 3,
    Chem.rdchem.BondStereo.STEREOTRANS: 4,
    Chem.rdchem.BondStereo.STEREOZ: 5,
}
"""Vocabulary of all RDKit bond stereo types mapped to integer values."""

CHIRAL_TYPE: List[Chem.rdchem.ChiralType] = [
    Chem.rdchem.ChiralType.CHI_OTHER,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
]
"""Vocabulary of all RDKit chiral types."""

CHIRAL_TYPE_TO_CHANNEL: Dict[Chem.rdchem.ChiralType, int] = {
    Chem.rdchem.ChiralType.CHI_OTHER: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 2,
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 3,
}
"""Vocabulary of all RDKit chiral types mapped to integer values."""
