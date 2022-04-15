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

ALLOWED_BOND_TYPE_TO_CHANNEL: Dict[Chem.BondType, int] = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}
"""Mapping of bondtypes to integer values."""


ALL_BOND_TYPES: List[Chem.BondType] = [
    Chem.BondType.AROMATIC,
    Chem.BondType.DATIVE,
    Chem.BondType.DATIVEL,
    Chem.BondType.DATIVER,
    Chem.BondType.DOUBLE,
    Chem.BondType.FIVEANDHALF,
    Chem.BondType.FOURANDHALF,
    Chem.BondType.HEXTUPLE,
    Chem.BondType.HYDROGEN,
    Chem.BondType.IONIC,
    Chem.BondType.ONEANDHALF,
    Chem.BondType.OTHER,
    Chem.BondType.QUADRUPLE,
    Chem.BondType.QUINTUPLE,
    Chem.BondType.SINGLE,
    Chem.BondType.THREEANDHALF,
    Chem.BondType.THREECENTER,
    Chem.BondType.TRIPLE,
    Chem.BondType.TWOANDHALF,
    Chem.BondType.UNSPECIFIED,
    Chem.BondType.ZERO,
]
"""Vocabulary of all RDkit BondTypes."""


ALL_BOND_TYPES_TO_CHANNEL: Dict[Chem.BondType, int] = {
    Chem.BondType.AROMATIC: 0,
    Chem.BondType.DATIVE: 1,
    Chem.BondType.DATIVEL: 2,
    Chem.BondType.DATIVER: 3,
    Chem.BondType.DOUBLE: 4,
    Chem.BondType.FIVEANDHALF: 5,
    Chem.BondType.FOURANDHALF: 6,
    Chem.BondType.HEXTUPLE: 7,
    Chem.BondType.HYDROGEN: 8,
    Chem.BondType.IONIC: 9,
    Chem.BondType.ONEANDHALF: 10,
    Chem.BondType.OTHER: 11,
    Chem.BondType.QUADRUPLE: 12,
    Chem.BondType.QUINTUPLE: 13,
    Chem.BondType.SINGLE: 14,
    Chem.BondType.THREEANDHALF: 15,
    Chem.BondType.THREECENTER: 16,
    Chem.BondType.TRIPLE: 17,
    Chem.BondType.TWOANDHALF: 18,
    Chem.BondType.UNSPECIFIED: 19,
    Chem.BondType.ZERO: 20,
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
    Chem.ChiralType.CHI_OTHER,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_UNSPECIFIED,
]
"""Vocabulary of all RDKit chiral types."""

CHIRAL_TYPE_TO_CHANNEL: Dict[Chem.rdchem.ChiralType] = {
    Chem.ChiralType.CHI_OTHER: 0,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 1,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: 2,
    Chem.ChiralType.CHI_UNSPECIFIED: 3,
}
"""Vocabulary of all RDKit chiral types mapped to integer values."""
