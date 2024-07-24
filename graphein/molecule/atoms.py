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

from loguru import logger

from graphein.utils.dependencies import import_message

try:
    import rdkit.Chem as Chem
except (ImportError, ModuleNotFoundError):
    logger.warning(
        import_message(__name__, "rdkit", "rdkit", True, extras=True)
    )

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


RDKIT_MOL_DESCRIPTORS: List[str] = [
    "MaxEStateIndex",
    "MinEStateIndex",
    "MaxAbsEStateIndex",
    "MinAbsEStateIndex",
    "qed",
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "NumValenceElectrons",
    "NumRadicalElectrons",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "BCUT2D_MWHI",
    "BCUT2D_MWLOW",
    "BCUT2D_CHGHI",
    "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI",
    "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",
    "BCUT2D_MRLOW",
    "BalabanJ",
    "BertzCT",
    "Chi0",
    "Chi0n",
    "Chi0v",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi3v",
    "Chi4n",
    "Chi4v",
    "HallKierAlpha",
    "Ipc",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "PEOE_VSA1",
    "PEOE_VSA10",
    "PEOE_VSA11",
    "PEOE_VSA12",
    "PEOE_VSA13",
    "PEOE_VSA14",
    "PEOE_VSA2",
    "PEOE_VSA3",
    "PEOE_VSA4",
    "PEOE_VSA5",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "SMR_VSA1",
    "SMR_VSA10",
    "SMR_VSA2",
    "SMR_VSA3",
    "SMR_VSA4",
    "SMR_VSA5",
    "SMR_VSA6",
    "SMR_VSA7",
    "SMR_VSA8",
    "SMR_VSA9",
    "SlogP_VSA1",
    "SlogP_VSA10",
    "SlogP_VSA11",
    "SlogP_VSA12",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA6",
    "SlogP_VSA7",
    "SlogP_VSA8",
    "SlogP_VSA9",
    "TPSA",
    "EState_VSA1",
    "EState_VSA10",
    "EState_VSA11",
    "EState_VSA2",
    "EState_VSA3",
    "EState_VSA4",
    "EState_VSA5",
    "EState_VSA6",
    "EState_VSA7",
    "EState_VSA8",
    "EState_VSA9",
    "VSA_EState1",
    "VSA_EState10",
    "VSA_EState2",
    "VSA_EState3",
    "VSA_EState4",
    "VSA_EState5",
    "VSA_EState6",
    "VSA_EState7",
    "VSA_EState8",
    "VSA_EState9",
    "FractionCSP3",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "RingCount",
    "MolLogP",
    "MolMR",
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_HOCCN",
    "fr_Imine",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_piperdine",
    "fr_piperzine",
    "fr_priamide",
    "fr_prisulfonamd",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfide",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiocyan",
    "fr_thiophene",
    "fr_unbrch_alkane",
    "fr_urea",
]
"""Vocabulary of easy-to-compute RDKit molecule descriptors"""
