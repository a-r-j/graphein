"""
Author: Eric J. Ma
Purpose: This is a set of utility variables and functions that can be used
across the PIN project.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

BACKBONE_ATOMS = ["N", "CA", "C", "O"]

AMINO_ACIDS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

BOND_TYPES = [
    "hydrophobic",
    "disulfide",
    "hbond",
    "ionic",
    "aromatic",
    "aromatic_sulphur",
    "cation_pi",
    "backbone",
    "delaunay",
]

RESI_NAMES = [
    "ALA",
    "ASX",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
    "GLX",
    "UNK",
]

HYDROPHOBIC_RESIS = [
    "ALA",
    "VAL",
    "LEU",
    "ILE",
    "MET",
    "PHE",
    "TRP",
    "PRO",
    "TYR",
]

DISULFIDE_RESIS = ["CYS"]

DISULFIDE_ATOMS = ["SG"]

IONIC_RESIS = ["ARG", "LYS", "HIS", "ASP", "GLU"]

POS_AA = ["HIS", "LYS", "ARG"]

NEG_AA = ["GLU", "ASP"]

AA_RING_ATOMS = dict()
AA_RING_ATOMS["PHE"] = ["CG", "CD", "CE", "CZ"]
AA_RING_ATOMS["TRP"] = ["CD", "CE", "CH", "CZ"]
AA_RING_ATOMS["HIS"] = ["CG", "CD", "CE", "ND", "NE"]
AA_RING_ATOMS["TYR"] = ["CG", "CD", "CE", "CZ"]

AROMATIC_RESIS = ["PHE", "TRP", "HIS", "TYR"]

CATION_PI_RESIS = ["LYS", "ARG", "PHE", "TYR", "TRP"]

CATION_RESIS = ["LYS", "ARG"]

PI_RESIS = ["PHE", "TYR", "TRP"]

SULPHUR_RESIS = ["MET", "CYS"]

ISOELECTRIC_POINTS = {
    "ALA": 6.11,
    "ARG": 10.76,
    "ASN": 10.76,
    "ASP": 2.98,
    "CYS": 5.02,
    "GLU": 3.08,
    "GLN": 5.65,
    "GLY": 6.06,
    "HIS": 7.64,
    "ILE": 6.04,
    "LEU": 6.04,
    "LYS": 9.74,
    "MET": 5.74,
    "PHE": 5.91,
    "PRO": 6.30,
    "SER": 5.68,
    "THR": 5.60,
    "TRP": 5.88,
    "TYR": 5.63,
    "VAL": 6.02,
    "UNK": 7.00,  # unknown so assign neutral
    "ASX": 6.87,  # the average of D and N
    "GLX": 4.35,  # the average of E and Q
}

scaler = StandardScaler()
scaler.fit(np.array([v for v in ISOELECTRIC_POINTS.values()]).reshape(-1, 1))

ISOELECTRIC_POINTS_STD = dict()
for k, v in ISOELECTRIC_POINTS.items():
    ISOELECTRIC_POINTS_STD[k] = scaler.transform(np.array([v]).reshape(-1, 1))

MOLECULAR_WEIGHTS = {
    "ALA": 89.0935,
    "ARG": 174.2017,
    "ASN": 132.1184,
    "ASP": 133.1032,
    "CYS": 121.1590,
    "GLU": 147.1299,
    "GLN": 146.1451,
    "GLY": 75.0669,
    "HIS": 155.1552,
    "ILE": 131.1736,
    "LEU": 131.1736,
    "LYS": 146.1882,
    "MET": 149.2124,
    "PHE": 165.1900,
    "PRO": 115.1310,
    "SER": 105.0930,
    "THR": 119.1197,
    "TRP": 204.2262,
    "TYR": 181.1894,
    "VAL": 117.1469,
    "UNK": 137.1484,  # unknown, therefore assign average of knowns
    "ASX": 132.6108,  # the average of D and N
    "GLX": 146.6375,  # the average of E and Q
}

MOLECULAR_WEIGHTS_STD = dict()

scaler.fit(np.array([v for v in MOLECULAR_WEIGHTS.values()]).reshape(-1, 1))
MOLECULAR_WEIGHTS_STD = dict()
for k, v in MOLECULAR_WEIGHTS.items():
    MOLECULAR_WEIGHTS_STD[k] = scaler.transform(np.array([v]).reshape(-1, 1))
