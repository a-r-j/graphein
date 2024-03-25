"""Constants for working with RNA Secondary Structure Graphs."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import Dict, List

RNA_BASES: List[str] = ["A", "U", "G", "C", "I"]
"""List of allowable RNA Bases."""

RNA_BASE_COLORS: Dict[str, str] = {
    "A": "r",
    "U": "b",
    "G": "g",
    "C": "y",
    "I": "m",
}
"""Maps RNA bases (:const:`~graphein.rna.constants.RNA_BASES`) to a colour for visualisations."""

CANONICAL_BASE_PAIRINGS: Dict[str, List[str]] = {
    "A": ["U"],
    "U": ["A"],
    "G": ["C"],
    "C": ["G"],
}
"""Maps standard RNA bases to their canonical base pairings."""

WOBBLE_BASE_PAIRINGS: Dict[str, List[str]] = {
    "A": ["I"],
    "U": ["G", "I"],
    "G": ["U"],
    "C": ["I"],
    "I": ["A", "C", "U"],
}
"""
Maps RNA bases (:const:`~graphein.rna.constants.RNA_BASES`) to their wobble base pairings.
"""

VALID_BASE_PAIRINGS = {
    key: CANONICAL_BASE_PAIRINGS.get(key, [])
    + WOBBLE_BASE_PAIRINGS.get(key, [])
    for key in set(
        list(CANONICAL_BASE_PAIRINGS.keys())
        + list(WOBBLE_BASE_PAIRINGS.keys())
    )
}
"""
Mapping of RNA bases (:const:`~graphein.rna.constants.RNA_BASES`) to their allowable pairings.
Amalgam of :const:`~graphein.rna.constants.CANONICAL_BASE_PAIRINGS` and :const:`~graphein.rna.constants.WOBBLE_BASE_PAIRINGS`.
"""

SIMPLE_DOTBRACKET_NOTATION: List[str] = ["(", ".", ")"]
"""List of characters in simplest dotbracket notation."""

PSEUDOKNOT_OPENING_SYMBOLS: List[str] = ["[", "{", "<"]
"""List of symbols denoting a pseudoknot opening."""

PSEUDOKNOT_CLOSING_SYMBOLS: List[str] = ["]", "}", ">"]
"""List of symbols denoting a pseudoknot closing."""

SUPPORTED_PSEUDOKNOT_NOTATION: List[str] = (
    PSEUDOKNOT_OPENING_SYMBOLS + PSEUDOKNOT_CLOSING_SYMBOLS
)
"""
List of characters denoting pseudoknots in dotbracket notation.
Amalgam of :const:`~graphein.rna.constants.PSEUDOKNOT_OPENING_SYMBOLS` and :const:`~graphein.rna.constants.PSEUDOKNOT_CLOSING_SYMBOLS`.
"""

SUPPORTED_DOTBRACKET_NOTATION = (
    SIMPLE_DOTBRACKET_NOTATION + SUPPORTED_PSEUDOKNOT_NOTATION
)
"""
List of all valid dotbracket symbols.
Amalgamation of :const:`~graphein.rna.constants.SIMPLE_DOTBRACKET_NOTATION` and :const:`~graphein.rna.constants.SUPPORTED_PSEUDOKNOT_NOTATION`.
"""


SS_BOND_TYPES: List[str] = [
    "phosphodiester_bond",
    "base_pairing",
    "pseudoknot",
]
"""List of valid secondary structure bond types."""

RNA_ATOMS: List[str] = [
    "C1'",
    "C2",
    "C2'",
    "C3'",
    "C4",
    "C4'",
    "C5",
    "C5'",
    "C6",
    "C8",
    "N1",
    "N2",
    "N3",
    "N4",
    "N6",
    "N7",
    "N9",
    "O2",
    "O2'",
    "O3'",
    "O4",
    "O4'",
    "O5'",
    "O6",
    "OP1",
    "OP2",
    "P",
]
"""List of valid RNA atoms found in PDB structures."""

PHOSPHORIC_ACID_ATOMS: List[str] = ["P", "OP1", "OP2"]

RIBOSE_ATOMS: List[str] = [
    "C1'",
    "O1'",
    "C2'",
    "O2'",
    "C3'",
    "O3'",
    "C4'",
    "O4'",
    "C5'",
    "O5'",
]

RNA_BACKBONE_ATOMS = RIBOSE_ATOMS + PHOSPHORIC_ACID_ATOMS

RNA_ATOMIC_RADII: Dict[str, float] = {
    "Csb": 0.77,
    "Cres": 0.72,
    "Cdb": 0.67,
    "Nsb": 0.70,
    "Ndb": 0.62,
    "H": 0.37,
    "Osb": 0.67,
    "Odb": 0.60,
    "P": 0.92,
}
"""List of atomic radii (in angstroms) from Heyrovska 2008."""

PHOSPHORIC_ACID_BOND_STATE: Dict[str, str] = {
    "P": "P",
    "OP1": "Osb",  # TODO check this? Maybe incorrect
    "OP2": "Odb",  # TODO check this? Maybe incorrect
}

RIBOSE_BOND_STATE: Dict[str, str] = {
    "C1'": "Cres",
    "O1'": "Osb",
    "C2'": "Cres",
    "O2'": "Osb",
    "C3'": "Cres",
    "O3'": "Osb",
    "C4'": "Cres",
    "O4'": "Osb",
    "C5'": "Cres",
    "O5'": "Osb",
    "H": "H",
}

RNA_BACKBONE_BOND_STATE: Dict[str, str] = {
    **RIBOSE_BOND_STATE,
    **PHOSPHORIC_ACID_BOND_STATE,
}

DEOXYRIBOSE_BOND_STATE: Dict[str, str] = {
    "C1'": "Cres",
    "O1'": "Osb",
    "C2'": "Cres",
    "O2'": "Osb",
    "C3'": "Cres",
    "O3'": "Osb",
    "C4'": "Cres",
    "O4'": "Osb",
    "C5'": "Cres",
    "O5'": "Osb",
    "H": "H",
}

A_ATOMS: List[str] = [
    "N1",
    "C2",
    "N3",
    "C4",
    "C5",
    "C6",
    "N6",
    "N7",
    "C8",
    "N9",
]

C_ATOMS: List[str] = [
    "N1",
    "C2",
    "O2",
    "N3",
    "C4",
    "N4",
    "C5",
    "C6",
]

G_ATOMS: List[str] = [
    "N1",
    "C2",
    "N2",
    "N3",
    "C4",
    "C5",
    "C6",
    "O6",
    "N7",
    "C8",
    "N9",
]

T_ATOMS: List[str] = []

RNA_ATOM_BOND_STATE: Dict[str, Dict[str, str]] = {
    "A": {
        "N1": "Ndb",
        "C2": "Cres",
        "N3": "Ndb",
        "C4": "Cres",
        "C5": "Cdb",
        "C6": "Cres",
        "N6": "Nsb",
        "N7": "Ndb",
        "C8": "Cdb",
        "N9": "Nsb",
        "H": "H",
        **RNA_BACKBONE_BOND_STATE,
    },
    "C": {
        "N1": "Nsb",
        "C2": "Cdb",
        "O2": "Odb",
        "N3": "Ndb",
        "C4": "Cdb",
        "N4": "Nsb",
        "C5": "Cres",
        "C6": "Cdb",
        "H": "H",
        **RNA_BACKBONE_BOND_STATE,
    },
    "G": {
        "N1": "Nsb",
        "C2": "Cdb",
        "N2": "Nsb",
        "N3": "Ndb",
        "C4": "Cdb",
        "C5": "Cres",
        "C6": "Cdb",
        "O6": "Odb",
        "N7": "Ndb",
        "C8": "Cdb",
        "N9": "Nsb",
        "H": "H",
        **RNA_BACKBONE_BOND_STATE,
    },
    "T": {
        "N1": "Nsb",
        "C2": "Cdb",
        "O2": "Odb",
        "N3": "Nsb",
        "C4": "Cdb",
        "O4": "Odb",
        "C5": "Cres",
        "C": "Csb",  # TODO check this? Maybe incorrect
        "C6": "Cdb",
        "H": "H",
        **RNA_BACKBONE_BOND_STATE,
    },
    "U": {
        "N1": "Nsb",
        "C2": "Cdb",
        "O2": "Odb",
        "N3": "Nsb",
        "C4": "Cdb",
        "O4": "Odb",
        "C5": "Cres",
        "C6": "Cdb",
        "H": "H",
        **RNA_BACKBONE_BOND_STATE,
    },
}
"""List of atoms mapped to bondstates from Heyrovska 2008."""
