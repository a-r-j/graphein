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
