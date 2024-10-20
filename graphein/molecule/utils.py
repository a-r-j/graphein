"""Utility Functions for working with Small Molecule Graphs in RDKit.

Many of these utilities are adapted from useful_rdkit_utils (https://github.com/PatWalters/useful_rdkit_utils) by Pat Walters.

Junction tree code adapted from Wengong Jin https://github.com/wengong-jin/icml18-jtnn
"""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT

# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from __future__ import annotations

import collections
from typing import Any, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from graphein.utils.dependencies import import_message, requires_python_libs

try:
    import rdkit
    from rdkit import Chem as Chem
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem.Descriptors import (
        TPSA,
        MolLogP,
        MolWt,
        NumHAcceptors,
        NumHDonors,
    )
    from rdkit.Chem.Descriptors3D import NPR1, NPR2
    from rdkit.Chem.rdMolTransforms import ComputeCentroid
except ImportError:
    import_message(
        "graphein.molecule.utils", "rdkit", "rdkit", True, extras=True
    )

try:
    import selfies as sf
except ImportError:
    import_message(
        "graphein.molecule.utils", "selfies", None, True, extras=True
    )


MST_MAX_WEIGHT: int = 100
MAX_NCAND: int = 2000


@requires_python_libs("rdkit")
def get_center(
    mol: Union[nx.Graph, Chem.Mol], weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute the centroid of the conformation.

    Hydrogens are ignored and no attention is paid to the difference in sizes
    of the heavy atoms; however, an optional vector of weights can be passed.

    :param mol: Molecular Graph or RDkit Mol to compute the center of.
    :type mol: Union[nx.Graph, Chem.Mol]
    :return: The centroid of the molecule.
    :rtype: np.ndarray
    """
    if isinstance(mol, nx.Graph):
        mol = mol.graph["rdmol"]
    assert (
        mol.GetNumConformers() > 0
    ), "Molecule must have at least one conformer."
    return np.array(ComputeCentroid(mol.GetConformer(0), weights=weights))


@requires_python_libs("rdkit")
def get_shape_moments(mol: Union[nx.Graph, Chem.Mol]) -> Tuple[float, float]:
    """Calculate principal moments of inertia as defined in https://pubs.acs.org/doi/10.1021/ci025599w

    :param mol: Molecular Graph or RDkit Mol to compute the moments of intertia of.
    :type mol: Union[nx.Graph, Chem.Mol]
    :return: First 2 moments as a tuple.
    :rtype: Tuple[float, float]
    """
    if isinstance(mol, nx.Graph):
        mol = mol.graph["rdmol"]
    assert (
        mol.GetNumConformers() > 0
    ), "Molecule must have at least one conformer."
    npr1 = NPR1(mol)
    npr2 = NPR2(mol)
    return npr1, npr2


@requires_python_libs("rdkit")
def count_fragments(mol: Union[nx.Graph, Chem.Mol]) -> int:
    """Counts the number of the disconnected fragments in a molecule.

    :param mol: The molecule or molecular graph.
    :type mol: Union[nx.Graph, Chem.Mol]
    :return: Number of fragments.
    :rtype: int
    """
    if isinstance(mol, nx.Graph):
        mol = mol.graph["rdmol"]
    return len(Chem.GetMolFrags(mol, asMols=True))


@requires_python_libs("rdkit")
def get_max_ring_size(mol: Union[nx.Graph, Chem.Mol]) -> int:
    """
    Get the size of the largest ring in a molecule.

    :param mol: Input Mol or molecular graph.
    :type mol: Union[nx.Graph, Chem.Mol]
    :return: size of the largest ring or 0 for an acyclic molecule.
    :rtype: int
    """
    if isinstance(mol, nx.Graph):
        mol = mol.graph["rdmol"]

    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    return 0 if len(atom_rings) == 0 else max(len(x) for x in ri.AtomRings())


@requires_python_libs("rdkit")
def label_rdmol_atoms(
    mol: Union[nx.Graph, Chem.Mol], labels: List[Any]
) -> Union[nx.Graph, Chem.Mol]:
    """Sets an atomNote label on each atom in the underlying RDKit Mol.

    :param mol: RDkit Mol or molecular graph containing a mol.
    :type mol: Union[nx.Graph, Chem.Mol]
    :param labels: A list of labels to set on each atom.
    :type labels: _type_
    :return: _description_
    :rtype: Union[nx.Graph, Chem.Mol]
    """
    if isinstance(mol, nx.Graph):
        mol = mol.graph["rdmol"]

    [atm.SetProp("atomNote", "") for atm in mol.GetAtoms()]
    for atm in mol.GetAtoms():
        idx = atm.GetIdx()
        mol.GetAtomWithIdx(idx).SetProp("atomNote", f"{labels[idx]}")
    return mol


@requires_python_libs("rdkit")
def tag_rdmol_atoms(
    mol, atoms_to_tag, tag: str = "x"
) -> Union[nx.Graph, Chem.Mol]:
    if isinstance(mol, nx.Graph):
        mol = mol.graph["rdmol"]
    [atm.SetProp("atomNote", "") for atm in mol.GetAtoms()]
    [mol.GetAtomWithIdx(idx).SetProp("atomNote", tag) for idx in atoms_to_tag]
    return mol


@requires_python_libs("rdkit")
def get_mol(smiles: str) -> rdkit.Chem.rdchem.Mol:
    """
    Function for getting rdmol from smiles. Applies kekulization.

    :param smiles: smiles string to get
    :type x: str
    :return: rdmol
    :rtype: rdkit.Chem.rdchem.Mol
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


@requires_python_libs("rdkit")
def get_smiles(mol: Union[nx.Graph, rdkit.Chem.rdchem.Mol]) -> str:
    """
    Function for getting smiles from rdmol. Applies kekulization.

    :param mol: rdmol to get
    :type mol: Union[rdkit.Chem.rdchem.Mol, nx.Graph]
    :return: smiles string
    :rtype: str
    """
    if isinstance(mol, nx.Graph):
        mol = mol.graph["rdmol"]
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


@requires_python_libs("rdkit")
def sanitize(mol: rdkit.Chem.rdchem.Mol) -> rdkit.Chem.rdchem.Mol:
    """
    Function for sanitizing a rdmol

    :param mol: rdmol to sanitize
    :type mol: rdkit.Chem.rdchem.Mol
    :return: sanitized rdmol
    :rtype: rdkit.Chem.rdchem.Mol
    """
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol


@requires_python_libs("rdkit")
def copy_edit_mol(mol: rdkit.Chem.rdchem.Mol) -> rdkit.Chem.rdchem.Mol:
    """
    Function for copying a rdmol

    :param mol: rdmol to copy
    :type mol: rdkit.Chem.rdchem.Mol
    :return: copied rdmol
    :rtype: rdkit.Chem.rdchem.Mol
    """
    new_mol = Chem.RWMol(Chem.MolFromSmiles(""))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


@requires_python_libs("rdkit")
def get_clique_mol(mol: rdkit.Chem.rdchem.Atom, atoms: List[int]):
    """
    Function for getting clique rdmol

    :param mol: rdmol to get clique
    :type mol: rdkit.Chem.rdchem.Mol
    :param atoms: atoms to cut
    :type atoms: List[int]
    :return: clique rdmol
    :rtype: rdkit.Chem.rdchem.Mol
    """
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    return new_mol


@requires_python_libs("rdkit")
def copy_rdmol_atom(atom: rdkit.Chem.rdchem.Atom) -> rdkit.Chem.rdchem.Atom:
    """
    Function for copying an atom

    :param mol: atom to copy
    :type mol: rdkit.Chem.rdchem.Atom
    :return: copied atom
    :rtype: rdkit.Chem.rdchem.Atom
    """
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


@requires_python_libs("rdkit")
def get_morgan_fp(
    mol: Union[nx.Graph, rdkit.Chem.rdchem.Mol],
    radius: int = 2,
    n_bits: int = 20,
) -> rdkit.DataStructs.cDataStructs.ExplicitBitVect:
    """
    Function for getting morgan fingerprint from an RDkit molecule or molecular graph.

    :param mol: RDKit molecule or molecular graph.
    :type mol: Union[nx.Graph, rdkit.Chem.rdchem.Mol]
    :param radius: Fingerprint radius.
    :type radius: int
    :param n_bits: Number of bits.
    :type n_bits: int
    :return: Morgan fingerprint.
    :rtype: rdkit.Chem.rdMolDescriptors.MorganFingerprint
    """
    if isinstance(mol, nx.Graph):
        mol = mol.graph["rdmol"]
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits
    )


@requires_python_libs("rdkit")
def get_morgan_fp_np(
    mol: Union[nx.Graph, rdkit.Chem.rdchem.Mol],
    radius: int = 2,
    n_bits: int = 20,
) -> rdkit.Chem.rdMolDescriptors.MorganFingerprint:
    """
    Function for getting morgan fingerprint from an RDkit molecule or molecular graph.

    :param mol: RDKit molecule or molecular graph.
    :type mol: Union[nx.Graph, rdkit.Chem.rdchem.Mol]
    :param radius: Fingerprint radius.
    :type radius: int
    :param n_bits: Number of bits.
    :type n_bits: int
    :return: Morgan fingerprint.
    :rtype: rdkit.Chem.rdMolDescriptors.MorganFingerprint
    """
    if isinstance(mol, nx.Graph):
        mol = mol.graph["rdmol"]

    arr = np.zeros((0,), dtype=np.int8)
    fp = get_morgan_fp(mol=mol, radius=radius, n_bits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


@requires_python_libs("rdkit")
def compute_fragments(mol: Union[nx.Graph, Chem.Mol]) -> List[Chem.Mol]:
    if isinstance(mol, nx.Graph):
        mol = mol.graph["rdmol"]
    return list(Chem.GetMolFrags(mol, asMols=True))


@requires_python_libs("rdkit")
def get_mol_weight(mol: Union[nx.Graph, Chem.Mol]) -> float:
    if isinstance(mol, nx.Graph):
        mol = mol.graph["rdmol"]
    return mol  # TDOO


@requires_python_libs("rdkit")
def get_qed_score(
    mol: Union[nx.Graph, rdkit.Chem.rdchem.Mol]
) -> Union[float, None]:
    """Computes the Quantitative Estimate of Druglikeness (QED) score for a molecule or molecular graph.

        Quantifying the chemical beauty of drugs
        G. Richard Bickerton, Gaia V. Paolini, Jérémy Besnard, Sorel Muresan & Andrew L. Hopkins
        Nature Chemistry volume 4, pages90–98 (2012

    :param mol: Molecule or molecular graph.
    :type mol: Union[nx.Graph, rdkit.Chem.rdchem.Mol]
    :return: QED score.
    :rtype: float or None if an exception is encountered
    """
    if isinstance(mol, nx.Graph):
        mol = mol.graph["rdmol"]

    try:
        return Chem.QED.qed(mol)
    except Exception as e:
        # log.warning(e)
        return None


def simplify_smile(smile: str) -> str:
    """
    Simplifies a SMILE string by removing hydrogen atoms (``H``),
    chiral specifications (``'@'``), charges (``+`` / ``-``), ``'#'``-characters,
    and square brackets (``'['``, ``']'``).

    :param smile_str: A smile string, e.g., ``C[C@H](CCC(=O)NCCS(=O)(=O)[O-])``
    :type smile_str: str
    :returns: Returns a simplified SMILE string, e.g., ``CC(CCC(=O)NCCS(=O)(=O)O)``.
    :rtype: str
    """
    remove_chars = ["@", "-", "+", "H", "[", "]", "#"]
    stripped_smile = []
    for sym in smile:
        if sym.isalpha():
            sym = sym.upper()
        if sym not in remove_chars:
            stripped_smile.append(sym)
    return "".join(stripped_smile)


@requires_python_libs("selfies")
def smile_to_selfies(smile: str) -> str:
    """Encodes a SMILES string into a Selfies string.

    :param smile: A valid SMILES string. E.g. ``"C1=CC=CC=C1"``.
    :type smile: str
    :return: Selfies string. E.g. ``"[C][=C][C][=C][C][=C][Ring1][=Branch1]"``.
    :rtype: str
    """
    return sf.encoder(smile)


@requires_python_libs("selfies")
def selfies_to_smile(selfie: str) -> str:
    """Decodes a selfies string into a SMILES string.

    :param selfie: The selfies string to decode. E.g. ``"[C][=C][C][=C][C][=C][Ring1][=Branch1]"``.
    :type selfie: str
    :return: The decoded SMILES string. E.g. ``"C1=CC=CC=C1"``.
    :rtype: str
    """
    return sf.decoder(selfie)


@requires_python_libs("rdkit")
def tree_decomp(mol: rdkit.Chem.rdchem.Mol) -> Tuple[List]:
    """
    Function for decomposing rdmol to a tree

    :param mol: rdmol to decompose
    :type mol: rdkit.Chem.rdchem.Mol
    :return: decomposed cliques and edges between them
    :rtype: Tuple[List]
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:  # special case
        return [[0]], []

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for _ in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Merge Rings with intersection > 2 atoms
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2:
            continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2:
                    continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []

    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for _ in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Build edges and add singleton cliques
    edges = collections.defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1:
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]
        if len(bonds) > 2 or (
            len(bonds) == 2 and len(cnei) > 2
        ):  # In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = 1
        elif len(rings) > 2:  # Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = MST_MAX_WEIGHT - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    edges[(c1, c2)] = max(edges[(c1, c2)], len(inter))
    edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]
    if not edges:
        return cliques, edges

    # Compute Maximum Spanning Tree
    row, col, data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]
    return (cliques, edges)
