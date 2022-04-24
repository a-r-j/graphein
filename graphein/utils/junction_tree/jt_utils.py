"""Utilities for working with graph objects."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# TODO: code adapted from Wengong Jin https://github.com/wengong-jin/icml18-jtnn
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import os
import platform
import subprocess
import sys
import collections
from typing import Any, Callable, Iterable, List, Optional, Tuple

import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import rdkit
import rdkit.Chem as Chem

MST_MAX_WEIGHT = 100 
MAX_NCAND = 2000

def get_mol(smiles: str) -> rdkit.Chem.rdchem.Mol:
    """
    Function for getting rdmol from smiles

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

def get_smiles(mol: rdkit.Chem.rdchem.Mol) -> str:
    """
    Function for getting smiles from rdmol

    :param mol: rdmol to get
    :type mol: rdkit.Chem.rdchem.Mol
    :return: smiles string
    :rtype: str
    """
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

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

def copy_edit_mol(mol: rdkit.Chem.rdchem.Mol) -> rdkit.Chem.rdchem.Mol:
    """
    Function for copying a rdmol

    :param mol: rdmol to copy
    :type mol: rdkit.Chem.rdchem.Mol
    :return: copied rdmol
    :rtype: rdkit.Chem.rdchem.Mol
    """
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

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
    new_mol = sanitize(new_mol) #We assume this is not None
    return new_mol

def copy_atom(atom: rdkit.Chem.rdchem.Atom) -> rdkit.Chem.rdchem.Atom:
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

def tree_decomp(mol: rdkit.Chem.rdchem.Mol) -> Tuple[List]:
    """
    Function for decomposing rdmol to a tree

    :param mol: rdmol to decompose
    :type mol: rdkit.Chem.rdchem.Mol
    :return: decomposed cliques and edges between them
    :rtype: Tuple[List]
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1: #special case
        return [[0]], []

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1,a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    
    #Merge Rings with intersection > 2 atoms
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2: continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2: continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []
    
    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    
    #Build edges and add singleton cliques
    edges = collections.defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1: 
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): #In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = 1
        elif len(rings) > 2: #Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = MST_MAX_WEIGHT - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1,c2 = cnei[i],cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1,c2)] < len(inter):
                        edges[(c1,c2)] = len(inter) #cnei[i] < cnei[j] by construction

    edges = [u + (MST_MAX_WEIGHT-v,) for u,v in edges.items()]
    if len(edges) == 0:
        return cliques, edges

    #Compute Maximum Spanning Tree
    row,col,data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
    junc_tree = minimum_spanning_tree(clique_graph)
    row,col = junc_tree.nonzero()
    edges = [(row[i],col[i]) for i in range(len(row))]
    return (cliques, edges)

