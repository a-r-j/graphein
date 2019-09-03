import pandas as pd
import numpy as np
# from sklear
import dgl
import networkx as nx
import torch as torch
from biopandas.pdb import PandasPdb


class ProteinGraph(object):

    def __init__(self, pdb_path, granularity, keep_hets, insertions):
        # self.seq_length =
        # self.mol_wt
        self.dgl_graph = create_dgl_graph(pdb_path, granularity, keep_hets)
        self.nx_graph = create_nx_graph(pdb_path, granularity, keep_hets)

        self.embedding_dict = {
            'meiler': {
                'ALA': [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
                       'GLY': [0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
                       'VAL': [3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
                       'LEU': [2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
                       'ILE': [4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
                       'PHE': [2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
                       'TYR': [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
                       'PTR': [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
                       'TRP': [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
                       'THR': [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
                       'TPO': [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
                       'SER': [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
                       'SEP': [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
                       'ARG': [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
                       'LYS': [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
                       'KCX': [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
                       'LLP': [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
                       'HIS': [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
                       'ASP': [1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
                       'GLU': [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
                       'PCA': [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
                       'ASN': [1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
                       'GLN': [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
                       'MET': [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
                       'MSE': [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
                       'PRO': [2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
                       'CYS': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
                       'CSO': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
                       'CAS': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
                       'CAF': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
                       'CSD': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
                       'UNKNOWN': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
                       },
            'kidera': {
                        'A': [-1.56, -1.67, -0.97, -0.27, -0.93, -0.78, -0.2, -0.08, 0.21, -0.48],
                        'C': [0.12, -0.89, 0.45, -1.05, -0.71, 2.41, 1.52, -0.69, 1.13, 1.1],
                        'E': [-1.45, 0.19, -1.61, 1.17, -1.31, 0.4, 0.04, 0.38, -0.35, -0.12],
                        'D': [0.58, -0.22, -1.58, 0.81, -0.92, 0.15, -1.52, 0.47, 0.76, 0.7],
                        'G': [1.46, -1.96, -0.23, -0.16, 0.1, -0.11, 1.32, 2.36, -1.66, 0.46],
                        'F': [-0.21, 0.98, -0.36, -1.43, 0.22, -0.81, 0.67, 1.1, 1.71, -0.44],
                        'I': [-0.73, -0.16, 1.79, -0.77, -0.54, 0.03, -0.83, 0.51, 0.66, -1.78],
                        'H': [-0.41, 0.52, -0.28, 0.28, 1.61, 1.01, -1.85, 0.47, 1.13, 1.63],
                        'K': [-0.34, 0.82, -0.23, 1.7, 1.54, -1.62, 1.15, -0.08, -0.48, 0.6],
                        'M': [-1.4, 0.18, -0.42, -0.73, 2.0, 1.52, 0.26, 0.11, -1.27, 0.27],
                        'L': [-1.04, 0.0, -0.24, -1.1, -0.55, -2.05, 0.96, -0.76, 0.45, 0.93],
                        'N': [1.14, -0.07, -0.12, 0.81, 0.18, 0.37, -0.09, 1.23, 1.1, -1.73],
                        'Q': [-0.47, 0.24, 0.07, 1.1, 1.1, 0.59, 0.84, -0.71, -0.03, -2.33],
                        'P': [2.06, -0.33, -1.15, -0.75, 0.88, -0.45, 0.3, -2.3, 0.74, -0.28],
                        'S': [0.81, -1.08, 0.16, 0.42, -0.21, -0.43, -1.89, -1.15, -0.97, -0.23],
                        'R': [0.22, 1.27, 1.37, 1.87, -1.7, 0.46, 0.92, -0.39, 0.23, 0.93],
                        'T': [0.26, -0.7, 1.21, 0.63, -0.1, 0.21, 0.24, -1.15, -0.56, 0.19],
                        'W': [0.3, 2.1, -0.72, -1.57, -1.16, 0.57, -0.48, -0.4, -2.3, -0.6],
                        'V': [-0.74, -0.71, 2.04, -0.4, 0.5, -0.81, -1.07, 0.06, -0.46, 0.65],
                        'Y': [1.38, 1.48, 0.8, -0.56, -0.0, -0.68, -0.31, 1.03, -0.05, 0.53],
                        'UNKNOWN': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
                       }
        }

    def create_dgl_graph(self, pdb_path, granularity, keep_hets=False):
        df = self.parse_pdb(pdb_path, granularity, insertions, keep_hets)
        df = self.get_chains(df)
        df = pd.concat(df)
        g = self.add_protein_nodes(df)
        e = self.get_protein_edges(pdb_path, granularity)
        g = self.add_edges(g, e)

        return dgl_graph

    def create_nx_graph(self, pdb_path, granularity, insertions, keep_hets):

        return nx_graph

    @staticmethod
    def parse_pdb(pdb_path, granularity, insertions, keep_hets):
        protein = PandasPdb().read_pdb(pdb_path)
        atoms = protein.df['ATOM']
        hetatms = protein.df['HETATM']

        if granularity != 'atom':
            atoms = atoms.loc[atoms['atom_name'] == granularity]

        if keep_hets:
            protein_df = pd.concat([atoms, hetatms])
        else:
            protein_df = atoms

        return protein_df

    @staticmethod
    def get_chains(protein_df):
        '''
        Args:
            protein_df (df): pandas dataframe of PDB subsetted to CA atoms
        Returns:
            chains (list): list of dataframes corresponding to each chain in protein
        '''
        chains = [protein_df.loc[protein_df['chain_id'] == chain] for chain in protein_df['chain_id'].unique()]
        return chains

    def add_protein_nodes(self, chain, granularity, embedding):
        '''
        Input:
            chain (list of dataframes): Contains a dataframe for each chain in the protein
        Output:
            g (DGLGraph): Graph of protein only populated by the nodes
        '''
        g = dgl.DGLGraph()

        residues = chain['chain_id'] + ':' + chain['residue_name'] + ':' + chain['residue_number'].apply(str)

        if granularity == 'atom':
            atoms = residues + ':' + chain['atom_name']

        if embedding:
            embedding = [self.aa_features(residue, embedding) for residue in chain['residue_name']]

            g.add_nodes(len(residues),
                        {'residue_id': residues,
                         'residue_name': chain['residue_name'],
                         'h': torch.stack(embedding).type('torch.FloatTensor')
                         })
        else:
            g.add_nodes(len(residues),
                        {'atom_id': atoms,
                         'node_name': chain['residue_name']
                         })
        return g

    def aa_features(self, residue, embedding):
        if residue not in self.embedding_dict[embedding].keys():
            residue = 'UNKNOWN'
        features = torch.Tensor(self.embedding_dict[embedding][residue]).double()
        return features
