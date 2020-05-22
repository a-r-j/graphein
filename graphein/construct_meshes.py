"""Class for creating Protein Meshes"""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: BSD 3 clause
# Project Website:
# Code Repository: https://github.com/a-r-j/graphein

import os
import glob
import re
import pandas as pd
import numpy as np
import dgl
import subprocess
import networkx as nx
import torch as torch
import torch.nn.functional as F
from biopandas.pdb import PandasPdb
from Bio.PDB import *
from Bio.PDB.DSSP import residue_max_acc
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.PDB.Polypeptide import aa1
from Bio.PDB.Polypeptide import one_to_three
from dgl.data.chem import mol_to_graph
import dgl.data.chem
from rdkit.Chem import MolFromPDBFile
from vmd import molecule
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

from ipymol import viewer as pymol


class ProteinMesh(object):

    def __init__(self):
        """
        Initialise ProteinGraph Generator Class
        """

    def get_obj_file(self, pdb_file=None, pdb_code=None, out_dir=None):
        file_name = "a"
        pymol.start()
        if not pdb_code and not pdb_file:
            print('Please pass a pdb_file or pdb_code argument')
        print(pdb_file)
        if pdb_code and pdb_file:
            print('Do not pass both a PDB code and PDB file. Choose one.')

        if not pdb_code:
            pymol.load(pdb_file)
            file_name = pdb_file[:-3] + '.obj'

        if not pdb_file:
            pymol.fetch(pdb_code)
            file_name = out_dir + pdb_code + '.obj'
        pymol.do('show_as surface')
        pymol.do(f'save {file_name}')
        print(f'Saved file to: {file_name}')
        return file_name

    def create_mesh(self, pdb_code, out_dir):
        obj_file = self.get_obj_file(pdb_code=pdb_code, out_dir=out_dir)
        verts, faces, aux = load_obj(obj_file)
        return verts, faces, aux


if __name__ == "__main__":
    p = ProteinMesh()
    print(p.create_mesh(pdb_code='3ACG', out_dir='../examples/meshes/'))

#pymol -R #-cKRQ