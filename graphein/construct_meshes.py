"""Class for creating Protein Meshes"""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website:
# Code Repository: https://github.com/a-r-j/graphein

from pytorch3d.io import load_obj, save_obj
from ipymol import viewer as pymol
from typing import Any, Dict, NamedTuple, List, Optional, Union


class ProteinMesh(object):
    def __init__(self) -> None:
        """
        Initialise ProteinGraph Generator Class
        """

    def get_obj_file(
        self,
        out_dir: str,
        pdb_file: Optional[str] = None,
        pdb_code: Optional[str] = None,
    ) -> str:
        """
        Produces .Obj file from PDB structure through IPyMol. pdb_code and pdb_file are optional arguments. Use one as suits your purposes

        :param pdb_file: Path to local .PDB file
        :type pdb_file: str
        :param pdb_code: 4 character PDB accession code
        :type pdb_code: str
        :param out_dir: Path to output directory
        :type out_dir: str
        :return:
        """
        file_name = "a"
        pymol.start()
        if not pdb_code and not pdb_file:
            print("Please pass a pdb_file or pdb_code argument")
        print(pdb_file)
        if pdb_code and pdb_file:
            print("Do not pass both a PDB code and PDB file. Choose one.")

        if pdb_file:
            pymol.load(pdb_file)
            file_name = pdb_file[:-3] + ".obj"

        if pdb_code:
            pymol.fetch(pdb_code)
            file_name = out_dir + pdb_code + ".obj"
        pymol.do("show_as surface")
        pymol.do(f"save {file_name}")
        print(f"Saved file to: {file_name}")
        return file_name

    def create_mesh(
        self, pdb_code: str = None, pdb_file: str = None, out_dir: str = None
    ):
        """
        #-> Tuple[torch.Tensor, _Face, _Aux]
        Creates a PyTorch3D Mesh from an .Obj file. pdb_code and pdb_file are optional arguments. Use one as suits your purposes

        :param pdb_code: 4-character PDB accession code
        :type pdb_code: str
        :param pdb_file: Path to local .PDB file
        :type pdb_file: str
        :param out_dir: Path to output directory
        :type out_dir: str
        :return: verts, faces, aux
        """
        obj_file = self.get_obj_file(
            pdb_code=pdb_code, pdb_file=pdb_file, out_dir=out_dir
        )
        verts, faces, aux = load_obj(obj_file)
        return verts, faces, aux


if __name__ == "__main__":
    p = ProteinMesh()
    print(p.create_mesh(pdb_code="3eiy", out_dir="../examples/meshes/"))
    # print(p.create_mesh(obj_file='../examples/meshes/3eiy.obj'))
