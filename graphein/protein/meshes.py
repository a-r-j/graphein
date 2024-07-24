"""Functions to create protein meshes via pymol."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import importlib.util
import os
import time
from typing import List, NamedTuple, Optional, Tuple

from loguru import logger as log

from graphein.protein.config import ProteinMeshConfig
from graphein.utils.dependencies import (
    import_message,
    requires_external_dependencies,
    requires_python_libs,
)
from graphein.utils.pymol import MolViewer

try:
    from pytorch3d.structures import Meshes
except ImportError:
    message = import_message(
        submodule="graphein.protein.meshes",
        package="pytorch3d",
        conda_channel="pytorch3d",
        pip_install=True,
    )
    log.debug(message)


def check_for_pymol_installation():
    """Checks for presence of a pymol installation"""
    spec = importlib.util.find_spec("pymol")
    if spec is None:
        log.error(
            "Please install pymol: conda install -c schrodinger pymol or conda install -c tpeulen pymol-open-source"
        )


def configure_pymol_session(
    config: Optional[ProteinMeshConfig] = None,
):
    """
    Configures a PyMol session based on ``config.parse_pymol_commands``. Uses default parameters ``"-cKq"``.

    See: https://pymolwiki.org/index.php/Command_Line_Options

    :param config: :class:`~graphein.protein.config.ProteinMeshConfig` to use. Defaults to ``None`` which uses default config.
    :type config: graphein.protein.config.ProteinMeshConfig
    """
    pymol = MolViewer()
    pymol.delete("all")  # delete all objects from other sessions if necessary.

    # If no config is provided, use default
    if config is None:
        config = ProteinMeshConfig()

    # Start PyMol session
    pymol.start([config.pymol_command_line_options])


@requires_external_dependencies("pymol")
def get_obj_file(
    pdb_file: Optional[str] = None,
    pdb_code: Optional[str] = None,
    out_dir: Optional[str] = None,
    config: Optional[ProteinMeshConfig] = None,
) -> str:
    """
    Runs PyMol to compute surface/mesh for a given protein.

    :param pdb_file:  path to ``pdb_file`` to use. Defaults to ``None``.
    :type pdb_file: str, optional
    :param pdb_code: 4-letter pdb accession code. Defaults to ``None``.
    :type pdb_code: str, optional
    :param out_dir: path to output. Defaults to ``None``.
    :type out_dir: str, optional
    :param config: :class:`~graphein.protein.config.ProteinMeshConfig` containing pymol commands to run. Default is ``None`` (``"show surface"``).
    :type config: graphein.protein.config.ProteinMeshConfig
    :raises: ValueError if both or neither ``pdb_file`` or ``pdb_code`` are provided.
    :return: returns path to ``.obj`` file (str)
    :rtype: str
    """
    pymol = MolViewer()

    # Check inputs
    if not pdb_code and not pdb_file:
        raise ValueError("Please pass either a pdb_file or pdb_code argument")
    if pdb_code and pdb_file:
        raise ValueError(
            "Please pass either a pdb_file or pdb_code argument. Not both"
        )

    if out_dir is None:
        out_dir = "/tmp/"

    configure_pymol_session()

    # Load structure
    pymol.load(pdb_file) if pdb_file else pymol.fetch(pdb_code)
    # Create file_name
    file_name = (
        f"{pdb_file[:-3]}obj" if pdb_file else out_dir + pdb_code + ".obj"
    )

    if config is None:
        config = ProteinMeshConfig()

    commands = parse_pymol_commands(config)
    print(commands)
    run_pymol_commands(commands)

    # Save surface object as .obj
    pymol.do(f"save {file_name}")
    log.debug(f"Saved {file_name}")
    return file_name


def parse_pymol_commands(config: ProteinMeshConfig) -> List[str]:
    """
    Parses pymol commands from config. At the moment users can only supply a list of string commands.

    :param config: ProteinMeshConfig containing pymol commands to run in ``config.pymol_commands``.
    :type config: ProteinMeshConfig
    :return: list of pymol commands to run
    :rtype: List[str]
    """
    if config is None:
        config = ProteinMeshConfig()

    # Todo parsing of individual pymol mesh calculation parameters. There's a lot of them so this is not a priority now.
    if config.pymol_commands is not None:
        return config.pymol_commands


@requires_external_dependencies("pymol")
def run_pymol_commands(commands: List[str]) -> None:
    """
    Runs Pymol Commands.

    :param commands: List of commands to pass to PyMol.
    :type commands: List[str]
    """
    pymol = MolViewer()

    for c in commands:
        log.debug(c)
        pymol.do(c)


@requires_python_libs("pytorch3d")
def create_mesh(
    pdb_file: Optional[str] = None,
    pdb_code: Optional[str] = None,
    out_dir: Optional[str] = None,
    config: Optional[ProteinMeshConfig] = None,
) -> Tuple[torch.FloatTensor, NamedTuple, NamedTuple]:
    """
    Creates a ``PyTorch3D`` mesh from a ``pdb_file`` or ``pdb_code``.

    :param pdb_file: path to ``pdb_file``. Defaults to ``None``.
    :type pdb_file: str, optional
    :param pdb_code: 4-letter PDB accession code. Defaults to None.
    :type pdb_code: str, optional
    :param out_dir: output directory to store ``.obj`` file. Defaults to ``None``.
    :type out_dir: str, optional
    :param config:  :class:`~graphein.protein.config.ProteinMeshConfig` config to use. Defaults to default config in ``graphein.protein.config``.
    :type config: graphein.protein.config.ProteinMeshConfig
    :return: ``verts``, ``faces``, ``aux``.
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    from pytorch3d.io import load_obj

    if config is None:
        config = ProteinMeshConfig()

    obj_file = get_obj_file(
        pdb_code=pdb_code, pdb_file=pdb_file, out_dir=out_dir, config=config
    )
    # Wait for PyMol to finish
    while os.path.isfile(obj_file) is False:
        time.sleep(0.1)

    verts, faces, aux = load_obj(obj_file)
    return verts, faces, aux


@requires_python_libs("torch")
def normalize_and_center_mesh_vertices(
    verts: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at ``(0,0,0)``.

    ``(scale, center)`` will be used to bring the predicted mesh to its original center and scale
    Note that normalizing the target mesh, speeds up the optimization but is not necessary!

    :param verts: Mesh vertices.
    :type verts: torch.FloatTensor
    :return: Normalized and centered vertices.
    :rtype: torch.FloatTensor
    """
    center = verts.mean()
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale
    return verts


@requires_python_libs("torch", "pytorch3d")
def convert_verts_and_face_to_mesh(
    verts: torch.FloatTensor, faces: NamedTuple
) -> Meshes:
    """
    Converts vertices and faces into a ``pytorch3d.structures`` Meshes object.

    :param verts: Vertices.
    :type verts: torch.FloatTensor
    :param faces: Faces.
    :type faces: NamedTuple
    :return: Meshes object.
    :rtype: pytorch3d.structures.Meshes
    """
    faces_idx = faces.verts_idx
    return Meshes(verts=[verts], faces=[faces_idx])


if __name__ == "__main__":
    from graphein.protein.visualisation import plot_pointcloud

    config = ProteinMeshConfig(
        pymol_commands=[
            "show surface",
            "set surface_solvent, on",
            "set solvent_radius, 10000",
        ]
    )
    verts, faces, aux = create_mesh(pdb_code="3eiy", config=config)

    trg_mesh = convert_verts_and_face_to_mesh(verts, faces)
    plot_pointcloud(trg_mesh)
