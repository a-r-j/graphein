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
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger as log

from graphein.protein.config import ProteinMeshConfig
from graphein.utils.dependencies import (
    import_message,
    requires_external_dependencies,
    requires_python_libs,
)
from graphein.utils.pymol import viewer

if TYPE_CHECKING:
    import torch


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
    config: ProteinMeshConfig | None = None,
):
    """
    Configures a PyMol session based on ``config.parse_pymol_commands``. Uses default parameters ``"-cKq"``.

    See: https://pymolwiki.org/index.php/Command_Line_Options

    :param config: :class:`~graphein.protein.config.ProteinMeshConfig` to use. Defaults to ``None`` which uses default config.
    :type config: graphein.protein.config.ProteinMeshConfig
    """
    pymol = viewer

    # If no config is provided, use default
    if config is None:
        config = ProteinMeshConfig()

    # Start PyMol session
    pymol.start(
        [config.pymol_command_line_options], timeout=config.max_wait_seconds
    )
    pymol.delete("all")  # delete all objects from other sessions if necessary.


@requires_external_dependencies("pymol")
def get_obj_file(
    pdb_file: str | None = None,
    pdb_code: str | None = None,
    out_dir: str | None = None,
    config: ProteinMeshConfig | None = None,
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
    pymol = viewer

    # Check inputs
    if not pdb_code and not pdb_file:
        raise ValueError("Please pass either a pdb_file or pdb_code argument")
    if pdb_code and pdb_file:
        raise ValueError(
            "Please pass either a pdb_file or pdb_code argument. Not both"
        )

    if out_dir is None:
        out_dir = "/tmp/"

    if config is None:
        config = ProteinMeshConfig()

    configure_pymol_session(config)

    # Load structure
    pymol.load(pdb_file) if pdb_file else pymol.fetch(pdb_code)
    # Create file_name
    # Absolute, as PyMOL resolves relative paths against its own cwd
    file_name = os.path.abspath(
        os.path.splitext(pdb_file)[0] + ".obj"
        if pdb_file
        else os.path.join(out_dir, pdb_code + ".obj")
    )

    commands = parse_pymol_commands(config)
    run_pymol_commands(commands)

    # Remove stale output so it can't be mistaken for this run's mesh
    if os.path.isfile(file_name):
        os.remove(file_name)

    # Save surface object as .obj
    pymol.do(f"save {file_name}")
    log.debug(f"Saved {file_name}")
    return file_name


def parse_pymol_commands(config: ProteinMeshConfig) -> list[str]:
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
    return config.pymol_commands or []


@requires_external_dependencies("pymol")
def run_pymol_commands(commands: list[str]) -> None:
    """
    Runs Pymol Commands.

    :param commands: List of commands to pass to PyMol.
    :type commands: List[str]
    """
    pymol = viewer

    for c in commands:
        log.debug(c)
        pymol.do(c)


@requires_python_libs("pytorch3d")
def create_mesh(
    pdb_file: str | None = None,
    pdb_code: str | None = None,
    out_dir: str | None = None,
    config: ProteinMeshConfig | None = None,
) -> tuple[torch.FloatTensor, NamedTuple, NamedTuple]:
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
    wait_for_obj_file(obj_file, max_wait_seconds=config.max_wait_seconds)

    verts, faces, aux = load_obj(obj_file)
    return verts, faces, aux


def wait_for_obj_file(
    obj_file: str, max_wait_seconds: float | None = None
) -> None:
    """Waits for PyMOL to finish writing an ``.obj`` mesh file.

    The file is taken as written once it is non-empty and its size is
    unchanged between two checks 0.1 s apart. If the wait times out, the PyMOL process is stopped so that a hung
    session isn't reused.

    :param obj_file: Path PyMOL is saving the mesh to.
    :type obj_file: str
    :param max_wait_seconds: Seconds to wait. ``None`` waits indefinitely.
    :type max_wait_seconds: float | None
    :raises RuntimeError: if the PyMOL process exits before writing the file.
    :raises TimeoutError: if the file is not written after
        ``max_wait_seconds``.
    """

    def size():
        return os.path.getsize(obj_file) if os.path.isfile(obj_file) else 0

    # ponytail: size-stable heuristic, a write stalled >0.1 s mid-file
    # passes. Unverified against real PyMOL whether `do("save ...")` returns
    # only after the save (then this wait is unneeded) or is queued.
    start_time = time.monotonic()
    last_size = 0
    while True:
        current_size = size()
        if current_size and current_size == last_size:
            return
        last_size = current_size
        if not viewer._process_is_running():
            raise RuntimeError(f"PyMOL exited before writing {obj_file}")
        if (
            max_wait_seconds is not None
            and time.monotonic() - start_time >= max_wait_seconds
        ):
            viewer.stop()
            raise TimeoutError(
                f"{obj_file} not written after {max_wait_seconds} seconds"
            )
        time.sleep(0.1)


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
    return verts / scale


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
