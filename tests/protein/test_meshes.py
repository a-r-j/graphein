"""Tests for graphein.protein.meshes."""

import pydantic
import pytest

from graphein.protein import meshes
from graphein.protein.config import ProteinMeshConfig
from graphein.utils.pymol import core


@pytest.fixture
def pymol_running(monkeypatch):
    monkeypatch.setattr(meshes.viewer, "_process_is_running", lambda: True)


def test_wait_for_obj_file_returns_when_file_is_found(tmp_path, pymol_running):
    obj_file = tmp_path / "test.obj"
    obj_file.touch()
    meshes.wait_for_obj_file(str(obj_file), max_wait_seconds=1.0)


def test_wait_for_obj_file_raises_timeout(tmp_path, pymol_running):
    with pytest.raises(TimeoutError, match="missing.obj not found"):
        meshes.wait_for_obj_file(
            str(tmp_path / "missing.obj"), max_wait_seconds=0.2
        )


def test_wait_for_obj_file_raises_if_pymol_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(meshes.viewer, "_process_is_running", lambda: False)
    with pytest.raises(RuntimeError, match="PyMOL exited"):
        meshes.wait_for_obj_file(str(tmp_path / "missing.obj"))


def test_molviewer_start_raises_if_pymol_exits(monkeypatch):
    class DeadProcess:
        returncode = 1

        def poll(self):
            return self.returncode

    class UnreachableServer:
        def __init__(self, uri):
            pass

        def bg_color(self, colour):
            raise ConnectionRefusedError

    monkeypatch.setattr(core.subprocess, "Popen", lambda *_: DeadProcess())
    monkeypatch.setattr(core, "Server", UnreachableServer)

    # Skip the pymol-on-PATH check; the process is mocked.
    with pytest.raises(RuntimeError, match="exited with code 1"):
        core.MolViewer.start.__wrapped__(core.MolViewer(), timeout=1.0)


def test_mesh_config_rejects_non_positive_wait():
    with pytest.raises(pydantic.ValidationError):
        ProteinMeshConfig(max_wait_seconds=0)
