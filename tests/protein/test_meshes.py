"""Tests for graphein.protein.meshes."""

import socket
import subprocess

import pydantic
import pytest

from graphein.protein import meshes
from graphein.protein.config import ProteinMeshConfig
from graphein.protein.meshes import convert_verts_and_face_to_mesh, create_mesh
from graphein.utils.pymol import core


@pytest.fixture
def pymol_running(monkeypatch):
    monkeypatch.setattr(meshes.viewer, "_process_is_running", lambda: True)
    monkeypatch.setattr(meshes.viewer, "stop", lambda: None)


def test_wait_for_obj_file_returns_when_file_is_found(tmp_path, pymol_running):
    obj_file = tmp_path / "test.obj"
    obj_file.write_text("v 0 0 0\n")
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
        def __init__(self, uri, **kwargs):
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


def test_create_mesh():
    pytest.importorskip("pymol")
    pytest.importorskip("pytorch3d")
    config = ProteinMeshConfig(
        pymol_commands=[
            "show surface",
            "set surface_solvent, on",
            "set solvent_radius, 10000",
        ]
    )
    verts, faces, _ = create_mesh(pdb_code="3eiy", config=config)
    mesh = convert_verts_and_face_to_mesh(verts, faces)
    assert mesh.verts_packed().shape[0] == verts.shape[0]


class FakeViewer:
    """Stands in for the shared PyMOL viewer in ``get_obj_file``."""

    def __init__(self):
        self.commands = []

    def load(self, path):
        self.commands.append(("load", path))

    def fetch(self, code):
        self.commands.append(("fetch", code))

    def do(self, command):
        self.commands.append(("do", command))


def test_obj_file_name_replaces_whole_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(meshes, "viewer", FakeViewer())
    monkeypatch.setattr(meshes, "configure_pymol_session", lambda _: None)
    monkeypatch.setattr(meshes, "run_pymol_commands", lambda _: None)
    unrelated = tmp_path / "model.mobj"
    unrelated.touch()

    obj_file = meshes.get_obj_file.__wrapped__(
        pdb_file=str(tmp_path / "model.mmtf")
    )

    assert obj_file == str(tmp_path / "model.obj")
    assert unrelated.exists()


def test_wait_for_obj_file_stops_pymol_on_timeout(
    tmp_path, pymol_running, monkeypatch
):
    stopped = []
    monkeypatch.setattr(meshes.viewer, "stop", lambda: stopped.append(True))
    with pytest.raises(TimeoutError):
        meshes.wait_for_obj_file(
            str(tmp_path / "missing.obj"), max_wait_seconds=0.2
        )
    assert stopped


def test_wait_for_obj_file_accepts_file_written_as_pymol_exits(
    tmp_path, monkeypatch
):
    obj_file = tmp_path / "test.obj"

    def write_then_exit():
        obj_file.write_text("v 0 0 0\n")
        return False

    monkeypatch.setattr(meshes.viewer, "_process_is_running", write_then_exit)
    meshes.wait_for_obj_file(str(obj_file), max_wait_seconds=1.0)


def test_wait_for_obj_file_waits_for_writing_to_finish(tmp_path, monkeypatch):
    obj_file = tmp_path / "test.obj"
    obj_file.write_text("v 0 0 0\n")
    writes = iter(["v 1 1 1\n", "f 1 1 1\n"])

    def still_writing():
        line = next(writes, None)
        if line is not None:
            with obj_file.open("a") as f:
                f.write(line)
        return True

    monkeypatch.setattr(meshes.viewer, "_process_is_running", still_writing)
    meshes.wait_for_obj_file(str(obj_file), max_wait_seconds=5.0)
    assert obj_file.read_text().endswith("f 1 1 1\n")
    assert next(writes, None) is None


def test_server_calls_time_out():
    # Accepts connections (via the backlog) but never answers.
    with socket.socket() as listener:
        listener.bind(("localhost", 0))
        listener.listen()
        v = core.MolViewer(port=listener.getsockname()[1])
        v._connect(timeout=0.2)
        with pytest.raises(TimeoutError):
            v._server.delete("all")


def test_server_call_timeout_stops_pymol(monkeypatch):
    class HungServer:
        def delete(self, name):
            raise TimeoutError

    v = core.MolViewer()
    v._server = HungServer()
    stopped = []
    monkeypatch.setattr(v, "_process_is_running", lambda: True)
    monkeypatch.setattr(v, "stop", lambda: stopped.append(True))
    with pytest.raises(TimeoutError):
        v.delete("all")
    assert stopped


def test_restart_reuses_start_args_and_timeout(monkeypatch):
    v = core.MolViewer()
    calls = []

    def fake_start(args, timeout=None):
        calls.append((list(args), timeout))
        v._server = object()

    monkeypatch.setattr(v, "start", fake_start)
    v._start_args, v._timeout = ["-cq"], 5.0
    v.__getattr__("__class__")
    assert calls == [(["-cq"], 5.0)]


def test_stop_waits_for_exit_and_forgets_process():
    class StubbornProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            pass

        def wait(self, timeout=None):
            if self.returncode is None and timeout is not None:
                raise subprocess.TimeoutExpired("pymol", timeout)
            return self.returncode

        def kill(self):
            self.returncode = -9

    v = core.MolViewer()
    process = v._process = StubbornProcess()
    v.stop()
    assert process.returncode == -9
    assert v._process is None
