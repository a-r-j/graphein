"""Tests for graphein.protein.meshes."""

import tempfile
from pathlib import Path

import pytest

from graphein.protein import meshes


def test_wait_for_obj_file_returns_when_file_is_found(monkeypatch):
    monkeypatch.setattr(meshes.os.path, "isfile", lambda _: True)
    monkeypatch.setattr(meshes.time, "sleep", lambda _: None)

    with tempfile.TemporaryDirectory() as tmp_dir:
        obj_path = Path(tmp_dir) / "test.obj"
        meshes.wait_for_obj_file(str(obj_path), max_wait_seconds=1.0)


def test_wait_for_obj_file_raises_timeout(monkeypatch):
    mock_timestamps = iter([0.0, 0.6, 1.2])

    monkeypatch.setattr(meshes.os.path, "isfile", lambda _: False)
    monkeypatch.setattr(meshes.time, "time", lambda: next(mock_timestamps))
    monkeypatch.setattr(meshes.time, "sleep", lambda _: None)

    with tempfile.TemporaryDirectory() as tmp_dir:
        missing_path = Path(tmp_dir) / "missing.obj"
        with pytest.raises(
            TimeoutError, match="missing.obj not found after 1.0 seconds"
        ):
            meshes.wait_for_obj_file(str(missing_path), max_wait_seconds=1.0)
