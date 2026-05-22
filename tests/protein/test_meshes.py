"""Tests for graphein.protein.meshes."""

import pytest

from graphein.protein import meshes


def test_wait_for_obj_file_returns_when_file_is_found(monkeypatch):
    monkeypatch.setattr(meshes.os.path, "isfile", lambda _: True)
    monkeypatch.setattr(meshes.time, "sleep", lambda _: None)

    meshes.wait_for_obj_file("/tmp/test.obj", max_wait_seconds=1.0)


def test_wait_for_obj_file_raises_timeout(monkeypatch):
    times = iter([0.0, 0.6, 1.2])

    monkeypatch.setattr(meshes.os.path, "isfile", lambda _: False)
    monkeypatch.setattr(meshes.time, "time", lambda: next(times))
    monkeypatch.setattr(meshes.time, "sleep", lambda _: None)

    with pytest.raises(
        TimeoutError, match="missing.obj not found after 1.0 seconds"
    ):
        meshes.wait_for_obj_file("/tmp/missing.obj", max_wait_seconds=1.0)
