"""Tests for graphein.ml.datasets.foldcomp_dataset."""

import importlib
import sys

import pytest

pytest.importorskip("torch_geometric")


def test_imports_without_lightning(monkeypatch):
    """lightning is optional: the module imports without it, and only the
    Lightning data module needs it."""
    monkeypatch.setitem(sys.modules, "lightning", None)  # import fails
    monkeypatch.delitem(
        sys.modules, "graphein.ml.datasets.foldcomp_dataset", raising=False
    )
    module = importlib.import_module("graphein.ml.datasets.foldcomp_dataset")

    assert module.FoldCompDataset
    with pytest.raises(ImportError, match="lightning"):
        module.FoldCompLightningDataModule(
            data_dir=".", database="afdb_swissprot_v4", batch_size=1
        )
