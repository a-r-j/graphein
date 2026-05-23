"""Tests for ligand map generation in the ML PDB dataset manager."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "graphein/ml/datasets/pdb_data.py"
)


class _DummyLogger:
    def info(self, *args, **kwargs):
        pass


def _load_pdb_data_module(monkeypatch):
    numpy_module = ModuleType("numpy")
    numpy_module.datetime64 = object

    pandas_module = ModuleType("pandas")
    pandas_module.DataFrame = object
    pandas_module.Series = object
    pandas_core_module = ModuleType("pandas.core")
    pandas_groupby_module = ModuleType("pandas.core.groupby")
    pandas_generic_module = ModuleType("pandas.core.groupby.generic")
    pandas_generic_module.DataFrameGroupBy = object

    wget_module = ModuleType("wget")
    wget_module.download = lambda *args, **kwargs: None

    biopandas_module = ModuleType("biopandas")
    biopandas_pdb_module = ModuleType("biopandas.pdb")
    biopandas_pdb_module.PandasPdb = object

    loguru_module = ModuleType("loguru")
    loguru_module.logger = _DummyLogger()

    tqdm_module = ModuleType("tqdm")
    tqdm_auto_module = ModuleType("tqdm.auto")
    tqdm_auto_module.tqdm = lambda iterable=None, *args, **kwargs: iterable

    graphein_module = ModuleType("graphein")
    graphein_ml_module = ModuleType("graphein.ml")
    graphein_ml_datasets_module = ModuleType("graphein.ml.datasets")
    graphein_utils_module = ModuleType("graphein.utils")
    graphein_protein_module = ModuleType("graphein.protein")

    dataset_utils_module = ModuleType("graphein.ml.datasets.utils")
    dataset_utils_module.generate_pdb_ligand_mappings = (
        lambda *args, **kwargs: None
    )

    protein_utils_module = ModuleType("graphein.protein.utils")
    protein_utils_module.cast_pdb_column_to_type = lambda *args, **kwargs: None
    protein_utils_module.download_pdb_multiprocessing = (
        lambda *args, **kwargs: None
    )
    protein_utils_module.extract_chains_to_file = (
        lambda *args, **kwargs: None
    )
    protein_utils_module.read_fasta = lambda *args, **kwargs: None

    dependencies_module = ModuleType("graphein.utils.dependencies")
    dependencies_module.is_tool = lambda *args, **kwargs: True

    monkeypatch.setitem(sys.modules, "numpy", numpy_module)
    monkeypatch.setitem(sys.modules, "pandas", pandas_module)
    monkeypatch.setitem(sys.modules, "pandas.core", pandas_core_module)
    monkeypatch.setitem(
        sys.modules, "pandas.core.groupby", pandas_groupby_module
    )
    monkeypatch.setitem(
        sys.modules, "pandas.core.groupby.generic", pandas_generic_module
    )
    monkeypatch.setitem(sys.modules, "wget", wget_module)
    monkeypatch.setitem(sys.modules, "biopandas", biopandas_module)
    monkeypatch.setitem(sys.modules, "biopandas.pdb", biopandas_pdb_module)
    monkeypatch.setitem(sys.modules, "loguru", loguru_module)
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)
    monkeypatch.setitem(sys.modules, "tqdm.auto", tqdm_auto_module)
    monkeypatch.setitem(sys.modules, "graphein", graphein_module)
    monkeypatch.setitem(sys.modules, "graphein.ml", graphein_ml_module)
    monkeypatch.setitem(
        sys.modules, "graphein.ml.datasets", graphein_ml_datasets_module
    )
    monkeypatch.setitem(
        sys.modules, "graphein.ml.datasets.utils", dataset_utils_module
    )
    monkeypatch.setitem(sys.modules, "graphein.protein", graphein_protein_module)
    monkeypatch.setitem(
        sys.modules, "graphein.protein.utils", protein_utils_module
    )
    monkeypatch.setitem(sys.modules, "graphein.utils", graphein_utils_module)
    monkeypatch.setitem(
        sys.modules, "graphein.utils.dependencies", dependencies_module
    )

    spec = spec_from_file_location(
        "test_graphein_ml_datasets_pdb_data", MODULE_PATH
    )
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_ligand_map_uses_distinct_intermediate_output(
    monkeypatch, tmp_path
):
    module = _load_pdb_data_module(monkeypatch)
    observed = {}

    def fake_generate_pdb_ligand_mappings(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(
        module, "generate_pdb_ligand_mappings", fake_generate_pdb_ligand_mappings
    )

    manager = module.PDBManager.__new__(module.PDBManager)
    manager.root_dir = tmp_path
    manager.ligand_map_filename = "cc-to-pdb.tdd"

    manager._generate_ligand_map()

    assert observed["cc_to_pdb_output_file"] == tmp_path / "cc-to-pdb.tdd"
    assert observed["generate_cc_extra_file"] is False
    assert observed["pdb_to_cc_output_file"] != observed["cc_to_pdb_output_file"]
    assert Path(observed["pdb_to_cc_output_file"]).name == "pdb-to-cc.tsv"
