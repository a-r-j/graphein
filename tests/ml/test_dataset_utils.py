"""Tests for ligand dataset utility helpers."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "graphein/ml/datasets/utils.py"
)


class _DummyLogger:
    def info(self, *args, **kwargs):
        pass


def _load_dataset_utils_module(monkeypatch):
    config = SimpleNamespace(DATA_API_MAX_CONCURRENT_REQUESTS=None)

    loguru_module = ModuleType("loguru")
    loguru_module.logger = _DummyLogger()

    rcsbapi_module = ModuleType("rcsbapi")
    config_module = ModuleType("rcsbapi.config")
    config_module.config = config

    data_module = ModuleType("rcsbapi.data")
    data_module.ALL_STRUCTURES = "ALL_STRUCTURES"

    class DummyQuery:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def exec(self, progress_bar=True):
            return {"data": {"entries": []}}

    data_module.DataQuery = DummyQuery
    rcsbapi_module.config = config_module
    rcsbapi_module.data = data_module

    monkeypatch.setitem(sys.modules, "loguru", loguru_module)
    monkeypatch.setitem(sys.modules, "rcsbapi", rcsbapi_module)
    monkeypatch.setitem(sys.modules, "rcsbapi.config", config_module)
    monkeypatch.setitem(sys.modules, "rcsbapi.data", data_module)

    spec = spec_from_file_location(
        "test_graphein_ml_datasets_utils", MODULE_PATH
    )
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, config


def test_process_chem_comp_results_writes_sorted_mapping_files(
    monkeypatch, tmp_path
):
    module, _ = _load_dataset_utils_module(monkeypatch)
    pdb_to_cc_output_file = tmp_path / "pdb-to-cc.tsv"
    cc_to_pdb_output_file = tmp_path / "cc-to-pdb.tsv"

    entry_chem_comp_results = [
        {
            "rcsb_id": "2DEF",
            "nonpolymer_entities": [
                {
                    "rcsb_nonpolymer_entity_container_identifiers": {
                        "nonpolymer_comp_id": "ATP"
                    }
                }
            ],
            "branched_entities": [
                {
                    "rcsb_branched_entity_container_identifiers": {
                        "chem_comp_monomers": ["NAG", "ATP"]
                    }
                }
            ],
            "polymer_entities": [
                {
                    "rcsb_polymer_entity_container_identifiers": {
                        "chem_comp_nstd_monomers": ["MSE"]
                    }
                }
            ],
        },
        {
            "rcsb_id": "1ABC",
            "nonpolymer_entities": [
                {
                    "rcsb_nonpolymer_entity_container_identifiers": {
                        "nonpolymer_comp_id": "HEM"
                    }
                },
                {
                    "rcsb_nonpolymer_entity_container_identifiers": {
                        "nonpolymer_comp_id": "ATP"
                    }
                },
            ],
        },
    ]

    module.process_chem_comp_results_and_write_to_file(
        entry_chem_comp_results=entry_chem_comp_results,
        pdb_to_cc_output_file=pdb_to_cc_output_file,
        cc_to_pdb_output_file=cc_to_pdb_output_file,
        cc_extras_output_file=tmp_path / "cc-counts-extra.tsv",
        generate_cc_extra_file=False,
    )

    assert pdb_to_cc_output_file.read_text() == (
        "2DEF\tATP MSE NAG\n1ABC\tATP HEM\n"
    )
    assert cc_to_pdb_output_file.read_text() == (
        "ATP\t1ABC 2DEF\nNAG\t2DEF\nMSE\t2DEF\nHEM\t1ABC\n"
    )


def test_generate_pdb_ligand_mappings_uses_requested_types(
    monkeypatch, tmp_path
):
    module, config = _load_dataset_utils_module(monkeypatch)
    observed = {}

    def fake_fetch_all_chem_comp_ids(chem_comp_types_to_include):
        observed["chem_comp_types_to_include"] = chem_comp_types_to_include
        return [
            {
                "rcsb_id": "1ABC",
                "nonpolymer_entities": [
                    {
                        "rcsb_nonpolymer_entity_container_identifiers": {
                            "nonpolymer_comp_id": "ATP"
                        }
                    }
                ],
            }
        ]

    monkeypatch.setattr(
        module, "fetch_all_chem_comp_ids", fake_fetch_all_chem_comp_ids
    )

    pdb_to_cc_output_file = tmp_path / "pdb-to-cc.tsv"
    cc_to_pdb_output_file = tmp_path / "cc-to-pdb.tsv"
    module.generate_pdb_ligand_mappings(
        chem_comp_types=("nonpolymer", "polymer_std"),
        pdb_to_cc_output_file=pdb_to_cc_output_file,
        cc_to_pdb_output_file=cc_to_pdb_output_file,
        max_concurrent_api_requests=7,
        generate_cc_extra_file=False,
    )

    assert observed["chem_comp_types_to_include"] == [
        module.CHEMICAL_COMPONENT_TYPES_ARG_MAPPINGS["nonpolymer"],
        module.CHEMICAL_COMPONENT_TYPES_ARG_MAPPINGS["polymer_std"],
    ]
    assert config.DATA_API_MAX_CONCURRENT_REQUESTS == 7
    assert pdb_to_cc_output_file.read_text() == "1ABC\tATP\n"
    assert cc_to_pdb_output_file.read_text() == "ATP\t1ABC\n"


def test_generate_pdb_ligand_mappings_rejects_invalid_types(
    monkeypatch, tmp_path
):
    module, config = _load_dataset_utils_module(monkeypatch)

    with pytest.raises(ValueError) as excinfo:
        module.generate_pdb_ligand_mappings(
            chem_comp_types=("nonpolymer", "invalid_type"),
            pdb_to_cc_output_file=tmp_path / "pdb-to-cc.tsv",
            cc_to_pdb_output_file=tmp_path / "cc-to-pdb.tsv",
        )

    assert str(excinfo.value) == (
        "Invalid chem_comp_types: ['invalid_type']. Allowed values are: "
        f"{module.ALLOWED_CHEM_COMP_TYPES}."
    )
    assert config.DATA_API_MAX_CONCURRENT_REQUESTS is None
