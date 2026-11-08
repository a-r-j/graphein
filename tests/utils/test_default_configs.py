"""Tests for built-in default graph construction configs."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from graphein import configs
from graphein.cli import main
from graphein.configs import available_configs, resolve_config_path
from graphein.protein.config import ProteinGraphConfig
from graphein.utils.config_parser import parse_config


def test_builtin_configs_are_discoverable():
    """A built-in config ships with the package."""
    assert "ca_distance_10" in available_configs()


def test_resolve_builtin_config_by_name():
    """A bare name resolves to a YAML file inside the package."""
    path = resolve_config_path("ca_distance_10")
    assert path.is_file()
    assert path.suffix == ".yml"


def test_resolve_existing_file_path(tmp_path):
    """An existing file path that isn't a built-in name is used as-is."""
    custom = tmp_path / "custom.yml"
    custom.write_text("!ProteinGraphConfig\ngranularity: CA\n")
    assert resolve_config_path(str(custom)) == custom


@pytest.mark.parametrize(
    "name",
    ["configs/missing.yml", "missing.yaml", "./missing", "ca_distance_10.yml"],
)
def test_resolve_missing_path_lists_builtins(name):
    """A missing path fails with an error that lists the built-ins."""
    with pytest.raises(ValueError, match="Built-in configs: ca_distance_10"):
        resolve_config_path(name)


@pytest.mark.parametrize("make", [Path.mkdir, Path.touch])
def test_builtin_name_wins_over_cwd_entry(tmp_path, monkeypatch, make):
    """A same-named folder or file in the cwd doesn't hide a built-in."""
    monkeypatch.chdir(tmp_path)
    make(tmp_path / "ca_distance_10")
    assert resolve_config_path("ca_distance_10").parent == configs.CONFIG_DIR


def test_resolve_rejects_names_outside_config_dir():
    """Only listed built-in names resolve; ../ can't escape the package."""
    with pytest.raises(ValueError):
        resolve_config_path("../configs/ca_distance_10")


def test_resolve_empty_name_raises():
    with pytest.raises(ValueError):
        resolve_config_path("")


def test_resolve_unknown_config_raises():
    """Unknown names fail with a helpful error listing available configs."""
    with pytest.raises(ValueError, match="ca_distance_10"):
        resolve_config_path("does-not-exist")


def test_builtin_config_parses_to_protein_graph_config():
    """The built-in config parses into a usable ProteinGraphConfig."""
    config = parse_config(resolve_config_path("ca_distance_10"))
    assert isinstance(config, ProteinGraphConfig)
    assert config.granularity == "CA"
    assert config.edge_construction_functions
    # Keeps residues with insertion codes, like ProteinGraphConfig().
    assert config.insertions


def test_resolve_directory_raises(tmp_path):
    """An existing directory is not a config file."""
    with pytest.raises(ValueError, match="Built-in configs"):
        resolve_config_path(str(tmp_path))


def test_yaml_suffix_configs_are_listed_and_resolved(tmp_path, monkeypatch):
    """Packaged .yaml configs are found as well as .yml ones."""
    monkeypatch.setattr(configs, "CONFIG_DIR", tmp_path)
    (tmp_path / "atom.yaml").write_text("!ProteinGraphConfig\n")
    assert configs.available_configs() == ["atom"]
    assert configs.resolve_config_path("atom") == tmp_path / "atom.yaml"


def test_cli_lists_builtin_configs():
    """--list-configs exposes built-in names without building a graph."""
    result = CliRunner().invoke(main, ["--list-configs"])
    assert result.exit_code == 0
    assert "ca_distance_10" in result.output
