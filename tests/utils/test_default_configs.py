"""Tests for built-in default graph construction configs."""

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from graphein.cli import main
from graphein.configs import available_configs, resolve_config_path
from graphein.protein.config import ProteinGraphConfig
from graphein.utils.config_parser import parse_config


def test_builtin_configs_are_discoverable():
    """A default config ships with the package."""
    assert "default" in available_configs()


def test_resolve_builtin_config_by_name():
    """A bare name resolves to a YAML file inside the package."""
    path = resolve_config_path("default")
    assert path.is_file()
    assert path.suffix == ".yml"


def test_resolve_explicit_path_takes_precedence(tmp_path):
    """An existing file path is used as-is, not looked up in built-ins."""
    custom = tmp_path / "custom.yml"
    custom.write_text("!ProteinGraphConfig\ngranularity: CA\n")
    assert resolve_config_path(str(custom)) == custom


def test_resolve_unknown_config_raises():
    """Unknown names fail with a helpful error listing available configs."""
    with pytest.raises(ValueError, match="default"):
        resolve_config_path("does-not-exist")


def test_default_config_parses_to_protein_graph_config():
    """The built-in default parses into a usable ProteinGraphConfig."""
    config = parse_config(resolve_config_path("default"))
    assert isinstance(config, ProteinGraphConfig)
    assert config.granularity == "CA"
    assert config.edge_construction_functions


def test_cli_lists_builtin_configs():
    """--list-configs exposes built-in names without building a graph."""
    result = CliRunner().invoke(main, ["--list-configs"])
    assert result.exit_code == 0
    assert "default" in result.output


def test_cli_accepts_builtin_config_name():
    """-c accepts a built-in name as well as a file path."""
    pdb = Path(__file__).resolve().parent.parent / "protein/test_data/4hhb.pdb"
    with tempfile.TemporaryDirectory() as tmp:
        result = CliRunner().invoke(
            main, ["-c", "default", "-p", str(pdb), "-o", tmp]
        )
        assert result.exit_code == 0, result.output
        assert (Path(tmp) / "4hhb.gpickle").is_file()
