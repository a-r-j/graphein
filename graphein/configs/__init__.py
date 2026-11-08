"""Built-in graph construction configs for the CLI.

Configs are YAML files parsed by
:func:`graphein.utils.config_parser.parse_config`. Pass their name to
``graphein -c <name>`` instead of a file path, or list them with
``graphein --list-configs``.
"""

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent


def _builtin_configs() -> dict[str, Path]:
    return {
        p.stem: p
        for p in CONFIG_DIR.iterdir()
        if p.suffix in {".yml", ".yaml"}
    }


def available_configs() -> list[str]:
    """Return the names of built-in configs, sorted alphabetically.

    :return: Config names without the ``.yml``/``.yaml`` suffix.
    :rtype: list[str]
    """
    return sorted(_builtin_configs())


def resolve_config_path(name_or_path: str | Path) -> Path:
    """Resolve a config name or an existing file path to a YAML file.

    An existing path always takes precedence over a built-in name.

    :param name_or_path: Built-in config name or path to a YAML file.
    :type name_or_path: str | pathlib.Path
    :return: Path to the YAML config file.
    :rtype: pathlib.Path
    :raises ValueError: If the argument is a directory, looks like a file
        path that doesn't exist, or is not a built-in name.
    """
    path = Path(name_or_path)
    if path.is_file():
        return path
    if str(name_or_path) and path.is_dir():
        raise ValueError(f"Config path is a directory: {name_or_path}")
    builtins = _builtin_configs()
    if str(name_or_path) in builtins:
        return builtins[str(name_or_path)]
    if path.suffix in {".yml", ".yaml"} or len(path.parts) > 1:
        raise ValueError(f"Config file not found: {name_or_path}")
    raise ValueError(
        f"Unknown config {name_or_path!r}. "
        f"Available built-in configs: {', '.join(sorted(builtins))}"
    )
