"""Built-in default graph construction configs for the CLI.

Configs are YAML files parsed by
:func:`graphein.utils.config_parser.parse_config`. Pass their name to
``graphein -c <name>`` instead of a file path, or list them with
``graphein --list-configs``.
"""

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent


def available_configs() -> list[str]:
    """Return the names of built-in configs, sorted alphabetically.

    :return: Config names without the ``.yml`` suffix.
    :rtype: list[str]
    """
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yml"))


def resolve_config_path(name_or_path: str | Path) -> Path:
    """Resolve a config name or an existing file path to a YAML file.

    An existing path always takes precedence over a built-in name.

    :param name_or_path: Built-in config name or path to a YAML file.
    :type name_or_path: str | pathlib.Path
    :return: Path to the YAML config file.
    :rtype: pathlib.Path
    :raises ValueError: If neither an existing file nor a built-in name
        matches.
    """
    path = Path(name_or_path)
    if path.is_file():
        return path
    builtin = CONFIG_DIR / f"{name_or_path}.yml"
    if builtin.is_file():
        return builtin
    raise ValueError(
        f"Unknown config {name_or_path!r}. "
        f"Available built-in configs: {', '.join(available_configs())}"
    )
