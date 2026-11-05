"""Command line interface for graphein."""

import pathlib
import pickle

import rich_click as click

from graphein import __version__
from graphein.configs import available_configs, resolve_config_path
from graphein.protein.graphs import construct_graph
from graphein.utils.config_parser import parse_config


@click.command()
@click.version_option(__version__)
@click.option(
    "-c",
    "--config_path",
    help="Built-in config name or path to a .yml config file",
    type=str,
)
@click.option(
    "--list-configs",
    is_flag=True,
    help="List built-in configs and exit",
)
@click.option(
    "-p",
    "--path",
    help="Path to input pdbs",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=True, path_type=pathlib.Path
    ),
)
@click.option(
    "-o",
    "--output_path",
    help="Path to output dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
)
def main(config_path, path, output_path, list_configs):
    """Build the graphs and save them in output dir."""
    if list_configs:
        click.echo("\n".join(available_configs()))
        return

    config = None
    if config_path:
        try:
            config_file = resolve_config_path(config_path)
        except ValueError as e:
            raise click.BadParameter(
                str(e), param_hint="-c/--config_path"
            ) from e
        config = parse_config(path=config_file)

    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = list(path.glob("*.pdb"))
    else:
        raise NotImplementedError(
            "Given PDB path needs to point to either a pdb file or a directory with pdb files."
        )

    for path in paths:
        g = construct_graph(config=config, path=str(path))

        with open(str(output_path / f"{path.stem}.gpickle"), "wb") as f:
            pickle.dump(g, f)


if __name__ == "__main__":
    main()
