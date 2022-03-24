"""Command line interface for graphein."""
import pathlib

import click
import networkx as nx

from graphein import __version__
from graphein.protein.graphs import construct_graph
from graphein.utils.config import parse_config


@click.command()
@click.version_option(__version__)
@click.option(
    "-c",
    "--config_path",
    help="The name the of .yml input config",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
)
@click.option(
    "-p",
    "--pdb_path",
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
def main(config_path, pdb_path, output_path):
    """Build the graphs and save them in output dir."""
    config = None
    if config_path:
        config = parse_config(path=config_path)

    if pdb_path.is_file():
        pdb_paths = [pdb_path]
    elif pdb_path.is_dir():
        pdb_paths = [pdb for pdb in pdb_path.glob("*.pdb")]
    else:
        raise NotImplementedError(
            "Given PDB path needs to point to either a pdb file or a directory with pdb files."
        )

    for path in pdb_paths:
        g = construct_graph(config=config, pdb_path=str(path))
        nx.write_gpickle(g, str(output_path / f"{path.stem}.pickle"))


if __name__ == "__main__":
    main()
