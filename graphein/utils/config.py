from pathlib import Path

from yaml import unsafe_load

from graphein.protein.config import ProteinGraphConfig


def parse_config(path: Path):
    """
    Parses a yaml configuration file into a config object
    :param path: Path to configuration file
    """
    with open(path) as file:
        config_dict = unsafe_load(file)

    print(config_dict)
    if config_dict["mode"] == "protein_graph":
        return parse_protein_graph_config(config_dict)
    elif config_dict["mode"] == "protein_mesh":
        raise NotImplementedError
    elif config_dict["mode"] == "rna":
        raise NotImplementedError
    elif config_dict["mode"] == "ppi":
        raise NotImplementedError


def parse_protein_graph_config(config_dict):

    config = ProteinGraphConfig(**config_dict)
    print(config)
    return config


def parse_dssp_config(config_dict):
    raise NotImplementedError
