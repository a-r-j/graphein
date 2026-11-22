"""Yaml parser for config objects"""

import importlib
from collections.abc import Callable
from functools import partial
from pathlib import Path

import yaml
from pydantic import BaseModel

from graphein.grn.config import GRNGraphConfig, RegNetworkConfig, TRRUSTConfig
from graphein.molecule.config import MoleculeGraphConfig
from graphein.ppi.config import BioGridConfig, PPIGraphConfig, STRINGConfig
from graphein.protein.config import (
    DSSPConfig,
    GetContactsConfig,
    ProteinGraphConfig,
    ProteinMeshConfig,
)


class ConfigLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` with graphein's config (``!ProteinGraphConfig``)
    and function (``!func:graphein.module.function``) tags. Subclassing keeps
    the tags off PyYAML's own loaders."""


CONFIGS = {
    config.__name__: config
    for config in (
        ProteinGraphConfig,
        DSSPConfig,
        ProteinMeshConfig,
        GetContactsConfig,
        TRRUSTConfig,
        RegNetworkConfig,
        GRNGraphConfig,
        STRINGConfig,
        BioGridConfig,
        PPIGraphConfig,
        MoleculeGraphConfig,
    )
}


def config_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
) -> BaseModel:
    """Construct a BaseModel config.

    :param loader: Given yaml loader
    :type loader: yaml.SafeLoader
    :param node: A mapping node
    :type node: yaml.nodes.MappingNode
    :return: The config named by the node's tag
    :rtype: BaseModel
    """
    arg_map = loader.construct_mapping(node, deep=True) if node.value else {}
    return CONFIGS[node.tag[1:]](**arg_map)


def function_constructor(
    loader: yaml.SafeLoader,
    tag_suffix: str,
    node: yaml.nodes.MappingNode | yaml.nodes.ScalarNode,
) -> Callable:
    """Construct a Callable. If function parameters are given, this returns a
    partial function. Only functions in ``graphein`` can be named, so a config
    file can't run arbitrary code (e.g. ``!func:os.system``).

    :param loader: Given yaml loader
    :type loader: yaml.SafeLoader
    :param tag_suffix: The name after the !func: tag
    :type tag_suffix: str
    :param node: A mapping node if function parameters are given, a scalar
        node if not
    :type node: yaml.nodes.MappingNode | yaml.nodes.ScalarNode
    :raises yaml.constructor.ConstructorError: If the function is not in
        ``graphein`` or can't be found.
    :return: The function, or a partial of it with the given parameters
    :rtype: Callable
    """
    module, _, name = tag_suffix.rpartition(".")
    if module != "graphein" and not module.startswith("graphein."):
        raise yaml.constructor.ConstructorError(
            None,
            None,
            f"!func: only accepts graphein functions, got {tag_suffix!r}",
            node.start_mark,
        )
    try:
        func = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as e:
        raise yaml.constructor.ConstructorError(
            None, None, f"Cannot find {tag_suffix!r}: {e}", node.start_mark
        ) from e
    if isinstance(node, yaml.nodes.MappingNode) and node.value:
        func = partial(func, **loader.construct_mapping(node, deep=True))
    return func


for _name in CONFIGS:
    ConfigLoader.add_constructor(f"!{_name}", config_constructor)
ConfigLoader.add_multi_constructor("!func:", function_constructor)


def get_loader() -> type[ConfigLoader]:
    """Return the PyYAML loader for graphein configs.

    :return: :class:`ConfigLoader`
    :rtype: type[ConfigLoader]
    """
    return ConfigLoader


def parse_config(path: Path) -> BaseModel:
    """
    Parses a yaml configuration file into a config object.

    :param path: Path to configuration file
    :type path: pathlib.Path
    """
    with open(path, "rb") as f:
        return yaml.load(f, Loader=ConfigLoader)
