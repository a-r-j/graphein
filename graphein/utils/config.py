"""Yaml parser for config objects"""
from functools import partial
from pathlib import Path
from typing import Callable, Union

import yaml
from deepdiff.operator import BaseOperator
from pydantic import BaseModel

from graphein.grn.config import GRNGraphConfig, RegNetworkConfig, TRRUSTConfig
from graphein.ppi.config import BioGridConfig, PPIGraphConfig, STRINGConfig
from graphein.protein.config import (
    DSSPConfig,
    GetContactsConfig,
    ProteinGraphConfig,
    ProteinMeshConfig,
)


def config_constructor(
    loader: yaml.FullLoader, node: yaml.nodes.MappingNode
) -> BaseModel:
    """Construct a BaseModel config.

    :param loader: Given yaml loader
    :param type: yaml.FullLoader
    :param loader: A mapping node
    :param type: yaml.nodes.MappingNode
    """
    arg_map = loader.construct_mapping(node, deep=True) if node.value else {}
    return eval(node.tag[1:])(**arg_map)


def function_constructor(
    loader: yaml.FullLoader,
    tag_suffix: str,
    node: Union[yaml.nodes.MappingNode, yaml.nodes.ScalarNode],
) -> Callable:
    """Construct a Callable. If function parameters are given, this returns a partial function.

    :param loader: Given yaml loader
    :param type: yaml.FullLoader
    :param tag_suffix: The name after the !func: tag
    :param type: str
    :param loader: A mapping node if function parameters are given, a scalar node if not
    :param type: Union[yaml.nodes.MappingNode, yaml.nodes.ScalarNode]
    """
    arg_map = None
    if isinstance(node, yaml.nodes.MappingNode):
        arg_map = (
            loader.construct_mapping(node, deep=True) if node.value else {}
        )
        node = yaml.nodes.ScalarNode(
            node.tag, "", node.start_mark, node.end_mark
        )
    func = loader.construct_python_name(tag_suffix, node)
    if arg_map:
        func = partial(func, **arg_map)
    return func


def get_loader() -> yaml.Loader:
    """Add constructors to PyYAML loader."""
    # For Python-specific tags, you can use full_load(), which resolves all tags except those known to be unsafe;
    # this includes all the tags listed here: https://pyyaml.org/wiki/PyYAMLDocumentation#yaml-tags-and-python-types
    loader = yaml.FullLoader
    configs = [
        ProteinGraphConfig.__name__,
        DSSPConfig.__name__,
        ProteinMeshConfig.__name__,
        GetContactsConfig.__name__,
        TRRUSTConfig.__name__,
        RegNetworkConfig.__name__,
        GRNGraphConfig.__name__,
        STRINGConfig.__name__,
        BioGridConfig.__name__,
        PPIGraphConfig.__name__,
    ]
    for config in configs:
        loader.add_constructor(f"!{config}", config_constructor)
    loader.add_multi_constructor("!func:", function_constructor)
    return loader


def parse_config(path: Path) -> BaseModel:
    """
    Parses a yaml configuration file into a config object.

    :param path: Path to configuration file
    :type path: pathlib.Path
    """
    with open(path, "rb") as f:
        yml_config = yaml.load(f, Loader=get_loader())
    return yml_config


def partial_functions_equal(func1: partial, func2: partial) -> bool:
    """
    Determine whether two partial functions are equal.

    :param func1: Partial function to check
    :type func1: partial
    :param func2: Partial function to check
    :type func2: partial
    :return: Whether the two functions are equal
    :rtype: bool
    """
    return (
        all(
            getattr(func1, attr) == getattr(func2, attr)
            for attr in ["func", "args", "keywords"]
        )
        if (isinstance(func1, partial) and isinstance(func2, partial))
        else False
    )


class PartialMatchOperator(BaseOperator):
    """Custom operator for deepdiff comparison. This operator compares whether the two partials are equal."""

    def give_up_diffing(self, level, diff_instance):
        return partial_functions_equal(level.t1, level.t2)


class PathMatchOperator(BaseOperator):
    """Custom operator for deepdiff comparison. This operator compares whether the two pathlib Paths are equal."""

    def give_up_diffing(self, level, diff_instance):
        return level.t1 == level.t2
