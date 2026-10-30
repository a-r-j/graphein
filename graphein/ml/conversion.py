"""Utilities for converting Graphein Networks to Geometric Deep Learning formats."""

# %%
# Graphein
# Author: Kexin Huang, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from typing import List, Optional

import networkx as nx
import numpy as np
import torch
from loguru import logger as log

from graphein.utils.dependencies import import_message

try:
    import torch
except ImportError:
    import_message(
        submodule="graphein.ml.conversion",
        package="torch",
        pip_install=True,
        conda_channel="pytorch",
    )

try:
    import torch_geometric
    from torch_geometric.data import Data
except ImportError:
    import_message(
        submodule="graphein.ml.conversion",
        package="torch_geometric",
        pip_install=True,
        conda_channel="rusty1s",
    )

try:
    import dgl
except ImportError:
    import_message(
        submodule="graphein.ml.conversion",
        package="dgl",
        pip_install=True,
        conda_channel="dglteam",
    )


SUPPORTED_FORMATS = ["nx", "pyg", "dgl"]
"""Supported conversion formats.

``"nx"``: NetworkX graph

``"pyg"``: PyTorch Geometric Data object

``"dgl"``: DGL graph
"""

SUPPORTED_VERBOSITY = ["gnn", "default", "all_info"]
"""Supported verbosity levels for preserving graph features in conversion."""


def _feature_lists(attr_dicts, keep=lambda key: True) -> dict:
    """Returns ``{key: [value per node/edge]}`` for the keys ``keep`` accepts.

    Every list has one entry per dict (``None`` where one lacks the key), so
    features stay aligned with the nodes / ``edge_index``. Missing entries of
    numeric features are ``NaN`` so the feature can still become a tensor.
    """
    attr_dicts = [{str(k): v for k, v in d.items()} for d in attr_dicts]
    keys = sorted({k for d in attr_dicts for k in d if keep(k)})
    features = {}
    for key in keys:
        values = [d.get(key) for d in attr_dicts]
        present = next((v for v in values if v is not None), None)
        if any(v is None for v in values) and np.asarray(present).dtype.kind in "biuf":
            nan = np.full(np.shape(present), np.nan)
            values = [nan if v is None else v for v in values]
        features[key] = values
    return features


def _symmetrize(edge_index: torch.Tensor, num_nodes: int):
    """Adds the reverse of each edge (except self-loops), sorted like
    ``to_undirected``. Unlike ``to_undirected``, parallel edges are kept.

    :return: Symmetrized ``edge_index`` and each edge's original position.
    """
    edge_id = torch.arange(edge_index.shape[1])
    rev = edge_index[0] != edge_index[1]
    edge_index = torch.cat([edge_index, edge_index[:, rev].flip(0)], dim=1)
    edge_id = torch.cat([edge_id, edge_id[rev]])
    order = torch.sort(edge_index[0] * num_nodes + edge_index[1], stable=True)
    return edge_index[:, order.indices], edge_id[order.indices]


class GraphFormatConvertor:
    """
    Provides conversion utilities between NetworkX Graphs and geometric deep learning library destination formats.
    Currently, we provide support for converstion from ``nx.Graph`` to ``dgl.DGLGraph`` and ``pytorch_geometric.Data``. Supported conversion
    formats can be retrieved from :const:`~graphein.ml.conversion.SUPPORTED_FORMATS`.

    :param src_format: The type of graph you'd like to convert from. Supported formats are available in :const:`~graphein.ml.conversion.SUPPORTED_FORMATS`
    :type src_format: Literal["nx", "pyg", "dgl"]
    :param dst_format: The type of graph format you'd like to convert to. Supported formats are available in:
        ``graphein.ml.conversion.SUPPORTED_FORMATS``
    :type dst_format:  Literal["nx", "pyg", "dgl"]
    :param verbose: Select from ``"gnn"``, ``"default"``, ``"all_info"`` to determine how much information is preserved (features)
        as some are unsupported by various downstream frameworks
    :type verbose: graphein.ml.conversion.SUPPORTED_VERBOSITY
    :param columns: List of columns in the node features to retain
    :type columns: List[str], optional
    """

    def __init__(
        self,
        src_format: str,
        dst_format: str,
        verbose: SUPPORTED_VERBOSITY = "gnn",
        columns: Optional[List[str]] = None,
    ):
        if (src_format not in SUPPORTED_FORMATS) or (
            dst_format not in SUPPORTED_FORMATS
        ):
            raise ValueError(
                "Please specify from supported format, "
                + "/".join(SUPPORTED_FORMATS)
            )
        self.src_format = src_format
        self.dst_format = dst_format

        # supported_verbose_format = ["gnn", "default", "all_info"]
        if (columns is None) and (verbose not in SUPPORTED_VERBOSITY):
            raise ValueError(
                "Please specify the supported verbose mode ("
                + "/".join(SUPPORTED_VERBOSITY)
                + ") or specify column names!"
            )

        if columns is None:
            if verbose == "gnn":
                columns = [
                    "edge_index",
                    "coords",
                    "name",
                    "node_id",
                ]
            elif verbose == "default":
                columns = [
                    "b_factor",
                    "chain_id",
                    "coords",
                    "edge_index",
                    "kind",
                    "name",
                    "node_id",
                    "residue_name",
                ]
            elif verbose == "all_info":
                columns = [
                    "atom_type",
                    "b_factor",
                    "chain_id",
                    "chain_ids",
                    "config",
                    "coords",
                    "dist_mat",
                    "edge_index",
                    "element_symbol",
                    "kind",
                    "name",
                    "node_id",
                    "node_type",
                    "pdb_df",
                    "raw_pdb_df",
                    "residue_name",
                    "residue_number",
                    "rgroup_df",
                    "sequence_A",
                    "sequence_B",
                ]
        self.columns = columns

        self.type2form = {
            "atom_type": "str",
            "b_factor": "float",
            "chain_id": "str",
            "coords": "np.array",
            "dist_mat": "np.array",
            "distance": "float",
            "element_symbol": "str",
            "node_id": "str",
            "residue_name": "str",
            "residue_number": "int",
            "edge_index": "torch.tensor",
            "kind": "str",
        }

    def convert_nx_to_dgl(self, G: nx.Graph) -> dgl.DGLGraph:
        """
        Converts ``NetworkX`` graph to ``DGL``

        :param G: ``nx.Graph`` to convert to ``DGLGraph``
        :type G: nx.Graph
        :return: ``DGLGraph`` object version of input ``NetworkX`` graph
        :rtype: dgl.DGLGraph
        """
        g = dgl.DGLGraph()
        node_id = list(G.nodes())
        G = nx.convert_node_labels_to_integers(G)

        ## add node level feat

        node_dict = _feature_lists(
            [d for _, d in G.nodes(data=True)], lambda k: k in self.columns
        )

        string_dict = {}
        node_dict_transformed = {}
        for i, j in node_dict.items():
            if i == "coords":
                node_dict_transformed[i] = torch.Tensor(np.asarray(j)).type(
                    "torch.FloatTensor"
                )
            elif i == "dist_mat":
                node_dict_transformed[i] = torch.Tensor(
                    np.asarray(j[0].values)
                ).type("torch.FloatTensor")
            elif self.type2form[i] == "str":
                string_dict[i] = j
            elif self.type2form[i] in ["float", "int"]:
                node_dict_transformed[i] = torch.Tensor(np.array(j))
        # DGL cannot add an empty feature dict, so pass None instead
        g.add_nodes(len(node_id), node_dict_transformed or None)

        edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

        # add edge level features
        edge_dict = _feature_lists(
            [d for _, _, d in G.edges(data=True)], lambda k: k in self.columns
        )

        edge_transform_dict = {}
        for i, j in edge_dict.items():
            if self.type2form[i] == "str":
                string_dict[i] = j
            elif self.type2form[i] in ["float", "int"]:
                edge_transform_dict[i] = torch.Tensor(np.array(j))
        g.add_edges(edge_index[0], edge_index[1], edge_transform_dict or None)

        # add graph level features
        graph_dict = {
            str(feat_name): [G.graph[feat_name]]
            for feat_name in G.graph
            if str(feat_name) in self.columns
        }

        return g

    def convert_nx_to_pyg(self, G: nx.Graph) -> Data:
        """
        Converts ``NetworkX`` graph to ``pytorch_geometric.data.Data`` object. Requires ``PyTorch Geometric`` (https://pytorch-geometric.readthedocs.io/en/latest/) to be installed.

        :param G: ``nx.Graph`` to convert to PyTorch Geometric ``Data`` object
        :type G: nx.Graph
        :return: ``Data`` object containing networkx graph data
        :rtype: pytorch_geometric.data.Data
        """

        # Initialise dict used to construct Data object & Assign node ids as a feature
        data = {"node_id": list(G.nodes())}
        G = nx.convert_node_labels_to_integers(G)

        # Construct Edge Index
        # G.edges() rather than G.edges: (u, v) pairs, also for multigraphs
        edge_index = (
            torch.LongTensor(list(G.edges())).t().contiguous().view(2, -1)
        )

        # Add node features
        data.update(
            _feature_lists(
                [d for _, d in G.nodes(data=True)], lambda k: k in self.columns
            )
        )

        # Add edge features, aligned with edge_index
        edge_features = _feature_lists(
            [d for _, _, d in G.edges(data=True)],
            lambda k: k in self.columns or k == "kind",
        )
        data.update(edge_features)

        # Add graph-level features, without replacing node or edge features
        for feat_name in G.graph:
            if str(feat_name) in self.columns and str(feat_name) not in data:
                data[str(feat_name)] = G.graph[feat_name]
        if "edge_index" in self.columns:
            data["edge_index"] = edge_index

        # Split edge index by edge kind. Kinds are sets, so sort them for a
        # name that doesn't depend on set iteration order.
        kind_strs = np.array(
            ["_".join(sorted(k or ())) for k in data.get("kind") or []]
        )
        kinds = set(kind_strs) - {""}  # edges without a kind
        for kind in kinds:
            key = f"edge_index_{kind}"
            if key in self.columns:
                mask = kind_strs == kind
                data[key] = edge_index[:, mask]
        if "kind" not in self.columns and data.get("kind"):
            del data["kind"]

        # Convert everything possible to torch.Tensors
        for key, val in data.items():
            try:
                if not isinstance(val, torch.Tensor):
                    data[key] = torch.tensor(np.array(val))
            except Exception as e:
                log.warning(e)
                pass

        # Construct PyG data
        data = Data.from_dict(data)
        data.num_nodes = G.number_of_nodes()

        # Symmetrize if undirected
        if not G.is_directed():
            # Reorder every edge feature (tensor or not, e.g. kind) to match
            # the symmetrized edge_index.
            edge_index, edge_id = _symmetrize(edge_index, data.num_nodes)
            if "edge_index" in self.columns:
                data.edge_index = edge_index
            for attr in edge_features:
                if attr not in data:
                    continue
                val = getattr(data, attr)
                if isinstance(val, torch.Tensor):
                    setattr(data, attr, val[edge_id])
                else:
                    setattr(data, attr, [val[i] for i in edge_id.tolist()])

            # Edge indices of different kinds
            for kind in kinds:
                key = f"edge_index_{kind}"
                if key in self.columns:
                    edge_index_kind, _ = _symmetrize(
                        getattr(data, key), data.num_nodes
                    )
                    setattr(data, key, edge_index_kind)

        return data

    @staticmethod
    def convert_nx_to_nx(G: nx.Graph) -> nx.Graph:
        """
        Converts NetworkX graph (``nx.Graph``) to NetworkX graph (``nx.Graph``) object. Redundant - returns itself.

        :param G: NetworkX Graph
        :type G: nx.Graph
        :return: NetworkX Graph
        :rtype: nx.Graph
        """
        return G

    @staticmethod
    def convert_dgl_to_nx(G: dgl.DGLGraph) -> nx.Graph:
        """
        Converts a DGL Graph (``dgl.DGLGraph``) to a NetworkX (``nx.Graph``) object. Preserves node and edge attributes.

        :param G: ``dgl.DGLGraph`` to convert to ``NetworkX`` graph.
        :type G: dgl.DGLGraph
        :return: NetworkX graph object.
        :rtype: nx.Graph
        """
        node_attrs = G.node_attr_schemes().keys()
        edge_attrs = G.edge_attr_schemes().keys()
        return dgl.to_networkx(G, node_attrs, edge_attrs)

    @staticmethod
    def convert_pyg_to_nx(G: Data) -> nx.Graph:
        """Converts PyTorch Geometric ``Data`` object to NetworkX graph (``nx.Graph``).

        :param G: Pytorch Geometric Data.
        :type G: torch_geometric.data.Data
        :returns: NetworkX graph.
        :rtype: nx.Graph
        """
        return torch_geometric.utils.to_networkx(G)

    def __call__(self, G: nx.Graph):
        nx_g = eval("self.convert_" + self.src_format + "_to_nx(G)")
        dst_g = eval("self.convert_nx_to_" + self.dst_format + "(nx_g)")
        return dst_g


def convert_nx_to_pyg_data(G: nx.Graph) -> Data:
    # Initialise dict used to construct Data object
    data = {"node_id": list(G.nodes())}

    G = nx.convert_node_labels_to_integers(G)

    # Construct Edge Index
    edge_index = torch.LongTensor(list(G.edges())).t().contiguous()

    # Add node and edge features
    data.update(_feature_lists([d for _, d in G.nodes(data=True)]))
    data.update(_feature_lists([d for _, _, d in G.edges(data=True)]))

    # Add graph-level features
    for feat_name in G.graph:
        data[str(feat_name)] = [G.graph[feat_name]]

    data["edge_index"] = edge_index.view(2, -1)
    data = Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data
