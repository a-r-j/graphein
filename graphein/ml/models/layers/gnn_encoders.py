from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import FiLMConv, GATConv, GCNConv, global_add_pool

from ..model_config import LayerType

log = logging.getLogger(__name__)


class GraphEncoder(nn.Module):
    def __init__(self, config, input_dim):
        super(GraphEncoder, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.layer_dims = config.layer_dims

        # Initialise layer module lists
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        # Build first layer
        self.layers.append(
            build_graph_layer(
                layer_type=config.layers[0],
                input_dim=self.input_dim,
                output_dim=self.layer_dims[0],
                add_self_loops=False,  # Todo add to config
            )
        )

        # Iteratively build remaining layers
        for i, _ in enumerate(self.layer_dims[:-1]):
            self.layers.append(
                build_graph_layer(
                    layer_type=config.layers[i + 1],
                    input_dim=self.layer_dims[i],
                    output_dim=self.layer_dims[i + 1],
                    add_self_loops=False,  # todo add to config
                )
            )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Iterate over layers
        for i, layer in enumerate(self.layers):
            if self.config.layers[i] is LayerType.GCN:
                x = layer(x, edge_index)
            elif self.config.layers[i] is LayerType.GNN_FILM:
                x = layer(x, edge_index)
            else:
                x = layer(x)

        # Do readout
        x = global_add_pool(x, batch)
        return x


def build_graph_layer(
    layer_type: LayerType,
    input_dim: int,
    output_dim: int,
    add_self_loops: bool,
):
    if layer_type == LayerType.GCN:
        return GCNConv(input_dim, output_dim, add_self_loops)
    elif layer_type == LayerType.GAT:
        return GATConv(input_dim, output_dim, add_self_loops)
    elif layer_type == LayerType.GNN_FILM:
        return FiLMConv(input_dim, output_dim, add_self_loops)
    elif layer_type == LayerType.LINEAR:
        return nn.Linear(in_features=input_dim, out_features=output_dim)
    else:
        message = f"Layer Type: {layer_type} not supported."
        log.error(message)
        raise ValueError(message)
