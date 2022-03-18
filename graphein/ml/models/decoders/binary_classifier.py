import torch
import torch.functional as F
import torch.nn as nn

from ..model_config import ClassifierConfig
from ..utils.activations import build_activation


class BinaryClassifier(nn.Module):
    def __init__(self, config: ClassifierConfig, input_dim: int = 256):
        super(BinaryClassifier, self).__init__()
        self.input_dim = input_dim
        self.layer_dims = config.layer_dims

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropout = nn.Dropout(config.dropout)

        # First layer
        self.layers.append(nn.Linear(self.input_dim, self.layer_dims[0]))
        # Construct subsequent layers
        for i, _ in enumerate(self.layer_dims[:-1]):
            self.layers.append(
                nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])
            )

        for act_type in config.activations:
            self.activations.append(build_activation(act_type))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Iterate over layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Do not apply dropout to output layer
            if i != len(self.layers):
                self.dropout(x)

            # Apply activations
            if self.activations[i] is not None:
                x = self.activations[i](x)

        return x
