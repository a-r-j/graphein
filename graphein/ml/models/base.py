import abc
import logging
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)


class Model(abc.ABC, nn.Module):
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load(self, file_path: str) -> None:
        """Load the model weights from disk."""
        self.load_state_dict(torch.load(file_path, map_location=self.device))

    def load_average_weights(self, file_paths: List[str]) -> None:
        """Load several weights and average them"""
        state: Dict[str, torch.Tensor] = {}
        for file_path in file_paths:
            state_new = torch.load(file_path, map_location=self.device)
            keys = state.keys()

            if len(keys) == 0:
                state = state_new
            else:
                for key in keys:
                    state[key] += state_new[key]

        num_weights = len(file_paths)
        for key in state.keys():
            state[key] = state[key] / num_weights

        self.load_state_dict(state)

    def save(self, file_path: str) -> None:
        """Save the model weights to disk."""
        torch.save(self.state_dict(), file_path)

    @staticmethod
    def assign_logits_to_classes(logits: List):
        """
        # Todo type hinting
        Takes output logits and assigns them to the predicted class
        """
        # Todo check
        return [int(logit > 0.5) for logit in logits]


class BinaryClassifierModel(nn.Module):
    def __init__(self, protein_encoder, ligand_encoder, classifier):
        super(BinaryClassifierModel, self).__init__()

        self.protein_encoder = protein_encoder
        # self.ligand_encoder = ligand_encoder
        self.classifier = classifier

    @abc.abstractmethod
    def forward(self, x_protein, x_ligand) -> torch.Tensor:
        protein_emb = self.protein_encoder(x_protein)
        # ligand_emb = self.ligand_encoder(x_ligand)
        # combi = torch.cat([protein_emb, ligand_emb])
        return self.classifier(protein_emb)

    def predict(self, input_a, input_b, targets, loss_fn, training=True):
        # Todo check list unwrapping
        target = targets[0]
        output = self(input_a, input_b)
        loss = loss_fn(target, output)

        # Todo check if we can move this into abc
        # Todo check if this is necessary
        pred = [int(o > 0.5) for o in output.detach().cpu().numpy()]
        real = [int(r > 0.5) for r in target]

        return pred, real, loss, output, target
