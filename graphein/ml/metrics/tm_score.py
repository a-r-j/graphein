from typing import Union

import numpy as np
import torch
from torch_geometric.utils import unbatch
from torchmetrics import Metric

from ...protein.tensor.types import AtomTensor, CoordTensor


def tm_score(
    y_hat: Union[CoordTensor, AtomTensor], y: Union[CoordTensor, AtomTensor]
) -> torch.Tensor:
    """Compute TMScore between ``y_hat`` and ``y``. Requires aligned structures.

    TM-score is a measure of similarity between two protein structures.
    The TM-score is intended as a more accurate measure of the global
    similarity of full-length protein structures than the often used RMSD
     measure. The TM-score indicates the similarity between two structures
    by a score between ``[0, 1]``, where 1 indicates a perfect match
    between two structures (thus the higher the better). Generally scores
    below 0.20 corresponds to randomly chosen unrelated proteins whereas
    structures with a score higher than 0.5 assume roughly the same fold.
    A quantitative study shows that proteins of TM-score = 0.5 have a
    posterior probability of 37% in the same CATH topology family and of
    13% in the same SCOP fold family. The probabilities increase rapidly
    when TM-score > 0.5. The TM-score is designed to be independent of
    protein lengths.

    https://en.wikipedia.org/wiki/Template_modeling_score

    :param y_hat: Tensor of atom positions (aligned to ``y``).
    :type y_hat: Union[graphein.protein.tensor.types.CoordTensor,
        graphein.protein.tensor.types.AtomTensor]
    :param y: Tensor of groundtruth/reference atom positions.
    :type y: Union[graphein.protein.tensor.types.CoordTensor,
        graphein.protein.tensor.types.AtomTensor]
    :returns: TMScore of aligned pair.
    :rtype: torch.Tensor
    """
    # Get CA
    if y_hat.ndim == 3:
        y_hat = y_hat[:, 1, :]
    if y.ndim == 3:
        y = y[:, 1, :]

    l_target = y.shape[0]

    d0_l_target = 1.24 * np.power(l_target - 15, 1 / 3) - 1.8

    di = torch.pairwise_distance(y_hat, y)

    return torch.sum(1 / (1 + (di / d0_l_target) ** 2)) / l_target


class TMScore(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "correct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        higher_is_better: bool = True
        full_state_update: bool = True

    @property
    def higher_is_better(self):
        return True

    def update(
        self,
        pred: Union[CoordTensor, AtomTensor],
        target: Union[CoordTensor, AtomTensor],
        batch: torch.Tensor,
    ):
        y = unbatch(target, batch)
        y_hat = unbatch(pred, batch)

        for i, j in zip(y, y_hat):
            self.correct += tm_score(i, j)
            self.total += 1

    def compute(self):
        return self.correct / self.total
