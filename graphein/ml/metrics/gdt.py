import torch
from torchmetrics import Metric

from ...protein.tensor.types import AtomTensor, CoordTensor


def gdt(
    y: Union[CoordTensor, AtomTensor],
    y_hat: Union[CoordTensor, AtomTensor],
    ca_only: bool = True,
    cutoff: float = 4,
    ts: bool = True,
) -> torch.Tensor:
    """Global Distance Deviation Test metric (GDDT).

    https://en.wikipedia.org/wiki/Global_distance_test


    The GDT score is calculated as the largest set of amino acid residues'
    alpha carbon atoms in the model structure falling within a defined
    distance cutoff of their position in the experimental structure, after
    iteratively superimposing the two structures. By the original design the
    GDT algorithm calculates 20 GDT scores, i.e. for each of 20 consecutive distance
    cutoffs (``0.5 Å, 1.0 Å, 1.5 Å, ... 10.0 Å``). For structure similarity assessment
    it is intended to use the GDT scores from several cutoff distances, and scores
    generally increase with increasing cutoff. A plateau in this increase may
    indicate an extreme divergence between the experimental and predicted structures,
    such that no additional atoms are included in any cutoff of a reasonable distance.
    The conventional GDT_TS total score in CASP is the average result of cutoffs at
    ``1``, ``2``, ``4``, and ``8`` Å.

    :param y: Tensor of groundtruth (reference) atom positions.
    :type y: Union[graphein.protein.tensor.CoordTensor, graphein.protein.tensor.AtomTensor]
    :param y_hat: Tensor of atom positions.
    :type y_hat: Union[graphein.protein.tensor.CoordTensor, graphein.protein.tensor.AtomTensor]
    :param ca_only: Whether or not to consider only Ca positions. Default is ``True``.
    :type ca_only: bool
    :param cutoff: Custom threshold to use.
    :type cutoff: float
    :param ts: Whether or not to use "Total Score" mode, where the scores over the thresholds
        ``1, 2, 4, 8`` are averaged (as per CASP).
    :type ts: bool
    :returns: GDT score (torch.FloatTensor)
    :rtype: torch.Tensor
    """
    if y.ndim == 3:
        y = y[:, 1, :] if ca_only else y.reshape(-1, 3)
    if y_hat.ndim == 3:
        y_hat = y_hat[:, 1, :] if ca_only else y_hat.reshape(-1, 3)
    # Get distance between points
    dist = torch.norm(y - y_hat, dim=1)

    if not ts:
        # Return fraction of distances below cutoff
        return (dist < cutoff).sum() / dist.numel()
    # Return mean fraction of distances below cutoff for each cutoff (1, 2, 4, 8)
    count_1 = (dist < 1).sum() / dist.numel()
    count_2 = (dist < 2).sum() / dist.numel()
    count_4 = (dist < 4).sum() / dist.numel()
    count_8 = (dist < 8).sum() / dist.numel()
    return torch.mean(torch.tensor([count_1, count_2, count_4, count_8]))


class GDT_TS(Metric):
    def __init__(self):
        """Torchmetrics implementation of GDT_TS."""
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "correct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        higher_is_better = True
        full_state_update = True

    @property
    def higher_is_better(self):
        return True

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, batch: torch.Tensor
    ):
        """Update method for metric.

        :param pred: Tensor of predictions.
        :type pred: torch.Tensor
        :param target: Tensor of target structures.
        :type target: torch.Tensor
        :param batch. Batch tensor, indicating which indices belong to which example in the batch.
            Assumes a PyTorch Geometric batching scheme.
        type batch: torch.Tensor.
        """
        y = unbatch(target, batch)
        y_hat = unbatch(preds, batch)

        for i, j in zip(y, y_hat):
            self.correct += gdt(i, j)
            self.total += 1

    def compute(self) -> float:
        return self.correct / self.total
