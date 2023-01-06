import torch

from graphein.protein.tensor.geometry import whole_protein_kabsch
from graphein.protein.tensor.reconstruction import dist_mat_to_coords


def test_dist_mat_to_coords():
    # Test that the distance matrix is recovered from the coordinates, with a
    # small error.
    for _ in range(10):
        coords = torch.rand((10, 3))
        d = torch.cdist(coords, coords)
        X = dist_mat_to_coords(d)
        assert torch.allclose(d, torch.cdist(X, X), atol=1e-4)
        X_aligned = whole_protein_kabsch(X, coords)
        assert torch.allclose(coords, X_aligned, atol=1e-4)
        return coords, X, X_aligned
