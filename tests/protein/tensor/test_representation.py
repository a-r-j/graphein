"""Tests for graphein.protein.tensor.representation."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import pytest

from graphein.protein.tensor.geometry import kabsch
from graphein.protein.tensor.reconstruction import dist_mat_to_coords
from graphein.protein.tensor.representation import (
    get_c_alpha,
    get_full_atom_coords,
)

try:
    import torch

    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_get_c_alpha():
    """Test get_c_alpha"""
    x = torch.rand((2, 37, 3))
    assert get_c_alpha(x).shape == (2, 3)
    assert torch.allclose(get_c_alpha(x), x[:, 1, :])

    x = torch.rand((2, 3))
    assert get_c_alpha(x).shape == (2, 3)
    assert torch.allclose(get_c_alpha(x), x)


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_get_full_atom_coords():
    # Test case 1
    atom_tensor = torch.Tensor(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    )
    coords, residue_index, atom_type = get_full_atom_coords(atom_tensor)
    assert coords.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    assert residue_index.tolist() == [0, 0, 1, 1]
    assert atom_type.tolist() == [0, 1, 0, 1]

    # Test case 2
    atom_tensor = torch.Tensor(
        [[[1, 2, 3], [4, 5, 6]], [[1e-5, 1e-5, 1e-5], [1e-5, 1e-5, 1e-5]]]
    )
    coords, residue_index, atom_type = get_full_atom_coords(atom_tensor)
    assert coords.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert residue_index.tolist() == [0, 0]
    assert atom_type.tolist() == [0, 1]

    # Test case 3
    atom_tensor = torch.Tensor(
        [[[1, 2, 3], [4, 5, 6]], [[1e-5, 1e-5, 1e-5], [10, 11, 12]]]
    )
    coords, residue_index, atom_type = get_full_atom_coords(atom_tensor)
    assert coords.tolist() == [[1, 2, 3], [4, 5, 6], [10, 11, 12]]
    assert residue_index.tolist() == [0, 0, 1]
    assert atom_type.tolist() == [0, 1, 1]

    # Test case 4
    atom_tensor = torch.Tensor(
        [[[1, 2, 3], [4, 5, 6]], [[1e-6, 1e-6, 1e-6], [1e-6, 1e-6, 1e-6]]]
    )
    coords, residue_index, atom_type = get_full_atom_coords(
        atom_tensor, fill_value=1e-6
    )
    assert coords.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert residue_index.tolist() == [0, 0]
    assert atom_type.tolist() == [0, 1]
