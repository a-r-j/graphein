"""Tests for graphein.protein.tensor.dssp."""

from pathlib import Path

import numpy as np
import pytest

from graphein.protein.resi_atoms import STANDARD_AMINO_ACIDS
from graphein.protein.tensor.dssp import (
    DSSP8_STATES,
    _hbond_energy,
    _hbonds,
    assign_dssp,
    assign_dssp8,
    assign_dssp8_np,
    assign_dssp_np,
    get_hbond_map,
    get_hbond_map_np,
)
from graphein.protein.tensor.io import protein_to_pyg

try:
    import torch

    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False

DATA = Path(__file__).parent.parent / "test_data"


def _coords(name):
    return protein_to_pyg(path=str(DATA / f"{name}.pdb")).coords.double()


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
@pytest.mark.parametrize("name", ["4hhb", "1lds", "P04629"])
def test_dssp_backends_and_batching_agree(name):
    x = _coords(name)
    x = torch.stack([x, x + 0.5 * torch.randn_like(x)])
    ss = assign_dssp(x)
    assert ss.shape == (2, x.shape[1], 3)
    assert (ss[..., 0] == ~(ss[..., 1] | ss[..., 2])).all()
    assert (ss[1] == assign_dssp(x[1])).all()
    assert (ss.numpy() == assign_dssp_np(x.numpy())).all()
    np.testing.assert_allclose(
        get_hbond_map(x).numpy(), get_hbond_map_np(x.numpy()), atol=1e-6
    )


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
@pytest.mark.parametrize("noise", [0.0, 1.0, 3.0])
def test_pruned_hbonds_equal_dense(noise):
    x = _coords("P04629")
    x = x + noise * torch.randn_like(x)
    assert (_hbonds(x, torch) == (_hbond_energy(x, torch) < -0.5)).all()


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_dssp_known_folds():
    loop, helix, strand = assign_dssp(_coords("4hhb")).sum(0)
    assert strand == 0 and helix > 0.7 * (loop + helix)  # all-alpha
    loop, helix, strand = assign_dssp(_coords("1lds")).sum(0)
    assert strand > 3 * helix  # beta-2-microglobulin, all-beta


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_dssp_donor_mask_and_grad():
    x = _coords("1lds").float().requires_grad_()
    hb = get_hbond_map(x)
    hb.sum().backward()
    assert torch.isfinite(x.grad).all()
    mask = torch.ones(x.shape[0], dtype=torch.bool)
    mask[10] = False
    assert (get_hbond_map(x, donor_mask=mask)[10] == 0).all()
    assert (
        assign_dssp(x, mask)
        == assign_dssp_np(x.detach().numpy(), mask.numpy())
    ).all()


# mkdssp 4.6.1 output (polyproline "P" mapped to loop)
MKDSSP = {
    "1lds": (
        "---B--EEEEEEE--GGGTT--EEEEEEEEEBSS--EEEEEETTEE-S---EEEEEESSTT-EE"
        "EEEEEE----TT--EEEEEE-TT-SS-EEEE--"
    ),
    "4hhb": (
        "---HHHHHHHHHHHHHHTTTHHHHHHHHHHHHHHH-GGGGGG-TTS--STT-HHHHHHHHHHHH"
        "HHHHHHHTGGGHHHHTHHHHHIIIIII---THHHHHHHHHHHHHHHHH-TTT--HHHHHHHHHH"
        "HHHHHHHHTTT------HHHHHHHHHHHTT--HHHHHHHHHHHHHHHSGGGGGG-GGG---SSH"
        "HHHHH-HHHHHHHHHHHHHHHHHHTTGGGHHHHHHHHHHIIIIII---THHHHHHHHHHHHHHH"
        "HHHGGGS-HHHHHHHHHHHHHHHHHHHTT-----HHHHHHHHHHHHHHGGGHHHHHHHHHHHHH"
        "HH-GGGGGG-TTS--STT-HHHHHHHHHHHHHHHHHHHTGGGHHHHTHHHHHHIIIII---THH"
        "HHHHHHHHHHHHHHH-TTT--HHHHHHHHHHHHHHHHHHHTT------HHHHHHHHHHHTT--H"
        "HHHHHHHHHHHHHHSGGGGGG-GGG----SHHHHHT-HHHHHHHHHHHHHHHHHHT-TT-HHHH"
        "HHHHHHIIIIIS---THHHHHHHHHHHHHHHHHHGGGS-HHHHHHHHHHHHHHHHHHHTT--"
    ),
}


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
@pytest.mark.parametrize("name", list(MKDSSP))
def test_dssp8_matches_mkdssp(name):
    protein = protein_to_pyg(path=str(DATA / f"{name}.pdb"))
    not_pro = protein.residue_type != STANDARD_AMINO_ACIDS.index("P")
    ss = assign_dssp8(protein.coords, donor_mask=not_pro)
    assert "".join(DSSP8_STATES[i] for i in ss) == MKDSSP[name]
    ss_np = assign_dssp8_np(protein.coords.numpy(), not_pro.numpy())
    assert (ss.numpy() == ss_np).all()
    batch = assign_dssp8(
        torch.stack([protein.coords] * 2), torch.stack([not_pro] * 2)
    )
    assert (batch == ss).all()
