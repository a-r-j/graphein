"""Tensor-based DSSP secondary structure assignment.

Optimised re-implementation of PyDSSP
(https://github.com/ShintaroMinami/PyDSSP; MIT License, Copyright (c) 2022
Shintaro Minami) for Graphein's tensor representation. Speedups come from:

* computing pairwise distances per component, never materialising
  ``(L, L, 3)`` arrays;
* for discrete assignment, evaluating the full H-bond energy only for O-N
  pairs within a provably sufficient radius (see :func:`_prune_radius`) and
  thresholding the energy directly instead of the smoothed H-bond map;
* replacing PyDSSP's ``(L, L, 3, 3)`` unfolded bridge maps with shifted slices
  of a boolean H-bond map.

Assignments match PyDSSP, except that PyDSSP drops H-bonds within ~1e-4
kcal/mol of the cutoff in float32 (its smoothed map rounds to 0 there); here
``E < -0.5`` is applied exactly.

NumPy (:func:`assign_dssp_np`, :func:`get_hbond_map_np`) and PyTorch
(:func:`assign_dssp`, :func:`get_hbond_map`) variants share one
implementation. The PyTorch variant runs on any device and
:func:`get_hbond_map` is differentiable.

:func:`assign_dssp8` (and :func:`assign_dssp8_np`) provide full 8-state DSSP
using mkdssp's own H-bond model and rules; output is identical to mkdssp
4.6.1 (polyproline aside). Since mkdssp keeps at most two H-bonds per donor,
everything after the H-bond energies works on an O(L) edge list.

:func:`accessibility` (and :func:`accessibility_np`) compute DSSP's solvent
accessible surface area per residue, identical to mkdssp 4.6.1.
"""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from loguru import logger as log
from scipy.spatial import cKDTree

from graphein.utils.dependencies import import_message

if TYPE_CHECKING:
    from graphein.protein.tensor.types import AtomTensor

try:
    import torch
except ImportError:
    message = import_message(
        "graphein.protein.tensor.dssp",
        package="torch",
        conda_channel="pytorch",
        pip_install=True,
    )
    log.debug(message)

CONST_Q1Q2 = 0.084
CONST_F = 332
DEFAULT_CUTOFF = -0.5
DEFAULT_MARGIN = 1.0


def _zeros(ref, shape, dtype=None):
    if isinstance(ref, np.ndarray):
        return np.zeros(shape, dtype=ref.dtype if dtype is None else dtype)
    return ref.new_zeros(shape, dtype=dtype)


def _unit(v, xp):
    return v / xp.sqrt((v * v).sum(-1))[..., None]


def _rsqrt(v, xp):
    # Fastest inverse sqrt per backend (np.power is ~7x slower than this).
    return 1 / np.sqrt(v) if xp is np else torch.rsqrt(v)


def _pdist2(a, b):
    # Summing per component avoids materialising an (..., L, L, 3) array.
    d2 = 0
    for i in range(3):
        d = a[..., :, None, i] - b[..., None, :, i]
        d2 = d2 + d * d
    return d2


def _check_shape(x):
    if x.ndim not in (3, 4) or x.shape[-1] != 3 or x.shape[-2] < 4:
        raise ValueError(
            "Expected coordinates of shape (L, A, 3) or (B, L, A, 3) with "
            f"A >= 4 (N, CA, C, O first), got {tuple(x.shape)}"
        )


def _backbone(x, xp):
    """N, pseudo-H of residues ``1..L-1``; C, O of residues ``0..L-2``."""
    _check_shape(x)
    n, ca = x[..., 1:, 0, :], x[..., 1:, 1, :]
    c, o = x[..., :-1, 2, :], x[..., :-1, 3, :]
    # Pseudo-H on the bisector of the C(i-1)->N and CA->N bonds (as in PyDSSP)
    h = n + 1.01 * _unit(_unit(n - c, xp) + _unit(n - ca, xp), xp)
    return n, h, c, o


def _hbond_energy(x, xp):
    """H-bond energy (kcal/mol) of shape ``(..., L-1, L-1)``.

    Entry ``[m, k]`` is the energy between the N-H of residue ``m + 1`` and
    the C=O of residue ``k``.
    """
    n, h, c, o = _backbone(x, xp)

    def inv(a, b):
        return _rsqrt(_pdist2(a, b), xp)

    return (CONST_Q1Q2 * CONST_F) * (
        inv(n, o) + inv(h, c) - inv(h, o) - inv(n, c)
    )


def _prune_radius(b: float, cutoff: float) -> float:
    """O-N distance beyond which ``|E| <= |cutoff|``, so no H-bond exists.

    With ``b`` the longest C=O bond and ``|N-H| = 1.01``, the triangle
    inequality bounds ``|E| / K`` by ``b / (r (r - b)) + b / ((r - 1.01)
    (r - 1.01 - b))``, which decreases in ``r``. ~13 Angstrom for real
    proteins.
    """
    K, s = CONST_Q1Q2 * CONST_F, b + 1.01

    def bound(r):
        return K * b * (1 / (r * (r - b)) + 1 / ((r - 1.01) * (r - s)))

    lo, hi = s, s + 1.0
    while bound(hi) > abs(cutoff):
        lo, hi = hi, 2 * hi
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if bound(mid) > abs(cutoff) else (lo, mid)
    return hi * 1.01  # headroom for floating point error in E


def _hbonds(x, xp, cutoff=DEFAULT_CUTOFF):
    """Boolean ``E < cutoff`` of shape ``(..., L-1, L-1)`` (see
    :func:`_hbond_energy`), evaluating the full energy only for O-N pairs
    within :func:`_prune_radius`. Exactly equal to the dense computation."""
    n, h, c, o = _backbone(x, xp)
    d2 = _pdist2(n, o)
    b = float(xp.nan_to_num(xp.sqrt(((c - o) ** 2).sum(-1))).max())
    near = d2 < _prune_radius(b, cutoff) ** 2
    idx = np.nonzero(near) if xp is np else torch.nonzero(near, as_tuple=True)
    don, acc = idx[:-1], idx[:-2] + idx[-1:]

    def inv(a, b):
        return _rsqrt(((a - b) ** 2).sum(-1), xp)

    e = (CONST_Q1Q2 * CONST_F) * (
        _rsqrt(d2[idx], xp)
        + inv(h[don], c[acc])
        - inv(h[don], o[acc])
        - inv(n[don], c[acc])
    )
    near[idx] = e < cutoff
    return near


def _allowed(ref, donor_mask):
    """``(..., L, L)`` bool mask [donor, acceptor] excluding local pairs
    (``0 <= donor - acceptor <= 2``) and masked donors (e.g. proline)."""
    L = ref.shape[-1]
    m = _zeros(ref, ref.shape, dtype=bool)
    m[..., 1:, :-1] = True
    for k in (0, 1, 2):
        m[..., range(k, L), range(L - k)] = False
    if donor_mask is not None:
        m &= donor_mask[..., :, None]
    return m


def _hbond_map(x, xp, donor_mask, cutoff, margin, return_e):
    e = _hbond_energy(x, xp)
    L = x.shape[-3]
    out = _zeros(e, e.shape[:-2] + (L, L))
    out[..., 1:, :-1] = e
    if return_e:
        return out
    hb = xp.clip(cutoff - margin - out, -margin, margin)
    hb = (xp.sin(hb / margin * (xp.pi / 2)) + 1.0) / 2
    return hb * _allowed(out, donor_mask)


def _spread(h, xp, n):
    """``out[i] = h[i] | h[i-1] | ... | h[i-n+1]``."""
    out = xp.zeros_like(h)
    for s in range(n):
        out[..., s:] |= h[..., : h.shape[-1] - s]
    return out


def _bridges(hb):
    """Parallel and antiparallel bridges between residues ``I`` (rows) and
    ``J`` (cols), both in ``[1, L-2]``, from ``hb[acceptor, donor]``."""
    hbT = hb.swapaxes(-1, -2)
    c = slice(1, -1)
    parallel = (hb[..., :-2, c] & hbT[..., 2:, c]) | (
        hbT[..., c, :-2] & hb[..., c, 2:]
    )
    antiparallel = (hb[..., c, c] & hbT[..., c, c]) | (
        hb[..., :-2, 2:] & hbT[..., 2:, :-2]
    )
    return parallel, antiparallel


def _assign(x, xp, donor_mask):
    L = x.shape[-3]
    # hb[i, j]: C=O of residue i accepts an H-bond from N-H of residue j.
    # Thresholding the energy is equivalent to PyDSSP's ``hbond_map > 0``.
    hb = _zeros(x, x.shape[:-3] + (L, L), dtype=bool)
    hb[..., :-1, 1:] = _hbonds(x, xp).swapaxes(-1, -2)
    for k in (0, 1, 2):
        hb[..., range(L - k), range(k, L)] = False
    if donor_mask is not None:
        hb &= donor_mask[..., None, :]

    # Helices: consecutive n-turns, with alpha (4) taking priority over 3/5.
    def helix_starts(k):
        t = xp.diagonal(hb, k, -2, -1)
        h = xp.zeros_like(hb[..., 0, :])
        h[..., 1 : L - k] = t[..., :-1] & t[..., 1:]
        return h

    helix4 = _spread(helix_starts(4), xp, 4)
    free = ~helix4
    free[..., :-1] &= ~helix4[..., 1:]
    helix = (
        helix4
        | _spread(helix_starts(3) & free, xp, 3)
        | _spread(helix_starts(5) & free, xp, 5)
    )

    parallel, antiparallel = _bridges(hb)
    strand = xp.zeros_like(helix)
    strand[..., 1:-1] = (parallel | antiparallel).any(-1)

    return xp.stack([~helix & ~strand, helix, strand], -1)


# 8-state DSSP, following mkdssp 4.x (https://github.com/PDB-REDO/dssp).
DSSP8_STATES = "-HBEGITS"
"""8-state codes, indexed by the output of :func:`assign_dssp8`. ``"-"`` is
loop; ``H``: alpha helix, ``B``: isolated bridge, ``E``: strand, ``G``: 3-10
helix, ``I``: pi helix, ``T``: turn, ``S``: bend."""
_L, _H, _B, _E, _G, _I, _T, _S = range(8)
# mkdssp constants (float32 in the original)
_DSSP_K = float(np.float32(27.888))
_DSSP_MAX_CA = 9.0
_DSSP_MIN_DIST = 0.5
_DSSP_MIN_E = -9.9
_DSSP_MAX_PEPTIDE = 2.5


def _dssp_energies(x, xp, donor_mask):
    """H-bond energies of every pair mkdssp evaluates, as NumPy ``(donor,
    acceptor, energy)`` sorted by donor, energy, then acceptor (mkdssp's tie
    order).

    Unlike PyDSSP: H lies 1 Angstrom from N along the preceding C=O; only
    pairs with CA-CA <= 9 Angstrom are evaluated; and energies are rounded to
    0.001 kcal/mol.
    """
    L = x.shape[0]
    n, ca, c, o = (x[:, k] for k in range(4))
    h = n * 1
    h[1:] = h[1:] + _unit(c[:-1] - o[:-1], xp)
    near_ca = _pdist2(ca, ca) <= _DSSP_MAX_CA**2

    # Donor/acceptor pairs mkdssp evaluates; it skips donor == acceptor + 1.
    pairs = near_ca | False  # copy
    pairs[range(L), range(L)] = False
    pairs[range(1, L), range(L - 1)] = False
    if donor_mask is not None:
        pairs &= donor_mask[:, None]
    don, acc = (
        np.nonzero(pairs) if xp is np else torch.nonzero(pairs, as_tuple=True)
    )

    def dist(a, b):  # float32 like mkdssp, then promoted to float64
        d = a[don] - b[acc]
        d2 = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]
        return _to_np(xp.sqrt(d2)).astype(np.float64)

    # mkdssp's operation order and float64 energy, so the 0.001 rounding
    # (and the -0.5 cutoff) match it exactly.
    ho, hc, nc, no = dist(h, o), dist(h, c), dist(n, c), dist(n, o)
    k = -_DSSP_K
    e = k / ho - k / hc + k / nc - k / no
    too_close = np.minimum(np.minimum(ho, hc), np.minimum(nc, no))
    e = np.where(too_close < _DSSP_MIN_DIST, _DSSP_MIN_E, e)
    e = np.sign(e) * np.floor(np.abs(e) * 1000 + 0.5) / 1000  # C++ round
    e = np.clip(e, _DSSP_MIN_E, None)

    don, acc = _to_np(don), _to_np(acc)
    order = np.lexsort((acc, e, don))
    return don[order], acc[order], e[order]


def _rank(group: np.ndarray) -> np.ndarray:
    """Position of each entry within its run of equal (sorted) ``group``."""
    return np.arange(len(group)) - np.searchsorted(group, group)


def _dssp_hbonds(don, acc, e):
    """H-bonds (``E < -0.5`` among each donor's two lowest-energy acceptors)
    as ``(acceptor, donor)`` index arrays."""
    keep = (_rank(don) < 2) & (e < DEFAULT_CUTOFF)
    return acc[keep], don[keep]


def _dssp_partners(don, acc, e, L):
    """mkdssp's two lowest-energy acceptors per donor (``N-H-->O``) and
    donors per acceptor (``O-->H-N``); only negative energies are kept.

    :return: ``((L, 2) acceptor indices, (L, 2) energies), ((L, 2) donor
        indices, (L, 2) energies)``; index ``-1`` and energy ``0`` if none.
    """

    def top2(key, other, en):  # arrays sorted by (key, energy, other)
        r = _rank(key)
        k = r < 2
        idx, out = np.full((L, 2), -1), np.zeros((L, 2))
        idx[key[k], r[k]], out[key[k], r[k]] = other[k], en[k]
        return idx, out

    neg = e < 0
    don, acc, e = don[neg], acc[neg], e[neg]
    order = np.lexsort((don, e, acc))
    return top2(don, acc, e), top2(acc[order], don[order], e[order])


def _to_np(a):
    return a if isinstance(a, np.ndarray) else a.cpu().numpy()


def _dssp_breaks(x: np.ndarray) -> np.ndarray:
    """``(L-1,)`` chain breaks between consecutive residues (C-N > 2.5)."""
    d = x[1:, 0] - x[:-1, 2]
    return d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2 > _DSSP_MAX_PEPTIDE**2


def _dssp_numbers(x: np.ndarray) -> np.ndarray:
    """mkdssp residue numbers: 1-based, skipping one at each chain break."""
    return np.arange(1, len(x) + 1) + np.r_[0, np.cumsum(_dssp_breaks(x))]


def _dihedral(p1, p2, p3, p4):
    """mkdssp's float32 dihedral angle (degrees); 360 if undefined."""

    def cross(a, b):
        return np.stack(
            [
                a[:, 1] * b[:, 2] - b[:, 1] * a[:, 2],
                a[:, 2] * b[:, 0] - b[:, 2] * a[:, 0],
                a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1],
            ],
            -1,
        )

    def dot(a, b):
        return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]

    z = p2 - p3
    p, x = cross(z, p1 - p2), cross(z, p4 - p3)
    y = cross(z, x)
    u, v = dot(x, x), dot(y, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        u2, v2 = dot(p, x) / np.sqrt(u), dot(p, y) / np.sqrt(v)
        angle = np.arctan2(v2, u2) * np.float32(180) / np.float32(np.pi)
    ok = (u > 0) & (v > 0) & ((u2 != 0) | (v2 != 0))
    return np.where(ok, angle, np.float32(360))


def _dssp_phi_psi(x: np.ndarray):
    """mkdssp phi and psi (degrees, 360 where undefined) of a float32
    ``(L, 4, 3)`` backbone."""
    phi = np.full(len(x), np.float32(360))
    psi = np.full(len(x), np.float32(360))
    ok = ~_dssp_breaks(x)
    n, ca, c = x[:, 0], x[:, 1], x[:, 2]
    phi[1:] = np.where(ok, _dihedral(c[:-1], n[1:], ca[1:], c[1:]), 360)
    psi[:-1] = np.where(ok, _dihedral(n[:-1], ca[:-1], c[:-1], n[1:]), 360)
    return phi, psi


def _assign8(x, xp, donor_mask, energies=None):
    # Everything after the H-bonds is O(L) (at most 2 bonds per donor), so
    # it runs in NumPy on the H-bond edge list rather than dense maps.
    L = x.shape[0]
    if energies is None:
        energies = _dssp_energies(x, xp, donor_mask)
    acc, don = _dssp_hbonds(*energies)
    keys = np.sort(acc * L + don)
    x = _to_np(x)

    def hb(a, d):  # vectorised Hbond(a, d): C=O of a accepts from N-H of d
        valid = (a >= 0) & (a < L) & (d >= 0) & (d < L)
        return valid & np.isin(a * L + d, keys)

    # Chain breaks: C(i-1)-N(i) > 2.5 Angstrom. seg[i] is i's segment id.
    seg = np.concatenate([[0], np.cumsum(_dssp_breaks(x))])

    def nobreak(a, b):
        return seg[a] == seg[b]

    # Bridges (i < j). Each bridge pattern contains a specific H-bond, so
    # candidates come from the edge list: for H-bond (a, d), parallel
    # Hbond(i-1, j) & Hbond(j, i+1) gives (a+1, d); Hbond(j-1, i) &
    # Hbond(i, j+1) gives (d, a+1); antiparallel Hbond(i, j) & Hbond(j, i)
    # gives (a, d); Hbond(i-1, j+1) & Hbond(j-1, i+1) gives (a+1, d-1).
    i = np.concatenate([acc + 1, don, acc, acc + 1])
    j = np.concatenate([don, acc + 1, don, don - 1])
    k = np.unique(np.minimum(i, j) * L + np.maximum(i, j))  # sorted (i, j)
    i, j = k // L, k % L
    d = x[i, 1] - x[j, 1]  # CA-CA, summed as in _pdist2
    ok = (i >= 1) & (i < j) & (j <= L - 2)
    ok &= d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2 <= _DSSP_MAX_CA**2
    i, j = i[ok], j[ok]
    ok = nobreak(i - 1, i + 1) & nobreak(j - 1, j + 1)
    par = ok & ((hb(i - 1, j) & hb(j, i + 1)) | (hb(j - 1, i) & hb(i, j + 1)))
    anti = (ok & ~par) & (
        (hb(i, j) & hb(j, i)) | (hb(i - 1, j + 1) & hb(j - 1, i + 1))
    )
    is_bridge = par | anti
    i, j = i[is_bridge].tolist(), j[is_bridge].tolist()
    is_par = par[is_bridge].tolist()

    # Ladders: consecutive bridges of one type. Bridges arrive sorted by
    # (i, j), so each ladder has a unique next (type, i, j) it can absorb.
    ladders, nxt = [], {}
    for bi, bj, p in zip(i, j, is_par):
        k = nxt.pop((p, bi, bj), None)
        if k is None:
            ladders.append((p, [bi], [bj]))
            k = len(ladders) - 1
        else:
            ladders[k][1].append(bi)
            ladders[k][2].append(bj) if p else ladders[k][2].insert(0, bj)
        J = ladders[k][2]
        nxt[(p, bi + 1, J[-1] + 1 if p else J[0] - 1)] = k

    # Join ladders across beta-bulges (mkdssp uses unsigned differences).
    def udiff(a, b):
        return a - b if a >= b else float("inf")

    ladders.sort(key=lambda lad: lad[1][0])
    a = 0
    while a < len(ladders):
        b = a + 1
        while b < len(ladders):
            p, I, J = ladders[a]
            p2, I2, J2 = ladders[b]
            if I2[0] >= I[-1] + 6:
                break  # sorted by start, so no later ladder can join
            di = udiff(I2[0], I[-1])
            dj = udiff(J2[0], J[-1]) if p else udiff(J[0], J2[-1])
            if (
                p != p2
                or not nobreak(min(I[0], I2[0]), max(I[-1], I2[-1]))
                or not nobreak(min(J[0], J2[0]), max(J[-1], J2[-1]))
                or di >= 6
                or (I[-1] >= I2[0] and I[0] <= I2[-1])
                or not ((dj < 6 and di < 3) or dj < 3)
            ):
                b += 1
                continue
            ladders[a] = (p, I + I2, J + J2 if p else J2 + J)
            del ladders[b]
        a += 1

    ss = np.zeros(L, dtype=np.int64)
    for _, I, J in ladders:
        code = _E if len(I) > 1 else _B
        for lo, hi in ((I[0], I[-1]), (J[0], J[-1])):
            r = ss[lo : hi + 1]
            r[r != _E] = code

    # n-turns at i: Hbond(i, i + n) within one segment.
    idx = np.arange(L)
    turn = {}
    for k in (3, 4, 5):
        t = np.zeros(L, dtype=bool)
        t[: max(L - k, 0)] = hb(idx[:-k], idx[k:]) & nobreak(idx[:-k], idx[k:])
        turn[k] = t

    def starts(k):  # consecutive n-turns at i - 1 and i
        s = np.zeros(L, dtype=bool)
        s[1:] = turn[k][1:] & turn[k][:-1]
        return s

    def window_any(m, k):  # any(m[i : i + k])
        return _spread(m[::-1].copy(), np, k)[::-1]

    ss[_spread(starts(4), np, 4)] = _H
    for k, code, blocked in ((3, _G, (_H, _B, _E)), (5, _I, (_B, _E, _G))):
        ok = starts(k) & ~window_any(np.isin(ss, blocked), k)
        ss[_spread(ok, np, k)] = code

    # Turns (inside an n-turn) and bends (kappa > 70) on remaining loops.
    in_turn = np.zeros(L, dtype=bool)
    for k in (3, 4, 5):
        in_turn[1:] |= _spread(turn[k], np, k - 1)[:-1]
    ca = x[:, 1]
    v1, v2 = ca[2:-2] - ca[:-4], ca[4:] - ca[2:-2]
    norm2 = (v1 * v1).sum(-1) * (v2 * v2).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(norm2 > 0, (v1 * v2).sum(-1) / np.sqrt(norm2), 0)
        kappa = np.degrees(np.arctan2(np.sqrt(1 - cos * cos), cos))
    bend = np.zeros(L, dtype=bool)
    bend[2:-2] = (kappa > 70) & nobreak(idx[:-4], idx[4:])
    loop = ss == _L
    loop[[0, -1]] = False
    ss[loop & in_turn] = _T
    ss[loop & ~in_turn & bend] = _S
    return ss


def _assign8_batched(x, xp, donor_mask):
    _check_shape(x)
    if x.ndim == 3:
        return _assign8(x, xp, donor_mask)
    masks = [None] * len(x) if donor_mask is None else donor_mask
    return np.stack([_assign8(xi, xp, m) for xi, m in zip(x, masks)])


def get_hbond_map(
    x: AtomTensor,
    donor_mask: Optional[torch.Tensor] = None,
    cutoff: float = DEFAULT_CUTOFF,
    margin: float = DEFAULT_MARGIN,
    return_e: bool = False,
) -> torch.Tensor:
    """Computes PyDSSP's continuous (differentiable) backbone H-bond map.

    .. code-block:: python

        import graphein.protein.tensor as gpt
        from graphein.protein.tensor.dssp import get_hbond_map

        protein = gpt.Protein().from_pdb_code("1lds")
        hbmap = get_hbond_map(protein.coords)  # (L, L)

    :param x: Atom coordinates of shape ``(L, A, 3)`` or ``(B, L, A, 3)``.
        The first four atoms must be ``N, CA, C, O`` (as in Graphein's
        ``AtomTensor``); remaining atoms are ignored. Hydrogens are placed
        from backbone geometry.
    :type x: AtomTensor
    :param donor_mask: Optional ``(..., L)`` mask of residues that can donate
        an H-bond (e.g. ``False`` for proline). Default ``None`` (all).
    :type donor_mask: Optional[torch.Tensor]
    :param cutoff: Energy cutoff (kcal/mol). Default ``-0.5``.
    :type cutoff: float
    :param margin: Smoothing margin around ``cutoff``. Default ``1.0``.
    :type margin: float
    :param return_e: Return the raw energy map instead. Default ``False``.
    :type return_e: bool
    :return: ``(..., L, L)`` map indexed ``[donor (N-H), acceptor (C=O)]``.
    :rtype: torch.Tensor
    """
    if donor_mask is not None:
        donor_mask = torch.as_tensor(
            donor_mask, dtype=torch.bool, device=x.device
        )
    return _hbond_map(
        x[..., :4, :], torch, donor_mask, cutoff, margin, return_e
    )


def assign_dssp(
    x: AtomTensor, donor_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Assigns 3-state (loop, helix, strand) secondary structure with DSSP.

    .. code-block:: python

        import graphein.protein.tensor as gpt
        from graphein.protein.tensor.dssp import assign_dssp

        protein = gpt.Protein().from_pdb_code("1lds")
        ss = assign_dssp(protein.coords)  # (L, 3) bool
        ss_idx = ss.int().argmax(-1)  # 0: loop, 1: helix, 2: strand

    :param x: Atom coordinates of shape ``(L, A, 3)`` or ``(B, L, A, 3)``.
        The first four atoms must be ``N, CA, C, O``.
    :type x: AtomTensor
    :param donor_mask: Optional ``(..., L)`` mask of residues that can donate
        an H-bond (e.g. ``False`` for proline). Default ``None`` (all).
    :type donor_mask: Optional[torch.Tensor]
    :return: Boolean tensor of shape ``(..., L, 3)`` ordered
        ``(loop, helix, strand)``. As in PyDSSP, a residue may be flagged as
        both helix and strand; loop is set when it is neither.
    :rtype: torch.Tensor
    """
    if donor_mask is not None:
        donor_mask = torch.as_tensor(
            donor_mask, dtype=torch.bool, device=x.device
        )
    with torch.no_grad():
        return _assign(x[..., :4, :], torch, donor_mask)


def get_hbond_map_np(
    x: np.ndarray,
    donor_mask: Optional[np.ndarray] = None,
    cutoff: float = DEFAULT_CUTOFF,
    margin: float = DEFAULT_MARGIN,
    return_e: bool = False,
) -> np.ndarray:
    """NumPy variant of :func:`get_hbond_map`."""
    if donor_mask is not None:
        donor_mask = np.asarray(donor_mask, dtype=bool)
    return _hbond_map(
        np.asarray(x)[..., :4, :], np, donor_mask, cutoff, margin, return_e
    )


def assign_dssp_np(
    x: np.ndarray, donor_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """NumPy variant of :func:`assign_dssp`."""
    if donor_mask is not None:
        donor_mask = np.asarray(donor_mask, dtype=bool)
    return _assign(np.asarray(x)[..., :4, :], np, donor_mask)


def assign_dssp8(
    x: AtomTensor, donor_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Assigns 8-state DSSP secondary structure, following mkdssp 4.x.

    Unlike :func:`assign_dssp` (PyDSSP), this uses DSSP's own H-bond model
    and rules (see :func:`_dssp_hbonds`), including chain breaks (C-N >
    2.5 Angstrom), beta-bulges, turns and bends. Polyproline (``P``, new in
    DSSP 4) is reported as loop.

    .. code-block:: python

        import graphein.protein.tensor as gpt
        from graphein.protein.tensor.dssp import DSSP8_STATES, assign_dssp8
        from graphein.protein.resi_atoms import STANDARD_AMINO_ACIDS

        protein = gpt.Protein().from_pdb_code("1lds")
        # DSSP never treats proline N as an H-bond donor
        not_pro = protein.residue_type != STANDARD_AMINO_ACIDS.index("P")
        ss = assign_dssp8(protein.coords, donor_mask=not_pro)  # (L,)
        "".join(DSSP8_STATES[i] for i in ss)  # "--EEEEE-SS..."

    To match mkdssp, ``x`` should only contain residues with all backbone
    atoms (mkdssp discards incomplete residues) and prolines should be
    excluded with ``donor_mask``.

    :param x: Atom coordinates of shape ``(L, A, 3)`` or ``(B, L, A, 3)``.
        The first four atoms must be ``N, CA, C, O``.
    :type x: AtomTensor
    :param donor_mask: Optional ``(..., L)`` mask of residues that can donate
        an H-bond (``False`` for proline). Default ``None`` (all).
    :type donor_mask: Optional[torch.Tensor]
    :return: ``(..., L)`` long tensor of indices into :data:`DSSP8_STATES`.
    :rtype: torch.Tensor
    """
    if donor_mask is not None:
        donor_mask = torch.as_tensor(
            donor_mask, dtype=torch.bool, device=x.device
        )
    with torch.no_grad():
        return torch.as_tensor(
            _assign8_batched(x[..., :4, :], torch, donor_mask),
            device=x.device,
        )


def assign_dssp8_np(
    x: np.ndarray, donor_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """NumPy variant of :func:`assign_dssp8`."""
    if donor_mask is not None:
        donor_mask = np.asarray(donor_mask, dtype=bool)
    return _assign8_batched(np.asarray(x)[..., :4, :], np, donor_mask)


# Solvent accessibility, following mkdssp 4.x. The arithmetic mirrors its
# float32/float64 mix so that results match it exactly.
_ASA_RADII = np.array([1.65, 1.87, 1.76, 1.4], dtype=np.float32)  # N CA C O
_ASA_RADIUS_SIDE = np.float32(1.8)
_ASA_RADIUS_WATER = np.float32(1.4)
_ASA_CHUNK = 4096  # atoms per batch of (atom, dot) pairs


def _surface_dots(n: int = 200):
    """mkdssp's Fibonacci sphere: ``(2n + 1, 3)`` unit dots, and the float32
    area fraction after ``k`` free dots (mkdssp sums the weight per dot)."""
    P = 2 * n + 1
    i = np.arange(-n, n + 1)
    phi = np.float32((1 + 5**0.5) / 2)
    lat = np.arcsin(np.float32(2.0) * i.astype(np.float32) / np.float32(P))
    lon = np.fmod(i, np.float64(phi)) * 2 * np.pi / np.float64(phi)
    lon = lon.astype(np.float32)
    dots = np.stack(
        [np.sin(lon) * np.cos(lat), np.cos(lon) * np.cos(lat), np.sin(lat)],
        -1,
    )
    weight = np.full(P, 4 * np.pi / P, dtype=np.float32)
    area = np.concatenate([[0], np.cumsum(weight, dtype=np.float32)])
    return dots, area.astype(np.float32)


def _accessibility(x: np.ndarray, present: np.ndarray) -> np.ndarray:
    res, atom = np.nonzero(present)
    xyz = x[res, atom].astype(np.float32)
    r = np.where(atom < 4, _ASA_RADII[np.minimum(atom, 3)], _ASA_RADIUS_SIDE)
    # atom radius + water, in float64 as mkdssp
    r_probe = r.astype(np.float64) + np.float64(_ASA_RADIUS_WATER)

    # Occluding neighbours: centres closer than both probe-extended radii.
    a, b = (
        cKDTree(xyz)
        .query_pairs(2 * r_probe.max() + 1e-3, output_type="ndarray")
        .T
    )
    d = xyz[a] - xyz[b]
    dist2 = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]
    near = (dist2 < (r_probe[a] + r_probe[b]) ** 2) & (dist2 > 0.0001)
    a, b, dist2 = a[near], b[near], dist2[near]
    i, j = np.concatenate([a, b]), np.concatenate([b, a])
    # Neighbour table, nearest first (as mkdssp), padded with -1.
    order = np.lexsort((np.concatenate([dist2, dist2]), i))
    i, j = i[order], j[order]
    first = np.searchsorted(i, np.arange(len(xyz)))
    degree = np.bincount(i, minlength=len(xyz))
    table = np.full((len(xyz), max(degree.max(initial=0), 1)), -1)
    table[i, np.arange(len(i)) - first[i]] = j
    r2 = r_probe**2

    # Track still-free (atom, dot) pairs and test them against each atom's
    # k-th nearest neighbour in turn: most dots are buried within a few
    # neighbours, so the work shrinks quickly (mkdssp exits early likewise).
    dots, area = _surface_dots()
    radius = (r + _ASA_RADIUS_WATER).astype(np.float32)
    n_dots = len(dots)
    # Dot positions per distinct probe radius, and neighbour offsets b - a.
    radii, kind = np.unique(radius, return_inverse=True)
    xx = (dots[None] * radii[:, None, None]).reshape(-1, 3).T.copy()
    loc = xyz[table] - xyz[:, None]  # (M, max degree, 3); padding unused
    loc = np.moveaxis(loc, -1, 0).reshape(3, -1)
    width = table.shape[1]
    n_free = np.zeros(len(xyz), dtype=np.int64)
    for s in range(0, len(xyz), _ASA_CHUNK):
        atoms = np.arange(s, min(s + _ASA_CHUNK, len(xyz)))
        free_here = n_free[s : s + len(atoms)]  # view
        fa = np.repeat(atoms, n_dots)
        fx = kind[fa] * n_dots + np.tile(np.arange(n_dots), len(atoms))
        for k in range(width):
            nb = table[fa, k]
            done = nb < 0  # no more neighbours: dot is free
            free_here += np.bincount(fa[done] - s, minlength=len(atoms))
            fa, fx, nb = fa[~done], fx[~done], nb[~done]
            if not len(fa):
                break
            fl = fa * width + k
            d2 = 0
            for c in range(3):
                diff = xx[c, fx] - loc[c, fl]
                d2 = diff * diff if c == 0 else d2 + diff * diff
            free = r2[nb] < d2
            fa, fx = fa[free], fx[free]
        free_here += np.bincount(fa - s, minlength=len(atoms))

    atom_area = area[n_free] * radius * radius
    acc = np.zeros(present.shape[0], dtype=np.float32)
    np.add.at(acc, res, atom_area)  # in order: N, CA, C, O, side chain
    return acc


def accessibility(x: AtomTensor, fill_value: float = 1e-5) -> torch.Tensor:
    """Solvent accessible surface area per residue, following mkdssp 4.x.

    Uses DSSP's radii (N 1.65, CA 1.87, C 1.76, O 1.4, other heavy atoms
    1.8, water 1.4 Angstrom) and its 401-dot Fibonacci sphere. Every heavy
    atom in ``x`` occludes, so to match mkdssp ``x`` should hold only protein
    residues with complete backbones (it ignores ligands, waters and
    incomplete residues).

    .. code-block:: python

        import graphein.protein.tensor as gpt
        from graphein.protein.tensor.dssp import accessibility

        protein = gpt.Protein().from_pdb_code("1lds")
        asa = accessibility(protein.coords)  # (L,) Angstrom^2

    :param x: Atom coordinates of shape ``(L, A, 3)``. The first four atoms
        must be ``N, CA, C, O``; any others are treated as side chain.
    :type x: AtomTensor
    :param fill_value: Value marking missing atoms. Default ``1e-5``.
    :type fill_value: float
    :return: ``(L,)`` accessible surface area in square Angstrom.
    :rtype: torch.Tensor
    """
    return torch.as_tensor(
        accessibility_np(x.detach().cpu().numpy(), fill_value),
        device=x.device,
    )


def accessibility_np(x: np.ndarray, fill_value: float = 1e-5) -> np.ndarray:
    """NumPy variant of :func:`accessibility`."""
    x = np.asarray(x)
    _check_shape(x)
    if x.ndim != 3:
        raise ValueError("accessibility expects a single protein (L, A, 3)")
    return _accessibility(x, x[..., 0] != fill_value)
