"""Nussinov algorithm for computing RNA secondary structure adopted from
adopted from https://github.com/cgoliver/Nussinov/blob/master/nussinov.py """

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Ryan Greenhalgh
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import Tuple

import numpy as np

MIN_LOOP_LENGTH = 10


def pair_check(
    base_pair_tuple: Tuple[str, str],
) -> bool:
    """Check if there is a base pair interaction between two bases.

    :param base_pair_tuple: Base pair tuple to test.
    :type base_pair_tuple: Tuple[str, str]
    :return: ``True`` if base pairs interact else ``False``.
    :rtype: bool
    """
    return base_pair_tuple in [("A", "U"), ("U", "A"), ("C", "G"), ("G", "C")]


def optimal_pairing(
    i: int,
    j: int,
    sequence: str,
) -> int:
    """Find the optimal pairing between two positions.

    :param i: Position ``i`` in sequence.
    :type i: int
    :param j: Position ``j`` in sequence.
    :type j: int
    :param sequence: RNA sequence.
    :type sequence: str
    :return:
    :rtype: int
    """

    # base case: no pairs allowed when i and j are less than 4 bases apart
    if i >= j - MIN_LOOP_LENGTH:
        return 0

    # i and j can either be paired or not be paired, if not paired then the optimal score is optimal_pairing(i,j-1)
    unpaired = optimal_pairing(i, j - 1, sequence)

    # check if j can be involved in a pairing with a position t
    pairing = [
        1
        + optimal_pairing(i, t - 1, sequence)
        + optimal_pairing(t + 1, j - 1, sequence)
        for t in range(i, j - MIN_LOOP_LENGTH)
        if pair_check((sequence[t], sequence[j]))
    ] or [0]
    paired = max(pairing)

    return max(unpaired, paired)


def traceback(
    i: int,
    j: int,
    structure: str,
    DP: np.ndarray,
    sequence: str,
) -> None:
    """Recursively check pairing and interactions between base pairs.

    :param i: Position ``i`` in sequence.
    :type i: int
    :param j: Position ``j`` in sequence.
    :type j: int
    :param structure: Dot-bracket notation for RNA seq.
    :type structure: str
    :param DP: Numpy matrix to cache
    :type DP: np.ndarray
    :param sequence: RNA sequence
    :type sequence: str
    """

    # in this case we've gone through the whole sequence. Nothing to do.
    if j <= i:
        return
    # if j is unpaired, there will be no change in score when we take it out, so we just recurse to the next index
    elif DP[i][j] == DP[i][j - 1]:
        traceback(i, j - 1, structure, DP, sequence)
    # consider cases where j forms a pair.
    else:
        # try pairing j with a matching index k to its left.
        for k in [
            b
            for b in range(i, j - MIN_LOOP_LENGTH)
            if pair_check((sequence[b], sequence[j]))
        ]:
            # if the score at i,j is the result of adding 1 from pairing (j,k) and whatever score
            # comes from the substructure to its left (i, k-1) and to its right (k+1, j-1)
            if k < 1:
                if DP[i][j] == DP[k + 1][j - 1] + 1:
                    structure.append((k, j))
                    traceback(k + 1, j - 1, structure, DP, sequence)
                    break
            elif DP[i][j] == DP[i][k - 1] + DP[k + 1][j - 1] + 1:
                # add the pair (j,k) to our list of pairs
                structure.append((k, j))
                # move the recursion to the two substructures formed by this pairing
                traceback(i, k - 1, structure, DP, sequence)
                traceback(k + 1, j - 1, structure, DP, sequence)
                break


def write_structure(
    sequence: str,
    structure: str,
) -> str:
    """Convert structure to string.

    :param sequence: RNA sequence.
    :type sequence: str
    :param structure: RNA dot-bracket.
    :type structure: str
    :return: Dot-bracket notation as a string.
    :rtype: str
    """
    dot_bracket = ["." for _ in range(len(sequence))]
    for s in structure:
        dot_bracket[min(s)] = "("
        dot_bracket[max(s)] = ")"
    return "".join(dot_bracket)


def initialize(
    N: int,
) -> np.ndarray:
    """Initialize DP matrix. ``NxN`` matrix that stores the scores of the optimal pairings.

    :param N: Length of RNA sequence
    :type N: int
    :return: DP matrix for Nussinov Algorithm
    :rtype: np.ndarray
    """

    DP = np.empty((N, N))
    DP[:] = np.nan
    for k in range(MIN_LOOP_LENGTH):
        for i in range(N - k):
            j = i + k
            DP[i][j] = 0
    return DP


def nussinov(
    sequence: str,
) -> str:
    """Nussinov algorithm for predicting RNA ss in dot-bracket notation.

    :param sequence: RNA sequence.
    :type sequence: str
    :return: Dot-bracket notation for RNA sequence.
    :rtype: str
    """
    N = len(sequence)
    DP = initialize(N)
    structure = []

    # fill the DP matrix diagonally
    for k in range(MIN_LOOP_LENGTH, N):
        for i in range(N - k):
            j = i + k
            DP[i][j] = optimal_pairing(i, j, sequence)

    # copy values to lower triangle to avoid null references
    for i in range(N):
        for j in range(i):
            DP[i][j] = DP[j][i]

    traceback(0, N - 1, structure, DP, sequence)

    return write_structure(sequence, structure)
