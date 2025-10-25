"""High-performance helpers for computing strand similarity matrices."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# Target size of the temporary boolean blocks used while computing pairwise matches.
# The block algorithm keeps the intermediate arrays around this size to avoid large
# allocations when libraries contain hundreds of sequences.
_TARGET_BLOCK_BYTES = 1 << 20  # 1 MiB

_RC_TRANSLATION = str.maketrans({
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "a": "t",
    "c": "g",
    "g": "c",
    "t": "a",
})


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of ``seq`` for standard DNA alphabets."""

    return seq.translate(_RC_TRANSLATION)[::-1]


def _encode_library(library: Sequence[str]) -> np.ndarray:
    """Encode a library of equal-length sequences as a 2D ``uint8`` array."""

    size = len(library)
    if size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    length = len(library[0])
    encoded = np.empty((size, length), dtype=np.uint8)
    for index, seq in enumerate(library):
        if len(seq) != length:
            raise ValueError("All sequences must have identical lengths.")
        encoded[index, :] = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return encoded


def _match_dtype(length: int) -> np.dtype:
    """Choose an integer dtype that can store ``length`` without overflow."""

    if length <= np.iinfo(np.int16).max:
        return np.int16
    return np.int32


def _pairwise_match_counts(
    left: np.ndarray,
    right: np.ndarray,
    *,
    symmetric: bool = False,
) -> np.ndarray:
    """Compute pairwise match counts between the sequences in ``left`` and ``right``.

    When ``symmetric`` is ``True`` the function assumes ``left is right`` and only
    computes the upper triangle before mirroring the result, reducing the amount of
    temporary memory required.
    """

    n_left, length = left.shape
    n_right = right.shape[0]
    dtype = _match_dtype(length)

    if n_left == 0 or n_right == 0 or length == 0:
        return np.zeros((n_left, n_right), dtype=dtype)

    if symmetric:
        if n_left != n_right:
            raise ValueError("Symmetric match computation requires equal shapes.")
        result = np.zeros((n_left, n_right), dtype=dtype)
        block = max(1, int(math.sqrt(max(1, _TARGET_BLOCK_BYTES // length))))
        for i_start in range(0, n_left, block):
            i_end = min(i_start + block, n_left)
            row_block = left[i_start:i_end]
            max_cols = max(1, min(n_right, _TARGET_BLOCK_BYTES // (max(1, row_block.shape[0] * length))))
            for j_start in range(i_start, n_right, max_cols):
                j_end = min(j_start + max_cols, n_right)
                col_block = right[j_start:j_end]
                matches = np.count_nonzero(
                    row_block[:, None, :] == col_block[None, :, :], axis=2
                ).astype(dtype, copy=False)
                result[i_start:i_end, j_start:j_end] = matches
                if i_start != j_start:
                    result[j_start:j_end, i_start:i_end] = matches.T
        return result

    result = np.zeros((n_left, n_right), dtype=dtype)
    block = max(1, min(n_left, _TARGET_BLOCK_BYTES // (n_right * length)))
    for i_start in range(0, n_left, block):
        i_end = min(i_start + block, n_left)
        row_block = left[i_start:i_end]
        matches = np.count_nonzero(
            row_block[:, None, :] == right[None, :, :], axis=2
        ).astype(dtype, copy=False)
        result[i_start:i_end, :] = matches
    return result


def sim_matrix(library: Sequence[str], duplex: bool) -> np.ndarray:
    """Return the similarity matrix for ``library``.

    ``duplex`` toggles whether reverse complements should be considered. The function
    is API-compatible with :func:`orthogonal.sim_matrix` but avoids Python-level loops
    by relying on vectorised NumPy operations and blockwise processing.
    """

    encoded = _encode_library(library)
    base = _pairwise_match_counts(encoded, encoded, symmetric=True)

    if not duplex:
        return base

    rc_library = [_reverse_complement(seq) for seq in library]
    encoded_rc = _encode_library(rc_library)

    base = np.maximum(base, _pairwise_match_counts(encoded, encoded_rc))
    base = np.maximum(base, _pairwise_match_counts(encoded_rc, encoded))
    base = np.maximum(base, _pairwise_match_counts(encoded_rc, encoded_rc, symmetric=True))
    return base


__all__ = [
    "sim_matrix",
    "_encode_library",
    "_pairwise_match_counts",
    "_reverse_complement",
]
