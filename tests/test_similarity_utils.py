import pytest

np = pytest.importorskip("numpy")

from similarity_utils import (
    _encode_library,
    _pairwise_match_counts,
    _reverse_complement,
    sim_matrix,
)


def _naive_matches(seq_a: str, seq_b: str) -> int:
    return sum(ch1 == ch2 for ch1, ch2 in zip(seq_a, seq_b))


def _naive_sim_matrix(library, duplex):
    size = len(library)
    result = np.zeros((size, size), dtype=int)
    rc_library = [_reverse_complement(seq) for seq in library] if duplex else library
    for i in range(size):
        seq_i = library[i]
        rc_i = rc_library[i]
        for j in range(i, size):
            seq_j = library[j]
            if duplex:
                rc_j = rc_library[j]
                best = max(
                    _naive_matches(seq_i, seq_j),
                    _naive_matches(seq_i, rc_j),
                    _naive_matches(rc_i, seq_j),
                    _naive_matches(rc_i, rc_j),
                )
            else:
                best = _naive_matches(seq_i, seq_j)
            result[i, j] = result[j, i] = best
    return result


def test_encode_library_requires_uniform_lengths():
    with pytest.raises(ValueError):
        _encode_library(["AAA", "AA"])


def test_sim_matrix_single_strand_matches_naive():
    library = ["AAAA", "AAAT", "TTTT", "AGTC"]
    expected = _naive_sim_matrix(library, duplex=False)
    actual = sim_matrix(library, duplex=False)
    np.testing.assert_array_equal(actual, expected)


def test_sim_matrix_duplex_matches_naive():
    library = ["AAAA", "AAAT", "TTTT", "AGTC"]
    expected = _naive_sim_matrix(library, duplex=True)
    actual = sim_matrix(library, duplex=True)
    np.testing.assert_array_equal(actual, expected)


def test_pairwise_block_logic_handles_large_library():
    # Construct a library where the block size logic must split work.
    library = ["ATCG" * 8 for _ in range(50)]
    encoded = _encode_library(library)
    counts = _pairwise_match_counts(encoded, encoded, symmetric=True)
    assert counts.shape == (50, 50)
    np.testing.assert_array_equal(counts.diagonal(), np.full(50, len(library[0])))


def test_sim_matrix_empty_library():
    result = sim_matrix([], duplex=False)
    assert result.shape == (0, 0)


def test_reverse_complement_round_trip():
    seq = "ACGTacgt"
    assert _reverse_complement(_reverse_complement(seq)) == seq
