"""Benchmark the similarity-matrix implementations."""
from __future__ import annotations

import argparse
import random
from statistics import mean
from time import perf_counter
from typing import List

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "NumPy is required for the similarity benchmark. Install numpy to run this script."
    ) from exc

from similarity_utils import _reverse_complement, sim_matrix
from timing_utils import TimingRecorder, format_duration


_ALPHABET = "ACGT"


def _random_library(size: int, length: int, *, seed: int) -> List[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(_ALPHABET) for _ in range(length)) for _ in range(size)]


def _naive_matches(seq_a: str, seq_b: str) -> int:
    return sum(ch1 == ch2 for ch1, ch2 in zip(seq_a, seq_b))


def _naive_sim_matrix(library: List[str], duplex: bool) -> np.ndarray:
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


def _measure(timer: TimingRecorder, label: str, func, repeats: int) -> List[float]:
    durations: List[float] = []
    descriptor = f"{label} ({repeats} run{'s' if repeats != 1 else ''})"
    with timer.time_block(descriptor):
        for _ in range(repeats):
            start = perf_counter()
            func()
            durations.append(perf_counter() - start)
    return durations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=120, help="Number of sequences to benchmark")
    parser.add_argument("--length", type=int, default=20, help="Length of each sequence")
    parser.add_argument("--duplex", action="store_true", help="Enable duplex (reverse-complement) mode")
    parser.add_argument("--repeats", type=int, default=3, help="Number of times to run each implementation")
    parser.add_argument("--seed", type=int, default=13, help="Seed for pseudo-random library generation")
    args = parser.parse_args()

    if args.size <= 0 or args.length <= 0:
        raise SystemExit("--size and --length must be positive integers")

    library = _random_library(args.size, args.length, seed=args.seed)
    print(f"Generated library with {args.size} sequences of length {args.length}.")
    print(f"Duplex mode: {'on' if args.duplex else 'off'}.\n")

    timer = TimingRecorder()
    vectorised_result: List[np.ndarray] = []
    naive_result: List[np.ndarray] = []

    def run_vectorised() -> None:
        result = sim_matrix(library, duplex=args.duplex)
        if not vectorised_result:
            vectorised_result.append(result)
        elif not np.array_equal(vectorised_result[0], result):
            raise AssertionError("Vectorised results varied between runs.")

    def run_naive() -> None:
        result = _naive_sim_matrix(library, duplex=args.duplex)
        if not naive_result:
            naive_result.append(result)
        elif not np.array_equal(naive_result[0], result):
            raise AssertionError("Naive results varied between runs.")

    vectorised_durations = _measure(timer, "Vectorised", run_vectorised, args.repeats)
    naive_durations = _measure(timer, "Naive", run_naive, args.repeats)

    if not vectorised_result or not naive_result:
        raise AssertionError("Benchmark did not execute the implementations.")

    reference = vectorised_result[0]
    if not np.array_equal(reference, naive_result[0]):
        raise AssertionError("Optimised and naive implementations disagree.")

    timer.report("Similarity benchmark timings")

    avg_vectorised = mean(vectorised_durations)
    avg_naive = mean(naive_durations)
    speedup = avg_naive / avg_vectorised if avg_vectorised else float("inf")

    print("Average per-run durations:")
    print(f"  Vectorised: {format_duration(avg_vectorised)}")
    print(f"  Naive     : {format_duration(avg_naive)}")
    if speedup == float("inf"):
        print("Vectorised implementation completed instantly relative to naive.")
    else:
        print(f"  Speed-up  : {speedup:.2f}x faster\n")


if __name__ == "__main__":
    main()
