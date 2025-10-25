"""Utility helpers for collecting wall-clock timings across pipeline steps."""
from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import warnings
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Iterable, Iterator, List, Optional


@dataclass
class TimingRecord:
    """Represents a single timing measurement."""

    label: str
    duration: float


@dataclass
class BaselineStats:
    """Running statistics for a timed label."""

    count: int = 0
    avg: float = 0.0

    def update(self, duration: float) -> None:
        """Update the running average with a new duration."""

        self.count += 1
        # Incremental running average to avoid precision loss.
        self.avg += (duration - self.avg) / self.count


@dataclass
class TimingRecorder:
    """Collects named duration measurements for later reporting.

    The recorder optionally keeps a baseline of historical durations per label. When a
    baseline file is supplied, entering a timed block will show the most recent average
    duration, providing a coarse *estimate* for how long the step may take. Each run
    updates the stored baseline so subsequent executions refine the estimate.
    """

    enabled: bool = True
    records: List[TimingRecord] = field(default_factory=list)
    baseline_path: Optional[str] = None
    autosave: bool = False
    _baseline: Dict[str, BaselineStats] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.enabled and self.baseline_path:
            self._load_baseline()

    @contextlib.contextmanager
    def time_block(self, label: str) -> Iterator[None]:
        """Context manager that measures the elapsed time of a code block.

        Args:
            label: Human-readable description of the timed operation.
        """

        if not self.enabled:
            yield
            return

        estimate = self.estimated_duration(label)
        if estimate is not None:
            print(f"{label} (est. {format_duration(estimate)})")
        else:
            print(label)

        start = perf_counter()
        try:
            yield
        finally:
            duration = perf_counter() - start
            record = TimingRecord(label=label, duration=duration)
            self.records.append(record)
            self._update_baseline(record)
            print(f"{label} completed in {format_duration(duration)}\n")

    def estimated_duration(self, label: str) -> Optional[float]:
        """Return the historical average duration for *label*, if available."""

        stats = self._baseline.get(label)
        if stats and stats.count:
            return stats.avg
        return None

    def extend(self, other: "TimingRecorder") -> None:
        """Merge another recorder's timings into this one."""

        if not self.enabled or not other.enabled:
            return
        self.records.extend(other.records)

    def report(self, header: str = "Timing summary") -> None:
        """Print an aggregated summary of all recorded timings."""

        if not self.enabled or not self.records:
            return

        total = sum(record.duration for record in self.records)
        widest = max(len(record.label) for record in self.records + [TimingRecord("Total", 0)])
        print(header)
        print(f"  {'Step'.ljust(widest)} | Duration    | Share   | Cumulative")
        print(f"  {'-' * widest}-+-------------+---------+-----------")

        cumulative = 0.0
        for record in self.records:
            cumulative += record.duration
            share = record.duration / total if total else 0.0
            print(
                "  "
                f"{record.label.ljust(widest)} | "
                f"{format_duration(record.duration).rjust(11)} | "
                f"{share * 100:6.2f}% | "
                f"{format_duration(cumulative).rjust(9)}"
            )

        print(
            "  "
            f"{'Total'.ljust(widest)} | "
            f"{format_duration(total).rjust(11)} | "
            f"{100:6.2f}% | "
            f"{format_duration(total).rjust(9)}\n"
        )

    def save_baseline(self, path: Optional[str] = None) -> None:
        """Persist the running averages to *path* (or ``baseline_path``)."""

        if not self.enabled:
            return
        target = path or self.baseline_path
        if not target:
            return

        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        serialisable = {
            label: {"count": stats.count, "avg": stats.avg}
            for label, stats in self._baseline.items()
            if stats.count
        }
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=2)

    def _update_baseline(self, record: TimingRecord) -> None:
        if not self.enabled:
            return
        stats = self._baseline.setdefault(record.label, BaselineStats())
        stats.update(record.duration)
        if self.autosave:
            self.save_baseline()

    def _load_baseline(self) -> None:
        if not self.baseline_path or not os.path.exists(self.baseline_path):
            return
        with open(self.baseline_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        for label, stats in raw.items():
            self._baseline[label] = BaselineStats(count=stats.get("count", 0), avg=stats.get("avg", 0.0))


def format_duration(seconds: float) -> str:
    """Convert seconds into a human-friendly string."""

    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, seconds_remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {seconds_remainder:.2f} s"
    hours, minutes_remainder = divmod(minutes, 60)
    return f"{int(hours)} h {int(minutes_remainder)} min {seconds_remainder:.2f} s"


def merge_reports(recorders: Iterable[TimingRecorder], header: str = "Timing summary") -> None:
    """Aggregate multiple TimingRecorder instances into a single report."""

    combined = TimingRecorder()
    for recorder in recorders:
        combined.extend(recorder)
    combined.report(header)


def configure_start_method(
    preferred: str = "forkserver",
    fallback: str = "spawn",
    *,
    verbose: bool = False,
) -> Optional[str]:
    """Ensure ``multiprocessing`` uses a safe start method for the current platform.

    On Unix-like systems ``fork`` is often the default, but using ``forkserver`` avoids the
    pickling and global-state hazards that accompany ``fork``. macOS, however, does not
    always expose ``forkserver``. This helper selects the preferred start method when
    available and gracefully falls back to a sensible alternative otherwise.

    Args:
        preferred: The ideal start method to request when supported by the runtime.
        fallback: The method to use when ``preferred`` is unavailable. If ``None`` the first
            available method will be selected instead.
        verbose: When ``True`` prints the selected method or fallback notice to stdout.

    Returns:
        The start method that is active after configuration, or ``None`` if no suitable
        method could be set.
    """

    available = multiprocessing.get_all_start_methods()
    if not available:
        if verbose:
            print("No multiprocessing start methods reported; continuing without changes.")
        else:
            warnings.warn("No multiprocessing start methods reported; continuing without changes.")
        return None

    target = None
    message: Optional[str] = None
    if preferred and preferred in available:
        target = preferred
    elif fallback and fallback in available:
        target = fallback
        message = f"Preferred start method '{preferred}' unavailable; falling back to '{fallback}'."
    else:
        target = available[0]
        message = (
            f"Preferred start method '{preferred}' unavailable; using '{target}' from available options."
        )

    current = multiprocessing.get_start_method(allow_none=True)
    if current == target:
        if verbose:
            print(f"Multiprocessing start method already configured as '{current}'.")
        return current

    force = current is not None and current != target
    if message:
        if verbose:
            print(message)
        else:
            warnings.warn(message)
    elif verbose:
        print(f"Using multiprocessing start method '{target}'.")

    multiprocessing.set_start_method(target, force=force)
    return target
