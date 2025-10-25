import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import timing_utils


def test_time_block_records_duration(monkeypatch):
    recorder = timing_utils.TimingRecorder()
    times = iter([1.0, 3.5])
    monkeypatch.setattr(timing_utils, "perf_counter", lambda: next(times))

    with recorder.time_block("Example Step"):
        pass

    assert len(recorder.records) == 1
    assert recorder.records[0].label == "Example Step"
    assert pytest.approx(recorder.records[0].duration, rel=0.01) == 2.5


def test_report_includes_percentages(capsys):
    recorder = timing_utils.TimingRecorder()
    recorder.records.extend(
        [
            timing_utils.TimingRecord(label="A", duration=1.0),
            timing_utils.TimingRecord(label="B", duration=1.0),
        ]
    )

    recorder.report("Summary")
    report_output = capsys.readouterr().out
    assert "Summary" in report_output
    assert "50.00%" in report_output
    assert "Total" in report_output


def test_baseline_round_trip(tmp_path: Path, monkeypatch):
    baseline_file = tmp_path / "baseline.json"
    recorder = timing_utils.TimingRecorder(baseline_path=str(baseline_file), autosave=False)

    times = iter([10.0, 11.0])
    monkeypatch.setattr(timing_utils, "perf_counter", lambda: next(times))

    with recorder.time_block("Stage"):
        pass

    recorder.save_baseline()
    stored = json.loads(baseline_file.read_text())
    assert stored["Stage"]["count"] == 1

    new_recorder = timing_utils.TimingRecorder(baseline_path=str(baseline_file))
    assert new_recorder.estimated_duration("Stage") == pytest.approx(
        recorder.records[0].duration, rel=0.01
    )

    times = iter([20.0, 21.0])
    monkeypatch.setattr(timing_utils, "perf_counter", lambda: next(times))
    with new_recorder.time_block("Stage"):
        pass

    # After the second run the average should be updated to the midpoint of the two durations.
    assert new_recorder.estimated_duration("Stage") == pytest.approx(
        (recorder.records[0].duration + 1.0) / 2, rel=0.01
    )


def test_configure_start_method_prefers_forkserver(monkeypatch):
    monkeypatch.setattr(timing_utils.multiprocessing, "get_all_start_methods", lambda: ["spawn", "forkserver"])
    monkeypatch.setattr(timing_utils.multiprocessing, "get_start_method", lambda allow_none=True: None)

    called = []

    def fake_set(method: str, force: bool = False) -> None:
        called.append((method, force))

    monkeypatch.setattr(timing_utils.multiprocessing, "set_start_method", fake_set)

    chosen = timing_utils.configure_start_method()
    assert chosen == "forkserver"
    assert called == [("forkserver", False)]


def test_configure_start_method_overrides_default_fork(monkeypatch):
    monkeypatch.setattr(
        timing_utils.multiprocessing,
        "get_all_start_methods",
        lambda: ["fork", "forkserver"],
    )
    monkeypatch.setattr(timing_utils.multiprocessing, "get_start_method", lambda allow_none=True: "fork")

    called = []

    def fake_set(method: str, force: bool = False) -> None:
        called.append((method, force))

    monkeypatch.setattr(timing_utils.multiprocessing, "set_start_method", fake_set)

    chosen = timing_utils.configure_start_method()
    assert chosen == "forkserver"
    assert called == [("forkserver", True)]


def test_configure_start_method_warns_when_falling_back(monkeypatch):
    monkeypatch.setattr(timing_utils.multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    monkeypatch.setattr(timing_utils.multiprocessing, "get_start_method", lambda allow_none=True: None)

    called = []

    def fake_set(method: str, force: bool = False) -> None:
        called.append((method, force))

    monkeypatch.setattr(timing_utils.multiprocessing, "set_start_method", fake_set)

    with pytest.warns(UserWarning, match="falling back to 'spawn'"):
        chosen = timing_utils.configure_start_method(verbose=False)

    assert chosen == "spawn"
    assert called == [("spawn", False)]


def test_configure_start_method_is_noop_when_already_set(monkeypatch):
    monkeypatch.setattr(timing_utils.multiprocessing, "get_all_start_methods", lambda: ["forkserver"])
    monkeypatch.setattr(timing_utils.multiprocessing, "get_start_method", lambda allow_none=True: "forkserver")

    def should_not_run(*args, **kwargs):
        pytest.fail("set_start_method should not be called when already configured")

    monkeypatch.setattr(timing_utils.multiprocessing, "set_start_method", should_not_run)

    chosen = timing_utils.configure_start_method()
    assert chosen == "forkserver"
