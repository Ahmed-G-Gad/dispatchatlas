"""The analysis runs the check of the campaign directory and carries what it found."""

from __future__ import annotations

import logging
import shutil
import tracemalloc
from dataclasses import replace
from pathlib import Path

import pytest

from dispatchatlas.analysis import AnalysisConfig, load_dataset, summarize, write_bundle
from dispatchatlas.campaigns import Protocol, SolverSpec, check_campaign, run
from dispatchatlas.campaigns.records import ErrorKind, RunRecord
from dispatchatlas.campaigns.store import CAMPAIGN_FILE, RUNS, SESSIONS
from dispatchatlas.core import StopCriteria, StopReason
from dispatchatlas.core.solver import TracePoint
from tests.campaigns.damage import edit, fail, rewrite
from tests.campaigns.solvers import registry, tiny_campaign

REGISTRY = registry()
RECORD = Path("random-search") / "flow-shop-3x2-1" / "r000-evals.json"
FAST = AnalysisConfig(resamples=50, protocol="evals")


@pytest.fixture(scope="module")
def clean(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A campaign of 20 runs under two protocols that bound the evaluations."""
    campaign = tiny_campaign(
        protocols=(
            Protocol("evals", StopCriteria(max_evaluations=15)),
            Protocol("more", StopCriteria(max_evaluations=30)),
        )
    )
    return run(campaign, tmp_path_factory.mktemp("clean"), registry=REGISTRY).directory


@pytest.fixture
def directory(clean: Path, tmp_path: Path) -> Path:
    return shutil.copytree(clean, tmp_path / clean.name)


def _topics(directory: Path, config: AnalysisConfig = FAST) -> dict[str, str]:
    summary = summarize(load_dataset(directory), config)
    return {item.topic: item.message for item in summary.limitations}


def test_a_loaded_dataset_holds_the_report_of_the_check(
    directory: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        dataset = load_dataset(directory)
    assert dataset.check is not None
    assert dataset.check.ok
    # The analysis does not derive the plan again, and the report says so.
    assert dataset.check.findings == check_campaign(directory, replan=False).findings
    assert [finding.code for finding in dataset.check.findings] == ["plan-not-compared"]
    assert caplog.text == ""
    assert dataset.select(solvers=["hc"]).check is dataset.check
    assert load_dataset(directory, check=False).check is None
    # A clean directory adds no limitation, and a note is never one.
    assert (
        summarize(dataset, FAST).limitations
        == summarize(load_dataset(directory, check=False), FAST).limitations
    )


def test_the_check_reads_the_records_before_the_dataset_and_keeps_none(tmp_path: Path) -> None:
    directory = run(tiny_campaign(replicates=10), tmp_path, registry=REGISTRY).directory

    def peak(check: bool) -> int:
        tracemalloc.start()
        dataset = load_dataset(directory, check=check)
        used = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert (dataset.check is not None) == check
        return used

    # A first load fills what the library caches, so that the two loads compare.
    load_dataset(directory)
    without, checked = peak(check=False), peak(check=True)
    # Holding the records of the check beside those of the dataset would need
    # about 1.7 times the memory of a load without the check.
    assert checked < 1.35 * without


def test_a_blocking_finding_becomes_a_limitation_and_a_warning(
    directory: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (directory / CAMPAIGN_FILE).unlink()
    with caplog.at_level(logging.WARNING):
        dataset = load_dataset(directory)
    assert "1 blocking finding(s)" in caplog.text
    assert "recipe-missing" in caplog.text
    for protocol in ("evals", "more"):
        topics = {
            item.topic: item.message
            for item in summarize(dataset, replace(FAST, protocol=protocol)).limitations
        }
        assert "campaign.json is missing" in topics["recipe-missing"]
        assert topics["recipe-missing"].endswith(
            "counts this as blocking: the results are not ready to be used."
        )


def test_missing_runs_are_named_by_cell_and_reported_once(directory: Path) -> None:
    (directory / RUNS / RECORD).unlink()
    topics = _topics(directory)
    assert "1 of 10 planned runs have no record yet" in topics["campaign"]
    assert topics["campaign"].endswith(
        "Their cells of solver and problem, each with its number of such runs: "
        "random-search on flow-shop-3x2-1 (1)."
    )
    # The analysis states the missing runs itself and names their cell, so neither
    # the finding on the runs nor the one on the partial cell is repeated.
    report = check_campaign(directory, replan=False)
    assert {"runs-unattempted", "cells-partial"} <= {item.code for item in report.findings}
    assert "runs-unattempted" not in topics
    assert "cells-partial" not in topics
    # The other protocol has every record.
    assert {"campaign", "cells-partial"}.isdisjoint(
        _topics(directory, replace(FAST, protocol="more"))
    )


def test_a_cell_with_failed_runs_is_carried_and_one_with_missing_runs_is_not(
    directory: Path,
) -> None:
    # The limitation on failures counts by solver and names no cell, so the cell
    # that is partial for a failed run is what the check adds.
    fail(directory, directory / RUNS / "hc" / RECORD.parent.name / RECORD.name, ErrorKind.TIMEOUT)
    topics = _topics(directory)
    assert "hc 1 timeout" in topics["failures"]
    assert topics["cells-partial"].endswith(
        "The items concerned: hc on flow-shop-3x2-1: 1 of 2 runs has a record."
    )
    # With a second cell that is partial for a missing run, only the first is carried.
    (directory / RUNS / RECORD).unlink()
    topics = _topics(directory)
    assert "random-search on flow-shop-3x2-1 (1)" in topics["campaign"]
    assert "2 of 6 cells" in topics["cells-partial"]
    assert topics["cells-partial"].endswith(
        "The items that no other limitation names: hc on flow-shop-3x2-1: 1 of 2 runs has a record."
    )


def test_findings_the_analysis_states_itself_are_not_repeated(directory: Path) -> None:
    for path in sorted((directory / RUNS / "hc").rglob("*-evals.json")):
        fail(directory, path, ErrorKind.TIMEOUT)
    report = check_campaign(directory, replan=False)
    assert {"runs-failed", "solver-without-records"} <= {item.code for item in report.findings}
    topics = _topics(directory)
    assert "hc 4 timeout" in topics["failures"]
    assert "runs-failed" not in topics
    # Every run of the solver is among the failures the analysis counts by solver.
    assert "solver-without-records" not in topics


def test_a_solver_without_records_for_runs_that_never_ran_is_carried(directory: Path) -> None:
    for path in sorted((directory / RUNS / "hc").rglob("*-evals.json")):
        path.unlink()
    topics = _topics(directory)
    assert "hc: 4 unattempted" in topics["solver-without-records"]
    assert "blocking" not in topics["solver-without-records"]
    assert "4 of 10 planned runs have no record yet" in topics["campaign"]


def test_an_overrun_and_divergent_stop_reasons_are_stated_once(tmp_path: Path) -> None:
    campaign = tiny_campaign(
        "timed",
        solvers=(SolverSpec("random-search"), SolverSpec("hill-climbing", "hc")),
        protocols=(Protocol("timed", StopCriteria(max_evaluations=15, time_limit=5.0)),),
    )
    directory = run(campaign, tmp_path, registry=REGISTRY).directory
    rewrite(
        sorted((directory / RUNS / "hc").rglob("*.json"))[0],
        lambda record: replace(
            record,
            provenance=replace(record.provenance, wall_seconds=9.0),
            outcome=replace(record.outcome, stop_reason=StopReason.STAGNATION),
        ),
    )
    report = check_campaign(directory, replan=False)
    found = {item.code: item.message for item in report.findings}
    assert {"budget-overrun", "stop-reasons-diverge"} <= set(found)
    summary = summarize(load_dataset(directory), replace(FAST, protocol="timed"))
    topics = [item.topic for item in summary.limitations]
    # The analysis states both with its own settings, under its own topics.
    assert (topics.count("budget"), topics.count("stop-reasons")) == (1, 1)
    assert {"budget-overrun", "stop-reasons-diverge"}.isdisjoint(topics)
    messages = [item.message for item in summary.limitations]
    assert sum("1.8 times" in message for message in messages) == 1
    assert sum("stagnation" in message for message in messages) == 1
    for code in ("budget-overrun", "stop-reasons-diverge"):
        assert not any(found[code] in message for message in messages)


def test_a_finding_on_one_protocol_qualifies_that_protocol_only(directory: Path) -> None:
    def backwards(record: RunRecord) -> RunRecord:
        value = record.outcome.value
        assert value is not None
        trace = (TracePoint(0.0002, 1, 0, value), TracePoint(0.0001, 2, 0, value))
        return replace(record, outcome=replace(record.outcome, trace=trace))

    rewrite(directory / RUNS / RECORD, backwards)
    topics = _topics(directory)
    assert "random-search: 1 of 4 records" in topics["trace-time-axis"]
    assert "trace-time-axis" not in _topics(directory, replace(FAST, protocol="more"))


def test_sessions_on_other_processors_limit_the_protocol_with_a_time_limit(tmp_path: Path) -> None:
    campaign = tiny_campaign(
        "both",
        protocols=(
            Protocol("evals", StopCriteria(max_evaluations=15)),
            Protocol("timed", StopCriteria(max_evaluations=15, time_limit=5.0)),
        ),
    )
    directory = run(campaign, tmp_path, registry=REGISTRY).directory
    sessions = directory / SESSIONS
    edit(sessions / "1.json", lambda document: document["environment"]["cpu"].update(allowed=[0]))
    shutil.copy(sessions / "1.json", sessions / "2.json")
    edit(sessions / "2.json", lambda document: document.update(session=2))
    edit(sessions / "2.json", lambda document: document["environment"]["cpu"].update(allowed=[1]))
    rewrite(
        (directory / RUNS / RECORD).with_name("r000-timed.json"),
        lambda record: replace(record, provenance=replace(record.provenance, session=2)),
    )
    timed = _topics(directory, replace(FAST, protocol="timed"))
    assert (
        "environment.cpu.allowed: [0] in session 1, [1] in session 2"
        in (timed["sessions-differ-processors"])
    )
    # The machine is a note of the check, which the analysis does not carry, and
    # the other protocol has no time limit to measure on it.
    assert "sessions-differ-machine" not in timed
    assert "sessions-differ-processors" not in _topics(directory)


def test_the_bundle_lists_what_the_check_found(directory: Path, tmp_path: Path) -> None:
    fail(directory, directory / RUNS / RECORD, ErrorKind.TIMEOUT)
    bundle = tmp_path / "bundle"
    write_bundle(directory, bundle, AnalysisConfig(resamples=50), figures=False)
    text = (bundle / "limitations.md").read_text(encoding="utf-8")
    assert "- **failures**: 1 runs failed" in text
    assert text.count("- **cells-partial**: Under protocol evals, 1 of 6 cells") == 1
    assert "cells-partial**: Under protocol more" not in text
