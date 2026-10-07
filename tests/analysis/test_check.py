"""The analysis runs the check of the campaign directory and carries what it found."""

from __future__ import annotations

import logging
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from dispatchatlas.analysis import AnalysisConfig, load_dataset, summarize, write_bundle
from dispatchatlas.campaigns import Protocol, check_campaign, run
from dispatchatlas.campaigns.records import ErrorKind, RunRecord
from dispatchatlas.campaigns.store import CAMPAIGN_FILE, RUNS
from dispatchatlas.core import StopCriteria
from dispatchatlas.core.solver import TracePoint
from tests.campaigns.damage import fail, rewrite
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
    # The analysis states the missing runs itself, so the finding is not repeated;
    # the partial cell is what the check adds.
    assert "runs-unattempted" not in topics
    assert "random-search on flow-shop-3x2-1: 1 of 2 runs have a record" in topics["cells-partial"]
    # The other protocol has every record.
    assert {"campaign", "cells-partial"}.isdisjoint(
        _topics(directory, replace(FAST, protocol="more"))
    )


def test_findings_the_analysis_states_itself_are_not_repeated(directory: Path) -> None:
    for path in sorted((directory / RUNS / "hc").rglob("*-evals.json")):
        fail(directory, path, ErrorKind.TIMEOUT)
    report = check_campaign(directory, replan=False)
    assert {"runs-failed", "solver-without-records"} <= {item.code for item in report.findings}
    topics = _topics(directory)
    assert "hc 4 timeout" in topics["failures"]
    assert "runs-failed" not in topics
    assert "hc: 4 failed" in topics["solver-without-records"]
    assert "blocking" not in topics["solver-without-records"]


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


def test_the_bundle_lists_what_the_check_found(directory: Path, tmp_path: Path) -> None:
    (directory / RUNS / RECORD).unlink()
    bundle = tmp_path / "bundle"
    write_bundle(directory, bundle, AnalysisConfig(resamples=50), figures=False)
    text = (bundle / "limitations.md").read_text(encoding="utf-8")
    assert "- **campaign**: 1 of 10 planned runs have no record yet" in text
    assert text.count("- **cells-partial**: Under protocol evals, 1 of 6 cells") == 1
    assert "cells-partial**: Under protocol more" not in text
