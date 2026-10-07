"""The check of a campaign directory: one finding per way a directory can be damaged.

Every test copies a small campaign directory that was run once, damages the
copy in one way, and reads the finding the damage earns. A changed record is
written back with its hashes computed anew where the test is not about the
hashes, so that only the change under test is found.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.campaigns import (
    CheckReport,
    FileSource,
    Finding,
    Protocol,
    SolverSpec,
    check_campaign,
    read_plan,
    run,
)
from dispatchatlas.campaigns.checking import CODES, SEVERITIES
from dispatchatlas.campaigns.execution import OBSERVATION_KINDS, observations, observe
from dispatchatlas.campaigns.records import Category, ErrorKind, RunRecord
from dispatchatlas.campaigns.store import (
    CAMPAIGN_FILE,
    FAILURES,
    PLAN_FILE,
    RUNS,
    SESSIONS,
    SUPERSEDED,
)
from dispatchatlas.core import ConfigError, StopCriteria, StopReason, save_problem
from dispatchatlas.core.solver import TracePoint
from tests.campaigns.damage import edit, fail, read, rewrite
from tests.campaigns.solvers import registry, tiny_campaign
from tests.helpers import random_problem

REGISTRY = registry()
DATA = Path(__file__).parent / "data"
RECORD = Path("random-search") / "flow-shop-3x2-1" / "r000-evals.json"
"""A record of the campaign every test starts from, under ``runs/``."""


@pytest.fixture(scope="module")
def clean(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A campaign of 10 runs: one protocol, two problems, three solvers, two replicates."""
    return run(tiny_campaign(), tmp_path_factory.mktemp("clean"), registry=REGISTRY).directory


@pytest.fixture(scope="module")
def timed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A campaign of 8 runs of two iterative solvers under a protocol with a time limit."""
    campaign = tiny_campaign(
        "timed",
        solvers=(SolverSpec("random-search"), SolverSpec("hill-climbing", "hc")),
        protocols=(Protocol("timed", StopCriteria(max_evaluations=15, time_limit=5.0)),),
    )
    return run(campaign, tmp_path_factory.mktemp("timed"), registry=REGISTRY).directory


@pytest.fixture
def directory(clean: Path, tmp_path: Path) -> Path:
    return shutil.copytree(clean, tmp_path / clean.name)


@pytest.fixture
def timed_directory(timed: Path, tmp_path: Path) -> Path:
    return shutil.copytree(timed, tmp_path / timed.name)


def check(directory: Path, **options: Any) -> CheckReport:
    return check_campaign(directory, registry=REGISTRY, **options)


def codes(report: CheckReport) -> set[str]:
    return {finding.code for finding in report.findings}


def one(report: CheckReport, code: str) -> Finding:
    (finding,) = report.find(code)
    assert finding.severity == CODES[code][0]
    return finding


def second_session(directory: Path, change: Callable[[dict[str, Any]], object]) -> None:
    """Record a second session that differs by ``change`` and executed one counted run."""

    def numbered(document: dict[str, Any]) -> None:
        document["session"] = 2
        change(document)

    shutil.copy(directory / SESSIONS / "1.json", directory / SESSIONS / "2.json")
    edit(directory / SESSIONS / "2.json", numbered)
    rewrite(
        directory / RUNS / RECORD,
        lambda record: replace(record, provenance=replace(record.provenance, session=2)),
    )


# ------------------------------------------------------------ whole reports


def test_a_clean_directory_has_no_finding(clean: Path) -> None:
    report = check(clean)
    assert report.ok
    assert report.findings == ()
    assert (report.campaign, report.config_hash) == ("tiny", read_plan(clean).config_hash)
    assert dict(report.counts) == {
        "planned": 10,
        "completed": 10,
        "failed": 0,
        "unsettled": 0,
        "unattempted": 0,
        "invalid": 0,
        "set_aside": 0,
    }


def test_the_check_writes_nothing(directory: Path) -> None:
    before = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    (directory / RUNS / RECORD).write_text("{ torn", encoding="utf-8")
    before[directory / RUNS / RECORD] = b"{ torn"
    assert not check(directory).ok
    assert {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()} == before


def test_a_directory_written_without_the_later_fields_gives_notes_and_no_crash(
    tmp_path: Path,
) -> None:
    # The files under data/earlier were written by the library before run records
    # stated notes and identity hashes and before session records stated the digest
    # of their plan, the changes from the last session, and the load. Only the
    # values of the session's environment that describe a machine were replaced.
    directory = shutil.copytree(DATA / "earlier" / "tiny", tmp_path / "tiny")
    assert "identity_hash" not in json.loads((directory / RUNS / RECORD).read_text("utf-8"))
    report = check_campaign(directory, replicate_floor=2)
    assert report.ok
    assert report.of("limitation") == ()
    assert codes(report) == {"records-without-identity-hash", "plan-digest-unstated"}
    assert (report.counts["planned"], report.counts["completed"]) == (6, 6)
    assert "6 of 6 records" in one(report, "records-without-identity-hash").message


def test_findings_are_ordered_by_severity_and_serialise(directory: Path) -> None:
    (directory / RUNS / "left-behind.tmp").write_text("x", encoding="utf-8")
    (directory / RUNS / RECORD).unlink()
    report = check(directory)
    severities = [SEVERITIES.index(finding.severity) for finding in report.findings]
    assert severities == sorted(severities)
    assert {finding.severity for finding in report.findings} == set(SEVERITIES)
    document = report.to_dict()
    assert json.loads(json.dumps(document)) == document
    assert document["ok"] is False
    assert document["severities"] == {"blocking": 1, "limitation": 1, "note": 1}
    assert document["counts"]["unattempted"] == 1
    assert [item["code"] for item in document["findings"]] == [
        "runs-unattempted",
        "cells-partial",
        "files-stray",
    ]
    assert set(document["findings"][0]) == {"code", "severity", "protocol", "message", "items"}


def test_every_code_has_a_severity_and_a_meaning() -> None:
    assert all(severity in SEVERITIES for severity, _ in CODES.values())
    assert all(meaning.endswith(".") for _, meaning in CODES.values())
    assert {severity for severity, _ in CODES.values()} == set(SEVERITIES)


def test_what_is_no_directory_or_no_floor_is_refused(directory: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="is not a directory"):
        check_campaign(tmp_path / "absent")
    for floor in (0, -1, 1.5, True):
        with pytest.raises(ConfigError, match="replicate_floor"):
            check(directory, replicate_floor=floor)
    with pytest.raises(ConfigError, match="overrun_tolerance"):
        check(directory, overrun_tolerance=-0.1)


def test_a_directory_without_its_runs_folder_is_checked(directory: Path) -> None:
    shutil.rmtree(directory / RUNS)
    report = check(directory)
    finding = one(report, "runs-unattempted")
    assert len(finding.items) == 6
    assert "10 of 10 planned runs" in finding.message
    assert one(report, "solver-without-records").items == (
        "input-order: 2 unattempted",
        "random-search: 4 unattempted",
        "hc: 4 unattempted",
    )
    assert (report.counts["unattempted"], report.counts["completed"]) == (10, 0)


def test_the_facts_of_a_record_are_the_notes_of_the_engine_with_a_kind(clean: Path) -> None:
    record = read(clean / RUNS / RECORD)
    value = record.outcome.value
    assert value is not None
    outcome = replace(record.outcome, trace=(TracePoint(-1.0, 1, 0, value + 1.0),))
    stop = StopCriteria(time_limit=0.1)
    facts = observations(outcome, stop, 5.0, iterative=True)
    assert [kind for kind, _ in facts] == ["overrun", "elapsed", "trace-time", "trace-value"]
    assert observe(outcome, stop, 5.0, iterative=True) == tuple(text for _, text in facts)
    untraced = replace(record.outcome, trace=())
    assert [kind for kind, _ in observations(untraced, stop, 0.001, iterative=True)] == ["no-trace"]
    assert observations(untraced, stop, 0.001, iterative=False) == ()
    stopped = replace(record.outcome, stop_reason=StopReason.TIME_LIMIT)
    (early,) = observations(stopped, StopCriteria(time_limit=60.0), 0.001, iterative=True)
    assert early[0] == "early-stop"
    assert {"overrun", "elapsed", "trace-time", "trace-value", "no-trace", "early-stop"} < set(
        OBSERVATION_KINDS
    )


# -------------------------------------------------------- campaign and plan


def test_a_missing_campaign_file_blocks(directory: Path) -> None:
    (directory / CAMPAIGN_FILE).unlink()
    report = check(directory)
    assert "is missing" in one(report, "recipe-missing").message
    assert [finding.code for finding in report.of("blocking")] == ["recipe-missing"]


def test_a_campaign_file_that_is_no_campaign_blocks(directory: Path) -> None:
    (directory / CAMPAIGN_FILE).write_text("{ not json", encoding="utf-8")
    assert [finding.code for finding in check(directory).of("blocking")] == ["recipe-unreadable"]
    (directory / CAMPAIGN_FILE).write_text('{"schema": "other"}', encoding="utf-8")
    assert "is not a" in one(check(directory), "recipe-unreadable").message


def test_a_campaign_file_edited_after_it_was_stored_blocks(directory: Path) -> None:
    recorded = json.loads((directory / CAMPAIGN_FILE).read_text("utf-8"))["config_hash"]
    edit(directory / CAMPAIGN_FILE, lambda document: document["campaign"].update(replicates=30))
    report = check(directory)
    finding = one(report, "recipe-altered")
    assert f"records {recorded[:12]}" in finding.message
    assert [item.code for item in report.of("blocking")] == ["recipe-altered"]


def test_a_missing_plan_ends_the_check(directory: Path) -> None:
    (directory / PLAN_FILE).unlink()
    report = check(directory)
    assert [finding.code for finding in report.findings] == ["plan-unreadable"]
    assert (report.campaign, dict(report.counts), report.ok) == (None, {}, False)
    assert report.config_hash is not None


def test_a_plan_of_another_config_hash_blocks(directory: Path) -> None:
    edit(directory / PLAN_FILE, lambda document: document.update(config_hash="f" * 64))
    finding = one(check(directory), "plan-foreign")
    assert "states config hash ffffffffffff" in finding.message


def test_a_plan_that_the_campaign_does_not_give_blocks(directory: Path) -> None:
    dropped: list[dict[str, Any]] = []
    edit(directory / PLAN_FILE, lambda document: dropped.append(document["runs"].pop()))
    report = check(directory)
    (difference,) = one(report, "plan-differs").items
    assert difference.startswith("1 runs of the derived plan are not in the stored plan: ")
    assert f"{dropped[0]['solver']} on {dropped[0]['problem']}" in difference
    # The record of the run that left the plan file is no longer read, and the
    # last session executed a plan of ten runs.
    assert len(one(report, "records-unplanned").items) == 1
    assert (
        "the plan file was changed after that session" in one(report, "plan-digest-differs").message
    )
    # Without deriving the plan again, the digest of the last session still tells.
    again = check(directory, replan=False)
    assert "plan-differs" not in codes(again)
    assert {"plan-not-compared", "plan-digest-differs"} <= codes(again)


def test_a_plan_that_cannot_be_derived_again_is_a_limitation(tmp_path: Path) -> None:
    campaign = tiny_campaign(solvers=(SolverSpec("input-order"), SolverSpec("test-scripted")))
    directory = run(campaign, tmp_path, registry=REGISTRY).directory
    # The scripted solver is declared iterative and returns no trace, which the
    # engine noted in its records and the check derives again from them.
    assert codes(check(directory)) == {"record-without-trace"}
    # The built-in registry has no solver of that id, so the plan is not derived.
    report = check_campaign(directory)
    finding = one(report, "plan-not-derived")
    assert "test-scripted" in finding.message
    assert report.ok
    assert codes(check_campaign(directory, replan=False)) == {
        "plan-not-compared",
        "record-without-trace",
    }


def test_a_plan_changed_after_the_last_session_blocks(directory: Path) -> None:
    edit(directory / SESSIONS / "1.json", lambda document: document.update(plan_digest="0" * 64))
    finding = one(check(directory), "plan-digest-differs")
    assert "session 1" in finding.message
    assert "000000000000" in finding.message


def test_a_session_without_the_digest_of_its_plan_is_noted(directory: Path) -> None:
    edit(directory / SESSIONS / "1.json", lambda document: document.pop("plan_digest"))
    report = check(directory)
    assert codes(report) == {"plan-digest-unstated"}
    assert report.ok


def test_a_plan_that_changed_between_sessions_is_a_limitation(directory: Path) -> None:
    run_document = read_plan(directory).runs[0].to_dict()
    edit(
        directory / SESSIONS / "1.json",
        lambda document: document.update(plan_left=[run_document], plan_entered=[run_document]),
    )
    report = check(directory)
    (item,) = one(report, "plan-changed").items
    assert item.startswith("session 1: 1 runs left, 1 entered, 0 got another seed")
    assert report.ok


# --------------------------------------------------------------------- runs


def test_missing_runs_are_listed_and_excluded_pairs_are_not(tmp_path: Path) -> None:
    deadlines = random_problem(10, tasks=6, max_modes=1, pools=0, deadlines=True)
    save_problem(deadlines, tmp_path / "deadlines.json")
    campaign = replace(
        tiny_campaign(
            solvers=(SolverSpec("random-search"), SolverSpec("test-no-deadlines")),
            instances=(FileSource(Path("deadlines.json")),),
        ),
        base=tmp_path,
    )
    directory = run(campaign, tmp_path / "out", registry=REGISTRY).directory
    report = check(directory)
    # The excluded solver has no run: it is reported apart and is not missing.
    assert codes(report) == {"solver-excluded-everywhere", "pairs-excluded"}
    assert one(report, "solver-excluded-everywhere").items == (
        "test-no-deadlines: excluded from 1 problems",
    )
    (pair,) = one(report, "pairs-excluded").items
    assert pair == f"test-no-deadlines does not model deadline (problem {deadlines.id})"
    assert report.ok
    assert report.counts["planned"] == 2

    (directory / RUNS / "random-search" / deadlines.id / "r001-evals.json").unlink()
    report = check(directory)
    finding = one(report, "runs-unattempted")
    assert finding.protocol == "evals"
    assert finding.items == (f"random-search on {deadlines.id}: 1 of 2 runs",)
    assert "1 of 2 planned runs have no record and no failed attempt" in finding.message
    assert [item.code for item in report.of("blocking")] == ["runs-unattempted"]
    assert report.counts["unattempted"] == 1


def test_partial_cell_is_named(directory: Path) -> None:
    (directory / RUNS / RECORD).unlink()
    finding = one(check(directory), "cells-partial")
    assert finding.items == ("random-search on flow-shop-3x2-1: 1 of 2 runs have a record",)
    assert "1 of 6 cells" in finding.message
    assert finding.protocol == "evals"


def test_settled_failures_are_counted_by_solver_and_kind(directory: Path) -> None:
    fail(directory, directory / RUNS / RECORD, ErrorKind.TIMEOUT)
    fail(directory, (directory / RUNS / RECORD).with_name("r001-evals.json"), ErrorKind.EXCEPTION)
    fail(directory, directory / RUNS / "hc" / RECORD.parent.name / RECORD.name, ErrorKind.TIMEOUT)
    report = check(directory)
    finding = one(report, "runs-failed")
    assert finding.items == ("random-search: 1 exception, 1 timeout", "hc: 1 timeout")
    assert "3 of 10 planned runs never completed" in finding.message
    # Failures that settle a run are a limitation to state, and block nothing.
    assert report.ok
    assert (report.counts["failed"], report.counts["completed"]) == (3, 7)
    assert one(report, "cells-partial").items == (
        "hc on flow-shop-3x2-1: 1 of 2 runs have a record",
    )


def test_unsettled_failure_is_not_a_disclosed_limitation(directory: Path, tmp_path: Path) -> None:
    # Without retries, one recoverable failure settles its run.
    fail(directory, directory / RUNS / RECORD, ErrorKind.EXCEPTION, Category.RECOVERABLE)
    report = check(directory)
    assert report.ok
    assert "runs-unsettled" not in codes(report)
    assert one(report, "runs-failed").items == ("random-search: 1 exception",)

    # With one retry, the same failure leaves the run owed a second attempt.
    retried = run(tiny_campaign(retries=1), tmp_path / "retried", registry=REGISTRY).directory
    fail(retried, retried / RUNS / RECORD, ErrorKind.EXCEPTION, Category.RECOVERABLE)
    report = check(retried)
    finding = one(report, "runs-unsettled")
    assert finding.items == ("random-search on flow-shop-3x2-1: 1 of 2 runs",)
    assert "runs-failed" not in codes(report)
    assert (report.ok, report.counts["unsettled"], report.counts["failed"]) == (False, 1, 0)
    # A terminal failure settles the run whatever the retries.
    fail(retried, retried / RUNS / "hc" / RECORD.parent.name / RECORD.name, ErrorKind.TIMEOUT)
    assert one(check(retried), "runs-failed").items == ("hc: 1 timeout",)


def test_a_solver_whose_every_run_failed_is_named(directory: Path) -> None:
    for path in sorted((directory / RUNS / "hc").rglob("*.json")):
        fail(directory, path, ErrorKind.WORKER_EXIT)
    report = check(directory)
    assert one(report, "solver-without-records").items == ("hc: 4 failed",)
    assert one(report, "runs-failed").items == ("hc: 4 worker-exit",)
    assert "cells-partial" not in codes(report)
    assert report.ok


# -------------------------------------------------------------------- files


def test_a_file_at_a_run_path_that_is_no_record_blocks(directory: Path) -> None:
    (directory / RUNS / RECORD).write_text("{ torn", encoding="utf-8")
    report = check(directory)
    (item,) = one(report, "record-unreadable").items
    assert item.startswith(f"{RUNS}/{RECORD.as_posix()}: ")
    assert (report.counts["invalid"], report.counts["completed"]) == (1, 9)
    # The file is reported once: its run is not listed as never attempted.
    assert "runs-unattempted" not in codes(report)


def test_a_record_of_another_configuration_blocks(directory: Path) -> None:
    rewrite(directory / RUNS / RECORD, lambda record: replace(record, config_hash="f" * 64))
    (item,) = one(check(directory), "record-foreign").items
    assert item.endswith("the record belongs to another configuration")


@pytest.mark.parametrize(
    ("change", "what"),
    [
        (lambda document: document["outcome"].update(value=1.0), "result hash"),
        (lambda document: document.update(seed=document["seed"] + 1), "identity hash"),
        (lambda document: document["schedule"][0].__setitem__(3, 1e6), "schedule digest"),
    ],
)
def test_a_record_changed_after_it_was_written_fails_a_hash(
    directory: Path, change: Callable[[dict[str, Any]], object], what: str
) -> None:
    edit(directory / RUNS / RECORD, change)
    report = check(directory)
    (item,) = one(report, "record-hash-mismatch").items
    assert what in item
    assert [finding.code for finding in report.of("blocking")] == ["record-hash-mismatch"]


def test_a_record_that_states_another_seed_than_the_plan_blocks(directory: Path) -> None:
    rewrite(directory / RUNS / RECORD, lambda record: replace(record, seed=record.seed + 1))
    (item,) = one(check(directory), "record-differs-from-plan").items
    assert item.endswith("the record states another seed than the plan")


def test_a_valid_record_under_superseded_blocks(directory: Path) -> None:
    record = read(directory / RUNS / RECORD)
    aside = directory / SUPERSEDED / RUNS / RECORD.parent / f"r000-evals.{record.run_id}.json"
    aside.parent.mkdir(parents=True)
    shutil.move(directory / RUNS / RECORD, aside)
    report = check(directory)
    (item,) = one(report, "records-set-aside").items
    assert item == f"{aside.relative_to(directory).as_posix()} belongs at runs/{RECORD.as_posix()}"
    assert report.counts["set_aside"] == 1
    assert "runs-unattempted" not in codes(report)


def test_records_and_files_beside_the_plan_are_reported_by_kind(directory: Path) -> None:
    shutil.copy(directory / RUNS / RECORD, (directory / RUNS / RECORD).with_name("r009-evals.json"))
    (directory / RUNS / "random-search" / "torn.json").write_text("{", encoding="utf-8")
    (directory / RUNS / "random-search" / "write.tmp").write_text("x", encoding="utf-8")
    report = check(directory)
    assert codes(report) == {"records-unplanned", "files-unreadable", "files-stray"}
    assert one(report, "records-unplanned").items == (
        "runs/random-search/flow-shop-3x2-1/r009-evals.json",
    )
    assert one(report, "files-unreadable").items == ("runs/random-search/torn.json",)
    assert one(report, "files-stray").items == ("runs/random-search/write.tmp",)
    assert report.ok


def test_failure_files_that_are_no_attempt_are_reported(directory: Path) -> None:
    torn = directory / FAILURES / "hc" / "flow-shop-3x2-1" / "r000-evals.attempt-1.json"
    torn.parent.mkdir(parents=True)
    torn.write_text("{", encoding="utf-8")
    foreign = fail(directory, directory / RUNS / RECORD, keep=True)
    edit(foreign, lambda document: document.update(config_hash="f" * 64))
    report = check(directory)
    assert codes(report) == {"failures-unreadable", "failures-foreign"}
    assert one(report, "failures-unreadable").items == (torn.relative_to(directory).as_posix(),)
    assert one(report, "failures-foreign").items == (foreign.relative_to(directory).as_posix(),)
    assert report.counts["completed"] == 10


# ------------------------------------------------------------------ records


def _with_trace(path: Path, build: Callable[[float], tuple[TracePoint, ...]]) -> None:
    def change(record: RunRecord) -> RunRecord:
        value = record.outcome.value
        assert value is not None
        return replace(record, outcome=replace(record.outcome, trace=build(value)))

    rewrite(path, change)


@pytest.mark.parametrize(
    "build",
    [
        lambda value: (TracePoint(-1.0, 1, 0, value),),
        lambda value: (TracePoint(0.0002, 1, 0, value), TracePoint(0.0001, 2, 0, value)),
        lambda value: (TracePoint(900.0, 1, 0, value),),
        lambda value: (TracePoint(0.0, 5, 0, value), TracePoint(0.0, 2, 0, value)),
    ],
    ids=["negative-time", "times-decrease", "time-beyond-elapsed", "evaluations-decrease"],
)
def test_a_trace_with_an_unusable_time_axis_is_a_limitation(
    directory: Path, build: Callable[[float], tuple[TracePoint, ...]]
) -> None:
    _with_trace(directory / RUNS / RECORD, build)
    report = check(directory)
    assert codes(report) == {"trace-time-axis"}
    finding = one(report, "trace-time-axis")
    assert (finding.items, finding.protocol) == (("random-search: 1 of 4 records",), "evals")


def test_a_trace_that_ends_at_another_value_than_the_outcome_is_a_limitation(
    directory: Path,
) -> None:
    _with_trace(directory / RUNS / RECORD, lambda value: (TracePoint(0.0, 1, 0, value + 5.0),))
    report = check(directory)
    assert codes(report) == {"trace-contradicts-outcome"}
    assert one(report, "trace-contradicts-outcome").items == ("random-search: 1 of 4 records",)


def test_untraced_records_of_an_iterative_solver_are_a_limitation(directory: Path) -> None:
    _with_trace(directory / RUNS / RECORD, lambda _: ())
    report = check(directory)
    assert codes(report) == {"record-without-trace"}
    assert one(report, "record-without-trace").items == ("random-search: 1 of 4 records",)
    # A solver that is not iterative owes no trace.
    path = directory / RUNS / "input-order" / RECORD.parent.name / RECORD.name
    assert not read(path).solver.iterative
    _with_trace(path, lambda _: ())
    assert one(check(directory), "record-without-trace").items == ("random-search: 1 of 4 records",)


def test_a_record_without_an_identity_hash_is_noted(directory: Path) -> None:
    edit(directory / RUNS / RECORD, lambda document: document.pop("identity_hash"))
    report = check(directory)
    assert codes(report) == {"records-without-identity-hash"}
    assert "1 of 10 records" in one(report, "records-without-identity-hash").message


def test_notes_of_the_engine_that_the_record_does_not_show_are_counted(directory: Path) -> None:
    note = "the solver calls its schedule optimal, the check finds it infeasible"
    rewrite(directory / RUNS / RECORD, lambda record: replace(record, notes=(note,)))
    report = check(directory)
    assert codes(report) == {"records-noted"}
    assert one(report, "records-noted").items == (
        "random-search: 1 of 4 records",
        f"for example: {note}",
    )


def _with_wall(path: Path, seconds: float) -> None:
    rewrite(
        path,
        lambda record: replace(record, provenance=replace(record.provenance, wall_seconds=seconds)),
    )


def test_an_elapsed_time_the_wall_time_contradicts_is_noted(directory: Path) -> None:
    _with_wall(directory / RUNS / RECORD, 2.0)
    report = check(directory)
    assert codes(report) == {"clock-disagreement"}
    assert report.ok
    assert report.of("limitation") == ()


def test_budget_overruns_are_counted_by_solver_with_the_largest_ratio(
    timed_directory: Path,
) -> None:
    assert check(timed_directory).findings == ()
    paths = sorted((timed_directory / RUNS / "hc").rglob("*.json"))
    _with_wall(paths[0], 9.0)
    _with_wall(paths[1], 6.0)
    _with_wall(sorted((timed_directory / RUNS / "random-search").rglob("*.json"))[0], 5.2)
    report = check(timed_directory)
    finding = one(report, "budget-overrun")
    assert finding.protocol == "timed"
    # 5.2 s is within the tolerance of 0.05 of the limit of 5 s.
    assert finding.items == ("hc: 2 of 4 runs, the longest 1.8 times the limit",)
    assert "whose time limit is 5 s" in finding.message
    assert "the longest took 1.8 times the limit" in finding.message
    assert report.ok
    # A caller that allows more leaves the shorter overrun out, and one that
    # allows nothing counts the run of the other solver too.
    assert one(check(timed_directory, overrun_tolerance=0.5), "budget-overrun").items == (
        "hc: 1 of 4 runs, the longest 1.8 times the limit",
    )
    assert len(one(check(timed_directory, overrun_tolerance=0.0), "budget-overrun").items) == 2


def test_hard_limits_that_differ_within_a_protocol_are_a_limitation(timed_directory: Path) -> None:
    path = sorted((timed_directory / RUNS / "hc").rglob("*.json"))[0]
    rewrite(
        path,
        lambda record: replace(record, provenance=replace(record.provenance, hard_limit=30.0)),
    )
    report = check(timed_directory)
    assert codes(report) == {"hard-limits-differ"}
    finding = one(report, "hard-limits-differ")
    assert finding.protocol == "timed"
    assert finding.items[0] == "30 s: 1 attempts"
    assert finding.items[1].endswith(": 7 attempts")


def test_stop_reasons_that_diverge_within_a_protocol_are_a_limitation(
    timed_directory: Path,
) -> None:
    path = sorted((timed_directory / RUNS / "hc").rglob("*.json"))[0]
    rewrite(
        path,
        lambda record: replace(
            record, outcome=replace(record.outcome, stop_reason=StopReason.STAGNATION)
        ),
    )
    report = check(timed_directory)
    assert codes(report) == {"stop-reasons-diverge"}
    finding = one(report, "stop-reasons-diverge")
    assert finding.protocol == "timed"
    assert finding.items == (
        "random-search: 4 evaluation-limit",
        "hc: 3 evaluation-limit, 1 stagnation",
    )


# ----------------------------------------------------------------- sessions


@pytest.mark.parametrize(
    ("change", "code", "field"),
    [
        (
            lambda document: document["environment"]["python"].update(version="3.99.0"),
            "sessions-differ-code",
            "environment.python.version: ",
        ),
        (
            lambda document: document["environment"]["library"].update(source_digest="0" * 64),
            "sessions-differ-code",
            "environment.library.source_digest: ",
        ),
        (
            lambda document: document["environment"]["plugins"].update(extra={"version": "1.0"}),
            "sessions-differ-plugins",
            "environment.plugins.extra",
        ),
        (
            lambda document: document["execution"].update(grace=9.0),
            "sessions-differ-execution",
            "execution.grace: 5.0 in session 1, 9.0 in session 2",
        ),
        (
            lambda document: document["environment"]["platform"].update(release="other"),
            "sessions-differ-machine",
            "environment.platform.release: ",
        ),
    ],
    ids=["python", "library", "plugins", "execution", "machine"],
)
def test_sessions_that_executed_counted_runs_are_compared(
    directory: Path, change: Callable[[dict[str, Any]], object], code: str, field: str
) -> None:
    second_session(directory, change)
    report = check(directory)
    assert codes(report) == {code}
    (item,) = one(report, code).items
    assert item.startswith(field)
    assert report.ok


def test_a_session_that_executed_nothing_counted_is_not_compared(directory: Path) -> None:
    shutil.copy(directory / SESSIONS / "1.json", directory / SESSIONS / "2.json")

    def other(document: dict[str, Any]) -> None:
        document["session"] = 2
        document["environment"]["python"]["version"] = "3.99.0"

    edit(directory / SESSIONS / "2.json", other)
    assert check(directory).findings == ()


def test_an_attempt_of_a_session_without_a_record_is_a_limitation(directory: Path) -> None:
    rewrite(
        directory / RUNS / RECORD,
        lambda record: replace(record, provenance=replace(record.provenance, session=7)),
    )
    report = check(directory)
    assert codes(report) == {"sessions-missing"}
    assert one(report, "sessions-missing").items == ("session 7",)


def test_a_session_record_that_cannot_be_read_is_a_limitation(directory: Path) -> None:
    (directory / SESSIONS / "2.json").write_text("{", encoding="utf-8")
    report = check(directory)
    assert "sessions-unreadable" in codes(report)
    assert report.of("blocking") == ()


def test_a_last_session_that_did_not_finish_is_noted(directory: Path) -> None:
    edit(directory / SESSIONS / "1.json", lambda document: document.update(status="interrupted"))
    report = check(directory)
    assert codes(report) == {"session-unfinished"}
    assert "recorded as interrupted" in one(report, "session-unfinished").message


# -------------------------------------------------------------- replication


def test_cells_below_the_replicate_floor_are_a_limitation(directory: Path) -> None:
    assert check(directory, replicate_floor=2).findings == ()
    report = check(directory, replicate_floor=3)
    finding = one(report, "replicates-below-floor")
    # The solver declared deterministic runs once by design and is left out.
    assert finding.items == (
        "random-search: 2 cells, the fewest with 2 completed replicates",
        "hc: 2 cells, the fewest with 2 completed replicates",
    )
    assert "4 of 4 cells" in finding.message
    assert finding.protocol == "evals"
    # The floor counts completed replicates, so a missing record lowers a cell.
    (directory / RUNS / RECORD).unlink()
    finding = one(check(directory, replicate_floor=2), "replicates-below-floor")
    assert finding.items == ("random-search: 1 cells, the fewest with 1 completed replicates",)


def test_a_seed_used_for_two_replicates_of_one_problem_blocks(directory: Path) -> None:
    plan = read_plan(directory)
    problem = "flow-shop-3x2-1"
    first = next(run for run in plan.runs if (run.problem, run.replicate) == (problem, 0)).seed

    def reseed(document: dict[str, Any]) -> None:
        for item in document["runs"]:
            if (item["problem"], item["replicate"]) == (problem, 1):
                item["seed"] = first

    edit(directory / PLAN_FILE, reseed)
    for solver in ("random-search", "hc"):
        rewrite(
            directory / RUNS / solver / problem / "r001-evals.json",
            lambda record: replace(record, seed=first),
        )
    report = check(directory, replan=False)
    assert one(report, "seed-reused").items == (f"{problem}: seed {first} for replicates 0, 1",)
    assert "seeds-unpaired" not in codes(report)
    assert not report.ok


def test_a_replicate_that_ran_with_two_seeds_is_a_limitation(directory: Path) -> None:
    record = rewrite(
        directory / RUNS / RECORD, lambda record: replace(record, seed=record.seed + 1)
    )
    planned = read_plan(directory).run(record.run_id)
    # A session that gave the run another seed names the earlier one, and a
    # record made with it still counts as the record of the run.
    edit(
        directory / SESSIONS / "1.json",
        lambda document: document.update(
            plan_reseeded=[{**planned.to_dict(), "earlier_seed": record.seed}]
        ),
    )
    report = check(directory)
    assert codes(report) == {"seeds-unpaired", "plan-changed"}
    assert one(report, "seeds-unpaired").items == (
        f"flow-shop-3x2-1, replicate 0: seeds {planned.seed}, {record.seed}",
    )
    assert report.ok
    assert report.counts["completed"] == 10
