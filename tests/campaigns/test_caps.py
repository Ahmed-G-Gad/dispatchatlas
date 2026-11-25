"""Time caps: a limit on the time of a solver's runs, and the record of a run that reaches one."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

from dispatchatlas.analysis.dataset import load_dataset
from dispatchatlas.campaigns import (
    Campaign,
    Execution,
    FamilySource,
    Plan,
    PlannedProblem,
    PlannedRun,
    Protocol,
    Provenance,
    RunRecord,
    SolverSpec,
    campaign_from_dict,
    campaign_to_dict,
    check_campaign,
    loads_campaign,
    plan,
    read_failures,
    read_runs,
    replay,
    run,
)
from dispatchatlas.campaigns.execution import observations
from dispatchatlas.campaigns.planning import prepare
from dispatchatlas.campaigns.records import ErrorKind, Outcome, ProblemRef, SolverEntry
from dispatchatlas.campaigns.runner import tasks_for
from dispatchatlas.campaigns.store import RUNS
from dispatchatlas.core import (
    ConfigError,
    Problem,
    SerializationError,
    SolveResult,
    SolverInfo,
    Status,
    StopCriteria,
    StopReason,
)
from dispatchatlas.core.serialization import canonical_json, digest
from dispatchatlas.solvers import SolverRegistry
from dispatchatlas.solvers.operators import random_topological_order
from dispatchatlas.solvers.search import Search
from tests.campaigns.damage import read, rewrite
from tests.campaigns.solvers import registry, tiny_campaign

FIRST, SECOND = "flow-shop-3x2-1", "flow-shop-3x2-2"
"""The ids of the two problems of the tiny campaigns."""

EVALS = Protocol("evals", StopCriteria(max_evaluations=15))
PINNED = Campaign(
    id="pinned",
    seed=5,
    instances=(FamilySource("flow-shop", (1, 2), {"jobs": 3, "machines": 2}),),
    solvers=(
        SolverSpec("random-search"),
        SolverSpec("hill-climbing", "hc", {"restart_after": 5}),
        SolverSpec("input-order", replicates=2),
    ),
    protocols=(EVALS, Protocol("clock", StopCriteria(time_limit=2.0, max_evaluations=100))),
    replicates=3,
    execution=Execution(backend="sequential"),
)
"""A campaign without caps, whose forms were hashed before caps existed."""


@dataclass(frozen=True)
class Slow:
    """Random sampling that pauses ``delay`` seconds before every evaluation."""

    delay: float = 0.01
    info: ClassVar[SolverInfo] = SolverInfo(
        id="test-slow",
        name="Slow sampling",
        family="random-sampling",
        summary="Samples random orders, pausing before each one.",
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        search = Search(problem, stop, seed, self.info)
        with search:
            while True:
                search.next_iteration()
                time.sleep(self.delay)
                search.evaluate(random_topological_order(problem, search.rng))
        return search.result()


def _registry() -> SolverRegistry:
    solvers = registry()
    solvers.register(Slow)
    return solvers


REGISTRY = _registry()


def _capped(caps: dict[str, float], *, evaluations: int = 40) -> Campaign:
    """A slow solver capped as ``caps`` says, and the random search, on two problems."""
    return tiny_campaign(
        "capped",
        solvers=(
            SolverSpec("test-slow", parameters={"delay": 0.01}, time_caps={"evals": caps}),
            SolverSpec("random-search"),
        ),
        protocols=(Protocol("evals", StopCriteria(max_evaluations=evaluations)),),
        replicates=1,
    )


def _with_caps(campaign: Campaign, caps: dict[str, dict[str, float]]) -> Campaign:
    """The campaign with the caps on its first solver."""
    first, *others = campaign.solvers
    return replace(campaign, solvers=(replace(first, time_caps=caps), *others))


def _records(directory: Path) -> list[Path]:
    return sorted((directory / RUNS).rglob("*.json"))


@pytest.fixture(scope="module")
def reached(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Capped runs: one that reaches its cap and one that does not."""
    campaign = _capped({FIRST: 0.15, SECOND: 60.0})
    return run(campaign, tmp_path_factory.mktemp("reached"), registry=REGISTRY).directory


# ------------------------------------------------------------- configuration


def test_a_campaign_without_caps_keeps_its_hash_and_its_documents() -> None:
    assert PINNED.config_hash == "bddc0cbd56976fda71e9ec496eef0bf3188ad1715e7651b06f9255fbc9875eae"
    assert [solver.to_dict() for solver in PINNED.solvers] == [
        {"id": "random-search", "label": None, "parameters": {}},
        {"id": "hill-climbing", "label": "hc", "parameters": {"restart_after": 5}},
        {"id": "input-order", "label": None, "parameters": {}, "replicates": 2},
    ]
    assert "time_caps" not in canonical_json(PINNED.identity())
    document = campaign_to_dict(PINNED)
    assert digest(document) == "9a06d2d1b057526f69b26b04e8b0058e90f5a5ec79086c4bc3e24b1417860132"
    assert campaign_from_dict(document) == PINNED


def test_a_record_and_a_plan_without_caps_keep_their_documents() -> None:
    entry = SolverEntry(
        "random-search", "random-search", None, {}, "random-sampling", True, True, False
    )
    provenance = Provenance(1, 1, "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", 1.0, 0.9, 1024)
    assert sorted(provenance.to_dict()) == [
        "attempt",
        "concurrent",
        "cpu",
        "cpu_seconds",
        "finished",
        "hard_limit",
        "peak_rss_bytes",
        "session",
        "shared_clock",
        "started",
        "waited_seconds",
        "wall_seconds",
        "worker",
    ]
    assert Provenance.from_dict(provenance.to_dict()) == provenance
    outcome = Outcome(
        status=Status.FEASIBLE,
        stop_reason=StopReason.EVALUATION_LIMIT,
        objective="makespan",
        value=5.0,
        objectives={"makespan": 5.0},
        bound=None,
        gap=None,
        feasible=True,
        violations=(),
        evaluations=20,
        iterations=20,
        elapsed=0.5,
    )
    record = RunRecord(
        run_id="run-0123456789abcdef",
        campaign="records",
        config_hash="0" * 64,
        problem=ProblemRef("p", None, "1" * 64),
        solver=entry,
        protocol=Protocol("evals", StopCriteria(max_evaluations=20)),
        replicate=0,
        seed=3,
        outcome=outcome,
        result_hash=outcome.result_hash(),
        schedule=None,
        provenance=provenance,
    )
    record = replace(record, identity_hash=record.identity_digest())
    assert record.identity_digest() == (
        "fe8668f89223035a16e69df52e79f298e9d248d22fab9a4922b133dbfca8ee0d"
    )
    assert digest(record.to_dict()) == (
        "4ed97a820264e3fd2667c968420dc5b4f91d0974aaa2b839eeb9a59c0f1aca67"
    )
    planned = Plan(
        campaign="pinned",
        config_hash=PINNED.config_hash,
        seed=5,
        replicates=3,
        problems=(PlannedProblem("p", None, "1" * 64, 0, 3, 2, ("makespan",), ()),),
        solvers=(entry,),
        protocols=PINNED.protocols,
        runs=(PlannedRun("run-0123456789abcdef", "p", "random-search", "evals", 0, 3),),
    )
    assert digest(planned.to_dict()) == (
        "7a7df9206d1234f012aef604ef8f158f88ec7c2de95982347281e543618d0080"
    )


def test_provenance_states_a_cap_and_refuses_one_that_is_no_time() -> None:
    provenance = Provenance(
        1, 1, "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", 1.0, hard_limit=7.5, time_cap=2.5
    )
    document = provenance.to_dict()
    assert document["time_cap"] == 2.5
    assert Provenance.from_dict(document) == provenance
    assert Provenance.from_dict({**document, "time_cap": None}) == replace(
        provenance, time_cap=None
    )
    refused = ((0, "above zero"), (-1.0, "above zero"), ("2", "number"), (True, "number"))
    for bad, message in refused:
        with pytest.raises(SerializationError, match=message):
            Provenance.from_dict({**document, "time_cap": bad})


def test_caps_are_part_of_the_document_and_the_hash_only_when_set() -> None:
    capped = _with_caps(PINNED, {"evals": {FIRST: 2.5, SECOND: 3}})
    assert capped.solvers[0].to_dict()["time_caps"] == {"evals": {FIRST: 2.5, SECOND: 3.0}}
    assert "time_caps" not in capped.solvers[1].to_dict()
    assert capped.identity()["solvers"][0]["time_caps"] == {"evals": {FIRST: 2.5, SECOND: 3.0}}
    assert capped.config_hash != PINNED.config_hash
    # The order of the entries and the type of a whole number change nothing.
    assert _with_caps(PINNED, {"evals": {SECOND: 3.0, FIRST: 2.5}}).config_hash == (
        capped.config_hash
    )
    other = _with_caps(PINNED, {"evals": {FIRST: 2.5, SECOND: 3.5}})
    assert other.config_hash not in {capped.config_hash, PINNED.config_hash}
    assert replace(PINNED.solvers[0], time_caps={}) == PINNED.solvers[0]


def test_caps_are_read_only_and_answer_by_protocol_and_problem() -> None:
    spec = SolverSpec("random-search", time_caps={"evals": {FIRST: 2}})
    assert spec.time_cap("evals", FIRST) == 2.0
    assert isinstance(spec.time_cap("evals", FIRST), float)
    assert spec.time_cap("evals", SECOND) is None
    assert spec.time_cap("clock", FIRST) is None
    assert SolverSpec("random-search").time_cap("evals", FIRST) is None
    with pytest.raises(TypeError):
        spec.time_caps["clock"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        spec.time_caps["evals"][SECOND] = 1.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("caps", "message"),
    [
        ("evals", "must be a table"),
        ({"evals": 5}, "must be a table of problem ids and seconds"),
        ({"evals": {}}, "not empty"),
        ({"": {FIRST: 1.0}}, "protocol name"),
        ({"evals": {1: 2.0}}, "problem id"),
        ({"evals": {FIRST: 0}}, "must be positive"),
        ({"evals": {FIRST: -1.0}}, "at least 0"),
        ({"evals": {FIRST: float("nan")}}, "finite"),
        ({"evals": {FIRST: float("inf")}}, "finite"),
        ({"evals": {FIRST: True}}, "must be a number"),
        ({"evals": {FIRST: "2"}}, "must be a number"),
    ],
)
def test_caps_that_are_no_tables_of_positive_seconds_are_refused(caps: Any, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        SolverSpec("random-search", time_caps=caps)


def test_caps_round_trip_through_campaign_documents_and_files() -> None:
    capped = _with_caps(PINNED, {"evals": {FIRST: 2.5}, "clock": {SECOND: 1.5}})
    document = campaign_to_dict(capped)
    assert document["solvers"][0]["time_caps"] == {"evals": {FIRST: 2.5}, "clock": {SECOND: 1.5}}
    assert "time_caps" not in document["solvers"][1]
    restored = campaign_from_dict(json.loads(json.dumps(document)))
    assert restored == capped
    assert restored.config_hash == capped.config_hash
    text = f"""
schema_version = 1
id = "capped"
seed = 3

[[instances]]
family = "flow-shop"
parameters = {{ jobs = 3, machines = 2 }}
seeds = [1, 2]

[[solvers]]
id = "random-search"

[solvers.time_caps.evals]
{FIRST} = 2.5
{SECOND} = 3

[[protocols]]
name = "evals"
stop = {{ max_evaluations = 20 }}
"""
    campaign = loads_campaign(text)
    assert campaign.solvers[0].time_caps == {"evals": {FIRST: 2.5, SECOND: 3.0}}
    assert loads_campaign(json.dumps(campaign_to_dict(campaign)), format="json") == campaign


@pytest.mark.parametrize(
    ("caps", "message"),
    [
        ("slow", r"solvers\[0\]\.time_caps must be a table"),
        ({"evals": 3}, r"solvers\[0\]\.time_caps\.evals must be a table"),
        ({"evals": {FIRST: "x"}}, rf"solvers\[0\]\.time_caps\.evals\.{FIRST} must be a number"),
        ({"evals": {FIRST: -2}}, r"solvers\[0\]: .*at least 0"),
    ],
)
def test_a_campaign_file_with_bad_caps_names_the_entry(caps: Any, message: str) -> None:
    document = campaign_to_dict(PINNED)
    document["solvers"][0]["time_caps"] = caps
    with pytest.raises(ConfigError, match=message):
        campaign_from_dict(document)


def test_the_stop_of_a_run_is_the_shorter_of_the_protocols_limit_and_its_cap() -> None:
    spec = SolverSpec("random-search", time_caps={"evals": {FIRST: 2.0}})
    bounded = Protocol("evals", StopCriteria(max_evaluations=15, schedule="binding"))
    assert spec.capped(bounded, FIRST) == (
        StopCriteria(time_limit=2.0, max_evaluations=15, schedule="binding"),
        2.0,
    )
    longer = Protocol("evals", StopCriteria(max_evaluations=15, time_limit=5.0))
    assert spec.capped(longer, FIRST) == (StopCriteria(time_limit=2.0, max_evaluations=15), 2.0)
    for protocol in (
        Protocol("evals", StopCriteria(max_evaluations=15, time_limit=1.0)),
        Protocol("evals", StopCriteria(max_evaluations=15, time_limit=2.0)),
    ):
        # No cap is stated where the protocol's own limit is as tight.
        assert spec.capped(protocol, FIRST) == (protocol.stop, None)
    assert spec.capped(bounded, SECOND) == (bounded.stop, None)
    assert spec.capped(replace(bounded, name="other"), FIRST) == (bounded.stop, None)


# ------------------------------------------------------------------ planning


def test_a_cap_for_a_protocol_or_problem_the_campaign_lacks_is_refused() -> None:
    def campaign(caps: dict[str, dict[str, float]]) -> Campaign:
        return tiny_campaign(
            solvers=(SolverSpec("random-search", time_caps=caps),), protocols=(EVALS,)
        )

    assert plan(campaign({"evals": {FIRST: 1.0, SECOND: 2.0}}), registry=REGISTRY).runs
    with pytest.raises(ConfigError, match="time caps for protocol 'clock'"):
        plan(campaign({"clock": {FIRST: 1.0}}), registry=REGISTRY)
    with pytest.raises(ConfigError, match=r"under protocol 'evals' for problems .*'flow-shop-9'"):
        plan(campaign({"evals": {FIRST: 1.0, "flow-shop-9": 2.0}}), registry=REGISTRY)


# ----------------------------------------------------------------- execution


def test_a_cap_is_the_time_limit_of_its_run_and_the_hard_limit_follows() -> None:
    campaign = tiny_campaign(
        solvers=(
            SolverSpec("random-search", time_caps={"evals": {FIRST: 4.0}}),
            SolverSpec("input-order"),
        ),
        protocols=(EVALS,),
        replicates=1,
        backend="process",
        timeout_factor=1.5,
        grace=0.5,
    )
    prepared = prepare(campaign, registry=REGISTRY)
    tasks = {
        (item.solver, item.problem): task
        for item, task in zip(
            prepared.plan.runs,
            tasks_for(prepared, prepared.plan.runs, campaign.execution, lambda _: 1),
            strict=True,
        )
    }
    capped = tasks["random-search", FIRST]
    assert capped.stop == StopCriteria(time_limit=4.0, max_evaluations=15)
    assert capped.hard_limit == 1.5 * 4.0 + 0.5
    # The run on the other problem and the other solver keep the protocol's criteria.
    for key in (("random-search", SECOND), ("input-order", FIRST), ("input-order", SECOND)):
        assert tasks[key].stop == EVALS.stop
        assert tasks[key].hard_limit is None


def test_a_run_that_reaches_its_cap_stops_for_time_and_is_censored(reached: Path) -> None:
    records = {(record.solver.name, record.problem.id): record for record in read_runs(reached)}
    stopped = records["test-slow", FIRST]
    assert stopped.outcome.stop_reason is StopReason.TIME_LIMIT
    assert stopped.outcome.evaluations < 40
    assert stopped.outcome.elapsed >= 0.15
    assert stopped.provenance.time_cap == 0.15
    assert not stopped.replayable
    # The run whose cap is far away ends at its evaluation limit, and states the cap it had.
    finished = records["test-slow", SECOND]
    assert finished.outcome.stop_reason is StopReason.EVALUATION_LIMIT
    assert finished.outcome.evaluations == 40
    assert finished.provenance.time_cap == 60.0
    assert finished.replayable
    for problem in (FIRST, SECOND):
        baseline = records["random-search", problem]
        assert baseline.outcome.stop_reason is StopReason.EVALUATION_LIMIT
        assert baseline.outcome.evaluations == 40
        assert baseline.provenance.time_cap is None
    dataset = load_dataset(reached)
    assert {(item.solver, item.problem): item.censored for item in dataset.runs} == {
        ("test-slow", FIRST): True,
        ("test-slow", SECOND): False,
        ("random-search", FIRST): False,
        ("random-search", SECOND): False,
    }
    # Every uncensored run ends at exactly its evaluation limit, on the evaluation limit.
    for item in dataset.runs:
        assert item.outcome is not None
        if not item.censored:
            assert item.outcome.evaluations == 40
            assert item.outcome.stop_reason is StopReason.EVALUATION_LIMIT


def test_a_record_names_a_cap_only_when_its_run_has_one(reached: Path) -> None:
    for path in _records(reached):
        document = json.loads(path.read_text(encoding="utf-8"))
        capped = document["solver"]["name"] == "test-slow"
        assert ("time_cap" in document["provenance"]) is capped
        assert RunRecord.from_dict(document).to_dict() == document


def test_the_check_of_a_capped_campaign_reports_the_stops_and_nothing_else(reached: Path) -> None:
    report = check_campaign(reached, registry=REGISTRY)
    assert report.ok
    # A run that ended for its time limit among runs that ended for their evaluation limit.
    assert {finding.code for finding in report.findings} == {"stop-reasons-diverge"}


def test_a_capped_campaign_resumes_and_a_changed_cap_is_another_configuration(
    reached: Path, tmp_path: Path
) -> None:
    shutil.copytree(reached, tmp_path / "capped")
    again = run(_capped({FIRST: 0.15, SECOND: 60.0}), tmp_path, registry=REGISTRY)
    assert (again.attempted, again.pending, again.done) == (0, 0, True)
    with pytest.raises(ConfigError, match="another configuration"):
        run(_capped({FIRST: 0.15, SECOND: 90.0}), tmp_path, registry=REGISTRY)


def test_a_replay_applies_the_caps_and_skips_the_run_that_stopped_on_one(
    reached: Path, tmp_path: Path
) -> None:
    shutil.copytree(reached, tmp_path / "capped")
    report = replay(_capped({FIRST: 0.15, SECOND: 60.0}), tmp_path, registry=REGISTRY)
    verdicts = {(entry.solver, entry.problem): entry.verdict.value for entry in report.entries}
    assert verdicts == {
        ("test-slow", FIRST): "skipped",
        ("test-slow", SECOND): "match",
        ("random-search", FIRST): "match",
        ("random-search", SECOND): "match",
    }
    assert report.ok


def test_hard_limits_that_follow_different_caps_are_no_limitation(
    reached: Path, tmp_path: Path
) -> None:
    directory = shutil.copytree(reached, tmp_path / "capped")
    paths = [path for path in _records(directory) if read(path).solver.name == "test-slow"]
    assert len(paths) == 2

    def arm(path: Path, limit: float | None, cap: float | None) -> None:
        rewrite(
            path,
            lambda held: replace(
                held, provenance=replace(held.provenance, hard_limit=limit, time_cap=cap)
            ),
        )

    # The process backend arms 1.5 * cap + 0.5 seconds for each run, so the limits differ.
    arm(paths[0], 1.5 * 0.15 + 0.5, 0.15)
    arm(paths[1], 1.5 * 60.0 + 0.5, 60.0)
    findings = {f.code for f in check_campaign(directory, registry=REGISTRY).findings}
    assert "hard-limits-differ" not in findings
    # Two attempts under one cap that were armed otherwise are a limitation.
    arm(paths[0], 7.0, 60.0)
    report = check_campaign(directory, registry=REGISTRY)
    (finding,) = report.find("hard-limits-differ")
    assert finding.protocol == "evals"
    assert "for the runs under a time cap of 60 s," in finding.message
    assert sorted(finding.items) == ["7 s: 1 attempt", "90.5 s: 1 attempt"]


def test_the_check_derives_the_notes_of_a_run_from_its_cap(reached: Path, tmp_path: Path) -> None:
    directory = shutil.copytree(reached, tmp_path / "capped")
    (path,) = (path for path in _records(directory) if read(path).provenance.time_cap == 60.0)
    record = read(path)
    # The run stopped for its time limit, the cap of 60 seconds, after one second.
    # The trace is cut to that second, so the times the real run took on a slow host stay within it.
    trace = tuple(replace(point, elapsed=min(point.elapsed, 1.0)) for point in record.outcome.trace)
    outcome = replace(record.outcome, stop_reason=StopReason.TIME_LIMIT, elapsed=1.0, trace=trace)
    facts = observations(
        outcome, replace(record.protocol.stop, time_limit=60.0), 1.0, iterative=True
    )
    assert [kind for kind, _ in facts] == ["early-stop"]
    rewrite(
        path,
        lambda held: replace(
            held,
            outcome=outcome,
            provenance=replace(held.provenance, wall_seconds=1.0),
            notes=tuple(text for _, text in facts),
        ),
    )
    findings = {f.code for f in check_campaign(directory, registry=REGISTRY).findings}
    # The note is one the check derives, not one it cannot account for.
    assert "clock-disagreement" in findings
    assert "records-noted" not in findings


def test_a_capped_run_that_never_stops_is_killed_at_its_hard_limit(tmp_path: Path) -> None:
    campaign = tiny_campaign(
        solvers=(
            SolverSpec(
                "test-sleeper", parameters={"seconds": 30.0}, time_caps={"evals": {FIRST: 0.3}}
            ),
        ),
        protocols=(EVALS,),
        instances=(FamilySource("flow-shop", (1,), {"jobs": 3, "machines": 2}),),
        backend="process",
        timeout_factor=1.0,
        grace=0.3,
    )
    summary = run(campaign, tmp_path, registry=REGISTRY)
    (failure,) = read_failures(summary.directory)
    assert failure.error.kind is ErrorKind.TIMEOUT
    assert failure.provenance.time_cap == 0.3
    assert failure.provenance.hard_limit == pytest.approx(0.6)
    (item,) = load_dataset(summary.directory).runs
    assert (item.killed, item.censored, item.status) == (True, False, "failed")
