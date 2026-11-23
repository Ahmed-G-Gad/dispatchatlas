"""A session deadline: no run starts that cannot end before it; the runs held back stay pending."""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.campaigns import Campaign, FamilySource, SolverSpec, read_runs, run
from dispatchatlas.campaigns import runner as runner_module
from dispatchatlas.campaigns.backends import SequentialBackend
from dispatchatlas.campaigns.planning import prepare
from dispatchatlas.campaigns.runner import RunSummary, _Deadline, tasks_for
from dispatchatlas.campaigns.store import LOGS
from dispatchatlas.core import ConfigError
from tests.campaigns.solvers import registry, tiny_campaign

FIRST, SECOND = "flow-shop-3x2-1", "flow-shop-3x2-2"
"""The ids of the two problems of the tiny campaigns."""

SESSION_KEYS = [
    "campaign",
    "changes",
    "config_hash",
    "counts",
    "environment",
    "execution",
    "finished",
    "load",
    "plan_digest",
    "plan_entered",
    "plan_left",
    "plan_reseeded",
    "plan_solvers",
    "schema",
    "session",
    "started",
    "status",
    "workers",
]
"""The keys of a session document, as sessions wrote them before they had a deadline."""

CONFIG_HASH = "1fedffa88fc74797d6a32ce612d0b7a237d5fe4aa23d3b8ae0f9d8d788522fab"
RECORD_HASHES = {
    "run-b056c20ee9329640": (
        "cb729113488aab8cf968e9732ecea06140b30807fdfe65e471bc55605950c11d",
        "5f16c3778391951150c19263bde32ab84eefe037ab97716616f88f9ea6b18271",
    ),
    "run-ea2c2b06ef20d000": (
        "8682882dbd83167b41a55cfa485d2076c0b28965494bea9b319ca3e0c2d99513",
        "bbb0aa9c39c204755d59b26e7acbf603fb0ee93b7b2280e93ce45e91e3e0df92",
    ),
    "run-371520626171b28f": (
        "7ce89d8505dc6f4b16767c6be925c72dd6c7650d6d8457248a39e144cc3c29fd",
        "c6a71baeeac0de7c8536bd4b6b3b4fab19001c309d69b30f4ddb20365f0ae8ee",
    ),
    "run-5cbf3fab5f6c34c3": (
        "05de72b1a79a68e2ec8a7caf539af031eb34d12a309216354824313f78569221",
        "ec7655f52c000680f75391013541cd3e8caa32bc9a7494b6e50a1c65cf374efb",
    ),
}
"""The config hash of ``_campaign(500.0)`` and the result and identity hash of each of its runs,
as the library made them before sessions had a deadline."""

REGISTRY = registry()


def _campaign(
    cap: float, problems: tuple[int, ...] = (1, 2), name: str = "baseline", **execution: Any
) -> Campaign:
    """Input order, which has no time limit, and a random search capped at ``cap`` seconds."""
    ids = {1: FIRST, 2: SECOND}
    capped = SolverSpec("random-search", time_caps={"evals": {ids[p]: cap for p in problems}})
    return tiny_campaign(
        name,
        solvers=(SolverSpec("input-order"), capped),
        instances=(FamilySource("flow-shop", problems, {"jobs": 3, "machines": 2}),),
        replicates=1,
        **execution,
    )


def _run_ids(campaign: Campaign, solver: str) -> set[str]:
    plan = prepare(campaign, registry=REGISTRY).plan
    return {item.run_id for item in plan.runs if item.solver == solver}


def _log(summary: RunSummary, session: int = 1) -> str:
    return (summary.directory / LOGS / f"session-{session}.log").read_text()


def _session(summary: RunSummary, session: int = 1) -> dict[str, Any]:
    document = json.loads((summary.directory / "sessions" / f"{session}.json").read_text())
    assert isinstance(document, dict)
    return document


def _hashes(summary: RunSummary) -> dict[str, tuple[str, str | None]]:
    return {r.run_id: (r.result_hash, r.identity_hash) for r in read_runs(summary.directory)}


# ------------------------------------------------------------------ the deadline


def test_a_deadline_holds_back_the_runs_whose_cap_cannot_end_before_it(tmp_path: Path) -> None:
    campaign = _campaign(500.0)
    capped, free = _run_ids(campaign, "random-search"), _run_ids(campaign, "input-order")
    assert len(capped) == len(free) == 2
    first = run(campaign, tmp_path, registry=REGISTRY, deadline_seconds=100.0)
    assert first.held_back == 2
    # A run without a time limit starts while the deadline has not passed.
    assert (first.attempted, first.completed, first.pending) == (2, 2, 2)
    assert {record.run_id for record in read_runs(first.directory)} == free
    lines = _log(first).splitlines()
    assert sum("deadline of 100 s from the start of the session" in line for line in lines) == 1
    assert lines[-2].endswith("session 1: the deadline held back 2 run(s), which stay pending")
    assert lines[-1].endswith("session 1 finished: 2 completed, 0 failed, 2 pending")
    # The runs held back had no attempt, so the next session makes their first, and the
    # records of the two sessions are those of one session without a deadline.
    second = run(campaign, tmp_path, registry=REGISTRY)
    assert second.done
    assert (second.attempted, second.held_back) == (2, 0)
    attempts = {record.run_id: record.provenance.attempt for record in read_runs(second.directory)}
    assert attempts == dict.fromkeys(capped | free, 1)
    assert _hashes(second) == RECORD_HASHES
    assert second.config_hash == CONFIG_HASH


def test_a_run_whose_cap_fits_before_the_deadline_starts(tmp_path: Path) -> None:
    summary = run(_campaign(500.0), tmp_path, registry=REGISTRY, deadline_seconds=1e6)
    assert summary.done
    assert (summary.attempted, summary.held_back) == (4, 0)
    assert "session 1: the deadline held back 0 run(s), which stay pending" in _log(summary)
    # The deadline changes no record, no hash and no document.
    assert _hashes(summary) == RECORD_HASHES
    assert summary.config_hash == CONFIG_HASH
    assert sorted(_session(summary)) == SESSION_KEYS
    assert "deadline" not in json.dumps(_session(summary)["execution"])


def test_a_deadline_that_has_passed_holds_back_every_run(tmp_path: Path) -> None:
    # Runs without a time limit start only while the deadline has not passed.
    summary = run(_campaign(500.0), tmp_path, registry=REGISTRY, deadline_seconds=1e-9)
    assert (summary.attempted, summary.held_back, summary.pending) == (0, 4, 4)
    assert not summary.done
    assert "the deadline held back 4 run(s)" in _log(summary)
    assert read_runs(summary.directory) == ()


def test_a_session_without_a_deadline_is_what_it_was(tmp_path: Path) -> None:
    campaign = _campaign(500.0)
    summary = run(campaign, tmp_path, registry=REGISTRY, deadline_seconds=None)
    assert summary.done
    assert summary.held_back == 0
    assert sorted(_session(summary)) == SESSION_KEYS
    assert "deadline" not in _log(summary)
    assert _log(summary).splitlines()[-1].endswith("finished: 4 completed, 0 failed, 0 pending")
    assert "deadline" not in json.dumps(_session(summary)["execution"])
    assert _hashes(summary) == RECORD_HASHES
    assert summary.config_hash == campaign.config_hash == CONFIG_HASH


@pytest.mark.parametrize("bad", [0, -1.0, math.inf, -math.inf, math.nan, True, "5", [5]])
def test_a_deadline_that_is_no_positive_finite_number_is_refused_before_anything_is_made(
    bad: Any, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match="deadline_seconds is a positive finite number"):
        run(_campaign(1.0), tmp_path, registry=REGISTRY, deadline_seconds=bad)
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------ a retry is tested too


def test_a_retry_that_the_deadline_forbids_stays_pending_for_the_next_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The clock reads the start of the session when the run first starts and a day later
    # when its retry would.
    offsets = iter([0.0, 86400.0, 86400.0])

    def deadline(begun: float, seconds: float) -> _Deadline:
        return _Deadline(begun, seconds, clock=lambda: begun + next(offsets))

    monkeypatch.setattr(runner_module, "_Deadline", deadline)
    (tmp_path / "marks").mkdir()
    flaky = SolverSpec("test-flaky", parameters={"marker": str(tmp_path / "marks"), "failures": 1})
    campaign = tiny_campaign(
        "retry",
        solvers=(flaky,),
        instances=(FamilySource("flow-shop", (1,), {"jobs": 3, "machines": 2}),),
        replicates=1,
        retries=1,
    )
    first = run(campaign, tmp_path, registry=REGISTRY, deadline_seconds=3600.0)
    assert (first.attempted, first.errors, first.held_back) == (1, 1, 1)
    assert (first.completed, first.pending) == (0, 1)
    assert not first.done
    monkeypatch.undo()
    second = run(campaign, tmp_path, registry=REGISTRY)
    assert (second.attempted, second.succeeded, second.held_back) == (1, 1, 0)
    assert second.done
    (record,) = read_runs(second.directory)
    assert record.provenance.attempt == 2  # the failure of the first session was attempt 1


# ------------------------------------------------------------------ the test itself


class _Clock:
    """A clock that a test sets."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _tasks() -> dict[str, Any]:
    """One task of each solver of a campaign with a free run and a run capped at 50 s."""
    prepared = prepare(_campaign(50.0), registry=REGISTRY)
    tasks = tasks_for(prepared, prepared.plan.runs, prepared.campaign.execution, lambda _: 1)
    by_run = {item.run_id: item.solver for item in prepared.plan.runs}
    return {by_run[task.run_id]: task for task in tasks}


def test_a_run_starts_when_the_time_that_is_left_covers_its_limit() -> None:
    clock = _Clock()
    deadline = _Deadline(clock.now, 100.0, clock=clock)
    capped = _tasks()["random-search"]
    assert capped.stop.time_limit == 50.0
    assert deadline.admits(capped)
    clock.now += 50.0  # 50 s are left: the limit ends exactly at the deadline
    assert deadline.admits(capped)
    clock.now += 0.001
    assert not deadline.admits(capped)
    assert deadline.held == {capped.run_id}
    clock.now += 100.0
    assert not deadline.admits(capped)
    assert deadline.held == {capped.run_id}  # a run is held back once, however often it is asked


def test_a_run_without_a_limit_starts_only_while_the_deadline_has_not_passed() -> None:
    clock = _Clock()
    deadline = _Deadline(clock.now, 100.0, clock=clock)
    free = _tasks()["input-order"]
    assert free.stop.time_limit is None
    clock.now += 99.0
    assert deadline.admits(free)
    clock.now += 1.0  # the deadline is now
    assert not deadline.admits(free)
    assert deadline.held == {free.run_id}


def test_the_backend_drops_a_run_that_is_refused_and_makes_no_attempt_for_it() -> None:
    tasks = list(_tasks().values())
    refused = {tasks[0].run_id}
    queue = deque(tasks)
    results = list(SequentialBackend().run(queue, lambda task: task.run_id not in refused))
    assert [result.run_id for result in results] == [t.run_id for t in tasks[1:]]
    assert not queue
    queue = deque(tasks)
    assert [result.run_id for result in SequentialBackend().run(queue)] == [t.run_id for t in tasks]


def test_the_process_backend_hands_over_only_the_runs_that_are_admitted(tmp_path: Path) -> None:
    campaign = _campaign(500.0, problems=(1,), backend="process", workers=1)
    summary = run(campaign, tmp_path, registry=REGISTRY, deadline_seconds=100.0)
    assert (summary.attempted, summary.completed, summary.held_back) == (1, 1, 1)
    assert {record.run_id for record in read_runs(summary.directory)} == _run_ids(
        campaign, "input-order"
    )
