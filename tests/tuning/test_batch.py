"""The process batch: experiments run in worker processes, and a killed job resumes from its rows.

Most tests replace :func:`~dispatchatlas.processes.calls.run_calls` by a function that runs the
calls in turn in the process of the test and appends the rows as it would, and that can stop
as a job killed by its wall time does. The tests with worker processes run a few experiments of the
synthetic solver of ``batch_targets.py``, and one slow test tunes a small budget in two jobs of
worker processes, the first killed in the middle of a batch, to the result of an uninterrupted run.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.core import ConfigError, StopCriteria
from dispatchatlas.processes.affinity import pinning_refusal_text
from dispatchatlas.processes.calls import Call, CallReport, CallsStoppedError, run_calls
from dispatchatlas.processes.child import locate_callable
from dispatchatlas.processes.rows import (
    RowWriter,
    TornRowsError,
    canonical_key,
    done_keys,
    read_rows,
    repair_rows,
)
from dispatchatlas.tuning import (
    Configuration,
    Experiment,
    ExperimentsFailedError,
    IteratedRacing,
    ProcessRunBatch,
    TuningResult,
    configuration_digest,
    experiment_key,
    process_run_batch,
)
from dispatchatlas.tuning import batch as batch_module
from dispatchatlas.tuning.batch import run_experiment
from tests.processes.helpers import ROOT, slot_cpus
from tests.tuning.batch_targets import FAIL_ABOVE, PROBLEM, SOLVER
from tests.tuning.targets import BEST, SPACE, noisy_cost

STOP = StopCriteria(max_evaluations=100)
INSTANCES = list(range(10))
PIN = pinning_refusal_text() is None
CPUS = slot_cpus(2)
BUDGET = 200
"""Enough runs for a tuning of the four parameters of the synthetic target to have races."""


class _KilledError(Exception):
    """The wall time of the job ended."""


class _Calls:
    """Stands for ``run_calls``: runs the calls in turn here, writes their rows, can be killed.

    The arguments and the rows go through JSON, as they do between the runner and a child.
    ``kill_after`` is the number of rows that this job may write before its wall time ends.
    A call whose arguments ``fails`` accepts writes no row, as a run that failed does, and the
    run stops, as ``run_calls`` does, once ``max_failures`` of them have failed.
    """

    def __init__(
        self,
        kill_after: int | None = None,
        *,
        fails: Callable[[dict[str, Any]], bool] | None = None,
        max_failures: int | None = None,
    ) -> None:
        self.kill_after = kill_after
        self.fails = fails or (lambda arguments: False)
        self.max_failures = max_failures
        self.written = 0
        self.invocations: list[dict[str, Any]] = []

    def __call__(
        self, calls: Sequence[Call], rows: Path, *, key_of: Any, stamp: Any = None, **options: Any
    ) -> CallReport:
        self.invocations.append({"calls": list(calls), "stamp": stamp, **options})
        done = done_keys(rows, key_of)
        written = failures = 0
        with RowWriter(rows, fsync=False) as writer:
            for call in calls:
                if canonical_key(call.key) in done:
                    continue
                if self.kill_after is not None and self.written >= self.kill_after:
                    raise _KilledError
                arguments = json.loads(json.dumps(call.arguments))
                if self.fails(arguments):
                    failures += 1
                    if self.max_failures is not None and failures >= self.max_failures:
                        raise CallsStoppedError(self._report(rows, written, failures, calls))
                    continue
                row = locate_callable(call.target)(**arguments)
                writer.write(json.loads(json.dumps({**row, **(stamp or {}), "run": {"fake": 1}})))
                self.written += 1
                written += 1
        return self._report(rows, written, failures, calls)

    @staticmethod
    def _report(rows: Path, written: int, failures: int, calls: Sequence[Call]) -> CallReport:
        return CallReport(
            rows_written=written,
            done_before=len(calls) - written - failures,
            skipped_failed=0,
            failures=failures,
            ballast_runs=0,
            stopped="max-failures" if failures else None,
            workers=1,
            cpus=(None,),
            elapsed_seconds=0.0,
            rows=rows,
            errors=rows.with_name(f"{rows.stem}.errors.jsonl"),
            ballast=None,
        )


def _batch(rows: Path, solver: str = SOLVER, **options: Any) -> ProcessRunBatch:
    settings: dict[str, Any] = {
        "problem_factory": PROBLEM,
        "stop": STOP,
        "cpus": CPUS,
        "pin": PIN,
        "cwd": ROOT,
        "stamp": {"revision": "abc"},
    }
    settings.update(options)
    return process_run_batch(solver, rows, **settings)


def _experiment(instance: int = 3, seed: int = 11, **changes: Any) -> Experiment[int]:
    configuration: Configuration = {**BEST, **changes}
    return Experiment(configuration, instance, seed)


def _costs(rows: Path) -> dict[str, Any]:
    return {canonical_key(row["key"]): row["cost"] for row in read_rows(rows)}


def _tune(batch: ProcessRunBatch, seed: int = 3) -> TuningResult[int]:
    return IteratedRacing(SPACE, BUDGET, seed=seed).tune(INSTANCES, batch)


# ------------------------------------------------------------------ the key


def test_the_key_of_an_experiment_is_the_digest_of_its_configuration_its_instance_and_seed() -> (
    None
):
    key = experiment_key(BEST, "ta01", 7)
    assert key == [configuration_digest(BEST), "ta01", 7]
    assert len(key[0]) == 64
    assert experiment_key(BEST, ["ta01", 3], 7)[1] == ["ta01", 3]
    # The digest does not depend on the order of the parameters, and depends on every value.
    shuffled = dict(reversed(list(BEST.items())))
    assert configuration_digest(shuffled) == configuration_digest(BEST)
    digests = {configuration_digest({**BEST, "x": 0.31}), configuration_digest({**BEST, "n": 8})}
    digests |= {configuration_digest({**BEST, "m": "a"}), configuration_digest(BEST)}
    assert len(digests) == 4
    assert configuration_digest({**BEST, "x": 1}) != configuration_digest({**BEST, "x": 1.0})


def test_a_configuration_or_an_instance_that_has_no_json_form_is_refused() -> None:
    with pytest.raises(ConfigError, match="has no digest"):
        configuration_digest({**BEST, "x": math.nan})
    with pytest.raises(ConfigError, match="cannot be written as JSON"):
        experiment_key(BEST, {1, 2}, 7)
    with pytest.raises(ConfigError, match="cannot be written as JSON"):
        experiment_key(BEST, math.inf, 7)


# ------------------------------------------------------------------ the row of a run


def test_a_run_returns_its_cost_its_evaluations_and_what_the_solver_reports() -> None:
    key = experiment_key(BEST, 3, 11)
    row = run_experiment(
        key=key,
        solver_factory=SOLVER,
        problem_factory=PROBLEM,
        stop={"max_evaluations": 100},
        configuration=BEST,
        instance=3,
        seed=11,
    )
    assert row == {
        "key": key,
        "configuration": BEST,
        "instance": 3,
        "seed": 11,
        "cost": noisy_cost(_experiment()),
        "status": "feasible",
        "evaluations": 100,
        "stop_reason": "evaluation-limit",
        "wall_seconds": 0.25,
        "first_schedule_digest": f"{11:016x}",
        "repairs": 0,
    }


def test_a_run_that_finds_no_schedule_has_no_cost_and_no_measures() -> None:
    configuration = {**BEST, "m": "c"}
    row = run_experiment(
        key=experiment_key(configuration, 3, 11),
        solver_factory=SOLVER,
        problem_factory=PROBLEM,
        stop={"time_limit": 1.0},
        configuration=configuration,
        instance=3,
        seed=11,
    )
    assert (row["cost"], row["status"], row["stop_reason"]) == (None, "unknown", "time-limit")
    assert (row["evaluations"], row["first_schedule_digest"], row["repairs"]) == (0, None, None)
    json.dumps(row, allow_nan=False)


# ------------------------------------------------------------------ the arguments


def test_the_batch_refuses_what_it_cannot_run_before_anything_starts(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    with pytest.raises(ConfigError, match=re.escape("solver_factory is 'package.module:function'")):
        process_run_batch("no_colon", rows, problem_factory=PROBLEM, stop=STOP, cpus=CPUS)
    for bad in ("", ":f", "m:", 5):
        with pytest.raises(
            ConfigError, match=re.escape("problem_factory is 'package.module:function'")
        ):
            process_run_batch(
                SOLVER,
                rows,
                problem_factory=bad,  # type: ignore[arg-type]
                stop=STOP,
                cpus=CPUS,
            )
    with pytest.raises(ConfigError, match="stop must be StopCriteria"):
        _batch(rows, stop={"max_evaluations": 5})
    with pytest.raises(ConfigError, match=r"the stamp may not use \['cost', 'experiment'\]"):
        _batch(rows, stamp={"cost": 1, "experiment": 2, "revision": "abc"})
    with pytest.raises(ConfigError, match="the stamp cannot be written as JSON"):
        _batch(rows, stamp={"when": math.nan})
    assert not rows.exists()


def test_the_run_of_calls_gets_the_options_the_batch_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _Calls()
    monkeypatch.setattr(batch_module, "run_calls", fake)
    rows = tmp_path / "rows.jsonl"
    batch = _batch(
        rows,
        workers=3,
        pin=False,
        ballast=True,
        ballast_seed=5,
        allow_changed_run=True,
        max_failures=2,
        timeout_seconds=9.5,
        environment={"A": "1"},
        clean_tokens=["secret"],
    )
    batch([_experiment()])
    (invocation,) = fake.invocations
    assert invocation["stamp"] == {
        "revision": "abc",
        "experiment": {
            "solver_factory": SOLVER,
            "problem_factory": PROBLEM,
            "stop": asdict(STOP),
        },
    }
    options = {name: invocation[name] for name in invocation if name not in ("calls", "stamp")}
    assert options == {
        "cpus": CPUS,
        "workers": 3,
        "pin": False,
        "ballast": True,
        "ballast_seed": 5,
        "allow_changed_run": True,
        "max_failures": 2,
        "timeout_seconds": 9.5,
        "python": None,
        "environment": {"A": "1"},
        "cwd": ROOT,
        "clean_tokens": ["secret"],
        "log": None,
    }
    (call,) = invocation["calls"]
    assert call.key == experiment_key(BEST, 3, 11)
    assert call.target == "dispatchatlas.tuning.batch:run_experiment"


# ------------------------------------------------------------------ costs and rows


def test_the_costs_come_back_in_the_order_of_the_experiments_and_a_row_is_asked_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _Calls()
    monkeypatch.setattr(batch_module, "run_calls", fake)
    rows = tmp_path / "rows.jsonl"
    batch = _batch(rows)
    experiments = [_experiment(instance=i, seed=100 + i) for i in (5, 2, 9)]
    costs = batch(experiments)
    assert costs == [noisy_cost(item) for item in experiments]
    assert [call.key[1] for call in fake.invocations[0]["calls"]] == [5, 2, 9]
    # Asked again, in another order, the rows answer and no run of calls is made.
    assert batch(experiments[::-1]) == costs[::-1]
    assert len(fake.invocations) == 1
    assert (batch.experiments, batch.executed, batch.replayed) == (6, 3, 3)
    assert batch([]) == []


def test_two_experiments_with_one_key_run_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _Calls()
    monkeypatch.setattr(batch_module, "run_calls", fake)
    batch = _batch(tmp_path / "rows.jsonl")
    twice = _experiment()
    costs = batch([twice, _experiment(instance=4), twice])
    assert costs[0] == costs[2] == noisy_cost(twice)
    assert len(fake.invocations[0]["calls"]) == 2
    assert (batch.experiments, batch.executed, batch.replayed) == (3, 2, 1)
    assert len(list(read_rows(tmp_path / "rows.jsonl"))) == 2


def test_a_run_without_a_schedule_costs_infinity_and_its_row_holds_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch_module, "run_calls", _Calls())
    rows = tmp_path / "rows.jsonl"
    costs = _batch(rows)([_experiment(m="c"), _experiment()])
    assert costs == [math.inf, noisy_cost(_experiment())]
    assert next(row["cost"] for row in read_rows(rows)) is None
    # A job that begins again reads the same infinity from the row.
    assert _batch(rows)([_experiment(m="c")]) == [math.inf]


def test_a_new_job_makes_the_run_of_calls_check_the_rows_before_it_replays_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = tmp_path / "rows.jsonl"
    monkeypatch.setattr(batch_module, "run_calls", _Calls())
    first = _batch(rows)([_experiment()])
    fake = _Calls()
    monkeypatch.setattr(batch_module, "run_calls", fake)
    batch = _batch(rows)
    assert batch([_experiment()]) == first
    # The one invocation, with nothing to run, is where a changed stamp is refused.
    (invocation,) = fake.invocations
    assert invocation["calls"] == []
    assert batch([_experiment()]) == first
    assert len(fake.invocations) == 1
    assert (batch.experiments, batch.executed, batch.replayed) == (2, 0, 2)


# ------------------------------------------------------------------ two jobs


def _uninterrupted(folder: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TuningResult[int], int]:
    """Tune without a kill, in a rows file of its own; return the result and the runs it made."""
    fake = _Calls()
    monkeypatch.setattr(batch_module, "run_calls", fake)
    result = _tune(_batch(folder / "whole.jsonl"))
    return result, fake.written


@pytest.mark.parametrize("share", [0.0, 0.02, 0.3, 0.55, 0.98])
def test_a_job_killed_after_some_experiments_resumes_to_the_result_of_an_uninterrupted_run(
    share: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    whole, total = _uninterrupted(tmp_path, monkeypatch)
    assert total > 50
    killed_after = int(total * share)
    rows = tmp_path / "rows.jsonl"
    # The first job ends when its wall time does, in the middle of a batch.
    first = _Calls(killed_after)
    monkeypatch.setattr(batch_module, "run_calls", first)
    with pytest.raises(_KilledError):
        _tune(_batch(rows))
    assert first.written == killed_after
    assert (len(list(read_rows(rows))) if rows.exists() else 0) == killed_after
    # The second begins again from the same seed, with the rows of the first.
    second = _Calls()
    monkeypatch.setattr(batch_module, "run_calls", second)
    batch = _batch(rows)
    resumed = _tune(batch)
    assert second.written == batch.executed == total - killed_after
    assert batch.replayed == batch.experiments - batch.executed
    assert resumed.elites == whole.elites
    assert resumed.records == whole.records
    assert resumed == whole
    assert _costs(rows) == _costs(tmp_path / "whole.jsonl")


def test_a_job_killed_again_in_its_turn_still_ends_where_the_uninterrupted_run_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    whole, total = _uninterrupted(tmp_path, monkeypatch)
    rows = tmp_path / "rows.jsonl"
    killed = [total // 5, total // 5, total // 4]
    for budget in killed:
        job = _Calls(budget)
        monkeypatch.setattr(batch_module, "run_calls", job)
        with pytest.raises(_KilledError):
            _tune(_batch(rows))
        assert job.written == budget
    last = _Calls()
    monkeypatch.setattr(batch_module, "run_calls", last)
    assert _tune(_batch(rows)) == whole
    assert last.written == total - sum(killed)


def test_a_resume_under_another_seed_asks_for_other_experiments_and_runs_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The rows are of experiments, not of a run, so another seed adds its own to the file.
    whole, total = _uninterrupted(tmp_path, monkeypatch)
    fake = _Calls()
    monkeypatch.setattr(batch_module, "run_calls", fake)
    other = _tune(_batch(tmp_path / "whole.jsonl"), seed=4)
    assert other.elites != whole.elites
    assert fake.written > 0
    assert len(list(read_rows(tmp_path / "whole.jsonl"))) == total + fake.written


# ------------------------------------------------------------------ failures


def test_an_experiment_without_a_row_stops_the_batch_and_a_new_job_makes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = tmp_path / "rows.jsonl"
    experiments = [_experiment(instance=i, seed=40 + i) for i in range(4)]
    failing = canonical_key(experiment_key(BEST, 2, 42))
    monkeypatch.setattr(
        batch_module, "run_calls", _Calls(fails=lambda arguments: arguments["instance"] == 2)
    )
    batch = _batch(rows, max_failures=0)
    with pytest.raises(ExperimentsFailedError, match=r"1 of 4 experiment\(s\)") as stopped:
        batch(experiments)
    assert stopped.value.missing == (failing,)
    assert stopped.value.errors == tmp_path / "rows.errors.jsonl"
    assert len(list(read_rows(rows))) == 3
    # The job that comes next finds the rows of the others and runs the one that failed.
    monkeypatch.setattr(batch_module, "run_calls", _Calls())
    again = _batch(rows)
    assert again(experiments) == [noisy_cost(item) for item in experiments]
    assert again.executed == 1


def test_the_stop_of_the_run_of_calls_after_its_failures_ends_the_batch_the_same_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = tmp_path / "rows.jsonl"
    experiments = [_experiment(instance=i, seed=40 + i) for i in range(4)]
    fake = _Calls(fails=lambda arguments: arguments["instance"] == 1, max_failures=1)
    monkeypatch.setattr(batch_module, "run_calls", fake)
    with pytest.raises(ExperimentsFailedError, match=r"3 of 4 experiment\(s\)") as stopped:
        _batch(rows)(experiments)
    # The run stopped at the first failure, so the experiments after it were never tried.
    assert isinstance(stopped.value.__cause__, CallsStoppedError)
    assert [json.loads(key)[1] for key in stopped.value.missing] == [1, 2, 3]
    assert [row["instance"] for row in read_rows(rows)] == [0]


# ------------------------------------------------------------------ worker processes


@pytest.fixture(scope="module")
def first_job(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The folder of a rows file that a first job wrote with worker processes: two rows."""
    folder = tmp_path_factory.mktemp("first-job")
    _batch(folder / "rows.jsonl")(DONE)
    return folder


DONE = [_experiment(instance=2, seed=21), _experiment(instance=3, seed=22, x=0.5)]


def _copy(folder: Path, into: Path) -> Path:
    """Copy the folder of a rows file and return the path of the copy of the rows."""
    shutil.copytree(folder, into / "job")
    return into / "job" / "rows.jsonl"


def test_worker_processes_write_the_costs_and_the_facts_of_their_runs(first_job: Path) -> None:
    rows = {row["instance"]: row for row in read_rows(first_job / "rows.jsonl")}
    assert sorted(rows) == [2, 3]
    for experiment in DONE:
        row = rows[experiment.instance]
        assert row["key"] == experiment_key(
            experiment.configuration, experiment.instance, row["seed"]
        )
        assert row["configuration"] == dict(experiment.configuration)
        assert row["cost"] == noisy_cost(experiment)
        assert (row["status"], row["evaluations"]) == ("feasible", 100)
        assert (row["stop_reason"], row["wall_seconds"]) == ("evaluation-limit", 0.25)
        assert (row["first_schedule_digest"], row["repairs"]) == (f"{experiment.seed:016x}", 0)
        assert row["revision"] == "abc"
        assert row["experiment"]["stop"] == asdict(STOP)
        assert row["run"]["workers"] == len(CPUS)
        assert row["run"]["ballast"] is False
        if PIN:
            assert row["run"]["cpu"] in CPUS


def test_a_second_job_replays_the_rows_of_the_first_and_runs_only_the_rest(
    first_job: Path, tmp_path: Path
) -> None:
    rows = _copy(first_job, tmp_path)
    new = _experiment(instance=4, seed=23)
    batch = _batch(rows)
    assert batch([*DONE, new]) == [noisy_cost(item) for item in [*DONE, new]]
    assert (batch.experiments, batch.executed, batch.replayed) == (3, 1, 2)
    assert sorted(row["instance"] for row in read_rows(rows)) == [2, 3, 4]
    first, second = read_rows(rows.with_name("rows.runs.jsonl"))
    assert first["stamp"] == second["stamp"]


@pytest.mark.parametrize("change", ["stamp", "stop", "solver", "problem", "workers", "cpus"])
def test_a_resume_that_differs_from_the_first_job_is_refused(
    change: str, first_job: Path, tmp_path: Path
) -> None:
    if change in ("workers", "cpus") and len(CPUS) < 2:
        pytest.skip("the machine gives this process one processor")
    options: dict[str, Any] = {
        "stamp": {"stamp": {"revision": "def"}},
        "stop": {"stop": StopCriteria(max_evaluations=200)},
        "solver": {"solver": f"{SOLVER}_2"},
        "problem": {"problem_factory": f"{PROBLEM}_2"},
        "workers": {"workers": 1},
        "cpus": {"cpus": CPUS[::-1]},
    }
    rows = _copy(first_job, tmp_path)
    before = rows.read_bytes()
    with pytest.raises(ConfigError, match="begun by a different run"):
        _batch(rows, **options[change])([_experiment(instance=7, seed=77)])
    assert rows.read_bytes() == before


def test_a_job_that_differs_from_the_first_is_refused_even_when_every_row_is_there(
    first_job: Path, tmp_path: Path
) -> None:
    # Nothing remains to run, so the refusal comes from the invocation that has no calls.
    rows = _copy(first_job, tmp_path)
    with pytest.raises(ConfigError, match="stamp revision: 'abc' then, 'def' now"):
        _batch(rows, stamp={"revision": "def"})(DONE)
    assert _batch(rows)(DONE) == [noisy_cost(item) for item in DONE]


def test_a_failed_run_ends_the_batch_and_the_next_job_runs_it(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    experiments = [_experiment(instance=1, seed=51, x=0.2), _experiment(instance=2, seed=52, x=0.9)]
    batch = _batch(rows, cpus=CPUS[:1], workers=1, environment={FAIL_ABOVE: "0.5"})
    with pytest.raises(ExperimentsFailedError) as failed:
        batch(experiments)
    assert failed.value.missing == (
        canonical_key(experiment_key(experiments[1].configuration, 2, 52)),
    )
    (error,) = read_rows(failed.value.errors)
    assert error["type"] == "RuntimeError"
    assert "the target refuses x = 0.9" in error["message"]
    assert [row["instance"] for row in read_rows(rows)] == [1]
    # The job that comes next, without the failure, reads the row of the first run.
    again = _batch(rows, cpus=CPUS[:1], workers=1)
    assert again(experiments) == [noisy_cost(item) for item in experiments]
    assert (again.executed, again.replayed) == (1, 1)


def test_a_torn_last_line_stops_a_resume_until_the_rows_are_repaired(
    first_job: Path, tmp_path: Path
) -> None:
    rows = _copy(first_job, tmp_path)
    rows.write_bytes(rows.read_bytes()[:-40])  # a kill in the middle of the last row
    batch = _batch(rows)
    with pytest.raises(TornRowsError):
        batch(DONE)
    repair_rows(rows)
    assert batch(DONE) == [noisy_cost(item) for item in DONE]
    assert (batch.executed, batch.replayed) == (1, 1)


@pytest.mark.skipif(len(CPUS) < 2, reason="ballast needs a processor with nothing to run")
def test_ballast_keeps_the_idle_processor_busy_while_the_experiment_runs(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    batch = _batch(rows, ballast=True)
    assert batch([_experiment()]) == [noisy_cost(_experiment())]
    assert batch.last_report is not None
    assert batch.last_report.ballast_runs >= 1
    assert len(list(read_rows(rows))) == 1  # the rows of the ballast are discarded


# ------------------------------------------------------------------ two jobs, worker processes


class _Killer:
    """Stands for ``run_calls`` in a job whose wall time ends after ``count`` experiments.

    It hands the first ``count`` calls it is given to the real ``run_calls``, so that worker
    processes run them and write their rows, and then raises, as a job killed in the middle of a
    batch does. A batch that its remaining count covers whole is served and returns its report.
    """

    def __init__(self, count: int) -> None:
        self.left = count

    def __call__(self, calls: Sequence[Call], rows: Path, **options: Any) -> CallReport:
        head = list(calls[: self.left])
        self.left -= len(head)
        report = run_calls(head, rows, **options)
        if len(head) < len(calls):
            raise _KilledError
        return report


def _small_tune(batch: ProcessRunBatch) -> TuningResult[int]:
    """Tune with a small budget, the least that gives two races, in a few batches."""
    racing = IteratedRacing(SPACE, 20, seed=3, first_test=2, iterations=2, min_survivors=2)
    return racing.tune(list(range(4)), batch)


@pytest.mark.slow
def test_a_job_killed_after_a_few_experiments_resumes_in_worker_processes_to_the_same_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The uninterrupted run needs no worker processes: a cost depends on the experiment alone.
    whole_calls = _Calls()
    monkeypatch.setattr(batch_module, "run_calls", whole_calls)
    whole = _small_tune(_batch(tmp_path / "whole.jsonl"))
    total, killed_after = whole_calls.written, 4
    assert whole.runs == total > 2 * killed_after
    assert len(whole_calls.invocations[0]["calls"]) > killed_after  # killed inside a batch
    # The first job runs four experiments in worker processes and is killed in its first batch.
    rows = tmp_path / "rows.jsonl"
    monkeypatch.setattr(batch_module, "run_calls", _Killer(killed_after))
    with pytest.raises(_KilledError):
        _small_tune(_batch(rows))
    assert len(list(read_rows(rows))) == killed_after
    # The second begins again from the same seed with the rows of the first, and runs the rest.
    monkeypatch.setattr(batch_module, "run_calls", run_calls)
    batch = _batch(rows)
    resumed = _small_tune(batch)
    assert (batch.executed, batch.replayed) == (total - killed_after, killed_after)
    assert resumed.elites == whole.elites
    assert resumed.records == whole.records
    assert resumed == whole
    assert _costs(rows) == _costs(tmp_path / "whole.jsonl")
    # Every row, those of both jobs, came from a worker process, and no experiment ran twice.
    assert len(_costs(rows)) == len(list(read_rows(rows))) == total
    assert all(row["run"]["workers"] == len(CPUS) for row in read_rows(rows))
    stamps = [item["stamp"] for item in read_rows(rows.with_name("rows.runs.jsonl"))]
    assert len(stamps) > 1
    assert all(stamp == stamps[0] for stamp in stamps)
