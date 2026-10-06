"""Processors and threads: the allowed set, pinned workers, thread limits, and what is recorded."""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import subprocess
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.campaigns import (
    Campaign,
    Execution,
    FamilySource,
    Protocol,
    SolverSpec,
    campaign_from_dict,
    campaign_to_dict,
    loads_campaign,
    read_runs,
    read_sessions,
    run,
)
from dispatchatlas.campaigns import backends as backends_module
from dispatchatlas.campaigns import config as config_module
from dispatchatlas.campaigns import environment as environment_module
from dispatchatlas.campaigns import processors as processors_module
from dispatchatlas.campaigns import runner as runner_module
from dispatchatlas.campaigns.backends import ProcessBackend, backend_for
from dispatchatlas.campaigns.config import THREAD_VARIABLES
from dispatchatlas.campaigns.environment import capture_environment
from dispatchatlas.campaigns.execution import Completion, Failure
from dispatchatlas.campaigns.fairness import assess
from dispatchatlas.campaigns.planning import plan, prepare
from dispatchatlas.campaigns.processors import (
    affinity_cpus,
    allowed_cpus,
    available_cpus,
    confined,
    describe_cpus,
    pin_process,
    pinning_refusal,
)
from dispatchatlas.campaigns.records import ErrorKind, Provenance, SessionStatus, SolverEntry
from dispatchatlas.campaigns.replaying import replay
from dispatchatlas.campaigns.runner import overridden, tasks_for
from dispatchatlas.core import ConfigError, SerializationError, StopCriteria
from tests.campaigns.solvers import registry, tiny_campaign

REGISTRY = registry()
ONE_PROBLEM = (FamilySource("flow-shop", (1,), {"jobs": 3, "machines": 2}),)
TIMED = (Protocol("clock", StopCriteria(time_limit=5.0, max_evaluations=10)),)
PROBES = (SolverSpec("test-environment", "a"), SolverSpec("test-environment", "b"))
needs_pinning = pytest.mark.skipif(
    pinning_refusal() is not None, reason="the platform cannot pin a process"
)

NARROWED = """
import json, os
from dispatchatlas.campaigns.config import Execution
from dispatchatlas.campaigns.processors import allowed_cpus, available_cpus, pin_process

before = allowed_cpus()
pin_process(os.getpid(), before[-1])
seen = {"before": before, "after": allowed_cpus(), "count": available_cpus()}
seen["auto"] = Execution(workers="auto").worker_count()
print(json.dumps(seen))
"""

LAUNCHER = """
# A launcher whose main module imports NumPy before it runs a campaign.

import json
import os
import sys
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy

from dispatchatlas.campaigns import (
    Campaign,
    Execution,
    FamilySource,
    Protocol,
    SolverSpec,
    read_runs,
    read_sessions,
    run,
)
from dispatchatlas.campaigns.config import THREAD_VARIABLES
from dispatchatlas.core import SolverInfo, StopCriteria
from dispatchatlas.solvers import default_registry
from dispatchatlas.solvers.rules import InputOrder


def limits():
    return ",".join(os.environ.get(name, "") for name in THREAD_VARIABLES)


AT_IMPORT = limits()


@dataclass(frozen=True)
class ImportProbe:
    info: ClassVar[SolverInfo] = SolverInfo(
        id="test-import-probe",
        name="import probe",
        family="random-sampling",
        summary="Reports the thread limits its module saw when it was imported.",
        stochastic=False,
        iterative=False,
    )

    def solve(self, problem, stop, seed=0):
        result = InputOrder().solve(problem, stop, seed)
        return replace(result, extras={"at_import": AT_IMPORT, "numpy": numpy.__version__})


if __name__ == "__main__":
    solvers = default_registry(plugins=False)
    solvers.register(ImportProbe)
    campaign = Campaign(
        id="launched",
        seed=1,
        instances=(FamilySource("flow-shop", (1,), {"jobs": 3, "machines": 2}),),
        solvers=(SolverSpec("test-import-probe", "a"), SolverSpec("test-import-probe", "b")),
        protocols=(Protocol("evals", StopCriteria(max_evaluations=5)),),
        replicates=1,
        execution=Execution(workers=2),
    )
    summary = run(campaign, sys.argv[1], registry=solvers)
    (session,) = read_sessions(summary.directory)
    seen = {
        "launcher": AT_IMPORT,
        "workers": [item.outcome.extras["at_import"] for item in read_runs(summary.directory)],
        "names": sorted(item.provenance.worker for item in read_runs(summary.directory)),
        "recorded": session.environment["threads"],
        "after": limits(),
    }
    print(json.dumps(seen))
"""


def _limits(threads: int) -> str:
    return ",".join([str(threads)] * len(THREAD_VARIABLES))


def _probes(campaign_id: str, **execution: Any) -> Campaign:
    return tiny_campaign(
        campaign_id, solvers=PROBES, instances=ONE_PROBLEM, backend="process", **execution
    )


def _foreign() -> int:
    allowed = allowed_cpus()
    return next(index for index in itertools.count() if index not in allowed)


# ------------------------------------------------------ the allowed processors


def test_processor_indices_are_described_as_runs() -> None:
    assert describe_cpus([3, 0, 1, 2, 8, 10, 11]) == "0-3, 8, 10-11"
    assert describe_cpus((5,)) == "5"
    assert describe_cpus([]) == "none"


def test_the_allowed_processors_are_sorted_indices_of_the_machine() -> None:
    allowed = allowed_cpus()
    assert allowed
    assert allowed == tuple(sorted(set(allowed)))
    assert set(allowed) <= set(range(os.cpu_count() or 1))
    assert available_cpus() == config_module.available_cpus() == len(allowed)
    scheduler = getattr(os, "sched_getaffinity", None)
    if scheduler is not None:
        assert allowed == tuple(sorted(scheduler(0)))


@needs_pinning
def test_a_narrowed_affinity_is_what_the_process_counts() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", NARROWED], capture_output=True, text=True, check=True, timeout=120
    )
    seen = json.loads(completed.stdout)
    assert tuple(seen["before"]) == allowed_cpus()
    assert seen["after"] == [seen["before"][-1]]
    assert (seen["count"], seen["auto"]) == (1, 1)


def test_the_environment_records_the_allowed_processors_and_the_thread_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    environment = capture_environment()
    assert environment["cpu"]["allowed"] == list(allowed_cpus())
    assert environment["cpu"]["available"] == len(allowed_cpus())
    assert set(environment["threads"]) == set(THREAD_VARIABLES)
    assert environment["threads"]["OMP_NUM_THREADS"] == "4"
    assert environment["threads"]["MKL_NUM_THREADS"] is None


# ------------------------------------------------------------- the settings


def test_the_processor_settings_load_from_campaign_files() -> None:
    text = """
schema_version = 1
id = "keys"
seed = 1

[[instances]]
family = "flow-shop"
parameters = { jobs = 3, machines = 2 }
seeds = [1]

[[solvers]]
id = "input-order"

[[protocols]]
name = "evals"
stop = { max_evaluations = 5 }

[execution]
workers = 2
affinity = [3, 1]
threads_per_worker = 2
"""
    campaign = loads_campaign(text)
    assert campaign.execution == Execution(workers=2, affinity=(3, 1), threads_per_worker=2)
    document = campaign_to_dict(campaign)
    assert document["execution"]["affinity"] == [3, 1]
    assert document["execution"]["threads_per_worker"] == 2
    assert campaign_from_dict(document) == campaign
    assert loads_campaign(text.replace("[3, 1]", '"pin"')).execution.affinity == "pin"
    del document["execution"]["affinity"], document["execution"]["threads_per_worker"]
    assert campaign_from_dict(document).execution == Execution(workers=2)
    assert (Execution().affinity, Execution().threads_per_worker) == ("inherit", 1)
    assert Execution().to_dict()["affinity"] == "inherit"
    assert campaign.config_hash == campaign_from_dict(document).config_hash
    with pytest.raises(ConfigError, match="affinity"):
        loads_campaign(text.replace("[3, 1]", "[3, 1.5]"))
    with pytest.raises(ConfigError, match="affinity must be"):
        loads_campaign(text.replace("[3, 1]", '"spread"'))


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"affinity": "spread"}, "affinity must be"),
        ({"affinity": []}, "affinity must be"),
        ({"affinity": [0, 0]}, "affinity must be"),
        ({"affinity": [-1]}, "affinity must be"),
        ({"affinity": [True]}, "affinity must be"),
        ({"affinity": 3}, "affinity must be"),
        ({"threads_per_worker": 0}, "threads_per_worker must be an integer of at least 1"),
        ({"threads_per_worker": True}, "threads_per_worker must be an integer"),
        ({"backend": "sequential", "affinity": "pin"}, "sequential backend"),
        ({"backend": "sequential", "affinity": [0]}, "sequential backend"),
    ],
)
def test_invalid_processor_settings_are_refused(settings: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Execution(**settings)


@needs_pinning
def test_pinning_needs_an_allowed_processor_for_every_worker(tmp_path: Path) -> None:
    allowed, foreign = allowed_cpus(), _foreign()
    assert Execution().worker_cpus() is None
    assert Execution(workers=len(allowed), affinity="pin").worker_cpus() == allowed
    assert Execution(affinity="pin").worker_cpus() == allowed[:1]
    assert Execution(affinity=(allowed[-1],)).worker_cpus() == (allowed[-1],)
    with pytest.raises(ConfigError, match=f"{len(allowed) + 1} workers exceed"):
        Execution(workers=len(allowed) + 1, affinity="pin").worker_cpus()
    with pytest.raises(
        ConfigError, match=f"processor\\(s\\) {foreign}, which this process may not"
    ):
        Execution(workers=2, affinity=(allowed[0], foreign)).worker_cpus()
    with pytest.raises(ConfigError, match="1 processor\\(s\\) for 2 workers"):
        Execution(workers=2, affinity=(allowed[0],)).worker_cpus()
    with pytest.raises(ConfigError, match="the calling process may not use"):
        pin_process(os.getpid(), foreign)
    assert allowed_cpus() == allowed
    crowded = tiny_campaign(backend="process", workers=len(allowed) + 1, affinity="pin")
    with pytest.raises(ConfigError, match="workers exceed"):
        run(crowded, tmp_path, registry=REGISTRY)
    assert read_sessions(tmp_path / "tiny") == ()
    with pytest.raises(ConfigError, match="processor of its own"):
        ProcessBackend(2, cpus=(allowed[0],))
    with pytest.raises(ConfigError, match="does not pin"):
        backend_for("sequential", 1, cpus=(allowed[0],))


def test_pinning_is_refused_where_the_platform_cannot_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    def refusal() -> str:
        return "this platform has no call that confines a process to a processor"

    monkeypatch.setattr(config_module, "pinning_refusal", refusal)
    monkeypatch.setattr(processors_module, "pinning_refusal", refusal)
    with pytest.raises(ConfigError, match="workers cannot be pinned: this platform has no call"):
        Execution(affinity="pin").worker_cpus()
    with pytest.raises(ConfigError, match="workers cannot be pinned"):
        Execution(affinity=(allowed_cpus()[0],)).worker_cpus()
    with pytest.raises(ConfigError, match="cannot pin process"):
        pin_process(os.getpid(), allowed_cpus()[0])
    assert Execution().worker_cpus() is None


# -------------------------------------------------------------- the workers


def test_one_worker_starts_with_the_thread_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    summary = run(_probes("one"), tmp_path, registry=REGISTRY)
    records = read_runs(summary.directory)
    assert len(records) == 2
    assert {item.outcome.extras["thread_variables"] for item in records} == {_limits(1)}
    assert {item.outcome.extras["cpus"] for item in records} == {
        ",".join(str(index) for index in allowed_cpus())
    }
    assert {
        (item.provenance.worker, item.provenance.cpu, item.provenance.concurrent)
        for item in records
    } == {("worker-1", None, 1)}
    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert "MKL_NUM_THREADS" not in os.environ
    (session,) = read_sessions(summary.directory)
    assert session.environment["threads"]["OMP_NUM_THREADS"] == "4"
    assert session.environment["threads"]["MKL_NUM_THREADS"] is None
    assert session.environment["cpu"]["allowed"] == list(allowed_cpus())
    assert (session.execution.affinity, session.execution.threads_per_worker) == ("inherit", 1)


@needs_pinning
def test_two_pinned_workers_run_on_their_processors_with_their_thread_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = allowed_cpus()
    if len(allowed) < 2:
        pytest.skip("needs two allowed processors")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    campaign = _probes("two", workers=2, affinity="pin", threads_per_worker=2)
    summary = run(campaign, tmp_path, registry=REGISTRY)
    records = read_runs(summary.directory)
    assert {item.outcome.extras["thread_variables"] for item in records} == {_limits(2)}
    assert {(item.provenance.worker, item.provenance.cpu) for item in records} == {
        ("worker-1", allowed[0]),
        ("worker-2", allowed[1]),
    }
    assert all(item.outcome.extras["cpus"] == str(item.provenance.cpu) for item in records)
    assert [item.provenance.concurrent for item in records] == [2, 2]
    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert allowed_cpus() == allowed
    (session,) = read_sessions(summary.directory)
    assert (session.workers, session.execution.affinity) == (2, "pin")
    assert session.execution.threads_per_worker == 2


@needs_pinning
def test_a_listed_processor_is_where_the_worker_runs(tmp_path: Path) -> None:
    last = allowed_cpus()[-1]
    campaign = _probes("listed", affinity=[last], threads_per_worker=2)
    summary = run(campaign, tmp_path, registry=REGISTRY)
    records = read_runs(summary.directory)
    assert {(item.outcome.extras["cpus"], item.provenance.cpu) for item in records} == {
        (str(last), last)
    }
    assert {item.outcome.extras["thread_cpus"] for item in records} == {str(last)}
    (session,) = read_sessions(summary.directory)
    assert session.execution.affinity == (last,)


def test_the_limits_precede_a_launching_module_that_imports_numpy(tmp_path: Path) -> None:
    script = tmp_path / "launcher.py"
    script.write_text(LAUNCHER, encoding="utf-8")
    environment = {**os.environ, "OMP_NUM_THREADS": "4"}
    environment.pop("MKL_NUM_THREADS", None)
    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "out")],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
        env=environment,
    )
    seen = json.loads(completed.stdout.strip().splitlines()[-1])
    assert seen["names"] == ["worker-1", "worker-2"]
    assert seen["workers"] == [_limits(1), _limits(1)]
    assert seen["launcher"].split(",")[0] == "4"
    assert seen["after"] == seen["launcher"]
    assert seen["recorded"]["OMP_NUM_THREADS"] == "4"
    assert seen["recorded"]["MKL_NUM_THREADS"] is None


# -------------------------------------------------------------- the records


def test_provenance_keeps_the_processor_and_the_concurrent_runs() -> None:
    provenance = Provenance(
        1, 1, "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", 1.0, cpu=3, concurrent=2
    )
    document = provenance.to_dict()
    assert (document["cpu"], document["concurrent"]) == (3, 2)
    assert Provenance.from_dict(document) == provenance
    del document["cpu"], document["concurrent"]
    earlier = Provenance.from_dict(document)
    assert (earlier.cpu, earlier.concurrent) == (None, None)
    assert Provenance.from_dict({**document, "cpu": None, "concurrent": None}) == earlier
    with pytest.raises(SerializationError, match="cpu"):
        Provenance.from_dict({**document, "cpu": "three"})


def test_a_store_written_without_the_new_fields_still_loads(tmp_path: Path) -> None:
    summary = run(tiny_campaign(), tmp_path, registry=REGISTRY)
    paths = sorted((summary.directory / "runs").rglob("*.json"))
    assert paths
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert (document["provenance"]["cpu"], document["provenance"]["concurrent"]) == (None, 1)
        del document["provenance"]["cpu"], document["provenance"]["concurrent"]
        path.write_text(json.dumps(document), encoding="utf-8")
    (path,) = (summary.directory / "sessions").glob("*.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["execution"]["affinity"], document["execution"]["threads_per_worker"]
    del document["environment"]["threads"], document["environment"]["cpu"]["allowed"]
    path.write_text(json.dumps(document), encoding="utf-8")
    path = summary.directory / "campaign.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["campaign"]["execution"]["affinity"]
    del document["campaign"]["execution"]["threads_per_worker"]
    path.write_text(json.dumps(document), encoding="utf-8")

    records = read_runs(summary.directory)
    assert len(records) == len(paths)
    assert {(item.provenance.cpu, item.provenance.concurrent) for item in records} == {(None, None)}
    (session,) = read_sessions(summary.directory)
    assert session.execution == Execution(backend="sequential")
    again = run(tiny_campaign(), tmp_path, registry=REGISTRY)
    assert (again.attempted, again.pending, again.completed) == (0, 0, len(paths))


# ------------------------------------------------- fairness and the warning


def test_a_time_limit_over_unequal_thread_counts_is_noted_and_stays_fair() -> None:
    solvers = (
        SolverSpec("test-threaded", "one"),
        SolverSpec("test-threaded", "four", {"threads": 4}),
    )
    protocols = (*TIMED, Protocol("evals", StopCriteria(max_evaluations=5)))
    timed = plan(
        tiny_campaign(solvers=solvers, protocols=protocols, instances=ONE_PROBLEM),
        registry=REGISTRY,
    )
    verdict = assess(timed)
    assert verdict.fair
    assert verdict.issues == ()
    (note,) = verdict.notes
    assert "protocol 'clock'" in note
    assert "one: 1, four: 4" in note
    assert timed.to_dict()["fairness"]["notes"] == [note]
    assert timed.to_dict()["fairness"]["fair"] is True

    equal = (
        SolverSpec("test-threaded", "a", {"threads": 2}),
        SolverSpec("test-threaded", "b", {"threads": 2}),
    )
    same = plan(
        tiny_campaign(solvers=equal, protocols=TIMED, instances=ONE_PROBLEM), registry=REGISTRY
    )
    assert assess(same).notes == ()
    untimed = plan(tiny_campaign(solvers=solvers, instances=ONE_PROBLEM), registry=REGISTRY)
    assert assess(untimed).notes == ()
    assert untimed.to_dict()["fairness"]["notes"] == []


def _warned(
    caplog: pytest.LogCaptureFixture,
    campaign: Campaign,
    workers: int,
    cpus: tuple[int, ...] | None = None,
) -> list[str]:
    prepared = prepare(campaign, registry=REGISTRY)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="dispatchatlas"):
        runner_module._warn_about_timing(prepared, campaign.execution, workers, cpus)
    return [item.getMessage() for item in caplog.records]


def _shared(
    caplog: pytest.LogCaptureFixture,
    campaign: Campaign,
    workers: int,
    cpus: tuple[int, ...] | None = None,
) -> list[str]:
    return [message for message in _warned(caplog, campaign, workers, cpus) if " share " in message]


def test_the_timing_warning_counts_threads_against_the_allowed_processors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(runner_module, "available_cpus", lambda: 4)
    solvers = (SolverSpec("input-order"), SolverSpec("test-threaded", parameters={"threads": 4}))
    threaded = tiny_campaign(
        solvers=solvers, protocols=TIMED, instances=ONE_PROBLEM, backend="process", workers=2
    )
    (message,) = _shared(caplog, threaded, 2)
    assert "2 worker(s) with up to 4 thread(s) each share 4 processor(s)" in message
    assert "clock" in message
    single = tiny_campaign(protocols=TIMED, instances=ONE_PROBLEM, backend="process", workers=4)
    assert _shared(caplog, single, 4) == []
    (message,) = _shared(caplog, single, 5)
    assert "5 worker(s) with up to 1 thread(s) each share 4 processor(s)" in message
    limited = tiny_campaign(
        protocols=TIMED, instances=ONE_PROBLEM, backend="process", workers=2, threads_per_worker=3
    )
    (message,) = _shared(caplog, limited, 2)
    assert "2 worker(s) with up to 3 thread(s) each share 4 processor(s)" in message
    untimed = tiny_campaign(instances=ONE_PROBLEM, backend="process", workers=8)
    assert _shared(caplog, untimed, 8) == []


def test_more_workers_than_allowed_processors_warn_under_a_time_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    allowed = available_cpus()
    campaign = tiny_campaign(
        protocols=TIMED, instances=ONE_PROBLEM, backend="process", workers=allowed + 1
    )
    (message,) = _shared(caplog, campaign, allowed + 1)
    assert f"{allowed + 1} worker(s) with up to 1 thread(s) each share {allowed} processor(s)" in (
        message
    )
    assert _shared(caplog, campaign, allowed) == []


# ------------------------------------------------ refusals leave the store alone


@needs_pinning
def test_a_refused_pinned_campaign_discards_nothing_and_creates_nothing(tmp_path: Path) -> None:
    allowed, foreign = allowed_cpus(), _foreign()
    kept = _probes("kept")
    summary = run(kept, tmp_path, registry=REGISTRY)
    files = sorted(path.name for path in summary.directory.rglob("*.json"))
    with pytest.raises(ConfigError, match="which this process may not use"):
        run(kept.with_execution(affinity=(foreign,)), tmp_path, registry=REGISTRY, fresh=True)
    with pytest.raises(ConfigError, match="workers exceed"):
        run(kept, tmp_path, registry=REGISTRY, fresh=True, workers=len(allowed) + 1, affinity="pin")
    assert len(read_runs(summary.directory)) == 2
    assert len(read_sessions(summary.directory)) == 1
    assert sorted(path.name for path in summary.directory.rglob("*.json")) == files
    with pytest.raises(ConfigError, match="which this process may not use"):
        run(_probes("new", affinity=(foreign,)), tmp_path, registry=REGISTRY)
    with pytest.raises(ConfigError, match="workers exceed"):
        run(_probes("new", workers=len(allowed) + 1, affinity="pin"), tmp_path, registry=REGISTRY)
    assert not (tmp_path / "new").exists()


@needs_pinning
def test_a_campaign_pinned_elsewhere_resumes_and_replays_under_an_override(
    tmp_path: Path,
) -> None:
    last, foreign = allowed_cpus()[-1], _foreign()
    done = run(_probes("moved", affinity=[last]), tmp_path, registry=REGISTRY)
    assert (done.attempted, done.completed) == (2, 2)
    elsewhere = _probes("moved", affinity=[foreign])
    again = run(elsewhere, tmp_path, registry=REGISTRY)
    assert (again.session, again.attempted, again.completed) == (2, 0, 2)
    with pytest.raises(ConfigError, match="which this process may not use"):
        replay(elsewhere, tmp_path, registry=REGISTRY)
    assert replay(elsewhere, tmp_path, registry=REGISTRY, runs=()).entries == ()
    assert replay(elsewhere, tmp_path, registry=REGISTRY, backend="sequential").ok
    report = replay(elsewhere, tmp_path, registry=REGISTRY, affinity=[last])
    assert report.ok
    assert len(report.entries) == 2


def test_the_sequential_override_drops_what_only_workers_have(tmp_path: Path) -> None:
    pinned = Execution(workers=2, affinity="pin", threads_per_worker=3)
    assert overridden(pinned, "sequential", None) == Execution(backend="sequential")
    assert overridden(pinned, None, None) is pinned
    assert overridden(pinned, None, None, "inherit").affinity == "inherit"
    assert overridden(Execution(workers=2), None, None, [3, 1]).affinity == (3, 1)
    assert overridden(pinned, None, 1, "pin") == Execution(affinity="pin", threads_per_worker=3)
    with pytest.raises(ConfigError, match="the sequential backend"):
        overridden(pinned, "sequential", None, "pin")
    campaign = _probes("debugged", workers=2, affinity=[4096, 4097], threads_per_worker=2)
    summary = run(campaign, tmp_path, registry=REGISTRY, backend="sequential")
    assert (summary.completed, summary.failed) == (2, 0)
    (session,) = read_sessions(summary.directory)
    assert session.execution == Execution(backend="sequential")
    assert {item.provenance.cpu for item in read_runs(summary.directory)} == {None}
    assert replay(campaign, tmp_path, registry=REGISTRY, backend="sequential").ok


# ------------------------------------------------------- the sequential backend


def test_the_sequential_backend_accepts_only_the_default_thread_limit() -> None:
    with pytest.raises(ConfigError, match="whose thread limits the engine does not change"):
        Execution(backend="sequential", threads_per_worker=2)
    with pytest.raises(ConfigError, match="OMP_NUM_THREADS, MKL_NUM_THREADS"):
        tiny_campaign(threads_per_worker=4)
    assert Execution(backend="sequential").threads_per_worker == 1
    assert Execution(threads_per_worker=2).threads_per_worker == 2


def test_a_timed_sequential_campaign_warns_about_unset_thread_variables(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv("OMP_NUM_THREADS")
    monkeypatch.delenv("MKL_NUM_THREADS")
    timed = tiny_campaign(protocols=TIMED, instances=ONE_PROBLEM)
    (message,) = [item for item in _warned(caplog, timed, 1) if "no thread limit" in item]
    assert "where OMP_NUM_THREADS, MKL_NUM_THREADS set(s) no thread limit" in message
    assert "OPENBLAS_NUM_THREADS" not in message
    assert "clock" in message
    untimed = tiny_campaign(instances=ONE_PROBLEM)
    assert _warned(caplog, untimed, 1) == []
    workers = tiny_campaign(protocols=TIMED, instances=ONE_PROBLEM, backend="process")
    assert not [item for item in _warned(caplog, workers, 1) if "no thread limit" in item]
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    assert not [item for item in _warned(caplog, timed, 1) if "no thread limit" in item]


# ----------------------------------------------------------- pinned workers


def test_a_pinned_worker_with_several_threads_warns_under_a_time_limit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(runner_module, "available_cpus", lambda: 48)
    limited = tiny_campaign(
        protocols=TIMED,
        instances=ONE_PROBLEM,
        backend="process",
        affinity="pin",
        threads_per_worker=2,
    )
    assert _shared(caplog, limited, 1) == []
    (message,) = _shared(caplog, limited, 1, (0,))
    assert "1 pinned worker(s) run up to 2 thread(s) each on one processor apiece" in message
    assert "clock" in message
    solvers = (SolverSpec("input-order"), SolverSpec("test-threaded", parameters={"threads": 8}))
    threaded = tiny_campaign(
        solvers=solvers,
        protocols=TIMED,
        instances=ONE_PROBLEM,
        backend="process",
        workers=4,
        affinity="pin",
    )
    assert _shared(caplog, threaded, 4) == []
    (message,) = _shared(caplog, threaded, 4, (0, 1, 2, 3))
    assert "4 pinned worker(s) run up to 8 thread(s) each" in message
    single = tiny_campaign(
        protocols=TIMED, instances=ONE_PROBLEM, backend="process", workers=2, affinity="pin"
    )
    assert _shared(caplog, single, 2, (0, 1)) == []
    untimed = tiny_campaign(
        instances=ONE_PROBLEM, backend="process", affinity="pin", threads_per_worker=2
    )
    assert _shared(caplog, untimed, 1, (0,)) == []


@needs_pinning
def test_a_process_created_while_confined_starts_on_that_processor() -> None:
    allowed = allowed_cpus()
    assert affinity_cpus() == allowed
    cpu = allowed[-1]
    code = "from dispatchatlas.campaigns.processors import allowed_cpus; print(allowed_cpus())"
    with confined(cpu):
        assert allowed_cpus() == (cpu,)
        child = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
        )
    assert allowed_cpus() == allowed
    assert child.stdout.strip() == str((cpu,))
    with pytest.raises(RuntimeError, match="inside"), confined(allowed[0]):
        raise RuntimeError("inside")
    assert allowed_cpus() == allowed
    with pytest.raises(ConfigError, match="the calling process may not use"), confined(_foreign()):
        pass
    assert allowed_cpus() == allowed


@needs_pinning
def test_a_worker_that_reports_other_processors_stops_the_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = allowed_cpus()
    if len(allowed) < 2:
        pytest.skip("needs two allowed processors")
    monkeypatch.setattr(backends_module, "confined", lambda cpu: contextlib.nullcontext())
    with pytest.raises(
        ConfigError, match=f"worker-1 was created on processor {allowed[0]} but reports"
    ):
        run(_probes("misplaced", affinity="pin"), tmp_path, registry=REGISTRY)
    assert allowed_cpus() == allowed
    assert len(read_runs(tmp_path / "misplaced")) == 0
    (session,) = read_sessions(tmp_path / "misplaced")
    assert session.status is SessionStatus.FAILED


@needs_pinning
def test_failures_of_a_pinned_worker_state_its_processor_and_its_replacement_is_pinned() -> None:
    last = allowed_cpus()[-1]
    solvers = (
        SolverSpec("test-sleeper", parameters={"seconds": 60.0}),
        SolverSpec("test-crasher"),
        SolverSpec("test-environment"),
    )
    campaign = tiny_campaign(
        solvers=solvers, protocols=TIMED, instances=ONE_PROBLEM, backend="process", affinity=[last]
    )
    prepared = prepare(campaign, registry=REGISTRY)
    runs = prepared.plan.runs
    built = tasks_for(prepared, runs, campaign.execution, lambda _: 1)
    tasks = {item.solver: task for item, task in zip(runs, built, strict=True)}
    probe = replace(tasks["test-environment"], hard_limit=None)
    queue = deque(
        [
            replace(tasks["test-sleeper"], hard_limit=1.0),
            probe,
            replace(tasks["test-crasher"], hard_limit=None),
            replace(probe, attempt=2),
        ]
    )
    timeout, first, crash, second = ProcessBackend(1, cpus=(last,)).run(queue)
    assert isinstance(timeout, Failure)
    assert timeout.error.kind is ErrorKind.TIMEOUT
    assert isinstance(crash, Failure)
    assert crash.error.kind is ErrorKind.WORKER_EXIT
    for failure in (timeout, crash):
        measured = failure.measurement
        assert (measured.worker, measured.cpu, measured.concurrent) == ("worker-1", last, 1)
    for completion in (first, second):
        assert isinstance(completion, Completion)
        assert completion.outcome.extras["cpus"] == str(last)
        assert completion.outcome.extras["thread_cpus"] == str(last)
        assert (completion.measurement.cpu, completion.measurement.concurrent) == (last, 1)
    assert allowed_cpus()[-1] == last


# -------------------------------------------------- what cannot be read or seen


def test_an_unread_affinity_is_recorded_as_unknown_and_warned_about(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    timed = tiny_campaign(protocols=TIMED, instances=ONE_PROBLEM, backend="process")
    if pinning_refusal() is None:
        assert affinity_cpus() == allowed_cpus()
        assert capture_environment()["cpu"]["allowed"] == list(allowed_cpus())
        assert not [item for item in _warned(caplog, timed, 1) if "cannot be read" in item]
    for module in (processors_module, environment_module, runner_module):
        monkeypatch.setattr(module, "affinity_cpus", lambda: None)
    machine = os.cpu_count() or 1
    assert allowed_cpus() == tuple(range(machine))
    assert available_cpus() == machine
    cpu = capture_environment()["cpu"]
    assert (cpu["allowed"], cpu["available"], cpu["logical"]) == (None, machine, machine)
    (message,) = [item for item in _warned(caplog, timed, 1) if "cannot be read" in item]
    assert f"so all {machine} of the machine are counted" in message
    assert "clock" in message
    untimed = tiny_campaign(instances=ONE_PROBLEM, backend="process")
    assert _warned(caplog, untimed, 1) == []


def test_the_campaign_engine_imports_without_ctypes() -> None:
    code = (
        "import sys\n"
        "sys.modules['_ctypes'] = None\n"
        "import dispatchatlas.campaigns\n"
        "print('ctypes' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert completed.stdout.strip() == "False"


def test_the_fairness_note_counts_the_threads_a_solver_holds_by_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    mixed = (SolverSpec("input-order", "plain"), SolverSpec("test-four-threads", "four"))
    campaign = tiny_campaign(
        solvers=mixed, protocols=TIMED, instances=ONE_PROBLEM, backend="process", workers=2
    )
    timed = plan(campaign, registry=REGISTRY)
    assert [entry.threads for entry in timed.solvers] == [None, 4]
    (note,) = assess(timed).notes
    assert "plain: 1, four: 4" in note
    assert timed.to_dict()["solvers"][1]["threads"] == 4
    assert timed.to_dict()["fairness"]["notes"] == [note]
    monkeypatch.setattr(runner_module, "available_cpus", lambda: 4)
    (message,) = _shared(caplog, campaign, 2)
    assert "2 worker(s) with up to 4 thread(s) each share 4 processor(s)" in message

    same = (
        SolverSpec("test-four-threads", "default"),
        SolverSpec("test-four-threads", "set", {"threads": 4}),
    )
    equal = plan(
        tiny_campaign(solvers=same, protocols=TIMED, instances=ONE_PROBLEM), registry=REGISTRY
    )
    assert [entry.threads for entry in equal.solvers] == [4, 4]
    assert assess(equal).notes == ()

    entry = timed.solvers[1]
    document = entry.to_dict()
    assert SolverEntry.from_dict(document) == entry
    del document["threads"]
    assert SolverEntry.from_dict(document) == replace(entry, threads=None)
    stored = replace(equal, solvers=tuple(replace(item, threads=None) for item in equal.solvers))
    (note,) = assess(stored).notes
    assert "default: 1, set: 4" in note
