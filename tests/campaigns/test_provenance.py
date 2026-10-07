"""What a resume reports, what the store keeps aside, the session log, and the identity hash."""

from __future__ import annotations

import json
import logging
import multiprocessing
import re
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from dispatchatlas.benchmarks import generate
from dispatchatlas.campaigns import (
    Campaign,
    Execution,
    FamilySource,
    FileSource,
    Protocol,
    SolverSpec,
    backends,
    environment,
    planning,
    read_failures,
    read_plan,
    read_runs,
    read_sessions,
    run,
    store,
)
from dispatchatlas.campaigns.estimation import estimate
from dispatchatlas.campaigns.execution import Failure
from dispatchatlas.campaigns.io import campaign_from_dict, campaign_to_dict
from dispatchatlas.campaigns.planning import prepare
from dispatchatlas.campaigns.records import ErrorKind, RunRecord, SessionRecord, SessionStatus
from dispatchatlas.campaigns.runner import session_changes, tasks_for
from dispatchatlas.campaigns.store import (
    FAILURES,
    LOGS,
    RUNS,
    SESSIONS,
    SUPERSEDED,
    Store,
    Strays,
    read_strays,
)
from dispatchatlas.core import ConfigError, Problem, SerializationError, SolveResult, SolverInfo
from dispatchatlas.core.io import read_json
from dispatchatlas.core.logs import Lowered, SessionLog, get_logger
from dispatchatlas.core.serialization import problem_from_dict, problem_to_dict, save_problem
from dispatchatlas.core.solver import StopCriteria
from dispatchatlas.solvers import SolverRegistry
from dispatchatlas.solvers.rules import InputOrder
from tests.campaigns.solvers import registry, tiny_campaign

LINE = re.compile(
    r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z (DEBUG|INFO|WARNING|ERROR) dispatchatlas\S*: "
)


@dataclass(frozen=True)
class Chatty:
    """Input order, which logs a warning, a line, and a detail in every solve."""

    info: ClassVar[SolverInfo] = SolverInfo(
        id="test-chatty",
        name="test-chatty",
        family="random-sampling",
        summary="A solver that logs while it solves.",
        stochastic=False,
        iterative=False,
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        logger = get_logger("tests.chatty")
        logger.warning("solving %s with seed %d", problem.id, seed)
        logger.info("working on %s", problem.id)
        logger.debug("a detail of %s", problem.id)
        return InputOrder().solve(problem, stop, seed)


@dataclass(frozen=True)
class Counter:
    """Loops for ``seconds``, whatever its stop criteria say, and writes how often it looped.

    With ``chatty`` every iteration logs a line at the level INFO. The
    count goes to the file ``chatty`` or ``silent`` under ``marker``.
    """

    marker: str = ""
    seconds: float = 0.3
    chatty: bool = True
    info: ClassVar[SolverInfo] = SolverInfo(
        id="test-counter",
        name="test-counter",
        family="random-sampling",
        summary="A solver that counts its iterations and may log each one.",
        stochastic=False,
        iterative=True,
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        logger = get_logger("tests.counter")
        count = 0
        begun = time.perf_counter()
        while time.perf_counter() - begun < self.seconds:
            if self.chatty:
                logger.info("iteration %d", count)
            count += 1
        (Path(self.marker) / ("chatty" if self.chatty else "silent")).write_text(str(count))
        return InputOrder().solve(problem, stop, seed)


def _registry() -> SolverRegistry:
    solvers = registry()
    solvers.register(Chatty)
    solvers.register(Counter)
    return solvers


REGISTRY = _registry()


@pytest.fixture
def quiet_library() -> Iterator[logging.Logger]:
    """The library logger at the level WARNING, whatever other tests left on it."""
    library = logging.getLogger("dispatchatlas")
    level = library.level
    library.setLevel(logging.WARNING)
    yield library
    library.setLevel(level)


def _write_problem(path: Path, seed: int) -> None:
    """Write a version of the problem ``shop``; versions of different seeds differ."""
    document = problem_to_dict(generate("flow-shop", seed, jobs=3, machines=2))
    save_problem(problem_from_dict({**document, "id": "shop"}), path)


def _filed(tmp_path: Path, **execution: Any) -> Campaign:
    """Return a two-run campaign on a problem file, written as its first version."""
    path = tmp_path / "shop.json"
    if not path.exists():
        _write_problem(path, 1)
    return tiny_campaign(
        "filed",
        solvers=(SolverSpec("input-order"), SolverSpec("random-search")),
        instances=(FileSource(path),),
        replicates=1,
        **execution,
    )


def _files(folder: Path) -> dict[str, str]:
    return {
        path.relative_to(folder).as_posix(): path.read_text()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _edit(path: Path, change: Any) -> None:
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _fields(session: SessionRecord) -> dict[str, tuple[Any, Any, str]]:
    return {item["field"]: (item["earlier"], item["now"], item["kind"]) for item in session.changes}


# ------------------------------------------------------------- configuration


def test_the_policies_are_optional_execution_keys() -> None:
    assert (Execution().on_changed_code, Execution().on_changed_problem) == ("warn", "rerun")
    with pytest.raises(ConfigError, match="on_changed_code must be one of"):
        Execution(on_changed_code="ignore")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="on_changed_problem must be one of"):
        Execution(on_changed_problem="warn")  # type: ignore[arg-type]
    campaign = tiny_campaign()
    document = campaign_to_dict(campaign)
    assert document["execution"]["on_changed_code"] == "warn"
    assert document["execution"]["on_changed_problem"] == "rerun"
    strict = {**document["execution"], "on_changed_code": "refuse", "on_changed_problem": "refuse"}
    read = campaign_from_dict({**document, "execution": strict})
    assert (read.execution.on_changed_code, read.execution.on_changed_problem) == (
        "refuse",
        "refuse",
    )
    assert read.config_hash == campaign.config_hash
    # A campaign document written without the keys reads with their defaults.
    optional = ("on_changed_code", "on_changed_problem", "log_level")
    earlier = {k: v for k, v in document["execution"].items() if k not in optional}
    assert campaign_from_dict({**document, "execution": earlier}) == campaign
    with pytest.raises(ConfigError, match="on_changed_code"):
        campaign_from_dict({**document, "execution": {**earlier, "on_changed_code": "never"}})


def test_the_log_level_is_an_optional_execution_key(tmp_path: Path) -> None:
    assert Execution().log_level == "inherit"
    with pytest.raises(ConfigError, match="log_level must be one of"):
        Execution(log_level="warning")  # type: ignore[arg-type]
    campaign = tiny_campaign()
    document = campaign_to_dict(campaign)
    assert document["execution"]["log_level"] == "inherit"
    detailed = campaign_from_dict(
        {**document, "execution": {**document["execution"], "log_level": "debug"}}
    )
    assert detailed.execution.log_level == "debug"
    assert detailed.config_hash == campaign.config_hash
    with pytest.raises(ConfigError, match="log_level"):
        campaign_from_dict({**document, "execution": {**document["execution"], "log_level": 10}})
    # A resume under another level names the change as one of the execution settings.
    first = run(campaign, tmp_path, registry=REGISTRY, max_runs=0)
    run(detailed, tmp_path, registry=REGISTRY, max_runs=0)
    assert _fields(read_sessions(first.directory)[1]) == {
        "execution.log_level": ("inherit", "debug", "execution")
    }


# -------------------------------------------------------- resume comparison


def test_a_resume_on_the_same_settings_and_code_reports_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    campaign = tiny_campaign()
    first = run(campaign, tmp_path, registry=REGISTRY, max_runs=4)
    with caplog.at_level(logging.WARNING):
        run(campaign, tmp_path, registry=REGISTRY)
    one, two = read_sessions(first.directory)
    assert one.changes == two.changes == ()
    assert "differs from session" not in caplog.text
    assert set(one.load) == {"start", "end"}
    assert one.load["start"]["seconds"] == environment.LOAD_SAMPLE_SECONDS
    # A session with nothing to attempt samples nothing.
    run(campaign, tmp_path, registry=REGISTRY)
    assert read_sessions(first.directory)[2].load == {}


def test_a_resume_with_other_workers_names_the_change_and_is_never_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    campaign = tiny_campaign(backend="process", workers=1, on_changed_code="refuse")
    first = run(campaign, tmp_path, registry=REGISTRY, max_runs=0)
    with caplog.at_level(logging.WARNING):
        run(campaign, tmp_path, registry=REGISTRY, max_runs=0, workers=2)
    session = read_sessions(first.directory)[1]
    assert _fields(session) == {"execution.workers": (1, 2, "execution")}
    assert "session 2 differs from session 1: execution.workers 1 -> 2" in caplog.text
    # Settings that only say what to do about a difference are no difference.
    relaxed = replace(campaign, execution=replace(campaign.execution, on_changed_code="warn"))
    run(relaxed, tmp_path, registry=REGISTRY, max_runs=0, workers=2)
    assert read_sessions(first.directory)[2].changes == ()


def test_a_resume_after_a_change_of_a_source_file_names_it_and_can_refuse(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tiny_campaign()
    strict = replace(campaign, execution=replace(campaign.execution, on_changed_code="refuse"))
    first = run(strict, tmp_path, registry=REGISTRY, max_runs=4)
    before = read_sessions(first.directory)[0].environment["library"]["source_digest"]
    monkeypatch.setattr(environment, "source_digest", lambda package=None: "f" * 64)
    kept = _files(first.directory)
    with pytest.raises(ConfigError, match="on_changed_code is 'refuse'") as refusal:
        run(strict, tmp_path, registry=REGISTRY)
    assert "environment.library.source_digest" in str(refusal.value)
    # A refused session writes nothing: no record, no log, no run.
    assert _files(first.directory) == kept
    with caplog.at_level(logging.WARNING):
        resumed = run(campaign, tmp_path, registry=REGISTRY)
    assert resumed.done
    session = read_sessions(first.directory)[1]
    assert _fields(session) == {
        "environment.library.source_digest": (before, "f" * 64, "code"),
    }
    assert f"environment.library.source_digest {before[:12]} -> ffffffffffff" in caplog.text


def test_a_resume_after_a_change_of_a_plugin_version_names_it_and_can_refuse(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def installed(version: str) -> dict[str, dict[str, Any]]:
        state = {"version": version, "origin": "index", "editable": False, "revision": None}
        return {"solver-pack": state}

    campaign = tiny_campaign()
    monkeypatch.setattr(environment, "plugin_states", lambda exclude=(): installed("1.0"))
    first = run(campaign, tmp_path, registry=REGISTRY, max_runs=4)
    monkeypatch.setattr(environment, "plugin_states", lambda exclude=(): installed("1.1"))
    strict = replace(campaign, execution=replace(campaign.execution, on_changed_code="refuse"))
    with pytest.raises(
        ConfigError, match=r"environment\.plugins\.solver-pack\.version 1\.0 -> 1\.1"
    ):
        run(strict, tmp_path, registry=REGISTRY)
    assert len(read_sessions(first.directory)) == 1
    with caplog.at_level(logging.WARNING):
        run(campaign, tmp_path, registry=REGISTRY, max_runs=1)
    assert _fields(read_sessions(first.directory)[1]) == {
        "environment.plugins.solver-pack.version": ("1.0", "1.1", "code"),
    }
    assert "environment.plugins.solver-pack.version 1.0 -> 1.1" in caplog.text
    # A plugin that is no longer installed differs as a whole.
    monkeypatch.setattr(environment, "plugin_states", lambda exclude=(): {})
    run(campaign, tmp_path, registry=REGISTRY, max_runs=1)
    assert _fields(read_sessions(first.directory)[2]) == {
        "environment.plugins.solver-pack": (installed("1.1")["solver-pack"], None, "code"),
    }


def test_differences_of_the_machine_are_told_from_differences_of_code(tmp_path: Path) -> None:
    summary = run(tiny_campaign(), tmp_path, registry=REGISTRY, max_runs=0)
    (previous,) = read_sessions(summary.directory)
    now = json.loads(json.dumps(previous.environment))
    now["cpu"]["model"] = "another processor"
    now["python"]["version"] = "3.99.0"
    now["packages"]["extra"] = "1.0"
    now["threads"]["OMP_NUM_THREADS"] = "8"
    now["git"]["working_directory"] = {"commit": "c" * 40}
    now["not recorded before"] = {"value": 1}
    changes = session_changes(previous, replace(previous.execution, retries=3), 1, now)
    kinds = {item["field"]: item["kind"] for item in changes}
    assert kinds["execution.retries"] == "execution"
    assert kinds["environment.cpu.model"] == "machine"
    assert kinds["environment.threads.OMP_NUM_THREADS"] == "machine"
    assert kinds["environment.python.version"] == "code"
    assert kinds["environment.packages.extra"] == "code"
    assert not any(field.startswith("environment.not recorded before") for field in kinds)
    assert any(field.startswith("environment.git.working_directory") for field in kinds)
    # The resolved worker count is compared when both sessions state the same setting.
    same = session_changes(previous, previous.execution, 5, previous.environment)
    assert [item["field"] for item in same] == ["workers"]


# ------------------------------------------------------ records kept aside


def test_a_record_of_an_earlier_problem_version_is_kept_aside(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    campaign = _filed(tmp_path)
    first = run(campaign, tmp_path / "out", registry=REGISTRY)
    earlier = {record.run_id: record for record in read_runs(first.directory)}
    files = _files(first.directory / RUNS)
    _write_problem(tmp_path / "shop.json", 2)
    with caplog.at_level(logging.WARNING):
        second = run(campaign, tmp_path / "out", registry=REGISTRY)
    assert second.config_hash == first.config_hash
    assert (second.attempted, second.completed) == (2, 2)
    # The session names the files when it starts, and each one when it is moved.
    assert "2 file(s) at the paths of planned runs are not their valid records" in caplog.text
    assert caplog.text.count("the record belongs to another version of its problem") == 4
    assert caplog.text.count("the earlier file is kept as") == 2
    # Every record of the first version is under superseded/, byte for byte.
    kept = _files(first.directory / SUPERSEDED)
    assert sorted(kept.values()) == sorted(files.values())
    for name, text in kept.items():
        record = RunRecord.from_dict(json.loads(text))
        assert record.verify() == ()
        assert record.run_id in earlier
        assert name.endswith(f".{record.run_id}.json")
        assert name.startswith(f"runs/{record.solver.name}/shop/r000-evals.")
    later = read_runs(first.directory)
    assert {record.run_id for record in later}.isdisjoint(earlier)
    assert {record.problem.digest for record in later} != {
        record.problem.digest for record in earlier.values()
    }
    assert not read_strays(first.directory)


def test_a_changed_problem_can_refuse_the_resume(tmp_path: Path) -> None:
    campaign = _filed(tmp_path, on_changed_problem="refuse")
    first = run(campaign, tmp_path / "out", registry=REGISTRY)
    kept = _files(first.directory)
    _write_problem(tmp_path / "shop.json", 2)
    with pytest.raises(ConfigError, match="on_changed_problem is 'refuse'") as refusal:
        run(campaign, tmp_path / "out", registry=REGISTRY)
    assert "2 record(s) made on another version of problem(s) shop" in str(refusal.value)
    assert _files(first.directory) == kept
    assert not (first.directory / SUPERSEDED).exists()
    # The first version of the problem resumes, and the other policy moves the records.
    _write_problem(tmp_path / "shop.json", 1)
    assert run(campaign, tmp_path / "out", registry=REGISTRY).attempted == 0
    _write_problem(tmp_path / "shop.json", 2)
    assert run(_filed(tmp_path), tmp_path / "out", registry=REGISTRY).attempted == 2
    assert len(_files(first.directory / SUPERSEDED)) == 2


def test_an_edited_and_an_unreadable_record_are_kept_aside(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    campaign = _filed(tmp_path, on_changed_problem="refuse")
    first = run(campaign, tmp_path / "out", registry=REGISTRY)
    edited, broken = sorted((first.directory / RUNS).rglob("*.json"))
    planned = {run.solver: run.run_id for run in read_plan(first.directory).runs}

    def spoil(document: dict[str, Any]) -> None:
        document["outcome"]["value"] -= 1.0

    for round_number in (1, 2):
        _edit(edited, spoil)
        broken.write_text("{ not a record")
        texts = (edited.read_text(), broken.read_text())
        with caplog.at_level(logging.WARNING):
            resumed = run(campaign, tmp_path / "out", registry=REGISTRY)
        assert (resumed.attempted, resumed.completed) == (2, 2)
        suffix = "" if round_number == 1 else f".{round_number}"
        for path, text in zip((edited, broken), texts, strict=True):
            solver = path.parent.parent.name
            aside = first.directory / SUPERSEDED / RUNS / solver / "shop"
            assert (aside / f"r000-evals.{planned[solver]}{suffix}.json").read_text() == text
    assert "the outcome does not match the result hash" in caplog.text
    assert len(_files(first.directory / SUPERSEDED)) == 4
    assert all(record.verify() == () for record in read_runs(first.directory))


def test_fresh_discards_the_files_kept_aside_and_the_logs(tmp_path: Path) -> None:
    campaign = _filed(tmp_path)
    first = run(campaign, tmp_path / "out", registry=REGISTRY)
    _write_problem(tmp_path / "shop.json", 2)
    run(campaign, tmp_path / "out", registry=REGISTRY)
    assert (first.directory / SUPERSEDED).is_dir()
    assert sorted(_files(first.directory / LOGS)) == ["session-1.log", "session-2.log"]
    again = run(campaign, tmp_path / "out", registry=REGISTRY, fresh=True)
    assert again.session == 1
    assert not (first.directory / SUPERSEDED).exists()
    assert sorted(_files(first.directory / LOGS)) == ["session-1.log"]


def test_a_session_that_repeats_no_run_moves_nothing_and_a_restored_problem_needs_no_rerun(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    campaign = _filed(tmp_path)
    first = run(campaign, tmp_path / "out", registry=REGISTRY)
    files = _files(first.directory / RUNS)
    _write_problem(tmp_path / "shop.json", 2)
    with caplog.at_level(logging.WARNING):
        nothing = run(campaign, tmp_path / "out", registry=REGISTRY, max_runs=0)
    assert (nothing.attempted, nothing.completed, nothing.pending) == (0, 0, 2)
    assert "2 file(s) at the paths of planned runs are not their valid records" in caplog.text
    assert _files(first.directory / RUNS) == files
    assert not (first.directory / SUPERSEDED).exists()
    # The records count again as soon as the problem is what it was.
    _write_problem(tmp_path / "shop.json", 1)
    back = run(campaign, tmp_path / "out", registry=REGISTRY, max_runs=0)
    assert (back.attempted, back.completed, back.pending) == (0, 2, 0)
    assert _files(first.directory / RUNS) == files
    assert not (first.directory / SUPERSEDED).exists()


def test_a_record_kept_aside_is_brought_back_when_its_problem_is_restored(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    campaign = _filed(tmp_path)
    first = run(campaign, tmp_path / "out", registry=REGISTRY)
    files = _files(first.directory / RUNS)
    # One run is repeated on another version of the problem, the other is not.
    _write_problem(tmp_path / "shop.json", 2)
    partial = run(campaign, tmp_path / "out", registry=REGISTRY, max_runs=1)
    assert (partial.attempted, partial.completed, partial.pending) == (1, 1, 1)
    (aside,) = _files(first.directory / SUPERSEDED)
    assert aside.startswith("runs/input-order/shop/r000-evals.run-")
    other = _files(first.directory / RUNS)
    assert (
        other["random-search/shop/r000-evals.json"] == files["random-search/shop/r000-evals.json"]
    )
    assert other["input-order/shop/r000-evals.json"] != files["input-order/shop/r000-evals.json"]

    def standing() -> tuple[int, int]:
        found = estimate(campaign, root=tmp_path / "out", registry=REGISTRY)
        return found.completed, found.remaining

    assert standing() == (1, 1)
    # The first version is restored: its record comes back and no run is repeated.
    _write_problem(tmp_path / "shop.json", 1)
    assert standing() == (2, 0)
    assert len(_files(first.directory / SUPERSEDED)) == 1  # an estimate moves nothing
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        restored = run(campaign, tmp_path / "out", registry=REGISTRY)
    assert (restored.attempted, restored.completed, restored.pending) == (0, 2, 0)
    assert caplog.text.count("is the valid record of the run and is back at") == 1
    assert "are not their valid records" not in caplog.text
    assert _files(first.directory / RUNS) == files
    # The record of the other version took its place under superseded/.
    (kept,) = _files(first.directory / SUPERSEDED).items()
    assert kept[0] != aside
    assert kept[1] == other["input-order/shop/r000-evals.json"]
    assert all(record.verify() == () for record in read_runs(first.directory))
    # A record that was edited while it was kept aside is not brought back.
    (spoiled,) = (first.directory / SUPERSEDED).rglob("*.json")
    _edit(spoiled, lambda document: document["outcome"].update(value=-1.0))
    _write_problem(tmp_path / "shop.json", 2)
    again = run(campaign, tmp_path / "out", registry=REGISTRY)
    assert (again.attempted, again.completed) == (2, 2)
    assert len(_files(first.directory / SUPERSEDED)) == 3


def test_a_failure_record_of_an_earlier_problem_version_is_kept_aside(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_problem(tmp_path / "shop.json", 1)
    campaign = tiny_campaign(
        "failing",
        solvers=(SolverSpec("test-refuser"),),
        instances=(FileSource(tmp_path / "shop.json"),),
        replicates=1,
    )
    first = run(campaign, tmp_path / "out", registry=REGISTRY)
    assert (first.attempted, first.failed) == (1, 1)
    (path,) = (first.directory / FAILURES).rglob("*.json")
    text = path.read_text()
    (earlier,) = read_failures(first.directory)
    _write_problem(tmp_path / "shop.json", 2)
    # A session that attempts nothing leaves the file where it is.
    assert run(campaign, tmp_path / "out", registry=REGISTRY, max_runs=0).pending == 1
    assert path.read_text() == text
    assert not (first.directory / SUPERSEDED).exists()
    with caplog.at_level(logging.WARNING):
        second = run(campaign, tmp_path / "out", registry=REGISTRY)
    assert (second.attempted, second.failed) == (1, 1)
    (later,) = read_failures(first.directory)
    assert later.run_id != earlier.run_id
    assert later.provenance.attempt == earlier.provenance.attempt == 1
    kept = _files(first.directory / SUPERSEDED)
    name = f"failures/test-refuser/shop/r000-evals.attempt-1.{earlier.run_id}.json"
    assert kept == {name: text}
    assert "the earlier file is kept as" in caplog.text
    # The other policy refuses a failure record of another version as it does a run record.
    _write_problem(tmp_path / "shop.json", 1)
    strict = replace(campaign, execution=replace(campaign.execution, on_changed_problem="refuse"))
    folders = (FAILURES, SUPERSEDED, SESSIONS)
    before = [_files(first.directory / folder) for folder in folders]
    with pytest.raises(ConfigError, match=r"1 record\(s\) made on another version of problem"):
        run(strict, tmp_path / "out", registry=REGISTRY)
    assert [_files(first.directory / folder) for folder in folders] == before


def test_a_stale_record_moves_aside_when_its_repeated_run_fails(tmp_path: Path) -> None:
    _write_problem(tmp_path / "shop.json", 1)
    marker = tmp_path / "marker"
    marker.mkdir()
    campaign = tiny_campaign(
        "flaky",
        solvers=(SolverSpec("test-flaky", None, {"marker": str(marker), "failures": 1}),),
        instances=(FileSource(tmp_path / "shop.json"),),
        replicates=1,
        retries=1,
    )
    first = run(campaign, tmp_path / "out", registry=REGISTRY)
    assert (first.attempted, first.completed) == (2, 1)  # one failed attempt, then the record
    (record,) = (first.directory / RUNS).rglob("*.json")
    (failure,) = (first.directory / FAILURES).rglob("*.json")
    texts = (record.read_text(), failure.read_text())
    run_id = json.loads(texts[0])["run_id"]
    # The problem changes, and the only attempt of the repeated run fails.
    _write_problem(tmp_path / "shop.json", 2)
    for count in marker.iterdir():
        count.unlink()
    once = replace(campaign, execution=replace(campaign.execution, retries=0))
    second = run(once, tmp_path / "out", registry=REGISTRY)
    assert (second.attempted, second.completed, second.failed) == (1, 0, 1)
    # Both files of the first version are aside, so no reader meets them at the
    # paths of the repeated run.
    assert not record.exists()
    assert _files(first.directory / SUPERSEDED) == {
        f"failures/test-flaky/shop/r000-evals.attempt-1.{run_id}.json": texts[1],
        f"runs/test-flaky/shop/r000-evals.{run_id}.json": texts[0],
    }
    assert read_runs(first.directory) == ()
    assert [item.run_id for item in read_failures(first.directory)] != [run_id]
    # The first version is restored: its record comes back and needs no run.
    _write_problem(tmp_path / "shop.json", 1)
    third = run(once, tmp_path / "out", registry=REGISTRY)
    assert (third.attempted, third.completed) == (0, 1)
    assert record.read_text() == texts[0]


def test_a_stale_file_that_cannot_be_moved_ends_the_session_as_failed_and_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _filed(tmp_path)
    first = run(campaign, tmp_path / "out", registry=REGISTRY)
    _write_problem(tmp_path / "shop.json", 2)
    held = first.directory / RUNS / "random-search" / "shop" / "r000-evals.json"
    text = held.read_text()
    move = Path.replace

    def refusing(self: Path, target: Path) -> Path:
        # As Windows answers while another program holds the file open.
        if self == held:
            raise PermissionError(13, "The process cannot access the file", str(self))
        return move(self, target)

    monkeypatch.setattr(Path, "replace", refusing)
    with pytest.raises(ConfigError, match="it cannot be moved under") as refusal:
        run(campaign, tmp_path / "out", registry=REGISTRY)
    assert str(held) in str(refusal.value)
    # The session is on record as failed, and the file is where it was.
    last = read_sessions(first.directory)[-1]
    assert (last.session, last.status) == (2, SessionStatus.FAILED)
    assert last.finished is not None
    assert last.counts["succeeded"] == 1
    assert held.read_text() == text
    assert len(_files(first.directory / SUPERSEDED)) == 1
    monkeypatch.setattr(Path, "replace", move)
    resumed = run(campaign, tmp_path / "out", registry=REGISTRY)
    assert (resumed.attempted, resumed.completed) == (1, 2)
    assert text in _files(first.directory / SUPERSEDED).values()


def test_the_reader_reports_what_the_runs_folder_holds_beside_the_plan(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _filed(tmp_path)
    first = run(campaign, tmp_path / "out", registry=REGISTRY)
    runs = first.directory / RUNS
    with caplog.at_level(logging.WARNING):
        assert len(read_runs(first.directory)) == 2
    assert caplog.text == ""
    assert read_strays(first.directory) == Strays()
    record = sorted(runs.rglob("*.json"))[0]
    unplanned = runs / "retired-solver" / "shop" / "r000-evals.json"
    unplanned.parent.mkdir(parents=True)
    unplanned.write_text(record.read_text())
    unreadable = runs / "retired-solver" / "shop" / "r001-evals.json"
    unreadable.write_text("{}")
    # A document that is not UTF-8 text, as a shell redirect can write one.
    exported = runs / "export.json"
    exported.write_bytes('{"a": "é"}'.encode("utf-16"))
    other = (runs / "notes.txt", record.with_name(f".{record.name}.partial"))
    for path in other:
        path.write_text("left behind")
    strays = read_strays(first.directory)
    assert strays == Strays((unplanned,), (exported, unreadable), tuple(sorted(other)))
    assert strays
    with caplog.at_level(logging.WARNING):
        assert len(read_runs(first.directory)) == 2
    assert "1 record(s) of runs the plan does not hold, which are not read: " in caplog.text
    assert "runs/retired-solver/shop/r000-evals.json" in caplog.text
    assert "2 file(s) that cannot be read as run records: runs/export.json, " in caplog.text
    assert "2 file(s) that are no records: " in caplog.text
    # A session leaves such files where they are.
    assert run(campaign, tmp_path / "out", registry=REGISTRY).attempted == 0
    assert read_strays(first.directory) == strays

    # A file the operating system does not let read is reported, not raised.
    def refusing(path: Path) -> Any:
        if path == unplanned:
            raise PermissionError(13, "Permission denied", str(path))
        return read_json(path)

    monkeypatch.setattr(store, "read_json", refusing)
    assert read_strays(first.directory) == Strays(
        (), (exported, unplanned, unreadable), tuple(sorted(other))
    )
    monkeypatch.setattr(store, "read_json", read_json)
    # A record at a planned path that cannot be read is an error of the reader.
    record.write_text("{}")
    with pytest.raises(SerializationError):
        read_runs(first.directory)


# ------------------------------------------------------------ identity hash


def test_an_edited_seed_is_refused_on_read_and_the_run_is_repeated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    summary = run(tiny_campaign(), tmp_path, registry=REGISTRY)
    records = read_runs(summary.directory)
    for record in records:
        assert record.identity_hash == record.identity_digest()
        assert record.verify() == ()
    path = sorted((summary.directory / RUNS).rglob("*.json"))[0]
    genuine = RunRecord.from_dict(json.loads(path.read_text()))
    _edit(path, lambda document: document.update(seed=document["seed"] + 1))
    with pytest.raises(SerializationError, match="the identity does not match the identity hash"):
        read_runs(summary.directory)
    assert len(read_runs(summary.directory, verify=False)) == len(records)
    with caplog.at_level(logging.WARNING):
        resumed = run(tiny_campaign(), tmp_path, registry=REGISTRY)
    assert (resumed.attempted, resumed.completed) == (1, len(records))
    assert "the identity does not match the identity hash" in caplog.text
    (aside,) = (summary.directory / SUPERSEDED).rglob("*.json")
    assert json.loads(aside.read_text())["seed"] == genuine.seed + 1
    assert RunRecord.from_dict(json.loads(path.read_text())).seed == genuine.seed


def test_the_identity_hash_covers_what_names_a_run(tmp_path: Path) -> None:
    summary = run(tiny_campaign(), tmp_path, registry=REGISTRY, max_runs=1)
    (record,) = read_runs(summary.directory)
    document = record.to_dict()

    def verified(**changes: Any) -> tuple[str, ...]:
        return RunRecord.from_dict({**json.loads(json.dumps(document)), **changes}).verify()

    issue = ("the identity does not match the identity hash",)
    assert verified() == ()
    assert verified(replicate=record.replicate + 1) == issue
    assert verified(problem={**document["problem"], "digest": "0" * 64}) == issue
    assert verified(solver={**document["solver"], "parameters": {"restart_after": 9}}) == issue
    assert verified(solver={**document["solver"], "id": "hill-climbing"}) == issue
    assert verified(protocol={"name": "evals", "stop": {"max_evaluations": 16}}) == issue
    # What the solver declares and what the attempt measured are no part of the identity.
    assert verified(solver={**document["solver"], "stochastic": True}) == ()
    assert verified(notes=["a note"]) == ()
    # A record that states no identity hash is not checked for one.
    assert verified(identity_hash=None, seed=record.seed + 1) == ()


def test_a_record_is_compared_with_what_the_plan_states_for_its_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    summary = run(tiny_campaign(), tmp_path, registry=REGISTRY)
    count = len(read_runs(summary.directory))
    path = sorted((summary.directory / RUNS).rglob("*.json"))[0]
    pristine = path.read_text()
    edits: dict[str, Any] = {
        "seed": lambda document: document.update(seed=document["seed"] + 1),
        "replicate": lambda document: document.update(replicate=document["replicate"] + 5),
        "problem": lambda document: document["problem"].update(id="another"),
        "problem digest": lambda document: document["problem"].update(digest="0" * 64),
        "solver": lambda document: document["solver"].update(name="another"),
        "solver id or parameters": lambda document: document["solver"]["parameters"].update(x=1),
        "protocol": lambda document: document["protocol"]["stop"].update(max_evaluations=16),
    }
    # The comparison needs no identity hash: a record written without one, and one
    # whose hash was removed with the edit, are checked as well.
    for field, change in edits.items():
        for stripped in (None, "absent"):

            def edit(document: dict[str, Any], change: Any = change, how: Any = stripped) -> None:
                change(document)
                if how is None:
                    document["identity_hash"] = None
                else:
                    del document["identity_hash"]

            _edit(path, edit)
            with pytest.raises(
                SerializationError, match=f"the record states another {field} than the plan"
            ):
                read_runs(summary.directory)
            assert len(read_runs(summary.directory, verify=False)) == count
            path.write_text(pristine)
    assert len(read_runs(summary.directory)) == count
    # A resume repeats the run of such a record and keeps the file aside.
    _edit(path, lambda document: document.update(seed=document["seed"] + 1, identity_hash=None))
    with caplog.at_level(logging.WARNING):
        resumed = run(tiny_campaign(), tmp_path, registry=REGISTRY)
    assert (resumed.attempted, resumed.completed) == (1, count)
    assert "the record states another seed than the plan" in caplog.text
    (aside,) = (summary.directory / SUPERSEDED).rglob("*.json")
    assert json.loads(aside.read_text())["identity_hash"] is None
    assert json.loads(path.read_text())["seed"] == json.loads(pristine)["seed"]


def test_a_record_made_with_a_seed_of_an_earlier_plan_still_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tiny_campaign()
    first = run(campaign, tmp_path, registry=REGISTRY)
    records = read_runs(first.directory)
    # The library that plans derives other seeds: the run ids stay, the seeds change.
    derive = planning.run_seed
    monkeypatch.setattr(
        planning, "run_seed", lambda *arguments, **options: derive(*arguments, **options) + 1
    )
    second = run(campaign, tmp_path, registry=REGISTRY)
    assert (second.attempted, second.completed) == (0, len(records))
    session = read_sessions(first.directory)[-1]
    assert len(session.plan_reseeded) == len(records)
    planned = {item.run_id: item.seed for item in read_plan(first.directory).runs}
    assert all(planned[record.run_id] == record.seed + 1 for record in records)
    # The session record names the earlier seeds, so the reader and a later session
    # accept the records that state them.
    assert read_runs(first.directory) == records
    assert run(campaign, tmp_path, registry=REGISTRY).attempted == 0
    assert not (first.directory / SUPERSEDED).exists()
    # A seed that no plan of the directory gave the run is refused.
    path = sorted((first.directory / RUNS).rglob("*.json"))[0]
    pristine = path.read_text()
    _edit(path, lambda document: document.update(seed=document["seed"] + 7, identity_hash=None))
    with pytest.raises(SerializationError, match="the record states another seed than the plan"):
        read_runs(first.directory)
    # Without a session record that can be read, the earlier seeds are not known,
    # and the seed is not compared.
    (first.directory / SESSIONS / "2.json").write_text("{ not a session record")
    assert len(read_runs(first.directory)) == len(records)
    path.write_text(pristine)
    assert read_runs(first.directory) == records


def test_records_and_sessions_that_lack_the_optional_fields_load_and_resume(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    campaign = tiny_campaign()
    summary = run(campaign, tmp_path, registry=REGISTRY)
    hashes = {record.run_id: record.result_hash for record in read_runs(summary.directory)}

    def without_policies(execution: dict[str, Any]) -> None:
        for key in ("on_changed_code", "on_changed_problem", "log_level"):
            execution.pop(key)

    def stripped_session(document: dict[str, Any]) -> None:
        for key in ("changes", "load"):
            document.pop(key)
        without_policies(document["execution"])
        captured = document["environment"]
        for key in ("library", "plugins"):
            captured.pop(key)
        for key in ("sockets", "cores"):
            captured["cpu"].pop(key)
        for state in captured["git"].values():
            for key in ("untracked", "digest"):
                state.pop(key)

    for path in (summary.directory / RUNS).rglob("*.json"):
        _edit(path, lambda document: document.pop("identity_hash"))
    _edit(summary.directory / "sessions" / "1.json", stripped_session)
    campaign_file = summary.directory / "campaign.json"
    _edit(campaign_file, lambda document: without_policies(document["campaign"]["execution"]))

    records = read_runs(summary.directory)
    assert {record.run_id: record.result_hash for record in records} == hashes
    assert all(record.identity_hash is None for record in records)
    (session,) = read_sessions(summary.directory)
    assert (session.changes, dict(session.load)) == ((), {})
    assert (session.execution.on_changed_code, session.execution.on_changed_problem) == (
        "warn",
        "rerun",
    )
    with caplog.at_level(logging.WARNING):
        resumed = run(campaign, tmp_path, registry=REGISTRY)
    assert (resumed.attempted, resumed.completed, resumed.session) == (0, len(hashes), 2)
    assert read_sessions(summary.directory)[1].changes == ()
    assert caplog.text == ""
    assert not (summary.directory / SUPERSEDED).exists()


# -------------------------------------------------------------- session log


@pytest.mark.parametrize("level", ["inherit", "info", "debug"])
def test_a_session_writes_a_log_file_with_time_stamps(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, quiet_library: logging.Logger, level: str
) -> None:
    library, earlier = quiet_library, quiet_library.level
    campaign = tiny_campaign(solvers=(SolverSpec("test-chatty"),), replicates=1, log_level=level)
    summary = run(campaign, tmp_path, registry=REGISTRY)
    lines = (summary.directory / LOGS / "session-1.log").read_text().splitlines()
    assert all(LINE.match(line) for line in lines)
    # The engine's own lines are in the file from INFO under every setting.
    assert (
        sum("session 1: 2 of 2 runs to attempt, sequential backend" in line for line in lines) == 1
    )
    assert (
        sum(" WARNING dispatchatlas.tests.chatty: solving flow-shop" in line for line in lines) == 2
    )
    assert lines[-1].endswith("session 1 finished: 2 completed, 0 failed, 0 pending")
    # A solver's records are in it from the level its logger lets through, which the
    # campaign can lower.
    assert sum(" INFO dispatchatlas.tests.chatty: working on" in line for line in lines) == (
        0 if level == "inherit" else 2
    )
    assert sum(" DEBUG dispatchatlas.tests.chatty: a detail" in line for line in lines) == (
        2 if level == "debug" else 0
    )
    # The handlers of the application show what they showed before, and the level is restored.
    assert library.level == earlier
    assert logging.getLogger("dispatchatlas.campaigns").level == logging.NOTSET
    assert {record.levelno for record in caplog.records} == {logging.WARNING}
    assert Store(summary.directory).log_path(1) == summary.directory / LOGS / "session-1.log"
    run(campaign, tmp_path, registry=REGISTRY)
    assert sorted(_files(summary.directory / LOGS)) == ["session-1.log", "session-2.log"]


@pytest.mark.parametrize("level", ["inherit", "info"])
def test_the_log_file_holds_what_a_worker_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, level: str
) -> None:
    campaign = tiny_campaign(
        solvers=(SolverSpec("test-chatty"),),
        replicates=1,
        backend="process",
        workers=1,
        log_level=level,
    )
    with caplog.at_level(logging.INFO, logger="dispatchatlas"):
        summary = run(campaign, tmp_path, registry=REGISTRY)
    assert summary.completed == 2
    text = (summary.directory / LOGS / "session-1.log").read_text()
    found = re.findall(
        r"^\S+Z WARNING dispatchatlas\.tests\.chatty: \[worker-1\] solving (\S+) with seed \d+$",
        text,
        flags=re.MULTILINE,
    )
    assert found == ["flow-shop-3x2-1", "flow-shop-3x2-2"]
    # A worker starts with the default level: its records below WARNING are made, and
    # sent, only when the campaign lowers the level.
    detailed = re.findall(
        r"^\S+Z INFO dispatchatlas\.tests\.chatty: \[worker-1\] working on (\S+)$",
        text,
        flags=re.MULTILINE,
    )
    assert detailed == ([] if level == "inherit" else ["flow-shop-3x2-1", "flow-shop-3x2-2"])
    assert "a detail of" not in text
    # The record of a worker goes to the file, and to no handler of this process.
    assert not any(record.name.endswith("tests.chatty") for record in caplog.records)
    # The workers started on the source the session started on.
    assert all(record.notes == () for record in read_runs(summary.directory))


def test_a_worker_that_starts_on_other_source_is_reported_and_its_runs_noted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The session states another digest than the one its workers compute.
    monkeypatch.setattr(environment, "source_digest", lambda package=None: "f" * 64)
    campaign = tiny_campaign(
        solvers=(SolverSpec("input-order"),), replicates=1, backend="process", workers=1
    )
    with caplog.at_level(logging.WARNING):
        summary = run(campaign, tmp_path, registry=REGISTRY)
    assert caplog.text.count("worker-1 started on other library source") == 1
    records = read_runs(summary.directory)
    assert len(records) == 2
    for record in records:
        (note,) = record.notes
        assert note.startswith("the worker started on other library source (digest ")
        assert note.endswith("than the session (digest ffffffffffff)")


class _Collector(logging.Handler):
    """Keeps the records it is given."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.seen.append(record)

    def messages(self) -> list[str]:
        return [record.getMessage() for record in self.seen]


def test_a_session_log_leaves_other_handlers_as_they_were(
    tmp_path: Path, quiet_library: logging.Logger
) -> None:
    library = quiet_library
    collector = _Collector()
    library.addHandler(collector)
    try:
        with SessionLog(lowered=("dispatchatlas",)) as log:
            get_logger("tests.early").info("kept until the file is named")
            log.open(tmp_path / "logs" / "a.log")
            get_logger("tests.late").warning("written at once")
            log.write(1.5, logging.WARNING, "dispatchatlas.elsewhere", "from another process")
            log.write(2.5, logging.DEBUG, "dispatchatlas.elsewhere", "too detailed")
        get_logger("tests.after").warning("after the block")
    finally:
        library.removeHandler(collector)
    lines = (tmp_path / "logs" / "a.log").read_text().splitlines()
    assert [line.split(" ", 1)[1] for line in lines] == [
        "INFO dispatchatlas.tests.early: kept until the file is named",
        "WARNING dispatchatlas.tests.late: written at once",
        "WARNING dispatchatlas.elsewhere: from another process",
    ]
    assert lines[2].startswith("1970-01-01T00:00:01.500Z ")
    assert collector.messages() == ["written at once", "after the block"]
    assert collector.filters == []
    with SessionLog():
        get_logger("tests.unnamed").warning("no file is named")
    assert sorted(path.name for path in (tmp_path / "logs").iterdir()) == ["a.log"]
    # Without a lowered logger the file receives what the loggers let through anyway.
    with SessionLog() as log:
        log.open(tmp_path / "logs" / "b.log")
        get_logger("tests.plain").info("not made")
        get_logger("tests.plain").warning("made")
        assert library.level == logging.WARNING
    assert [
        line.split(" ", 2)[2] for line in (tmp_path / "logs" / "b.log").read_text().splitlines()
    ] == ["dispatchatlas.tests.plain: made"]


def test_lowered_loggers_show_no_more_and_the_loggers_of_the_application_no_less(
    quiet_library: logging.Logger,
) -> None:
    # The usual configuration of an application: a handler on the root logger, the root
    # at WARNING, and its own logger at INFO.
    root, library = logging.getLogger(), quiet_library
    application = logging.getLogger("tests_provenance_application")
    engine = logging.getLogger("dispatchatlas.campaigns")
    verbose = get_logger("tests.verbose")
    shown, below, received = _Collector(), _Collector(), _Collector()
    levels = (root.level, application.level, engine.level, verbose.level)
    root.setLevel(logging.WARNING)
    application.setLevel(logging.INFO)
    engine.setLevel(logging.ERROR)  # a level of its own, above the library's
    verbose.setLevel(logging.DEBUG)  # a logger the application asked details of
    root.addHandler(shown)
    verbose.addHandler(below)
    try:
        lowered = Lowered(("dispatchatlas", "dispatchatlas.campaigns"), logging.INFO)
        with lowered:
            assert (library.level, engine.level) == (logging.INFO, logging.INFO)
            library.addHandler(received)  # as the session's file: added after, so not filtered
            application.info("of the application")
            get_logger("tests.solver").info("of a solver")
            get_logger("tests.solver").warning("a warning of a solver")
            engine.info("of the engine")
            engine.warning("a warning of the engine")
            engine.error("an error of the engine")
            verbose.debug("a detail that was asked for")
            lowered.apply()  # applying twice changes nothing
        library.removeHandler(received)
        assert (library.level, engine.level) == (logging.WARNING, logging.ERROR)
        assert shown.filters == below.filters == []
        application.info("after the block")
        engine.warning("below the level of the engine again")
    finally:
        root.removeHandler(shown)
        verbose.removeHandler(below)
        library.removeHandler(received)
        for logger, level in zip((root, application, engine, verbose), levels, strict=True):
            logger.setLevel(level)
    # The handlers of the application show what they would show without the block.
    assert shown.messages() == [
        "of the application",
        "a warning of a solver",
        "an error of the engine",
        "a detail that was asked for",
        "after the block",
    ]
    assert below.messages() == ["a detail that was asked for"]
    # The lowered loggers made every record from INFO.
    assert received.messages() == [
        "of a solver",
        "a warning of a solver",
        "of the engine",
        "a warning of the engine",
        "an error of the engine",
        "a detail that was asked for",
    ]
    # Loggers that let the level through already are left alone.
    library.setLevel(logging.DEBUG)
    with Lowered(("dispatchatlas",), logging.INFO):
        assert library.level == logging.DEBUG
    assert library.level == logging.DEBUG


def test_a_log_file_is_buffered_and_limited(tmp_path: Path, quiet_library: logging.Logger) -> None:
    path = tmp_path / "burst.log"
    with SessionLog() as log:
        log.open(path)
        for number in range(3):
            log.write(float(number), logging.INFO, "dispatchatlas.elsewhere", f"line {number}")
        # A burst below WARNING waits in the buffer behind its first record.
        assert path.read_text().count("\n") == 1
        log.write(3.0, logging.WARNING, "dispatchatlas.elsewhere", "a warning")
        assert path.read_text().count("\n") == 4
        log.write(4.0, logging.INFO, "dispatchatlas.elsewhere", "the last line")
    assert path.read_text().splitlines()[-1].endswith("INFO dispatchatlas.elsewhere: the last line")
    # Past the limit records are counted, and the last line states how many were left out.
    limited = tmp_path / "limited.log"
    with SessionLog(limit=300) as log:
        log.open(limited)
        for number in range(50):
            log.write(float(number), logging.WARNING, "dispatchatlas.elsewhere", f"line {number}")
    lines = limited.read_text().splitlines()
    written = len(lines) - 1
    assert 1 <= written < 50
    assert sum(len(line) + 1 for line in lines[:-1]) < 300 + len(lines[-2]) + 1
    assert lines[-1].endswith(
        f"WARNING dispatchatlas.core.logs: {50 - written} later record(s) were left out: "
        "the file reached its limit of 300 characters"
    )
    assert LINE.match(lines[-1])


class _Pipe:
    """The sending end of a worker's pipe, which holds every message until it is opened."""

    def __init__(self) -> None:
        self.open = threading.Event()
        self.sent: list[tuple[str, Any]] = []

    def send(self, message: tuple[str, Any]) -> None:
        self.open.wait()
        self.sent.append(message)


def test_a_worker_that_logs_never_waits_for_the_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backends, "_LOG_QUEUE", 20)
    monkeypatch.setattr(backends, "_LOG_BATCH", 8)
    pipe = _Pipe()
    forwarder = backends._Forwarder(pipe, logging.INFO)  # type: ignore[arg-type]
    logger = logging.getLogger("tests_provenance_forwarder")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(forwarder)

    def log() -> None:
        for number in range(200):
            logger.info("record %d", number)
        logger.debug("below the level of the forwarder")

    try:
        # The supervisor reads nothing: the thread that logs goes on all the same.
        logging_thread = threading.Thread(target=log, daemon=True)
        logging_thread.start()
        logging_thread.join(20.0)
        assert not logging_thread.is_alive(), "the thread that logs waits for the supervisor"
        assert pipe.sent == []
        pipe.open.set()
        forwarder.drain()
    finally:
        pipe.open.set()
        logger.removeHandler(forwarder)
    assert {kind for kind, _ in pipe.sent} == {"log"}
    records = [record for _, (batch, _) in pipe.sent for record in batch]
    dropped = sum(count for _, (_, count) in pipe.sent)
    assert all(len(batch) <= 8 for _, (batch, _) in pipe.sent)
    # Every record was either handed over, in order, or counted as dropped.
    assert len(records) + dropped == 200
    assert 20 <= len(records) <= 29  # the queue, and the batch its thread held meanwhile
    numbers = [int(message.split()[1]) for _, _, _, message in records]
    assert numbers == sorted(numbers)
    assert all(
        (level, name) == (logging.INFO, "tests_provenance_forwarder")
        for _, level, name, _ in records
    )


def test_the_supervisor_reads_the_log_records_of_a_worker_in_bounded_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[str, float, int, str, str]] = []
    context = multiprocessing.get_context("spawn")
    ours, theirs = context.Pipe(duplex=True)
    slot = backends._Slot(
        1, context, {}, None, 60.0, log=lambda *record: received.append(record), log_level=None
    )
    connection: Any = ours  # the class of a pipe's end differs between platforms
    slot._connection = connection
    slot._process = SimpleNamespace(exitcode=None)  # type: ignore[assignment]
    try:
        record = (1.5, logging.WARNING, "dispatchatlas.tests", "a warning")
        for _ in range(5):
            theirs.send(("log", ([record, record], 0)))
        theirs.send(("log", ([], 3)))
        # With no time for log records, one reading handles one message and returns.
        monkeypatch.setattr(backends, "LOG_READ_SECONDS", 0.0)
        assert slot.collect() is None
        assert len(received) == 2
        assert slot.readable()
        # With time, it handles everything the pipe holds.
        monkeypatch.setattr(backends, "LOG_READ_SECONDS", 60.0)
        assert slot.collect() is None
        assert not slot.readable()
    finally:
        ours.close()
        theirs.close()
    assert received[:10] == [("worker-1", *record)] * 10
    (worker, _, level, name, message) = received[10]
    assert (worker, level, name) == (
        "worker-1",
        logging.WARNING,
        "dispatchatlas.campaigns.backends",
    )
    assert message.startswith("3 log record(s) of the worker were dropped")
    assert len(received) == 11


def test_a_solver_that_logs_below_the_level_loses_little_of_its_time(
    tmp_path: Path, quiet_library: logging.Logger
) -> None:
    # Under the default setting a record below WARNING is not made, in the calling
    # process and in a worker, so logging in the loop costs a fraction of the loop.
    for backend in ("sequential", "process"):
        marker = tmp_path / backend
        marker.mkdir()
        campaign = tiny_campaign(
            backend,
            solvers=(
                SolverSpec("test-counter", "silent", {"marker": str(marker), "chatty": False}),
                SolverSpec("test-counter", "chatty", {"marker": str(marker), "chatty": True}),
            ),
            instances=(FamilySource("flow-shop", (1,), {"jobs": 3, "machines": 2}),),
            replicates=1,
            backend=backend,
        )
        summary = run(campaign, tmp_path / "out", registry=REGISTRY)
        assert summary.completed == 2
        silent = int((marker / "silent").read_text())
        chatty = int((marker / "chatty").read_text())
        assert chatty > silent / 10, (backend, silent, chatty)
        text = (summary.directory / LOGS / "session-1.log").read_text()
        assert "tests.counter" not in text


def test_a_solver_that_floods_the_log_is_killed_at_its_hard_limit(tmp_path: Path) -> None:
    # The solver ignores its time limit for eight seconds and logs every iteration,
    # which the campaign asks to record. Its hard limit is 0.5 + 0.5 = 1 second.
    marker = tmp_path / "marker"
    marker.mkdir()
    campaign = tiny_campaign(
        "flood",
        solvers=(SolverSpec("test-counter", None, {"marker": str(marker), "seconds": 8.0}),),
        protocols=(Protocol("timed", StopCriteria(time_limit=0.5)),),
        instances=(FamilySource("flow-shop", (1,), {"jobs": 3, "machines": 2}),),
        replicates=1,
        backend="process",
        workers=1,
        timeout_factor=1.0,
        grace=0.5,
        log_level="info",
    )
    summary = run(campaign, tmp_path / "out", registry=REGISTRY)
    (failure,) = read_failures(summary.directory)
    assert failure.error.kind is ErrorKind.TIMEOUT
    assert failure.provenance.hard_limit == 1.0
    # The kill comes within a second of the limit while the campaign's own sink records
    # the flood. Whether records that wait in the pipe postpone a kill is the matter of
    # the next test, whose sink is slow enough to keep some waiting.
    assert 1.0 <= failure.provenance.wall_seconds < 2.0
    text = (summary.directory / LOGS / "session-1.log").read_text()
    assert " INFO dispatchatlas.tests.counter: [worker-1] iteration 0" in text


def test_log_records_that_wait_in_the_pipe_do_not_postpone_a_kill(tmp_path: Path) -> None:
    # The same solver, and a sink so slow that the worker's pipe holds log records
    # whenever the supervisor looks at it, also when the hard limit of 1 second passes.
    campaign = tiny_campaign(
        "flood",
        solvers=(SolverSpec("test-counter", None, {"marker": str(tmp_path), "seconds": 8.0}),),
        protocols=(Protocol("timed", StopCriteria(time_limit=0.5)),),
        instances=(FamilySource("flow-shop", (1,), {"jobs": 3, "machines": 2}),),
        replicates=1,
        backend="process",
        workers=1,
        timeout_factor=1.0,
        grace=0.5,
    )
    prepared = prepare(campaign, registry=REGISTRY)
    (task,) = tasks_for(prepared, prepared.plan.runs, campaign.execution, lambda _: 1)
    assert task.hard_limit == 1.0
    received: list[str] = []
    taken: list[tuple[float, float]] = []  # when the sink took each record, and when it returned

    def slow(worker: str, created: float, level: int, name: str, message: str) -> None:
        # Two milliseconds of computing: a sleep that short lasts as long as the timer
        # of the system makes it.
        began = time.perf_counter()
        while time.perf_counter() - began < 0.002:
            pass
        received.append(message)
        taken.append((began, time.perf_counter()))

    backend = backends.ProcessBackend(1, log=slow, log_level=logging.INFO)
    (result,) = backend.run(deque([task]))
    assert isinstance(result, Failure)
    assert result.error.kind is ErrorKind.TIMEOUT
    assert received[0] == "iteration 0"
    # The supervisor ends the message it is handing to the sink, one batch of records and
    # at most one warning about dropped ones, and then looks at the limit. So the kill is
    # late by what the sink took for one message, measured here on this machine, and not
    # by what the pipe held.
    size = backends._LOG_BATCH + 1
    message = max(
        taken[min(first + size, len(taken)) - 1][1] - taken[first][0]
        for first in range(max(1, len(taken) - size + 1))
    )
    late = result.measurement.wall_seconds - task.hard_limit
    assert 0.0 <= late < message + backends.LOG_READ_SECONDS + 2.0
    # A kill that waited for the pipe to be empty would have come after the eight
    # seconds of the solver, which then had written its count.
    assert not (tmp_path / "chatty").exists()


SCRIPT = '''
"""An application that configures logging the usual way, in its main module."""

import logging
import sys
from dataclasses import dataclass
from typing import ClassVar

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
logging.getLogger("myapp").setLevel(logging.INFO)

from dispatchatlas.campaigns import Campaign, Execution, FamilySource, Protocol, SolverSpec, run
from dispatchatlas.core import SolverInfo, StopCriteria
from dispatchatlas.core.logs import get_logger
from dispatchatlas.solvers import default_registry
from dispatchatlas.solvers.rules import InputOrder


@dataclass(frozen=True)
class Talker:
    info: ClassVar[SolverInfo] = SolverInfo(
        id="talker",
        name="talker",
        family="random-sampling",
        summary="A solver that logs while it solves.",
        stochastic=False,
        iterative=False,
    )

    def solve(self, problem, stop, seed=0):
        logging.getLogger("myapp").info("APP-INFO %s", problem.id)
        get_logger("talker").info("LIB-INFO %s", problem.id)
        get_logger("talker").warning("LIB-WARNING %s", problem.id)
        return InputOrder().solve(problem, stop, seed)


if __name__ == "__main__":
    backend, level, root = sys.argv[1:4]
    solvers = default_registry(plugins=False)
    solvers.register(Talker)
    campaign = Campaign(
        id="talk",
        seed=1,
        instances=(FamilySource("flow-shop", (1,), {"jobs": 3, "machines": 2}),),
        solvers=(SolverSpec("talker"),),
        protocols=(Protocol("evals", StopCriteria(max_evaluations=5)),),
        replicates=1,
        execution=Execution(backend=backend, workers=1, log_level=level),
    )
    logging.getLogger("myapp").info("APP-INFO before")
    run(campaign, root, registry=solvers)
    logging.getLogger("myapp").info("APP-INFO after")
'''


@pytest.mark.parametrize(
    ("backend", "level"), [("sequential", "inherit"), ("process", "inherit"), ("process", "info")]
)
def test_standard_error_is_the_same_with_the_session_log(
    tmp_path: Path, backend: str, level: str
) -> None:
    script = tmp_path / "application.py"
    script.write_text(SCRIPT, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script), backend, level, str(tmp_path / "out")],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    shown = [line for line in completed.stderr.splitlines() if "APP-INFO" in line or "LIB-" in line]
    # What the application's handler shows without a session log: the records of its
    # own logger from INFO, in this process and in a worker, whose main module is the
    # application's as well, and the library's from WARNING.
    assert shown == [
        "INFO myapp APP-INFO before",
        "INFO myapp APP-INFO flow-shop-3x2-1",
        "WARNING dispatchatlas.talker LIB-WARNING flow-shop-3x2-1",
        "INFO myapp APP-INFO after",
    ]
    text = (tmp_path / "out" / "talk" / LOGS / "session-1.log").read_text()
    assert ("LIB-INFO flow-shop-3x2-1" in text) == (level == "info")
    assert "LIB-WARNING flow-shop-3x2-1" in text
    assert "APP-INFO" not in text
