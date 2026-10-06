"""What a resume reports, what the store keeps aside, the session log, and the identity hash."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

from dispatchatlas.benchmarks import generate
from dispatchatlas.campaigns import (
    Campaign,
    Execution,
    FileSource,
    SolverSpec,
    environment,
    read_plan,
    read_runs,
    read_sessions,
    run,
)
from dispatchatlas.campaigns.io import campaign_from_dict, campaign_to_dict
from dispatchatlas.campaigns.records import RunRecord, SessionRecord
from dispatchatlas.campaigns.runner import session_changes
from dispatchatlas.campaigns.store import LOGS, RUNS, SUPERSEDED, Store, Strays, read_strays
from dispatchatlas.core import ConfigError, Problem, SerializationError, SolveResult, SolverInfo
from dispatchatlas.core.logs import SessionLog, get_logger
from dispatchatlas.core.serialization import problem_from_dict, problem_to_dict, save_problem
from dispatchatlas.core.solver import StopCriteria
from dispatchatlas.solvers import SolverRegistry
from dispatchatlas.solvers.rules import InputOrder
from tests.campaigns.solvers import registry, tiny_campaign

LINE = re.compile(
    r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z (INFO|WARNING|ERROR) dispatchatlas\S*: "
)


@dataclass(frozen=True)
class Chatty:
    """Input order, which logs a warning and a detail in every solve."""

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
        logger.debug("a detail of %s", problem.id)
        return InputOrder().solve(problem, stop, seed)


def _registry() -> SolverRegistry:
    solvers = registry()
    solvers.register(Chatty)
    return solvers


REGISTRY = _registry()


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
    earlier = {k: v for k, v in document["execution"].items() if not k.startswith("on_changed")}
    assert campaign_from_dict({**document, "execution": earlier}) == campaign
    with pytest.raises(ConfigError, match="on_changed_code"):
        campaign_from_dict({**document, "execution": {**earlier, "on_changed_code": "never"}})


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
    monkeypatch.setattr(environment, "plugin_states", lambda: installed("1.0"))
    first = run(campaign, tmp_path, registry=REGISTRY, max_runs=4)
    monkeypatch.setattr(environment, "plugin_states", lambda: installed("1.1"))
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
    monkeypatch.setattr(environment, "plugin_states", dict)
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
    assert caplog.text.count("the record belongs to another version of its problem") == 2
    assert caplog.text.count("the earlier file is kept as") == 2
    # Every record of the first version is under superseded/, byte for byte.
    kept = _files(first.directory / SUPERSEDED)
    assert sorted(kept.values()) == sorted(files.values())
    for name, text in kept.items():
        record = RunRecord.from_dict(json.loads(text))
        assert record.verify() == ()
        assert record.run_id in earlier
        assert name.endswith(f".{record.run_id}.json")
        assert name.startswith(f"{record.solver.name}/shop/r000-evals.")
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
            aside = first.directory / SUPERSEDED / solver / "shop"
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


def test_the_reader_reports_what_the_runs_folder_holds_beside_the_plan(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
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
    other = (runs / "notes.txt", record.with_name(f".{record.name}.partial"))
    for path in other:
        path.write_text("left behind")
    strays = read_strays(first.directory)
    assert strays == Strays((unplanned,), (unreadable,), tuple(sorted(other)))
    assert strays
    with caplog.at_level(logging.WARNING):
        assert len(read_runs(first.directory)) == 2
    assert "1 record(s) of runs the plan does not hold, which are not read: " in caplog.text
    assert "runs/retired-solver/shop/r000-evals.json" in caplog.text
    assert "1 file(s) that cannot be read as run records: " in caplog.text
    assert "2 file(s) that are no records: " in caplog.text
    # A session leaves such files where they are.
    assert run(campaign, tmp_path / "out", registry=REGISTRY).attempted == 0
    assert read_strays(first.directory) == strays
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


def test_records_and_sessions_written_without_the_new_fields_still_load(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    campaign = tiny_campaign()
    summary = run(campaign, tmp_path, registry=REGISTRY)
    hashes = {record.run_id: record.result_hash for record in read_runs(summary.directory)}

    def without_policies(execution: dict[str, Any]) -> None:
        for key in ("on_changed_code", "on_changed_problem"):
            execution.pop(key)

    def session_as_before(document: dict[str, Any]) -> None:
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
    _edit(summary.directory / "sessions" / "1.json", session_as_before)
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


def test_a_session_writes_a_log_file_with_time_stamps(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    library = logging.getLogger("dispatchatlas")
    level = library.level
    assert library.getEffectiveLevel() == logging.WARNING
    campaign = tiny_campaign(solvers=(SolverSpec("test-chatty"),), replicates=1)
    summary = run(campaign, tmp_path, registry=REGISTRY)
    lines = (summary.directory / LOGS / "session-1.log").read_text().splitlines()
    assert all(LINE.match(line) for line in lines)
    assert (
        sum("session 1: 2 of 2 runs to attempt, sequential backend" in line for line in lines) == 1
    )
    assert (
        sum(" WARNING dispatchatlas.tests.chatty: solving flow-shop" in line for line in lines) == 2
    )
    assert lines[-1].endswith("session 1 finished: 2 completed, 0 failed, 0 pending")
    assert not any("a detail of" in line for line in lines)
    # The handlers of the application show what they showed before, and the level is restored.
    assert library.level == level
    assert {record.levelno for record in caplog.records} == {logging.WARNING}
    assert Store(summary.directory).log_path(1) == summary.directory / LOGS / "session-1.log"
    run(campaign, tmp_path, registry=REGISTRY)
    assert sorted(_files(summary.directory / LOGS)) == ["session-1.log", "session-2.log"]


def test_the_log_file_holds_what_a_worker_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    campaign = tiny_campaign(
        solvers=(SolverSpec("test-chatty"),), replicates=1, backend="process", workers=1
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


def test_a_session_log_leaves_other_handlers_as_they_were(tmp_path: Path) -> None:
    library = logging.getLogger("dispatchatlas")
    seen: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record)

    collector = Collector()
    library.addHandler(collector)
    try:
        with SessionLog() as log:
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
    assert [record.getMessage() for record in seen] == ["written at once", "after the block"]
    assert collector.filters == []
    with SessionLog():
        get_logger("tests.unnamed").warning("no file is named")
    assert sorted(path.name for path in (tmp_path / "logs").iterdir()) == ["a.log"]
