"""Running caller-defined functions, each once in a pinned child process of its own.

:func:`run_calls` runs a list of :class:`Call` objects. Each call is one
function, named ``"package.module:function"``, called with keyword
arguments in a fresh child process started with the running interpreter
(:mod:`dispatchatlas.processes.child`). The mapping it returns becomes one
line of a rows file (:class:`~dispatchatlas.processes.rows.RowWriter`),
written by the runner alone.

- **Processors.** The run owns a set of processors; worker slot ``i`` owns
  processor ``cpus[i]``. The runner pins itself to the whole set, creates each
  child while confined to the slot's processor, so the child (and a launcher
  executable in front of the interpreter) starts there, and the child then
  pins itself and reads its mask back: a mismatch is a failure, not a row.
- **Groups.** Each slot has a kill-on-close process group
  (:class:`~dispatchatlas.processes.groups.ProcessGroup`) that the child also
  joins by name; whatever a call leaves running ends when the call ends, and
  every child ends when the runner dies.
- **Resume.** The rows file is the record of what is done: calls whose key
  (read back through ``key_of``) has a row are skipped. A torn last line is
  refused before anything starts.
- **Failures.** A call that raises, returns something that is not a row,
  dies, or passes its time limit is recorded in the errors sidecar
  (``<rows stem>.errors.jsonl``), never in the rows file, and is retried on
  the next invocation unless ``skip_failed``. After ``max_failures`` failures
  the run stops at once and raises :class:`CallsStoppedError`.
- **Load.** Every row records the number of slots, how many were busy when
  its call started, and the least that were busy during its life; ``ballast``
  keeps idle slots busy with discarded repeat calls until the last real call
  ends.

Next to the rows file the runner writes ``<stem>.errors.jsonl``,
``<stem>.ballast.jsonl`` (with ``ballast``), ``<stem>.children.log`` (the
children's standard error, tracebacks included), and ``<stem>.report.json``.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from dispatchatlas.core.errors import ConfigError, DispatchAtlasError
from dispatchatlas.core.io import atomic_write_json
from dispatchatlas.core.logs import get_logger
from dispatchatlas.processes.affinity import (
    confined,
    cpus_of_mask,
    current_cpus,
    describe_cpus,
    parse_cpus,
    pin_self,
    pinning_refusal_text,
)
from dispatchatlas.processes.child import SCHEMA, THREAD_VARIABLES, clean_message
from dispatchatlas.processes.groups import ProcessGroup, group_name
from dispatchatlas.processes.rows import Key, RowWriter, canonical_key, done_keys, read_rows

__all__ = [
    "ERROR_SCHEMA",
    "Call",
    "CallError",
    "CallFacts",
    "CallReport",
    "CallsStoppedError",
    "interleaved",
    "run_calls",
]

ERROR_SCHEMA = "dispatchatlas.call-error/1"
"""The schema of an error record of the errors sidecar."""

_LOG = get_logger(__name__)
_JOIN_SECONDS = 30.0
_STOP_SECONDS = 60.0


@dataclass(frozen=True)
class Call:
    """One function to call once in a child process.

    Attributes:
        key: What identifies the call: a string, an integer, or a tuple or
            list of keys. The row the call returns must give it back through
            the runner's ``key_of``.
        target: ``"package.module:function"``, importable in the child.
        arguments: Keyword arguments of the function; JSON-able.
        timeout_seconds: Wall-clock limit of the child, ``None`` for the
            runner's default.
        label: Free text for logs.
    """

    key: Key
    target: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None
    label: str | None = None


@dataclass(frozen=True)
class CallFacts:
    """What the runner and the child recorded about one call.

    Attributes:
        key: The call's key.
        attempt: 1 for a first attempt, 1 + earlier error records otherwise.
        cpu: The slot's processor, ``None`` when the run does not pin.
        affinity_inherited: The child's mask before it pinned itself.
        affinity_mask: The child's mask read back after pinning.
        pid: The id of the process that ran the function.
        group: How the child joined its group (``"job"``, ``"job
            (already)"``, ``"session"``).
        threads: The thread-limit variables the child saw.
        peak_rss_bytes: The child's peak resident set, where readable.
        started_utc: When the child began, in UTC.
        ended_utc: When the function returned, in UTC.
        wall_seconds: Seconds from the child's start to the function's end.
        workers: Worker slots of the run.
        workers_live_start: Busy slots once the runner had filled every
            slot it could when this call started.
        workers_live_min: The least busy slots during the call's life.
        ballast: Whether this was a ballast call.
    """

    key: Any
    attempt: int
    cpu: int | None
    affinity_inherited: int | None
    affinity_mask: int | None
    pid: int | None
    group: str | None
    threads: Mapping[str, str | None]
    peak_rss_bytes: int | None
    started_utc: str | None
    ended_utc: str | None
    wall_seconds: float | None
    workers: int
    workers_live_start: int
    workers_live_min: int
    ballast: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the facts as a JSON-able dictionary."""
        document = asdict(self)
        document["threads"] = dict(self.threads)
        return document


@dataclass(frozen=True)
class CallError:
    """One failed attempt of a call, a line of the errors sidecar.

    Attributes:
        key: The call's key.
        attempt: The attempt number.
        type: The error type: the class the function raised, a step of the
            child (``PinningRefused``, ``GroupRefused``, ``TargetNotFound``,
            ``RowNotMapping``, ``RowNotSerializable``, ``BadDocument``), or
            the runner's (``ProcessDied``, ``Timeout``, ``KeyMismatch``,
            ``RowRefused``, ``StartFailed``).
        message: The message, without file paths or caller-named tokens.
        returncode: The child's exit code; negative for a POSIX signal.
        last_line: Whether the child wrote its reply line.
        facts: The call's facts, as far as known.
        cpu: The slot's processor.
        started_utc: When the runner started the child.
        ended_utc: When the runner saw it end.
        schema: :data:`ERROR_SCHEMA`.
    """

    key: Any
    attempt: int
    type: str
    message: str
    returncode: int | None
    last_line: bool
    facts: Mapping[str, Any] | None
    cpu: int | None
    started_utc: str
    ended_utc: str
    schema: str = ERROR_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-able dictionary."""
        document = asdict(self)
        document["facts"] = None if self.facts is None else dict(self.facts)
        return document

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> CallError:
        """Read a record written by :meth:`to_dict`."""
        facts = document.get("facts")
        returncode = document.get("returncode")
        cpu = document.get("cpu")
        return cls(
            key=document.get("key"),
            attempt=int(document.get("attempt", 1)),
            type=str(document.get("type", "")),
            message=str(document.get("message", "")),
            returncode=None if returncode is None else int(returncode),
            last_line=bool(document.get("last_line")),
            facts=dict(facts) if isinstance(facts, Mapping) else None,
            cpu=None if cpu is None else int(cpu),
            started_utc=str(document.get("started_utc", "")),
            ended_utc=str(document.get("ended_utc", "")),
            schema=str(document.get("schema", ERROR_SCHEMA)),
        )


@dataclass(frozen=True)
class CallReport:
    """What one invocation of :func:`run_calls` did.

    Attributes:
        rows_written: Rows appended by this invocation.
        done_before: Calls skipped because their key already had a row.
        skipped_failed: Calls skipped because of earlier error records.
        failures: Failed attempts of real calls in this invocation.
        ballast_runs: Ballast calls started.
        stopped: ``None`` when every call was attempted, else
            ``"max-failures"`` or ``"interrupted"``.
        workers: Worker slots.
        cpus: The slots' processors, ``None`` entries when unpinned.
        elapsed_seconds: Wall-clock seconds of the invocation.
        rows: The rows file.
        errors: The errors sidecar.
        ballast: The ballast sidecar, ``None`` without ballast.
    """

    rows_written: int
    done_before: int
    skipped_failed: int
    failures: int
    ballast_runs: int
    stopped: str | None
    workers: int
    cpus: tuple[int | None, ...]
    elapsed_seconds: float
    rows: Path
    errors: Path
    ballast: Path | None

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-able dictionary."""
        document = asdict(self)
        document.update(
            cpus=list(self.cpus),
            rows=str(self.rows),
            errors=str(self.errors),
            ballast=None if self.ballast is None else str(self.ballast),
        )
        return document


class CallsStoppedError(DispatchAtlasError):
    """A run of calls stopped before every call was attempted.

    Attributes:
        report: The report of the run, also written next to the rows file.
    """

    def __init__(self, report: CallReport) -> None:
        super().__init__(
            f"the run stopped ({report.stopped}) after {report.failures} failure(s) and "
            f"{report.rows_written} row(s); the errors are in {report.errors}"
        )
        self.report = report


def interleaved(
    calls: Sequence[Call], *, seed: int, classes: Callable[[Call], int] | None = None
) -> list[Call]:
    """Return the calls in a seeded random order, optionally class by class.

    Args:
        calls: The calls; not changed.
        seed: Seed of the shuffle.
        classes: Gives each call a class; calls are ordered by class (lower
            first) and shuffled within each class.
    """
    generator = random.Random(seed)
    if classes is None:
        shuffled = list(calls)
        generator.shuffle(shuffled)
        return shuffled
    ordered: list[Call] = []
    groups: dict[int, list[Call]] = {}
    for call in calls:
        groups.setdefault(classes(call), []).append(call)
    for cls in sorted(groups):
        members = list(groups[cls])
        generator.shuffle(members)
        ordered.extend(members)
    return ordered


# ------------------------------------------------------------------ runner


@dataclass
class _Running:
    """One child in a slot."""

    slot: int
    call: Call
    canonical: str
    attempt: int
    ballast: bool
    cpu: int | None
    process: subprocess.Popen[bytes] | None
    started_utc: str
    deadline: float | None
    live_start: int = 0
    live_min: int = 0
    timed_out: bool = False
    stopped: bool = False
    last_line: bytes = b""
    returncode: int | None = None
    start_error: str | None = None


class _Sidecar:
    """A rows writer opened at its first record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._writer: RowWriter | None = None

    def write(self, record: Mapping[str, Any]) -> None:
        if self._writer is None:
            self._writer = RowWriter(self.path)
        self._writer.write(record)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


def run_calls(
    calls: Sequence[Call],
    rows: str | Path,
    *,
    key_of: Callable[[Mapping[str, Any]], Any],
    cpus: Sequence[int] | int | str,
    workers: int | None = None,
    pin: bool = True,
    stamp: Mapping[str, Any] | None = None,
    finish: Callable[[dict[str, Any], CallFacts], Mapping[str, Any]] | None = None,
    max_failures: int = 3,
    skip_failed: bool = False,
    ballast: bool | Sequence[Call] = False,
    python: str | Path | None = None,
    entry: Sequence[str] = ("-m", "dispatchatlas.processes.child"),
    environment: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    clean_tokens: Sequence[str] = (),
    timeout_seconds: float | None = None,
    progress_every: int = 100,
    log: logging.Logger | None = None,
) -> CallReport:
    """Run every call not yet done, each in a child process, and append its row.

    Args:
        calls: The calls, in the order to start them (see
            :func:`interleaved`). Keys must be distinct.
        rows: The rows file; appended to, and read first for a resume.
        key_of: Gives a row's key; it must equal the call's key for the row
            the call returns (after ``finish`` and ``stamp``).
        cpus: The run's processors: indices, a mask, or text for
            :func:`~dispatchatlas.processes.affinity.parse_cpus`. Each must
            be one the calling process may use. With ``pin=False`` only
            their number matters.
        workers: Worker slots, at most ``len(cpus)``; slot ``i`` owns
            ``cpus[i]``. Defaults to ``len(cpus)``.
        pin: Pin the runner to the set and each child to its slot's
            processor. Refused where the platform cannot pin.
        stamp: Constant fields added to every row, such as provenance; a
            field the row already has is a ``RowRefused`` failure.
        finish: Turns the function's row and the call's facts into the row
            written; by default the facts go under ``row["run"]``.
        max_failures: Failed attempts of real calls after which the run
            stops at once; 0 for no limit.
        skip_failed: Skip calls with an earlier error record instead of
            retrying them.
        ballast: ``True`` to keep idle slots busy with repeats of ``calls``
            in a seeded order, or the calls to repeat; their rows are
            discarded and their facts go to the ballast sidecar.
        python: The interpreter of the children; ``sys.executable`` by
            default.
        entry: Arguments after the interpreter that run
            :func:`~dispatchatlas.processes.child.child_main`.
        environment: Variables added to the children's environment, after
            the thread limits (each ``1``), ``PYTHONUNBUFFERED=1``, and
            ``PYTHONIOENCODING=utf-8``.
        cwd: Working directory of the children.
        clean_tokens: Words removed from error messages, ignoring case.
        timeout_seconds: Default wall-clock limit of a child.
        progress_every: Log a progress line every this many rows.
        log: Logger for progress; this module's by default.

    Returns:
        The report, also written to ``<rows stem>.report.json``.

    Raises:
        ConfigError: If the arguments are invalid or the platform cannot
            do what they ask; nothing is read or started then.
        TornRowsError: If the rows file or the errors sidecar ends in a
            torn line; nothing is started then.
        CallsStoppedError: When ``max_failures`` is reached; every child has
            ended and the report is written.
        KeyboardInterrupt: After every child has ended and the report is
            written with ``stopped="interrupted"``.
    """
    began = time.monotonic()
    logger = log or _LOG
    rows_path = Path(rows)
    stem = str(rows_path.with_suffix(""))
    errors_path = Path(f"{stem}.errors.jsonl")
    ballast_path = Path(f"{stem}.ballast.jsonl") if ballast is not False else None
    children_log = Path(f"{stem}.children.log")
    report_path = Path(f"{stem}.report.json")

    slots_cpus = _slot_cpus(cpus, workers, pin=pin)
    canonical = [canonical_key(call.key) for call in calls]
    if len(set(canonical)) != len(canonical):
        raise ConfigError("the calls' keys are not distinct")
    if max_failures < 0:
        raise ConfigError(f"max_failures is 0 (no limit) or positive, not {max_failures}")
    for limit in (timeout_seconds, *(call.timeout_seconds for call in calls)):
        if limit is not None and not limit > 0:
            raise ConfigError(f"a time limit is positive, not {limit}")
    stamp_fields = dict(stamp or {})

    done = done_keys(rows_path, key_of)
    earlier: dict[str, int] = {}
    if errors_path.exists():
        for record in read_rows(errors_path):
            name = canonical_key(record.get("key"))
            earlier[name] = earlier.get(name, 0) + 1
    pending: deque[tuple[Call, str]] = deque()
    done_before = skipped_failed = 0
    for call, name in zip(calls, canonical, strict=True):
        if name in done:
            done_before += 1
        elif skip_failed and name in earlier:
            skipped_failed += 1
        else:
            pending.append((call, name))
    repeats: Iterator[Call] | None = None
    if ballast is not False:
        source = list(calls) if ballast is True else list(ballast)
        if source:
            ordered = interleaved(source, seed=zlib.crc32(rows_path.name.encode("utf-8")))
            repeats = itertools.cycle(ordered)

    owned = [cpu for cpu in slots_cpus if cpu is not None]
    with _runner_on(owned):
        return _run(
            pending=pending,
            repeats=repeats,
            slots_cpus=slots_cpus,
            earlier=earlier,
            key_of=key_of,
            finish=finish,
            stamp=stamp_fields,
            tokens=tuple(clean_tokens),
            timeout=timeout_seconds,
            command=[str(python or sys.executable), *entry],
            variables=_child_environment(environment),
            cwd=None if cwd is None else str(cwd),
            paths=(rows_path, errors_path, ballast_path, children_log, report_path),
            max_failures=max_failures,
            progress_every=max(1, progress_every),
            log=logger,
            counts=(done_before, skipped_failed),
            began=began,
        )


@contextlib.contextmanager
def _runner_on(cpus: Sequence[int]) -> Iterator[None]:
    """Pin the runner to the run's processors, read back, and restore its set on exit."""
    with contextlib.ExitStack() as stack:
        if cpus:
            try:
                stack.enter_context(confined(cpus))
                pin_self(cpus)
            except RuntimeError as error:
                raise ConfigError(f"cannot pin the runner: {error}") from None
        yield


def _run(
    *,
    pending: deque[tuple[Call, str]],
    repeats: Iterator[Call] | None,
    slots_cpus: list[int | None],
    earlier: dict[str, int],
    key_of: Callable[[Mapping[str, Any]], Any],
    finish: Callable[[dict[str, Any], CallFacts], Mapping[str, Any]] | None,
    stamp: dict[str, Any],
    tokens: tuple[str, ...],
    timeout: float | None,
    command: list[str],
    variables: dict[str, str],
    cwd: str | None,
    paths: tuple[Path, Path, Path | None, Path, Path],
    max_failures: int,
    progress_every: int,
    log: logging.Logger,
    counts: tuple[int, int],
    began: float,
) -> CallReport:
    """Run the pending calls with the runner already on its processors."""
    rows_path, errors_path, ballast_path, children_log, report_path = paths
    done_before, skipped_failed = counts
    logger = log
    count = len(slots_cpus)
    run_id = f"{time.time_ns():x}"
    groups: list[ProcessGroup] = []
    try:
        for slot in range(count):
            group = ProcessGroup.create(f"{group_name(os.getpid(), slot)}-{run_id}")
            groups.append(group)
            if group.platform == "none":
                raise ConfigError(f"cannot create the process group of slot {slot}: {group.error}")
        state = _State(
            calls=pending,
            repeats=repeats,
            slots=[None] * count,
            cpus=slots_cpus,
            groups=groups,
            earlier=earlier,
            key_of=key_of,
            finish=finish,
            stamp=stamp,
            tokens=tokens,
            timeout=timeout,
            command=command,
            variables=variables,
            cwd=cwd,
            children_log=children_log,
            rows=RowWriter(rows_path),
            errors=_Sidecar(errors_path),
            ballast=_Sidecar(ballast_path) if ballast_path is not None else None,
            max_failures=max_failures,
            progress_every=progress_every,
            log=logger,
        )
        logger.info(
            "running %d call(s) on %d slot(s); %d done before, %d skipped after failures",
            len(pending),
            count,
            done_before,
            skipped_failed,
        )
        stopped: str | None = None
        try:
            stopped = state.run()
        except KeyboardInterrupt:
            state.stop_all()
            stopped = "interrupted"
            report = state.report(
                done_before, skipped_failed, stopped, began, rows_path, errors_path, ballast_path
            )
            atomic_write_json(report_path, report.to_dict())
            raise
        finally:
            state.close()
    finally:
        for group in groups:
            group.close()
    report = state.report(
        done_before, skipped_failed, stopped, began, rows_path, errors_path, ballast_path
    )
    atomic_write_json(report_path, report.to_dict())
    logger.info(
        "%d row(s) written, %d failure(s)%s",
        report.rows_written,
        report.failures,
        "" if stopped is None else f"; stopped: {stopped}",
    )
    if stopped is not None:
        raise CallsStoppedError(report)
    return report


def _slot_cpus(
    cpus: Sequence[int] | int | str, workers: int | None, *, pin: bool
) -> list[int | None]:
    """Return each slot's processor, or raise :class:`ConfigError`."""
    try:
        if isinstance(cpus, str):
            chosen = parse_cpus(cpus)
        elif isinstance(cpus, int):
            chosen = cpus_of_mask(cpus)
        else:
            chosen = tuple(int(cpu) for cpu in cpus)
    except ValueError as error:
        raise ConfigError(str(error)) from None
    if not chosen:
        raise ConfigError("the run needs at least one processor")
    if len(set(chosen)) != len(chosen):
        raise ConfigError("the run's processors are not distinct")
    count = len(chosen) if workers is None else workers
    if not 1 <= count <= len(chosen):
        raise ConfigError(
            f"workers is between 1 and the number of processors ({len(chosen)}), not {count}"
        )
    if not pin:
        return [None] * count
    refusal = pinning_refusal_text()
    if refusal is not None:
        raise ConfigError(f"cannot pin the run's processes: {refusal}; pass pin=False")
    allowed = current_cpus()
    if allowed is not None:
        outside = sorted(set(chosen) - set(allowed))
        if outside:
            raise ConfigError(
                f"processor(s) {outside} are outside the set this process may use "
                f"({describe_cpus(allowed)})"
            )
    return list(chosen[:count])


def _child_environment(extra: Mapping[str, str] | None) -> dict[str, str]:
    variables = dict(os.environ)
    variables.update(dict.fromkeys(THREAD_VARIABLES, "1"))
    variables.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    variables.update(extra or {})
    return variables


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass
class _State:
    """The runner's bookkeeping; :meth:`run` is the main loop."""

    calls: deque[tuple[Call, str]]
    repeats: Iterator[Call] | None
    slots: list[_Running | None]
    cpus: list[int | None]
    groups: list[ProcessGroup]
    earlier: dict[str, int]
    key_of: Callable[[Mapping[str, Any]], Any]
    finish: Callable[[dict[str, Any], CallFacts], Mapping[str, Any]] | None
    stamp: dict[str, Any]
    tokens: tuple[str, ...]
    timeout: float | None
    command: list[str]
    variables: dict[str, str]
    cwd: str | None
    children_log: Path
    rows: RowWriter
    errors: _Sidecar
    ballast: _Sidecar | None
    max_failures: int
    progress_every: int
    log: logging.Logger
    events: queue.Queue[int] = field(default_factory=queue.Queue)
    failures: int = 0
    ballast_runs: int = 0
    real_running: int = 0
    stderr: IO[bytes] | None = None

    def run(self) -> str | None:
        """Run until every call is done; return why the run stopped early, or ``None``."""
        self.children_log.parent.mkdir(parents=True, exist_ok=True)
        self.stderr = self.children_log.open("ab")
        self._fill()
        while any(running is not None for running in self.slots):
            try:
                slot = self.events.get(timeout=self._wait_seconds())
            except queue.Empty:
                self._expire()
                continue
            ended = [slot]
            with contextlib.suppress(queue.Empty):
                while True:
                    ended.append(self.events.get_nowait())
            for item in ended:
                self._finish(item)
            if self.max_failures and self.failures >= self.max_failures:
                self.stop_all()
                return "max-failures"
            if not self.calls and self.real_running == 0:
                self._end_ballast()
            self._fill()
        return None

    # -- starting

    def _fill(self) -> None:
        """Start a call in every free slot that has one to run, then update the load."""
        started: list[_Running] = []
        for slot, running in enumerate(self.slots):
            if running is not None:
                continue
            if self.calls:
                call, name = self.calls.popleft()
                started.append(self._start(slot, call, name, ballast=False))
            elif self.repeats is not None and self.real_running > 0:
                call = next(self.repeats)
                started.append(self._start(slot, call, canonical_key(call.key), ballast=True))
        live = sum(running is not None for running in self.slots)
        for running in started:
            running.live_start = running.live_min = live
        for running in self.slots:
            if running is not None:
                running.live_min = min(running.live_min, live)

    def _start(self, slot: int, call: Call, name: str, *, ballast: bool) -> _Running:
        cpu = self.cpus[slot]
        attempt = 1 if ballast else self.earlier.get(name, 0) + 1
        limit = call.timeout_seconds if call.timeout_seconds is not None else self.timeout
        running = _Running(
            slot=slot,
            call=call,
            canonical=name,
            attempt=attempt,
            ballast=ballast,
            cpu=cpu,
            process=None,
            started_utc=_utc_now(),
            deadline=None if limit is None else time.monotonic() + limit,
        )
        self.slots[slot] = running
        if ballast:
            self.ballast_runs += 1
        else:
            self.real_running += 1
        document = {
            "schema": SCHEMA,
            "key": call.key,
            "target": call.target,
            "arguments": dict(call.arguments),
            "cpu": cpu,
            "group": self.groups[slot].name,
            "attempt": attempt,
            "ballast": ballast,
            "clean_tokens": list(self.tokens),
        }
        group = self.groups[slot]
        try:
            with confined(cpu) if cpu is not None else contextlib.nullcontext():
                process = subprocess.Popen(
                    self.command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=self.stderr,
                    cwd=self.cwd,
                    env=self.variables,
                    **group.popen_options(),
                )
        except (OSError, RuntimeError, ValueError) as error:
            running.start_error = f"{type(error).__name__}: {error}"
            self.events.put(slot)  # the slot frees at once; the failure is recorded then
            return running
        running.process = process
        refused = group.assign(process.pid)
        if refused is not None:
            self.log.debug("slot %d: the parent could not assign the child: %s", slot, refused)
        payload = json.dumps(document, ensure_ascii=False, allow_nan=False).encode("utf-8")
        threading.Thread(
            target=_attend,
            args=(running, payload, group, self.events),
            name=f"dispatchatlas-call-{slot}",
            daemon=True,
        ).start()
        return running

    # -- ending

    def _wait_seconds(self) -> float:
        deadlines = [
            running.deadline
            for running in self.slots
            if running is not None and running.deadline is not None and not running.timed_out
        ]
        if not deadlines:
            return 1.0
        return min(1.0, max(0.0, min(deadlines) - time.monotonic()))

    def _expire(self) -> None:
        now = time.monotonic()
        for running in self.slots:
            if (
                running is not None
                and running.deadline is not None
                and not running.timed_out
                and now >= running.deadline
            ):
                running.timed_out = True
                self._kill(running)

    def _kill(self, running: _Running) -> None:
        self.groups[running.slot].terminate()
        process = running.process
        if process is not None:
            if sys.platform != "win32":
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(OSError):
                process.kill()

    def _finish(self, slot: int) -> None:
        running = self.slots[slot]
        if running is None:
            return
        self.slots[slot] = None
        if not running.ballast:
            self.real_running -= 1
        if running.process is None:
            message = running.start_error or "the child did not start"
            self._fail(running, "StartFailed", message, None, last_line=False)
            return
        reply, problem = _reply(running.last_line)
        facts = self._facts(running, reply.get("facts") if reply else None)
        if running.stopped:
            if running.ballast and self.ballast is not None:
                self.ballast.write(
                    {"key": running.call.key, "outcome": "stopped", "facts": facts.to_dict()}
                )
            return
        if running.timed_out:
            limit = running.call.timeout_seconds or self.timeout
            self._fail(running, "Timeout", f"the call passed its limit of {limit} s", facts)
            return
        if reply is None:
            code = running.returncode
            self._fail(running, "ProcessDied", _death(code, problem), facts, last_line=False)
            return
        if "error" in reply:
            error = reply.get("error") or {}
            self._fail(
                running,
                str(error.get("type", "Error")),
                str(error.get("message", "")),
                facts,
            )
            return
        row = reply.get("row")
        if not isinstance(row, dict):
            self._fail(running, "RowRefused", "the reply holds no row object", facts)
            return
        if running.ballast:
            if self.ballast is not None:
                self.ballast.write(
                    {"key": running.call.key, "outcome": "row", "facts": facts.to_dict()}
                )
            return
        self._write(running, row, facts)

    def _write(self, running: _Running, row: dict[str, Any], facts: CallFacts) -> None:
        try:
            if self.finish is None:
                final = dict(row)
                final["run"] = facts.to_dict()
            else:
                final = dict(self.finish(row, facts))
        except Exception as error:  # the caller's hook failed: the row is refused
            self._fail(running, "RowRefused", f"finish failed: {error}", facts)
            return
        collision = sorted(set(final) & set(self.stamp))
        if collision:
            self._fail(
                running, "RowRefused", f"the row already has the stamp field(s) {collision}", facts
            )
            return
        final.update(self.stamp)
        try:
            given = canonical_key(self.key_of(final))
        except Exception as error:  # a key function that cannot read the row
            self._fail(running, "KeyMismatch", f"key_of failed on the row: {error}", facts)
            return
        if given != running.canonical:
            self._fail(
                running,
                "KeyMismatch",
                f"the row's key {given} is not the call's key {running.canonical}",
                facts,
            )
            return
        try:
            written = self.rows.write(final)
        except (TypeError, ValueError) as error:
            self._fail(running, "RowRefused", str(error), facts)
            return
        if written % self.progress_every == 0:
            self.log.info("%d row(s) written, %d failure(s)", written, self.failures)

    def _fail(
        self,
        running: _Running,
        kind: str,
        message: str,
        facts: CallFacts | None,
        *,
        last_line: bool = True,
    ) -> None:
        record = CallError(
            key=running.call.key,
            attempt=running.attempt,
            type=kind,
            message=clean_message(message, tokens=self.tokens),
            returncode=running.returncode,
            last_line=last_line and running.process is not None,
            facts=None if facts is None else facts.to_dict(),
            cpu=running.cpu,
            started_utc=running.started_utc,
            ended_utc=_utc_now(),
        )
        if running.ballast:
            if self.ballast is not None:
                self.ballast.write(
                    {"key": running.call.key, "outcome": "error", "error": record.to_dict()}
                )
            return
        self.errors.write(record.to_dict())
        self.failures += 1
        self.log.warning(
            "call %s failed (%s, attempt %d): %s",
            running.canonical,
            kind,
            running.attempt,
            record.message,
        )

    def _facts(self, running: _Running, reported: Any) -> CallFacts:
        child: Mapping[str, Any] = reported if isinstance(reported, Mapping) else {}
        threads = child.get("threads")
        return CallFacts(
            key=running.call.key,
            attempt=running.attempt,
            cpu=running.cpu,
            affinity_inherited=child.get("affinity_inherited"),
            affinity_mask=child.get("affinity_mask"),
            pid=child.get("pid"),
            group=child.get("group"),
            threads=dict(threads) if isinstance(threads, Mapping) else {},
            peak_rss_bytes=child.get("peak_rss_bytes"),
            started_utc=child.get("started_utc"),
            ended_utc=child.get("ended_utc"),
            wall_seconds=child.get("wall_seconds"),
            workers=len(self.slots),
            workers_live_start=running.live_start,
            workers_live_min=running.live_min,
            ballast=running.ballast,
        )

    # -- stopping

    def _end_ballast(self) -> None:
        """End every ballast call once no real call runs."""
        busy = [running for running in self.slots if running is not None and running.ballast]
        for running in busy:
            running.stopped = True
            self._kill(running)
        self._await(busy)

    def stop_all(self) -> None:
        """End every child, discard what is queued, and wait for the slots to free."""
        self.calls.clear()
        self.repeats = None
        busy = [running for running in self.slots if running is not None]
        for running in busy:
            running.stopped = True
            self._kill(running)
        self._await(busy)

    def _await(self, busy: Sequence[_Running]) -> None:
        deadline = time.monotonic() + _STOP_SECONDS
        waiting = {running.slot for running in busy}
        while waiting and time.monotonic() < deadline:
            try:
                slot = self.events.get(timeout=1.0)
            except queue.Empty:
                continue
            waiting.discard(slot)
            self._finish(slot)
        if waiting:
            self.log.warning("slot(s) %s did not end within %.0f s", sorted(waiting), _STOP_SECONDS)

    def close(self) -> None:
        self.rows.close()
        self.errors.close()
        if self.ballast is not None:
            self.ballast.close()
        if self.stderr is not None:
            self.stderr.close()

    def report(
        self,
        done_before: int,
        skipped_failed: int,
        stopped: str | None,
        began: float,
        rows: Path,
        errors: Path,
        ballast: Path | None,
    ) -> CallReport:
        return CallReport(
            rows_written=self.rows.count,
            done_before=done_before,
            skipped_failed=skipped_failed,
            failures=self.failures,
            ballast_runs=self.ballast_runs,
            stopped=stopped,
            workers=len(self.slots),
            cpus=tuple(self.cpus),
            elapsed_seconds=time.monotonic() - began,
            rows=rows,
            errors=errors,
            ballast=ballast,
        )


def _attend(
    running: _Running, payload: bytes, group: ProcessGroup, events: queue.Queue[int]
) -> None:
    """Feed a child its document, collect its last line, wait for it, and clean its group."""
    process = running.process
    if process is None:
        events.put(running.slot)
        return
    reader = threading.Thread(target=_read_last_line, args=(running,), daemon=True)
    reader.start()
    if process.stdin is not None:
        with contextlib.suppress(OSError):
            process.stdin.write(payload)
        with contextlib.suppress(OSError):
            process.stdin.close()
    if sys.platform != "win32":
        # Wait without reaping, so the group id cannot be reused before it is killed.
        with contextlib.suppress(ChildProcessError, OSError):
            os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        running.returncode = process.wait()
    else:
        running.returncode = process.wait()
        group.terminate()  # what the call left running in the slot's job
    reader.join(_JOIN_SECONDS)
    events.put(running.slot)


def _read_last_line(running: _Running) -> None:
    process = running.process
    if process is None or process.stdout is None:
        return
    with contextlib.suppress(OSError, ValueError):
        for line in process.stdout:
            if line.strip():
                running.last_line = line
    with contextlib.suppress(OSError):
        process.stdout.close()


def _reply(line: bytes) -> tuple[dict[str, Any] | None, str]:
    """Return the reply of a last line, or ``None`` with why it is not one."""
    if not line.strip():
        return None, "no reply line"
    try:
        reply = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None, "the last line is not a reply"
    if not isinstance(reply, dict) or not ("row" in reply or "error" in reply):
        return None, "the last line is not a reply"
    return reply, ""


def _death(code: int | None, problem: str) -> str:
    if code is not None and code < 0 and sys.platform != "win32":
        with contextlib.suppress(ValueError):
            return f"killed by {signal.Signals(-code).name} ({problem})"
    return f"exit code {code} ({problem})"
