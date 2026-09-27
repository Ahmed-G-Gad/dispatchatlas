"""The records a campaign writes: completed runs, failed attempts, and sessions.

Each record is a JSON document whose ``schema`` field names its type and
version:

- ``dispatchatlas.run/1`` (:class:`RunRecord`): one completed run. It names
  the run (campaign, config hash, problem, solver, protocol, replicate, seed),
  holds its :class:`Outcome`, the ``result_hash`` of that outcome, the best
  schedule as ``[task, mode, start, end]`` rows unless the campaign disables
  it, and the :class:`Provenance` of the attempt that produced it.
- ``dispatchatlas.failure/1`` (:class:`FailureRecord`): one failed attempt,
  with its error and :class:`Category`.
- ``dispatchatlas.session/1`` (:class:`SessionRecord`): one invocation of the
  runner, with its execution settings and the environment it ran in.

The **result hash** is the SHA-256 digest of the canonical JSON of
:meth:`Outcome.hashed`: every outcome field except those that measure wall
time (``elapsed`` and the elapsed column of the trace) and ``extras``, whose
solver-specific facts carry no determinism guarantee. Nothing about the
machine, the session, or the attempt enters it, so a replay on any machine
compares results directly (see :mod:`~dispatchatlas.campaigns.replaying`). The
schedule enters it through its digest, so a record with or without stored
schedule rows hashes alike.

These classes convert records to and from documents. Where the files live is
the business of :mod:`~dispatchatlas.campaigns.store`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from dispatchatlas.campaigns.config import Execution, Protocol
from dispatchatlas.core.errors import ConfigError, ModelError, SerializationError
from dispatchatlas.core.model import Scalar
from dispatchatlas.core.schedule import Assignment, Schedule
from dispatchatlas.core.serialization import digest
from dispatchatlas.core.solver import Status, StopCriteria, StopReason, TracePoint

__all__ = [
    "FAILURE_SCHEMA",
    "RUN_SCHEMA",
    "SESSION_SCHEMA",
    "TRACE_COLUMNS",
    "Category",
    "ErrorKind",
    "FailureRecord",
    "Outcome",
    "ProblemRef",
    "Provenance",
    "RunError",
    "RunRecord",
    "SessionRecord",
    "SessionStatus",
    "SolverEntry",
    "protocol_from_dict",
    "schedule_digest",
    "schedule_rows",
    "settled",
]

RUN_SCHEMA = "dispatchatlas.run/1"
FAILURE_SCHEMA = "dispatchatlas.failure/1"
SESSION_SCHEMA = "dispatchatlas.session/1"

TRACE_COLUMNS: tuple[str, ...] = ("elapsed", "evaluations", "iteration", "value", "feasible")
"""Column order of the rows of a recorded trace."""


class Category(StrEnum):
    """Whether a failed attempt may succeed when tried again.

    Typed DispatchAtlas errors (an invalid configuration, a feature the solver
    does not model, a missing optional dependency) and hard timeouts are
    **terminal**: the instance, seed, and budget are fixed, so another attempt
    would fail the same way. Any other exception, and a worker process that
    dies, is **recoverable**: memory pressure or an I/O error may not recur.
    """

    TERMINAL = "terminal"
    RECOVERABLE = "recoverable"


class ErrorKind(StrEnum):
    """What ended a failed attempt."""

    EXCEPTION = "exception"
    """The solver, or the checks of its result, raised an exception."""
    TIMEOUT = "timeout"
    """The run passed its hard time limit and its worker was killed."""
    WORKER_EXIT = "worker-exit"
    """The worker process died during the run."""
    TRANSFER = "transfer"
    """The run could not be sent to, or loaded in, a worker process."""


class SessionStatus(StrEnum):
    """How a session of the runner ended."""

    RUNNING = "running"
    FINISHED = "finished"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


def settled(failures: Sequence[FailureRecord], max_attempts: int) -> bool:
    """Return whether a run's failures settle it, so it gets no further attempt.

    A run that never completed is settled when its latest failure is
    terminal, or when it has failed ``max_attempts`` times.
    """
    if not failures:
        return False
    latest = max(failures, key=lambda failure: failure.provenance.attempt)
    return latest.category is Category.TERMINAL or len(failures) >= max_attempts


def schedule_rows(schedule: Schedule) -> list[list[Any]]:
    """Return a schedule as ``[task, mode, start, end]`` rows in schedule order."""
    return [
        [item.task, int(item.mode), float(item.start), float(item.end)]
        for item in schedule.assignments
    ]


def schedule_digest(schedule: Schedule) -> str:
    """Return the SHA-256 digest of the canonical JSON of :func:`schedule_rows`."""
    return digest(schedule_rows(schedule))


# ------------------------------------------------------------------ outcome


@dataclass(frozen=True)
class Outcome:
    """What a run found, as recorded.

    Attributes:
        status: What the solver states about its schedule.
        stop_reason: Why the solver stopped.
        objective: Name of the primary objective.
        value: The primary objective's value, or ``None`` without a schedule.
        objectives: Every problem objective's value, computed from the
            schedule by :func:`~dispatchatlas.core.objectives.evaluate`.
        bound: Proven bound on the primary objective, for exact solvers.
        gap: Relative distance between value and bound, as
            :attr:`~dispatchatlas.core.solver.SolveResult.gap` defines it.
        feasible: Whether the schedule breaks no constraint, as checked by
            :func:`~dispatchatlas.core.validation.check_schedule`.
        violations: Codes of the constraints the schedule breaks, sorted.
        evaluations: Schedules the solver evaluated.
        iterations: Main-loop iterations of the solver.
        elapsed: Wall-clock seconds the solver reports.
        trace: Improvements of the best solution during the run.
        front: Objective vectors of the non-dominated solutions of a
            multi-objective solver, in the problem's objective order.
        schedule_digest: :func:`schedule_digest` of the schedule, or ``None``
            without one.
        extras: Solver-specific facts.
    """

    status: Status
    stop_reason: StopReason
    objective: str
    value: float | None
    objectives: Mapping[str, float]
    bound: float | None
    gap: float | None
    feasible: bool
    violations: tuple[str, ...]
    evaluations: int
    iterations: int
    elapsed: float
    trace: tuple[TracePoint, ...] = ()
    front: tuple[tuple[float, ...], ...] = ()
    schedule_digest: str | None = None
    extras: Mapping[str, Scalar] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        # Numbers are normalised on construction, so an outcome built from a
        # solver result and one read back from its record serialise alike.
        values = {
            "status": Status(self.status),
            "stop_reason": StopReason(self.stop_reason),
            "value": _optional_float(self.value),
            "objectives": {str(k): float(v) for k, v in self.objectives.items()},
            "bound": _optional_float(self.bound),
            "gap": _optional_float(self.gap),
            "violations": tuple(str(code) for code in self.violations),
            "evaluations": int(self.evaluations),
            "iterations": int(self.iterations),
            "elapsed": float(self.elapsed),
            "trace": tuple(
                TracePoint(
                    float(p.elapsed),
                    int(p.evaluations),
                    int(p.iteration),
                    float(p.value),
                    bool(p.feasible),
                )
                for p in self.trace
            ),
            "front": tuple(tuple(float(v) for v in point) for point in self.front),
            "extras": dict(self.extras),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def replayable(self) -> bool:
        """Whether a replay can reproduce the run.

        A run stopped by its wall-clock limit cannot: how far it got depends
        on the speed and load of the machine.
        """
        return self.stop_reason is not StopReason.TIME_LIMIT

    def to_dict(self) -> dict[str, Any]:
        """Return the outcome as a JSON-ready dictionary."""
        return {
            "status": self.status.value,
            "stop_reason": self.stop_reason.value,
            "objective": self.objective,
            "value": self.value,
            "objectives": dict(self.objectives),
            "bound": self.bound,
            "gap": self.gap,
            "feasible": self.feasible,
            "violations": list(self.violations),
            "evaluations": self.evaluations,
            "iterations": self.iterations,
            "elapsed": self.elapsed,
            "trace": [
                [p.elapsed, p.evaluations, p.iteration, p.value, p.feasible] for p in self.trace
            ],
            "front": [list(point) for point in self.front],
            "schedule_digest": self.schedule_digest,
            "extras": dict(self.extras),
        }

    def hashed(self) -> dict[str, Any]:
        """Return the fields the result hash covers: all but wall times and extras."""
        document = self.to_dict()
        del document["elapsed"], document["extras"]
        document["trace"] = [row[1:] for row in document["trace"]]
        return document

    def result_hash(self) -> str:
        """Return the SHA-256 digest of the canonical JSON of :meth:`hashed`.

        Raises:
            SerializationError: If a value is not finite.
        """
        return digest(self.hashed())

    @classmethod
    def from_dict(cls, document: object) -> Outcome:
        """Build an outcome from :meth:`to_dict` output.

        Raises:
            SerializationError: If the document is malformed.
        """
        fields = _Fields(document, "an outcome")
        trace = tuple(_trace_point(row) for row in fields.array("trace"))
        front = tuple(
            tuple(_number(value, "a front value") for value in _list(point, "a front point"))
            for point in fields.array("front")
        )
        violations = tuple(_text(code, "a violation code") for code in fields.array("violations"))
        objectives = {
            key: _number(value, "an objective value")
            for key, value in fields.mapping("objectives").items()
        }
        status, stop_reason = fields.text("status"), fields.text("stop_reason")
        return _build(
            "an outcome",
            lambda: cls(
                status=Status(status),
                stop_reason=StopReason(stop_reason),
                objective=fields.text("objective"),
                value=fields.optional_number("value"),
                objectives=objectives,
                bound=fields.optional_number("bound"),
                gap=fields.optional_number("gap"),
                feasible=fields.boolean("feasible"),
                violations=violations,
                evaluations=fields.integer("evaluations"),
                iterations=fields.integer("iterations"),
                elapsed=fields.number("elapsed"),
                trace=trace,
                front=front,
                schedule_digest=fields.optional_text("schedule_digest"),
                extras=fields.mapping("extras"),
            ),
        )


# ------------------------------------------------------------------- blocks


@dataclass(frozen=True)
class ProblemRef:
    """The problem a run solved.

    Attributes:
        id: Problem id.
        family: Benchmark family of the problem, if known.
        digest: :func:`~dispatchatlas.core.serialization.problem_digest` of
            the problem as solved, after any objective override.
    """

    id: str
    family: str | None
    digest: str

    def to_dict(self) -> dict[str, Any]:
        """Return the reference as a JSON-ready dictionary."""
        return {"id": self.id, "family": self.family, "digest": self.digest}

    @classmethod
    def from_dict(cls, document: object) -> ProblemRef:
        """Build a reference from :meth:`to_dict` output."""
        fields = _Fields(document, "a problem reference")
        return cls(fields.text("id"), fields.optional_text("family"), fields.text("digest"))


@dataclass(frozen=True)
class SolverEntry:
    """A solver of a campaign, with the facts its metadata declares.

    Attributes:
        name: Label, or id when the campaign gives no label.
        id: Registered solver id.
        label: Campaign label, if any.
        parameters: Parameter values set by the campaign.
        family: Solver family.
        stochastic: Whether its results depend on the seed.
        iterative: Whether it runs until a stop criterion fires.
        multi_objective: Whether it optimises every objective at once.
    """

    name: str
    id: str
    label: str | None
    parameters: Mapping[str, Any]
    family: str
    stochastic: bool
    iterative: bool
    multi_objective: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the entry as a JSON-ready dictionary."""
        return {
            "name": self.name,
            "id": self.id,
            "label": self.label,
            "parameters": dict(self.parameters),
            "family": self.family,
            "stochastic": self.stochastic,
            "iterative": self.iterative,
            "multi_objective": self.multi_objective,
        }

    @classmethod
    def from_dict(cls, document: object) -> SolverEntry:
        """Build an entry from :meth:`to_dict` output."""
        fields = _Fields(document, "a solver entry")
        return cls(
            name=fields.text("name"),
            id=fields.text("id"),
            label=fields.optional_text("label"),
            parameters=fields.mapping("parameters"),
            family=fields.text("family"),
            stochastic=fields.boolean("stochastic"),
            iterative=fields.boolean("iterative"),
            multi_objective=fields.boolean("multi_objective"),
        )


def protocol_from_dict(document: object) -> Protocol:
    """Build a protocol from the dictionary its ``to_dict`` method returns.

    Raises:
        SerializationError: If the document is malformed.
    """
    fields = _Fields(document, "a protocol")
    stop = fields.mapping("stop")
    return _build("a protocol", lambda: Protocol(fields.text("name"), StopCriteria.from_dict(stop)))


@dataclass(frozen=True)
class Provenance:
    """When, where, and at what cost one attempt of a run executed.

    Attributes:
        attempt: Attempt number, counted from 1 across sessions.
        session: Number of the runner session that executed it.
        started: UTC start time in ISO 8601.
        finished: UTC end time in ISO 8601.
        wall_seconds: Wall-clock seconds of the solve.
        cpu_seconds: CPU seconds the executing process spent on the solve,
            including any threads the solver started. ``None`` when not
            measured.
        peak_rss_bytes: Peak resident memory of the executing process during
            the solve, where the platform can reset and read it (Linux).
            It includes the interpreter and the loaded libraries.
        worker: ``"main"`` for the sequential backend, else the worker slot.
        hard_limit: Seconds after which the run would have been killed, or
            ``None`` when no hard limit was armed.
    """

    attempt: int
    session: int
    started: str
    finished: str
    wall_seconds: float
    cpu_seconds: float | None = None
    peak_rss_bytes: int | None = None
    worker: str = "main"
    hard_limit: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the provenance as a JSON-ready dictionary."""
        return {
            "attempt": self.attempt,
            "session": self.session,
            "started": self.started,
            "finished": self.finished,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "worker": self.worker,
            "hard_limit": self.hard_limit,
        }

    @classmethod
    def from_dict(cls, document: object) -> Provenance:
        """Build provenance from :meth:`to_dict` output."""
        fields = _Fields(document, "a provenance block")
        rss = fields.raw("peak_rss_bytes")
        return cls(
            attempt=fields.integer("attempt"),
            session=fields.integer("session"),
            started=fields.text("started"),
            finished=fields.text("finished"),
            wall_seconds=fields.number("wall_seconds"),
            cpu_seconds=fields.optional_number("cpu_seconds"),
            peak_rss_bytes=None if rss is None else fields.integer("peak_rss_bytes"),
            worker=fields.text("worker"),
            hard_limit=fields.optional_number("hard_limit"),
        )


# ------------------------------------------------------------------- records


@dataclass(frozen=True)
class RunRecord:
    """One completed run.

    Attributes:
        run_id: Run id from the plan.
        campaign: Campaign id.
        config_hash: Config hash of the campaign.
        problem: The problem solved.
        solver: The solver and its parameters.
        protocol: The protocol and its stop criteria.
        replicate: Replicate number, from 0.
        seed: Seed the solver received.
        outcome: What the run found.
        result_hash: :meth:`Outcome.result_hash` at the time of the run.
        schedule: The best schedule, unless the campaign did not record it.
        provenance: How the attempt that completed the run executed.
    """

    run_id: str
    campaign: str
    config_hash: str
    problem: ProblemRef
    solver: SolverEntry
    protocol: Protocol
    replicate: int
    seed: int
    outcome: Outcome
    result_hash: str
    schedule: Schedule | None
    provenance: Provenance

    def verify(self) -> tuple[str, ...]:
        """Return the integrity problems of the record, empty when it is intact.

        The outcome must hash to :attr:`result_hash`, and stored schedule rows
        must match the outcome's schedule digest.
        """
        issues: list[str] = []
        if self.outcome.result_hash() != self.result_hash:
            issues.append("the outcome does not match the result hash")
        if self.schedule is not None and schedule_digest(self.schedule) != (
            self.outcome.schedule_digest
        ):
            issues.append("the schedule does not match the outcome's schedule digest")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-ready document."""
        return {
            "schema": RUN_SCHEMA,
            "run_id": self.run_id,
            "campaign": self.campaign,
            "config_hash": self.config_hash,
            "problem": self.problem.to_dict(),
            "solver": self.solver.to_dict(),
            "protocol": self.protocol.to_dict(),
            "replicate": self.replicate,
            "seed": self.seed,
            "outcome": self.outcome.to_dict(),
            "result_hash": self.result_hash,
            "schedule": None if self.schedule is None else schedule_rows(self.schedule),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, document: object) -> RunRecord:
        """Build a record from :meth:`to_dict` output.

        Raises:
            SerializationError: If the document is not a run record of this
                schema version, or is malformed.
        """
        fields = _Fields(document, "a run record")
        _require_schema(fields, RUN_SCHEMA)
        rows = fields.raw("schedule")
        schedule = None
        if rows is not None:
            if not isinstance(rows, list):
                raise SerializationError("the schedule of a run record must be a list of rows")
            schedule = _schedule(rows)
        return cls(
            run_id=fields.text("run_id"),
            campaign=fields.text("campaign"),
            config_hash=fields.text("config_hash"),
            problem=ProblemRef.from_dict(fields.raw("problem")),
            solver=SolverEntry.from_dict(fields.raw("solver")),
            protocol=protocol_from_dict(fields.raw("protocol")),
            replicate=fields.integer("replicate"),
            seed=fields.integer("seed"),
            outcome=Outcome.from_dict(fields.raw("outcome")),
            result_hash=fields.text("result_hash"),
            schedule=schedule,
            provenance=Provenance.from_dict(fields.raw("provenance")),
        )


@dataclass(frozen=True)
class RunError:
    """The error that ended a failed attempt.

    Attributes:
        kind: What ended the attempt.
        message: Human-readable description.
        type: Qualified exception class, for exceptions.
        traceback: Formatted traceback, for exceptions.
    """

    kind: ErrorKind
    message: str
    type: str | None = None
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the error as a JSON-ready dictionary."""
        return {
            "kind": self.kind.value,
            "message": self.message,
            "type": self.type,
            "traceback": self.traceback,
        }

    @classmethod
    def from_dict(cls, document: object) -> RunError:
        """Build an error from :meth:`to_dict` output."""
        fields = _Fields(document, "an error")
        return _build(
            "an error",
            lambda: cls(
                kind=ErrorKind(fields.text("kind")),
                message=fields.text("message"),
                type=fields.optional_text("type"),
                traceback=fields.optional_text("traceback"),
            ),
        )


@dataclass(frozen=True)
class FailureRecord:
    """One failed attempt of a run.

    Attributes:
        run_id: Run id from the plan.
        campaign: Campaign id.
        config_hash: Config hash of the campaign.
        problem: The problem.
        solver: The solver and its parameters.
        protocol: The protocol and its stop criteria.
        replicate: Replicate number, from 0.
        seed: Seed the solver received.
        category: Whether another attempt may succeed.
        error: What ended the attempt.
        provenance: How the attempt executed. CPU time and memory are only
            known when the attempt raised an exception.
    """

    run_id: str
    campaign: str
    config_hash: str
    problem: ProblemRef
    solver: SolverEntry
    protocol: Protocol
    replicate: int
    seed: int
    category: Category
    error: RunError
    provenance: Provenance

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-ready document."""
        return {
            "schema": FAILURE_SCHEMA,
            "run_id": self.run_id,
            "campaign": self.campaign,
            "config_hash": self.config_hash,
            "problem": self.problem.to_dict(),
            "solver": self.solver.to_dict(),
            "protocol": self.protocol.to_dict(),
            "replicate": self.replicate,
            "seed": self.seed,
            "category": self.category.value,
            "error": self.error.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, document: object) -> FailureRecord:
        """Build a record from :meth:`to_dict` output.

        Raises:
            SerializationError: If the document is not a failure record of
                this schema version, or is malformed.
        """
        fields = _Fields(document, "a failure record")
        _require_schema(fields, FAILURE_SCHEMA)
        category = fields.text("category")
        return cls(
            run_id=fields.text("run_id"),
            campaign=fields.text("campaign"),
            config_hash=fields.text("config_hash"),
            problem=ProblemRef.from_dict(fields.raw("problem")),
            solver=SolverEntry.from_dict(fields.raw("solver")),
            protocol=protocol_from_dict(fields.raw("protocol")),
            replicate=fields.integer("replicate"),
            seed=fields.integer("seed"),
            category=_build("a failure record", lambda: Category(category)),
            error=RunError.from_dict(fields.raw("error")),
            provenance=Provenance.from_dict(fields.raw("provenance")),
        )


@dataclass(frozen=True)
class SessionRecord:
    """One invocation of the runner on a campaign.

    Attributes:
        session: Session number, from 1.
        campaign: Campaign id.
        config_hash: Config hash of the campaign.
        started: UTC start time in ISO 8601.
        finished: UTC end time, or ``None`` while running or after a crash.
        status: How the session ended.
        execution: The execution settings in force.
        workers: Number of workers, with ``"auto"`` resolved.
        environment: The environment, as
            :func:`~dispatchatlas.campaigns.environment.capture_environment`
            records it.
        counts: Run counts at the end of the session: ``planned``,
            ``completed``, ``failed`` (settled by failures), ``pending``, and
            this session's ``attempted``, ``succeeded``, and ``errors``.
    """

    session: int
    campaign: str
    config_hash: str
    started: str
    finished: str | None
    status: SessionStatus
    execution: Execution
    workers: int
    environment: Mapping[str, Any]
    counts: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-ready document."""
        return {
            "schema": SESSION_SCHEMA,
            "session": self.session,
            "campaign": self.campaign,
            "config_hash": self.config_hash,
            "started": self.started,
            "finished": self.finished,
            "status": self.status.value,
            "execution": self.execution.to_dict(),
            "workers": self.workers,
            "environment": dict(self.environment),
            "counts": dict(self.counts),
        }

    @classmethod
    def from_dict(cls, document: object) -> SessionRecord:
        """Build a record from :meth:`to_dict` output.

        Raises:
            SerializationError: If the document is not a session record of
                this schema version, or is malformed.
        """
        fields = _Fields(document, "a session record")
        _require_schema(fields, SESSION_SCHEMA)
        execution = fields.mapping("execution")
        status = fields.text("status")
        return cls(
            session=fields.integer("session"),
            campaign=fields.text("campaign"),
            config_hash=fields.text("config_hash"),
            started=fields.text("started"),
            finished=fields.optional_text("finished"),
            status=_build("a session record", lambda: SessionStatus(status)),
            execution=_build("a session record", lambda: Execution(**execution)),
            workers=fields.integer("workers"),
            environment=fields.mapping("environment"),
            counts={
                str(key): _integer(value, "a session count")
                for key, value in fields.mapping("counts").items()
            },
        )


# ------------------------------------------------------------------- helpers


class _Fields:
    """Typed access to the fields of a record document."""

    def __init__(self, document: object, what: str) -> None:
        if not isinstance(document, Mapping):
            raise SerializationError(f"{what} must be a JSON object")
        self._document: Mapping[str, Any] = document
        self._what = what

    def raw(self, key: str) -> Any:
        if key not in self._document:
            raise SerializationError(f"{self._what} lacks the field {key!r}")
        return self._document[key]

    def text(self, key: str) -> str:
        value = self.raw(key)
        if not isinstance(value, str):
            raise SerializationError(f"{self._what} field {key!r} must be a string")
        return value

    def optional_text(self, key: str) -> str | None:
        return None if self.raw(key) is None else self.text(key)

    def integer(self, key: str) -> int:
        return _integer(self.raw(key), f"{self._what} field {key!r}")

    def number(self, key: str) -> float:
        return _number(self.raw(key), f"{self._what} field {key!r}")

    def optional_number(self, key: str) -> float | None:
        return None if self.raw(key) is None else self.number(key)

    def boolean(self, key: str) -> bool:
        value = self.raw(key)
        if not isinstance(value, bool):
            raise SerializationError(f"{self._what} field {key!r} must be true or false")
        return value

    def array(self, key: str) -> list[Any]:
        value = self.raw(key)
        if not isinstance(value, list):
            raise SerializationError(f"{self._what} field {key!r} must be a list")
        return value

    def mapping(self, key: str) -> dict[str, Any]:
        value = self.raw(key)
        if not isinstance(value, Mapping):
            raise SerializationError(f"{self._what} field {key!r} must be an object")
        return dict(value)


def _require_schema(fields: _Fields, expected: str) -> None:
    found = fields.raw("schema")
    if found != expected:
        raise SerializationError(f"expected a {expected!r} document, found schema {found!r}")


def _build[T](what: str, factory: Callable[[], T]) -> T:
    """Call ``factory``, turning an invalid value into a :class:`SerializationError`."""
    try:
        return factory()
    except SerializationError:
        raise
    except (ValueError, TypeError, ConfigError, ModelError) as error:
        raise SerializationError(f"{what} holds an invalid value: {error}") from error


def _integer(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SerializationError(f"{what} must be an integer")
    return value


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SerializationError(f"{what} must be a number")
    if not math.isfinite(value):
        raise SerializationError(f"{what} must be finite")
    return float(value)


def _text(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise SerializationError(f"{what} must be a string")
    return value


def _list(value: object, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise SerializationError(f"{what} must be a list")
    return value


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _trace_point(row: object) -> TracePoint:
    items = _list(row, "a trace row")
    if len(items) != len(TRACE_COLUMNS):
        raise SerializationError(f"a trace row must list {', '.join(TRACE_COLUMNS)}")
    elapsed, evaluations, iteration, value, feasible = items
    if not isinstance(feasible, bool):
        raise SerializationError("the feasibility of a trace row must be true or false")
    return TracePoint(
        _number(elapsed, "a trace time"),
        _integer(evaluations, "a trace evaluation count"),
        _integer(iteration, "a trace iteration"),
        _number(value, "a trace value"),
        feasible,
    )


def _schedule(rows: list[Any]) -> Schedule:
    assignments = []
    for row in rows:
        items = _list(row, "a schedule row")
        if len(items) != 4:
            raise SerializationError("a schedule row must be [task, mode, start, end]")
        task, mode, start, end = items
        assignments.append(
            Assignment(
                _text(task, "the task of a schedule row"),
                _integer(mode, "the mode of a schedule row"),
                _number(start, "the start of a schedule row"),
                _number(end, "the end of a schedule row"),
            )
        )
    return Schedule.of(assignments)
