"""Running the experiments of iterated racing in worker processes, keeping rows for a resume.

:func:`process_run_batch` returns a ``run_batch`` for
:meth:`~dispatchatlas.tuning.irace.IteratedRacing.tune` (or
:func:`~dispatchatlas.tuning.racing.race`) that runs the experiments of a step in
worker processes, through :func:`~dispatchatlas.processes.calls.run_calls`, and
appends one row for each to a rows file.

- **Experiments.** One experiment is one :class:`~dispatchatlas.processes.calls.Call`:
  a child process of its own, pinned to a processor of its own unless ``pin`` is false. The child
  (:func:`run_experiment`) builds the solver with ``solver_factory(configuration)``
  and the problem with ``problem_factory(instance)``, two functions the caller names
  as ``"package.module:function"``, runs ``solver.solve(problem, stop, seed)``, and
  returns the row. An instance is a JSON value, such as the name of a benchmark
  instance, which ``problem_factory`` turns into a problem.
- **Rows.** The key of a row is ``[digest, instance, seed]`` (:func:`experiment_key`),
  with the canonical digest of the configuration (:func:`configuration_digest`). The
  row records the ``configuration``, the ``instance`` and the ``seed``, then the
  ``cost``, the primary objective of the best schedule, which is ``None`` when the
  run found none and is read back as ``math.inf``; the ``status``; the
  ``evaluations``; the ``stop_reason``; the ``wall_seconds`` of the solve; the
  ``first_schedule_digest`` and the ``repairs`` that the solver reports among the
  extras of its result, ``None`` when it reports none; the fields of the ``stamp``;
  and, under ``run``, the facts of the call (processor, process, load, memory).
- **Resume.** The rows file is the record of what is done. Each call of the batch
  reads it, returns the recorded cost of every experiment that has a row, and runs
  the others. :meth:`~dispatchatlas.tuning.irace.IteratedRacing.tune` is a
  deterministic function of the costs it receives, so a job that begins again with the
  same seed, space, and instances asks for the same experiments in the same order,
  receives the recorded costs of those that have rows without running them, and runs
  the rest. A run whose job was killed, by its wall time for instance, therefore ends
  with the elites and the records of a run that was never interrupted, provided that
  the cost of an experiment depends only on its configuration, instance, and seed:
  a budget in evaluations does, a time limit or the load of the machine does not.
  Two experiments of a batch with the same key, the same configuration on the same
  instance under the same seed, run once. A killed job can leave a torn last line,
  which :func:`~dispatchatlas.processes.rows.repair_rows` removes before the resume.
- **The same run.** :func:`~dispatchatlas.processes.calls.run_calls` refuses a resume
  whose stamp, number of workers, or processors differ from those of the first
  invocation on the rows file. The batch adds to the stamp, under ``experiment``, the
  two factories and the stop criteria, so that rows made under another budget are never
  taken for these. The stamp that the caller gives must therefore carry nothing that
  differs between the jobs of one tuning run: no job id, no host name, no time. Keep
  those in a file beside the rows.
- **Failures.** A run that fails (the solver raises, the child dies, the limit
  ``timeout_seconds`` passes) has no row, and the next invocation on the rows file
  retries it. A batch with an experiment that has no row raises
  :class:`ExperimentsFailedError`, which also stands for the stop of the run of calls
  after ``max_failures`` failures. The tuning then ends, the errors are in the sidecar
  of the rows file, and a new job resumes once the cause is fixed. Nothing makes up a
  cost for a run that failed: it would be recorded nowhere, and a resume could not
  reproduce it.
- **Ballast.** With ``ballast=True`` the processors that have no experiment to run
  are kept busy with repeats of the experiments of the batch, whose rows are discarded,
  until the last real experiment ends, so that the time of an experiment is never
  measured on a half-idle node. A calibration job that measures the cost of an
  evaluation passes it; a tuning run need not.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dispatchatlas.core.errors import ConfigError, DispatchAtlasError, SerializationError
from dispatchatlas.core.serialization import canonical_json, digest
from dispatchatlas.core.solver import StopCriteria
from dispatchatlas.processes.calls import Call, CallReport, CallsStoppedError, run_calls
from dispatchatlas.processes.child import locate_callable
from dispatchatlas.processes.rows import canonical_key, read_rows
from dispatchatlas.tuning.racing import Experiment
from dispatchatlas.tuning.space import Configuration

__all__ = [
    "ExperimentsFailedError",
    "ProcessRunBatch",
    "configuration_digest",
    "experiment_key",
    "process_run_batch",
    "run_experiment",
]

_TARGET = f"{__name__}:run_experiment"
_ROW_FIELDS = frozenset(
    {
        "key",
        "configuration",
        "instance",
        "seed",
        "cost",
        "status",
        "evaluations",
        "stop_reason",
        "wall_seconds",
        "first_schedule_digest",
        "repairs",
        "run",
    }
)
"""The fields a row holds itself, which a stamp may not use."""
_RESERVED = "experiment"
"""The field that the stamp of a batch holds for the factories and the stop criteria."""


def configuration_digest(configuration: Configuration) -> str:
    """Return the canonical digest of a configuration: the SHA-256 of its canonical JSON.

    The canonical JSON has sorted keys, no spaces, and the shortest text of each number
    that reads back as the same float, so the digest depends on the names and values of
    the configuration and not on the order of its parameters. A value of another type
    has another text: the integer ``1`` and the float ``1.0`` differ.

    Raises:
        ConfigError: If a value cannot be written as standard JSON, such as a NaN.
    """
    try:
        return digest(dict(configuration))
    except SerializationError as error:
        raise ConfigError(
            f"the configuration {dict(configuration)!r} has no digest: {error}"
        ) from None


def experiment_key(configuration: Configuration, instance: Any, seed: int) -> list[Any]:
    """Return the key of an experiment: ``[configuration digest, instance, seed]``.

    This is the key of the row the experiment writes, so it identifies the rows of a
    configuration on an instance under a seed when a rows file is read.

    Raises:
        ConfigError: If the configuration or the instance cannot be written as JSON.
    """
    key = [configuration_digest(configuration), instance, seed]
    try:
        canonical_key(key)
    except ValueError as error:
        raise ConfigError(f"the instance or the seed cannot be written as JSON: {error}") from None
    return key


def run_experiment(
    *,
    key: Any,
    solver_factory: str,
    problem_factory: str,
    stop: Mapping[str, Any],
    configuration: Mapping[str, Any],
    instance: Any,
    seed: int,
) -> dict[str, Any]:
    """Run one experiment and return its row; the function a child process calls.

    Args:
        key: The key of the experiment (:func:`experiment_key`), returned in the row.
        solver_factory: ``"package.module:function"`` of a function that takes the
            configuration, a dictionary, and returns the solver.
        problem_factory: ``"package.module:function"`` of a function that takes the
            instance and returns the problem.
        stop: The fields of the :class:`~dispatchatlas.core.solver.StopCriteria` of the run.
        configuration: The configuration to run.
        instance: The instance, a JSON value.
        seed: The seed of the run.

    Returns:
        The row, with the fields that the module description lists up to
        ``repairs``. The ``cost`` is ``None`` when the run found no schedule or its
        objective value is infinite.

    Raises:
        ValueError: If the objective value is a NaN or negative infinity, which no
            cost can stand for.
    """
    solver = locate_callable(solver_factory)(dict(configuration))
    problem = locate_callable(problem_factory)(instance)
    result = solver.solve(problem, StopCriteria(**stop), seed)
    value = result.value
    if value is not None and (math.isnan(value) or value == -math.inf):
        raise ValueError(f"the solver reported the objective value {value}, which is no cost")
    return {
        "key": key,
        "configuration": dict(configuration),
        "instance": instance,
        "seed": seed,
        "cost": value if value is not None and math.isfinite(value) else None,
        "status": str(result.status.value),
        "evaluations": result.evaluations,
        "stop_reason": str(result.stop_reason.value),
        "wall_seconds": result.elapsed,
        "first_schedule_digest": result.extras.get("first_schedule_digest"),
        "repairs": result.extras.get("repairs"),
    }


class ExperimentsFailedError(DispatchAtlasError):
    """Experiments of a batch have no row: their runs failed, or the run of calls stopped.

    The rows of the experiments that did run are kept, so a new job resumes from them
    once the cause is fixed.

    Attributes:
        missing: The canonical keys of the experiments that have no row.
        errors: The errors sidecar of the rows file, which says why each run failed.
    """

    def __init__(self, missing: Sequence[str], total: int, errors: Path) -> None:
        super().__init__(
            f"{len(missing)} of {total} experiment(s) of the batch have no row; the errors "
            f"are in {errors}. Fix the cause and run the job again: it resumes from the rows"
        )
        self.missing = tuple(missing)
        self.errors = errors


class ProcessRunBatch:
    """A ``run_batch`` that runs each experiment in a worker process and keeps its row.

    Make one with :func:`process_run_batch`, which documents the arguments, and pass
    it to :meth:`~dispatchatlas.tuning.irace.IteratedRacing.tune`. Called with the
    experiments of a step, it returns their costs in order: the recorded cost of an
    experiment that has a row in the rows file, and the cost of its run for the
    others, which it runs together, as many at a time as there are workers. A run that
    returns no schedule costs ``math.inf``.

    Attributes:
        rows: The rows file.
        experiments: The experiments asked for so far, whether run or recorded.
        executed: The experiments this object ran in worker processes, which wrote rows.
        last_report: The report of its latest run of calls, ``None`` before the first.
    """

    def __init__(
        self,
        rows: Path,
        arguments: Mapping[str, Any],
        stamp: Mapping[str, Any],
        options: Mapping[str, Any],
    ) -> None:
        self.rows = rows
        self.experiments = 0
        self.executed = 0
        self.last_report: CallReport | None = None
        self._arguments = dict(arguments)
        self._stamp = dict(stamp)
        self._options = dict(options)
        self._checked = False

    @property
    def replayed(self) -> int:
        """The experiments answered without a new run: from a row an earlier job wrote."""
        return self.experiments - self.executed

    def __call__(self, experiments: Sequence[Experiment[Any]]) -> list[float]:
        """Return the cost of each experiment, running those that have no row.

        Raises:
            ExperimentsFailedError: If an experiment has no row after its run.
            ConfigError: If an instance or a configuration cannot be written as JSON, or
                :func:`~dispatchatlas.processes.calls.run_calls` refuses the run: the
                rows file is in use, holds a key twice, or was begun by another run.
            TornRowsError: If the rows file ends in a torn line.
        """
        names: list[str] = []
        wanted: dict[str, Call] = {}
        for experiment in experiments:
            key = experiment_key(experiment.configuration, experiment.instance, experiment.seed)
            names.append(canonical_key(key))
            if names[-1] not in wanted:
                wanted[names[-1]] = self._call(key, experiment)
        known = self._costs()
        missing = [call for name, call in wanted.items() if name not in known]
        if missing or (wanted and not self._checked):
            known = self._run(missing, wanted)
        self.experiments += len(experiments)
        return [known[name] for name in names]

    def _call(self, key: list[Any], experiment: Experiment[Any]) -> Call:
        configuration = dict(experiment.configuration)
        return Call(
            key=key,
            target=_TARGET,
            arguments={
                **self._arguments,
                "key": key,
                "configuration": configuration,
                "instance": experiment.instance,
                "seed": experiment.seed,
            },
            label=f"{key[0][:12]} on {experiment.instance!r}, seed {experiment.seed}",
        )

    def _run(self, calls: list[Call], wanted: Mapping[str, Call]) -> dict[str, float]:
        """Run the calls that have no row and return the costs of the rows then."""
        try:
            report = run_calls(calls, self.rows, key_of=_key_of, stamp=self._stamp, **self._options)
        except CallsStoppedError as stopped:
            absent = [name for name in wanted if name not in self._costs()]
            raise ExperimentsFailedError(absent, len(wanted), stopped.report.errors) from stopped
        self._checked = True
        self.last_report = report
        self.executed += report.rows_written
        known = self._costs()
        absent = [name for name in wanted if name not in known]
        if absent:
            raise ExperimentsFailedError(absent, len(wanted), report.errors)
        return known

    def _costs(self) -> dict[str, float]:
        """Return the cost in each row of the rows file, by the canonical text of its key."""
        costs: dict[str, float] = {}
        if self.rows.exists():
            for row in read_rows(self.rows):
                if "key" not in row or "cost" not in row:
                    raise ConfigError(
                        f"a row of {self.rows} has no key or no cost: no experiment's"
                    )
                cost = row["cost"]
                costs[canonical_key(row["key"])] = math.inf if cost is None else float(cost)
        return costs


def _key_of(row: Mapping[str, Any]) -> Any:
    return row["key"]


def process_run_batch(
    solver_factory: str,
    rows: str | Path,
    *,
    problem_factory: str,
    stop: StopCriteria,
    cpus: Sequence[int] | int | str,
    workers: int | None = None,
    pin: bool = True,
    stamp: Mapping[str, Any] | None = None,
    ballast: bool = False,
    ballast_seed: int | None = None,
    allow_changed_run: bool = False,
    max_failures: int = 1,
    timeout_seconds: float | None = None,
    python: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    clean_tokens: Sequence[str] = (),
    log: logging.Logger | None = None,
) -> ProcessRunBatch:
    """Return a ``run_batch`` that runs experiments in worker processes and resumes from rows.

    The module description gives the rows, the resume, and the failures. The batch it
    returns takes the experiments of a step of a race, in order, and returns their
    costs. It runs each experiment that has no row in a child process of its own, in
    the order given, as many at a time as there are ``workers``, and appends its row to
    ``rows``.

    Args:
        solver_factory: ``"package.module:function"``, importable in the child
            processes, of a function that takes a configuration, a dictionary of
            parameter names and values, and returns a solver.
        rows: The rows file, appended to and read first for a resume. It has one
            runner at a time.
        problem_factory: ``"package.module:function"`` of a function that takes an
            instance, as the caller gave it to ``tune``, and returns the problem. An
            instance must be a JSON value: a string, a number, or a list or dictionary
            of them.
        stop: The stop criteria of every run, such as a number of evaluations, which
            keeps the cost of an experiment independent of the speed of the machine.
        cpus: The processors of the run: indices, a mask, or text for
            :func:`~dispatchatlas.processes.affinity.parse_cpus`.
        workers: Worker slots, at most the number of ``cpus``; by default all of them.
        pin: Pin each child to the processor of its slot, as
            :func:`~dispatchatlas.processes.calls.run_calls` does.
        stamp: Constant fields added to every row, such as the revision of the code.
            A resume whose stamp differs from that of the first invocation, in the
            workers, or in the processors is refused unless ``allow_changed_run``, so
            the stamp must carry no job id, host name, or time: a job that resumes
            another's has different ones. The names of the fields of a row, and
            ``experiment``, are taken.
        ballast: Keep the processors that have no experiment to run busy with repeats
            of the experiments of the batch, whose rows are discarded, until the last
            real experiment ends. For a calibration job, so that no cost is measured on
            a node that is half idle.
        ballast_seed: The seed of the order of the repeats; by default one derived from
            the name of the rows file.
        allow_changed_run: Resume although the stamp, the workers, or the processors
            differ from the first invocation's. Rows made under other stop criteria or
            factories are then taken for these, so use it knowingly.
        max_failures: Failed runs after which the batch stops at once; 1 by default,
            since a failed run ends the tuning, and 0 for no limit.
        timeout_seconds: Wall-clock limit of a child; ``None`` for none.
        python: The interpreter of the children; the running one by default.
        environment: Variables added to the environment of the children.
        cwd: Working directory of the children.
        clean_tokens: Words removed from the error messages of the sidecar.
        log: Logger for progress.

    Returns:
        The batch, a :class:`ProcessRunBatch`.

    Raises:
        ConfigError: If a factory is no ``"package.module:function"``, ``stop`` is no
            :class:`~dispatchatlas.core.solver.StopCriteria`, or the stamp cannot be
            written as JSON or uses a name of a row.
    """
    for name, factory in (("solver_factory", solver_factory), ("problem_factory", problem_factory)):
        module, colon, function = str(factory).partition(":")
        if not (isinstance(factory, str) and colon and module and function):
            raise ConfigError(f"{name} is 'package.module:function', not {factory!r}")
    if not isinstance(stop, StopCriteria):
        raise ConfigError(f"stop must be StopCriteria, not {stop!r}")
    fields = dict(stamp or {})
    taken = sorted(set(fields) & (_ROW_FIELDS | {_RESERVED}))
    if taken:
        raise ConfigError(f"the stamp may not use {taken}: a row holds those fields itself")
    arguments = {
        "solver_factory": solver_factory,
        "problem_factory": problem_factory,
        "stop": asdict(stop),
    }
    try:
        canonical_json(fields)
    except SerializationError as error:
        raise ConfigError(f"the stamp cannot be written as JSON: {error}") from None
    options = {
        "cpus": cpus,
        "workers": workers,
        "pin": pin,
        "ballast": ballast,
        "ballast_seed": ballast_seed,
        "allow_changed_run": allow_changed_run,
        "max_failures": max_failures,
        "timeout_seconds": timeout_seconds,
        "python": python,
        "environment": environment,
        "cwd": cwd,
        "clean_tokens": clean_tokens,
        "log": log,
    }
    return ProcessRunBatch(Path(rows), arguments, {**fields, _RESERVED: arguments}, options)
