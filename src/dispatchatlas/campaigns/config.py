"""Campaign configuration: the instances, solvers, protocols, and settings of an experiment.

A :class:`Campaign` states what determines an experiment's results and,
separately, how to execute it:

- ``instances``: one or more instance sources, see :data:`InstanceSource`;
- ``solvers``: registered solver ids with parameter values. A label tells two
  parameter settings of one solver apart;
- ``protocols``: named stop criteria. Every solver runs under every protocol
  with the same criteria, so each protocol is one fair comparison;
- ``replicates``: runs per stochastic solver, problem, and protocol.
  Deterministic solvers run once, unless ``replicate_timed`` gives them every
  replicate under a time limit or their entry states its own count;
- ``seed``: the root of every run seed, and ``seed_namespace``, an optional
  name that separates the seeds of campaigns that share a root;
- ``objectives``: objectives that replace those the problems declare, if given;
- ``execution`` and ``metadata``: how to execute, and free annotations.

The **config hash** identifies an experiment. It is the SHA-256 digest of the
canonical JSON of :meth:`Campaign.identity`: the fields ``seed``,
``instances``, ``solvers``, ``protocols``, ``replicates``, and ``objectives``
as written, so reordering a list also changes it, and each optional setting
that is set to other than its default. Three things stay outside
it. The campaign ``id`` only names where the results are stored.
``execution`` and ``metadata`` do not change what a run computes. And the
locations of instance files and directories are left out of ``instances``:
a problem read from a file is identified by its content, whose digest enters
the id of every run on it (see :mod:`~dispatchatlas.campaigns.planning`), so
a campaign keeps its identity when its files move or live elsewhere on
another machine.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from dispatchatlas.campaigns.processors import (
    allowed_cpus,
    available_cpus,
    describe_cpus,
    pinning_refusal,
)
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.serialization import digest
from dispatchatlas.core.solver import StopCriteria

__all__ = [
    "BACKENDS",
    "CPU_RESERVE",
    "DEFAULT_REPLICATES",
    "DEFAULT_TIMEOUT_FACTOR",
    "GRACE_SECONDS",
    "SCHEMA_VERSION",
    "STARTUP_SECONDS",
    "THREAD_VARIABLES",
    "Campaign",
    "CatalogSource",
    "DirectorySource",
    "Execution",
    "FamilySource",
    "FileSource",
    "InstanceSource",
    "Protocol",
    "SolverSpec",
    "SuiteSource",
    "allowed_cpus",
    "available_cpus",
    "check_name",
    "jsonable",
]

SCHEMA_VERSION = 1
"""Version of the campaign file format this release reads and writes."""

DEFAULT_REPLICATES = 30
"""Runs per stochastic solver, problem, and protocol when a campaign does not say."""

DEFAULT_TIMEOUT_FACTOR = 1.0
"""Default multiple of a protocol's time limit after which a run is killed."""

GRACE_SECONDS = 5.0
"""Default seconds a run may exceed ``timeout_factor * time_limit`` before it is killed.

The grace lets a solver that honours its limit finish the step it is in and
return its result.
"""

STARTUP_SECONDS = 120.0
"""Default seconds a worker process has, from the hand-over of a run, to begin its solve.

They cover the start of the worker's interpreter, the loading of plugins, and
the rebuilding of the solver and the problem, on a machine that may be busy.
"""

CPU_RESERVE = 1
"""Processors that ``workers = "auto"`` leaves to the supervising process and the system."""

BACKENDS = ("sequential", "process")
"""Execution backends: in the calling process, or in supervised worker processes."""

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
"""Environment variables that limit the thread pools of numerical libraries.

Every worker process starts with each of them set to the campaign's
``threads_per_worker``.
"""

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def check_name(value: object, what: str) -> str:
    """Return ``value`` if it can name a file or directory on every platform.

    Campaign ids, solver names, protocol names, and problem ids name files in
    the result store, so they must start with a letter or digit and contain
    only letters, digits, ``.``, ``_``, and ``-``.

    Raises:
        ConfigError: If ``value`` is not such a string.
    """
    if not isinstance(value, str) or not _SAFE_NAME.match(value):
        raise ConfigError(
            f"{what} {value!r} must start with a letter or digit and contain only "
            "letters, digits, '.', '_', and '-'"
        )
    return value


def jsonable(value: Any) -> Any:
    """Return a frozen configuration value as plain JSON types.

    Tables become dictionaries and sequences become lists, recursively.
    """
    if isinstance(value, Mapping):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def _plain(value: Any) -> Any:
    """Return a frozen value with tables as dictionaries and sequences kept as tuples."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    return value


def _frozen(value: Any, where: str) -> Any:
    """Return an immutable copy of a JSON-compatible value."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{where} must be a finite number, got {value!r}")
        return value
    if isinstance(value, Mapping):
        return _frozen_table(value, where)
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item, f"{where}[{i}]") for i, item in enumerate(value))
    raise ConfigError(
        f"{where} must be a string, number, boolean, list, or table, got {type(value).__name__}"
    )


def _frozen_table(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a table")
    items: dict[str, Any] = {}
    for key in value:
        if not isinstance(key, str) or not key:
            raise ConfigError(f"{where} keys must be non-empty strings")
        items[key] = _frozen(value[key], f"{where}.{key}")
    return MappingProxyType(items)


def _check_integer(value: object, what: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{what} must be an integer of at least {minimum}, got {value!r}")
    return value


def _check_affinity(value: object) -> Literal["inherit", "pin"] | tuple[int, ...]:
    if value == "inherit":
        return "inherit"
    if value == "pin":
        return "pin"
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and min(value) >= 0
        and len(set(value)) == len(value)
    ):
        return tuple(value)
    raise ConfigError(
        f"affinity must be 'inherit', 'pin', or a list of distinct processor indices, got {value!r}"
    )


def _check_number(value: object, what: str, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{what} must be a number, got {value!r}")
    if not math.isfinite(value) or value < minimum:
        raise ConfigError(f"{what} must be a finite number of at least {minimum:g}, got {value!r}")
    return float(value)


def _check_text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ConfigError(f"{what} must be a non-empty string without surrounding spaces")
    return value


def _check_path(value: object, what: str) -> Path:
    if isinstance(value, str | os.PathLike) and str(value).strip():
        return Path(value)
    raise ConfigError(f"{what} must be a non-empty path")


def _texts(values: object, what: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ConfigError(f"{what} must be a list of strings")
    items = tuple(_check_text(value, what) for value in values)
    if not items:
        raise ConfigError(f"{what} must not be empty")
    if len({item.lower() for item in items}) != len(items):
        raise ConfigError(f"{what} lists an entry twice")
    return items


# ------------------------------------------------------------------- sources


@dataclass(frozen=True)
class FamilySource:
    """Problems generated by a registered benchmark family, one per seed.

    Attributes:
        family: Id of a family in the benchmark registry, for example
            ``"flow-shop"``.
        seeds: Generator seeds. Each gives one problem, identified by the id
            the family assigns.
        parameters: Family parameters; the others keep their defaults.
    """

    family: str
    seeds: tuple[int, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _check_text(self.family, "the family of a family source")
        given: object = self.seeds
        if isinstance(given, str) or not isinstance(given, Sequence):
            raise ConfigError(f"family source {self.family!r} needs a list of seeds")
        seeds = tuple(_check_integer(seed, "a family seed", 0) for seed in given)
        if not seeds:
            raise ConfigError(f"family source {self.family!r} needs at least one seed")
        if len(set(seeds)) != len(seeds):
            raise ConfigError(f"family source {self.family!r} lists a seed twice")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "parameters", _frozen_table(self.parameters, "parameters"))

    def to_dict(self) -> dict[str, Any]:
        """Return the source as a JSON-ready dictionary."""
        return {
            "family": self.family,
            "parameters": jsonable(self.parameters),
            "seeds": list(self.seeds),
        }

    def identity(self) -> dict[str, Any]:
        """Return the fields of the source that the config hash covers."""
        return {"source": "family", **self.to_dict()}

    def arguments(self) -> dict[str, Any]:
        """Return the parameters as keyword arguments: tables as dictionaries, lists as tuples."""
        return {key: _plain(value) for key, value in self.parameters.items()}


@dataclass(frozen=True)
class SuiteSource:
    """Instances of a published suite from :mod:`dispatchatlas.benchmarks.suites`.

    Attributes:
        suite: Suite id, for example ``"taillard-flow-shop"``.
        instances: Instance names, for example ``("ta001", "ta002")``.
        directory: Directory holding the downloaded files of a file suite.
            Generated suites need none.
    """

    suite: str
    instances: tuple[str, ...]
    directory: Path | None = None

    def __post_init__(self) -> None:
        _check_text(self.suite, "the suite of a suite source")
        object.__setattr__(self, "instances", _texts(self.instances, f"suite {self.suite!r}"))
        if self.directory is not None:
            object.__setattr__(self, "directory", _check_path(self.directory, "directory"))

    def to_dict(self) -> dict[str, Any]:
        """Return the source as a JSON-ready dictionary."""
        return {
            "suite": self.suite,
            "instances": list(self.instances),
            "directory": None if self.directory is None else self.directory.as_posix(),
        }

    def identity(self) -> dict[str, Any]:
        """Return the fields of the source that the config hash covers: not the directory."""
        return {"source": "suite", "suite": self.suite, "instances": list(self.instances)}


@dataclass(frozen=True)
class CatalogSource:
    """Every instance of a built-in catalog, see :func:`~dispatchatlas.benchmarks.build_catalog`.

    Attributes:
        catalog: Catalog name, for example ``"smoke"``.
        seed: Root seed the catalog derives its instance seeds from.
    """

    catalog: str
    seed: int = 0

    def __post_init__(self) -> None:
        _check_text(self.catalog, "the catalog of a catalog source")
        _check_integer(self.seed, "a catalog seed", 0)

    def to_dict(self) -> dict[str, Any]:
        """Return the source as a JSON-ready dictionary."""
        return {"catalog": self.catalog, "seed": self.seed}

    def identity(self) -> dict[str, Any]:
        """Return the fields of the source that the config hash covers."""
        return {"source": "catalog", **self.to_dict()}


@dataclass(frozen=True)
class DirectorySource:
    """Problems written by :func:`~dispatchatlas.benchmarks.materialize`.

    Every file is checked against the SHA-256 digest its manifest records.

    Attributes:
        directory: The directory holding ``manifest.json``.
    """

    directory: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", _check_path(self.directory, "directory"))

    def to_dict(self) -> dict[str, Any]:
        """Return the source as a JSON-ready dictionary."""
        return {"directory": self.directory.as_posix()}

    def identity(self) -> dict[str, Any]:
        """Return the fields of the source that the config hash covers: not the directory."""
        return {"source": "directory"}


@dataclass(frozen=True)
class FileSource:
    """One instance file.

    Attributes:
        file: A DispatchAtlas problem document (``.json``), or an instance in
            a published format read by
            :func:`~dispatchatlas.benchmarks.formats.load_instance`.
        format: Format name for :func:`~dispatchatlas.benchmarks.formats.load_instance`,
            when the extension does not imply it.
    """

    file: Path
    format: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "file", _check_path(self.file, "file"))
        if self.format is not None:
            _check_text(self.format, "format")

    def to_dict(self) -> dict[str, Any]:
        """Return the source as a JSON-ready dictionary."""
        return {"file": self.file.as_posix(), "format": self.format}

    def identity(self) -> dict[str, Any]:
        """Return the fields of the source that the config hash covers: not the path."""
        return {"source": "file", "format": self.format}


type InstanceSource = FamilySource | SuiteSource | CatalogSource | DirectorySource | FileSource
"""Where a campaign's problems come from."""

_SOURCE_TYPES = (FamilySource, SuiteSource, CatalogSource, DirectorySource, FileSource)


# ------------------------------------------------------------------- solvers


@dataclass(frozen=True)
class SolverSpec:
    """A registered solver with parameter values.

    Attributes:
        id: Id in the solver registry, for example ``"simulated-annealing"``.
        label: Name that tells this setting apart from other settings of the
            same solver. Defaults to the id.
        parameters: Solver parameters; the others keep their defaults.
        replicates: Runs of this solver per problem and protocol, whatever
            the solver declares and whatever the protocol bounds. ``None``
            leaves the count to the campaign: its ``replicates`` for a
            stochastic solver, and for one declared deterministic a single
            run, or every replicate under a time limit when the campaign
            sets ``replicate_timed``.
    """

    id: str
    label: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict, hash=False)
    replicates: int | None = None

    def __post_init__(self) -> None:
        _check_text(self.id, "a solver id")
        if self.label is not None:
            check_name(self.label, "solver label")
        check_name(self.name, "solver name")
        object.__setattr__(self, "parameters", _frozen_table(self.parameters, "parameters"))
        if self.replicates is not None:
            _check_integer(self.replicates, f"the replicates of solver {self.name!r}", 1)

    @property
    def name(self) -> str:
        """The label, or the id when there is none. Unique within a campaign."""
        return self.label if self.label is not None else self.id

    def to_dict(self) -> dict[str, Any]:
        """Return the specification as a JSON-ready dictionary.

        ``replicates`` is part of it only when the specification sets one.
        """
        document = {"id": self.id, "label": self.label, "parameters": jsonable(self.parameters)}
        if self.replicates is not None:
            document["replicates"] = self.replicates
        return document

    def arguments(self) -> dict[str, Any]:
        """Return the parameters as keyword arguments: tables as dictionaries, lists as tuples.

        Solver instances built from them can be pickled, which the process
        backend needs.
        """
        return {key: _plain(value) for key, value in self.parameters.items()}


@dataclass(frozen=True)
class Protocol:
    """Named stop criteria that every solver of a campaign runs under.

    Attributes:
        name: Protocol name, used in result file names.
        stop: The criteria. They must bound the run with a time, evaluation,
            or iteration limit, because iterative solvers never stop otherwise.
    """

    name: str
    stop: StopCriteria

    def __post_init__(self) -> None:
        check_name(self.name, "protocol name")
        if not isinstance(self.stop, StopCriteria):
            raise ConfigError(f"protocol {self.name!r} needs StopCriteria")
        if not self.stop.bounded:
            raise ConfigError(
                f"protocol {self.name!r} must set time_limit, max_evaluations, or "
                "max_iterations; without one, iterative solvers never stop"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the protocol as a JSON-ready dictionary."""
        return {"name": self.name, "stop": self.stop.to_dict()}


# ----------------------------------------------------------------- execution


@dataclass(frozen=True)
class Execution:
    """How a campaign executes. None of these settings changes a run's result.

    Attributes:
        backend: ``"process"`` runs each run in one of ``workers`` supervised
            worker processes, which enforces hard time limits and survives a
            crashing solver. ``"sequential"`` runs in the calling process, one
            run at a time, relying on solvers to honour their time limits; it
            suits debugging and solvers that cannot be sent to another process.
        workers: Worker processes, or ``"auto"`` for the processors the
            calling process may use minus :data:`CPU_RESERVE`. Must be 1 for
            the sequential backend.
        timeout_factor: A run whose protocol sets ``time_limit`` is killed
            ``timeout_factor * time_limit + grace`` seconds after its solve
            begins. At least 1. ``None`` disables hard limits.
        grace: Seconds added to the hard limit.
        retries: Further attempts for a run whose failure may be transient.
        record_schedules: Store every run's best schedule in its record. Its
            digest is stored either way.
        affinity: The processors the worker processes run on. ``"inherit"``
            leaves every worker on the processors the calling process may
            use. ``"pin"`` confines worker ``k`` to the ``k``-th of those
            processors, in index order, whichever cores they belong to. A
            list of processor indices confines worker ``k`` to its ``k``-th
            entry. A pinned campaign needs the process backend and a
            processor of its own for every worker.
        threads_per_worker: Threads the numerical libraries of a worker
            process may use. Every variable of :data:`THREAD_VARIABLES` is
            set to it in the environment a worker starts with, for any
            number of workers and whatever the calling environment holds.
            The sequential backend has no worker processes and leaves the
            environment of the calling process as it is, so it accepts only
            the default, which then states nothing about that process: set
            the variables before Python starts to limit such a run.
        startup_limit: Seconds a worker process has, from the hand-over of a
            run, to begin its solve. A worker that takes longer is killed,
            and the attempt is a start-up failure, which ``retries`` covers
            and which is no timeout of the solver. Positive.
    """

    backend: str = "process"
    workers: int | Literal["auto"] = 1
    timeout_factor: float | None = DEFAULT_TIMEOUT_FACTOR
    grace: float = GRACE_SECONDS
    retries: int = 0
    record_schedules: bool = True
    affinity: Literal["inherit", "pin"] | tuple[int, ...] = "inherit"
    threads_per_worker: int = 1
    startup_limit: float = STARTUP_SECONDS

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ConfigError(f"backend must be one of {list(BACKENDS)}, got {self.backend!r}")
        if self.workers != "auto" and (
            isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1
        ):
            raise ConfigError(f"workers must be a positive integer or 'auto', got {self.workers!r}")
        if self.backend == "sequential" and self.workers != 1:
            raise ConfigError(
                "the sequential backend runs one run at a time; use backend 'process' "
                "for several workers"
            )
        if self.timeout_factor is not None:
            object.__setattr__(
                self, "timeout_factor", _check_number(self.timeout_factor, "timeout_factor", 1.0)
            )
        object.__setattr__(self, "grace", _check_number(self.grace, "grace", 0.0))
        startup = _check_number(self.startup_limit, "startup_limit", 0.0)
        if startup <= 0.0:
            raise ConfigError(f"startup_limit must be positive, got {self.startup_limit!r}")
        object.__setattr__(self, "startup_limit", startup)
        _check_integer(self.retries, "retries", 0)
        if not isinstance(self.record_schedules, bool):
            raise ConfigError(
                f"record_schedules must be true or false, got {self.record_schedules!r}"
            )
        object.__setattr__(self, "affinity", _check_affinity(self.affinity))
        if self.backend == "sequential" and self.affinity != "inherit":
            raise ConfigError(
                "the sequential backend runs in the calling process, whose processors the "
                "engine does not change; use affinity 'inherit', or backend 'process'"
            )
        _check_integer(self.threads_per_worker, "threads_per_worker", 1)
        if self.backend == "sequential" and self.threads_per_worker != 1:
            raise ConfigError(
                "the sequential backend runs in the calling process, whose thread limits the "
                "engine does not change; leave threads_per_worker at 1 and set "
                f"{', '.join(THREAD_VARIABLES)} before Python starts, or use backend 'process'"
            )

    @property
    def max_attempts(self) -> int:
        """Attempts a run gets before a failure that may be transient settles it."""
        return 1 + self.retries

    def worker_count(self) -> int:
        """Return the number of workers, resolving ``"auto"`` on this machine."""
        if isinstance(self.workers, str):
            return max(1, available_cpus() - CPU_RESERVE)
        return self.workers

    def worker_cpus(self, workers: int | None = None) -> tuple[int, ...] | None:
        """Return the processor each worker is pinned to, resolved on this machine.

        Args:
            workers: Number of workers. Defaults to :meth:`worker_count`.

        Returns:
            One processor index per worker, in worker order, or ``None`` when
            ``affinity`` is ``"inherit"``.

        Raises:
            ConfigError: If the platform cannot pin a process, the workers
                outnumber the processors they are pinned to, or a listed
                processor is not one the calling process may use.
        """
        if self.affinity == "inherit":
            return None
        count = self.worker_count() if workers is None else workers
        refusal = pinning_refusal()
        if refusal is not None:
            raise ConfigError(f"workers cannot be pinned: {refusal}; use affinity 'inherit'")
        allowed = allowed_cpus()
        chosen: tuple[int, ...]
        if isinstance(self.affinity, str):
            chosen = allowed
            if count > len(chosen):
                raise ConfigError(
                    f"affinity 'pin' gives every worker a processor of its own, but {count} "
                    f"workers exceed the {len(chosen)} processor(s) this process may use "
                    f"({describe_cpus(chosen)})"
                )
        else:
            chosen = self.affinity
            foreign = [index for index in chosen if index not in allowed]
            if foreign:
                raise ConfigError(
                    f"affinity lists processor(s) {describe_cpus(foreign)}, which this process "
                    f"may not use; it may use {describe_cpus(allowed)}"
                )
            if count > len(chosen):
                raise ConfigError(
                    f"affinity lists {len(chosen)} processor(s) for {count} workers; every "
                    "worker needs a processor of its own"
                )
        return chosen[:count]

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a JSON-ready dictionary."""
        return {
            "backend": self.backend,
            "workers": self.workers,
            "timeout_factor": self.timeout_factor,
            "grace": self.grace,
            "retries": self.retries,
            "record_schedules": self.record_schedules,
            "affinity": self.affinity if isinstance(self.affinity, str) else list(self.affinity),
            "threads_per_worker": self.threads_per_worker,
            "startup_limit": self.startup_limit,
        }


# ------------------------------------------------------------------ campaign


@dataclass(frozen=True)
class Campaign:
    """A reproducible experiment: problems, solvers, protocols, and replicates.

    Attributes:
        id: Campaign name. Results go to ``<root>/<id>/``.
        seed: Root seed of every run seed.
        instances: Where the problems come from.
        solvers: The solvers to compare, with unique names.
        protocols: Stop criteria every solver runs under, with unique names.
        replicates: Runs per stochastic solver, problem, and protocol.
        objectives: Objectives, primary first, that replace those every
            problem declares. ``None`` keeps the problems' own.
        execution: How to execute. Not part of the config hash.
        metadata: Free annotations. Not part of the config hash.
        base: Directory that relative paths in instance sources refer to;
            the directory of the campaign file when read by
            :func:`~dispatchatlas.campaigns.io.load_campaign`, else the
            working directory. Not part of the config hash.
        replicate_timed: Whether a solver declared deterministic runs
            ``replicates`` times, like a stochastic one, under every protocol
            that sets a time limit. Where the clock ends such a run, its
            result depends on the speed and load of the machine, and a single
            run does not show by how much.
        seed_namespace: A name that separates the seeds of this campaign
            from those of every campaign with the same ``seed`` and another
            namespace, or none. ``None`` leaves every seed as the campaign
            seed alone gives it.
    """

    id: str
    seed: int
    instances: tuple[InstanceSource, ...]
    solvers: tuple[SolverSpec, ...]
    protocols: tuple[Protocol, ...]
    replicates: int = DEFAULT_REPLICATES
    objectives: tuple[str, ...] | None = None
    execution: Execution = field(default_factory=Execution)
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)
    base: Path | None = None
    replicate_timed: bool = False
    seed_namespace: str | None = None

    def __post_init__(self) -> None:
        check_name(self.id, "campaign id")
        _check_integer(self.seed, "seed", 0)
        _check_integer(self.replicates, "replicates", 1)
        if not isinstance(self.replicate_timed, bool):
            raise ConfigError(
                f"replicate_timed must be true or false, got {self.replicate_timed!r}"
            )
        if self.seed_namespace is not None:
            _check_text(self.seed_namespace, "seed_namespace")
        instances = tuple(self.instances)
        if not instances:
            raise ConfigError(f"campaign {self.id!r} needs at least one instance source")
        for source in instances:
            if not isinstance(source, _SOURCE_TYPES):
                raise ConfigError(f"campaign {self.id!r} has an invalid instance source {source!r}")
        object.__setattr__(self, "instances", instances)
        object.__setattr__(self, "solvers", _unique(self.solvers, SolverSpec, "solver"))
        object.__setattr__(self, "protocols", _unique(self.protocols, Protocol, "protocol"))
        if self.objectives is not None:
            object.__setattr__(self, "objectives", _texts(self.objectives, "objectives"))
        if not isinstance(self.execution, Execution):
            raise ConfigError(f"campaign {self.id!r} needs Execution settings")
        object.__setattr__(self, "metadata", _frozen_table(self.metadata, "metadata"))
        if self.base is not None:
            object.__setattr__(self, "base", _check_path(self.base, "base"))

    def identity(self) -> dict[str, Any]:
        """Return the fields that determine the results, as JSON-ready values.

        These are exactly the fields the config hash covers: ``seed``,
        ``instances`` without file-system locations, ``solvers``,
        ``protocols``, ``replicates``, and ``objectives``, and the optional
        settings that are set: ``replicate_timed`` when it is true,
        ``seed_namespace`` when there is one, a solver's own ``replicates``,
        and a protocol's ``schedule`` when it is not the default. A setting
        left at its default is absent, so a campaign that sets none of them
        has the config hash it has without them.
        """
        identity: dict[str, Any] = {
            "seed": self.seed,
            "instances": [source.identity() for source in self.instances],
            "solvers": [solver.to_dict() for solver in self.solvers],
            "protocols": [protocol.to_dict() for protocol in self.protocols],
            "replicates": self.replicates,
            "objectives": None if self.objectives is None else list(self.objectives),
        }
        if self.replicate_timed:
            identity["replicate_timed"] = True
        if self.seed_namespace is not None:
            identity["seed_namespace"] = self.seed_namespace
        return identity

    @cached_property
    def config_hash(self) -> str:
        """SHA-256 hex digest of the canonical JSON of :meth:`identity`."""
        return digest(self.identity())

    def solver(self, name: str) -> SolverSpec:
        """Return the solver with the given name.

        Raises:
            ConfigError: If no solver has that name.
        """
        for solver in self.solvers:
            if solver.name == name:
                return solver
        raise ConfigError(f"campaign {self.id!r} has no solver named {name!r}")

    def protocol(self, name: str) -> Protocol:
        """Return the protocol with the given name.

        Raises:
            ConfigError: If no protocol has that name.
        """
        for protocol in self.protocols:
            if protocol.name == name:
                return protocol
        raise ConfigError(f"campaign {self.id!r} has no protocol named {name!r}")

    def resolve(self, path: Path) -> Path:
        """Return ``path`` resolved against :attr:`base` when it is relative."""
        if path.is_absolute() or self.base is None:
            return path
        return self.base / path

    def with_execution(self, **changes: Any) -> Campaign:
        """Return a copy with some execution settings changed.

        The copy has the same config hash, so it continues the same results.
        """
        return replace(self, execution=replace(self.execution, **changes))


def _unique[T: (SolverSpec, Protocol)](items: object, kind: type[T], what: str) -> tuple[T, ...]:
    if isinstance(items, str) or not isinstance(items, Sequence):
        raise ConfigError(f"the {what}s must be a list")
    values = tuple(items)
    if not values:
        raise ConfigError(f"a campaign needs at least one {what}")
    names: set[str] = set()
    for item in values:
        if not isinstance(item, kind):
            raise ConfigError(f"invalid {what} {item!r}")
        # Names become directory and file names, which may be case-insensitive.
        key = item.name.lower()
        if key in names:
            raise ConfigError(f"two {what}s are named {item.name!r}; give each a unique name")
        names.add(key)
    return values
