"""Planning: from a campaign to the runs it needs.

:func:`plan` loads every problem the campaign's sources name, checks every
solver id and parameter set against the registry, and expands the grid
problem x replicate x solver x protocol into :class:`PlannedRun` items:

- **Replicates.** A stochastic solver runs ``campaign.replicates`` times per
  problem and protocol, as replicates ``0, 1, ...``. A deterministic one runs
  once, as replicate 0, unless the campaign sets ``replicate_timed``: it then
  runs every replicate under each protocol that sets a time limit, where its
  result depends on the machine. A solver whose entry carries its own
  ``replicates`` runs that many times under every protocol, whatever it
  declares.
- **Seeds.** Seeds are common random numbers: replicate ``r`` of problem
  ``p`` receives ``derive_seed(campaign.seed, "problem:" + p, r)`` under every
  solver and protocol. Results are therefore paired across solvers, and
  adding, removing, or reordering solvers never changes a seed. A campaign
  with a ``seed_namespace`` derives them from the root
  ``derive_seed(campaign.seed, "namespace:" + namespace)`` in place of its
  seed, so two campaigns that share a seed and differ in namespace share no
  run seed.
- **Run ids.** ``"run-"`` and the first 16 hexadecimal digits of the SHA-256
  digest of the canonical JSON of the config hash, the problem id and digest,
  the solver id, label, and parameters, the protocol name, and the replicate.
  Equal campaigns give equal run ids on every machine, and a problem whose
  content changes gets new ones.
- **Compatibility.** A pair whose solver does not model a feature of the
  problem is excluded, and the plan records why (see
  :mod:`~dispatchatlas.campaigns.compatibility`).

The problem digest is that of the problem as solved, after any objective
override. :func:`~dispatchatlas.campaigns.fairness.assess` verifies every plan
before it is returned.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from dispatchatlas.benchmarks.catalog import build_catalog, load_directory
from dispatchatlas.benchmarks.formats import load_instance
from dispatchatlas.benchmarks.registry import FamilyRegistry
from dispatchatlas.benchmarks.registry import default_registry as default_families
from dispatchatlas.benchmarks.suites import suite as find_suite
from dispatchatlas.campaigns import compatibility, fairness
from dispatchatlas.campaigns.compatibility import Caveat, Compatibility, Exclusion
from dispatchatlas.campaigns.config import (
    Campaign,
    CatalogSource,
    DirectorySource,
    FamilySource,
    InstanceSource,
    Protocol,
    SolverSpec,
    SuiteSource,
    check_name,
    jsonable,
)
from dispatchatlas.campaigns.records import SolverEntry, protocol_from_dict
from dispatchatlas.core.errors import ConfigError, DispatchAtlasError, SerializationError
from dispatchatlas.core.logs import get_logger
from dispatchatlas.core.model import Problem
from dispatchatlas.core.seeds import derive_seed
from dispatchatlas.core.serialization import digest, load_problem, problem_digest
from dispatchatlas.solvers.registry import SolverRegistry
from dispatchatlas.solvers.registry import default_registry as default_solvers

__all__ = [
    "PLAN_SCHEMA",
    "Plan",
    "PlannedProblem",
    "PlannedRun",
    "Prepared",
    "check_compatibility",
    "plan",
    "prepare",
    "run_id",
    "run_seed",
]

PLAN_SCHEMA = "dispatchatlas.plan/1"

_LOG = get_logger(__name__)


def run_seed(root: int, problem_id: str, replicate: int, namespace: str | None = None) -> int:
    """Return the seed of every run of one problem and replicate.

    Args:
        root: The campaign seed.
        problem_id: Id of the problem.
        replicate: Replicate number, from 0.
        namespace: The campaign's seed namespace, or ``None`` without one.

    Returns:
        ``derive_seed(root, "problem:" + problem_id, replicate)``. With a
        namespace, ``derive_seed(root, "namespace:" + namespace)`` takes the
        place of ``root``.
    """
    if namespace is not None:
        root = derive_seed(root, f"namespace:{namespace}")
    return derive_seed(root, f"problem:{problem_id}", replicate)


def run_id(
    config_hash: str,
    problem_id: str,
    problem_hash: str,
    solver: SolverSpec,
    protocol: str,
    replicate: int,
) -> str:
    """Return the id of one run.

    Args:
        config_hash: Config hash of the campaign.
        problem_id: Id of the problem.
        problem_hash: Digest of the problem as solved.
        solver: The solver specification.
        protocol: Name of the protocol.
        replicate: Replicate number.

    Returns:
        ``"run-"`` and 16 hexadecimal digits.
    """
    identity = {
        "config_hash": config_hash,
        "problem": problem_id,
        "problem_digest": problem_hash,
        "solver": solver.id,
        "label": solver.label,
        "parameters": jsonable(solver.parameters),
        "protocol": protocol,
        "replicate": replicate,
    }
    return "run-" + digest(identity)[:16]


@dataclass(frozen=True)
class PlannedProblem:
    """A problem of a plan.

    Attributes:
        id: Problem id.
        family: Benchmark family, if known.
        digest: Digest of the problem as solved.
        source: Index of the instance source that produced it.
        tasks: Number of tasks.
        resources: Number of resources.
        objectives: Objectives it is solved for, primary first.
        features: Its structural features, sorted.
    """

    id: str
    family: str | None
    digest: str
    source: int
    tasks: int
    resources: int
    objectives: tuple[str, ...]
    features: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the entry as a JSON-ready dictionary."""
        return {
            "id": self.id,
            "family": self.family,
            "digest": self.digest,
            "source": self.source,
            "tasks": self.tasks,
            "resources": self.resources,
            "objectives": list(self.objectives),
            "features": list(self.features),
        }


@dataclass(frozen=True)
class PlannedRun:
    """One run of a plan: a solver on a problem under a protocol.

    Attributes:
        run_id: Run id.
        problem: Problem id.
        solver: Solver name.
        protocol: Protocol name.
        replicate: Replicate number, from 0.
        seed: Seed the solver receives.
    """

    run_id: str
    problem: str
    solver: str
    protocol: str
    replicate: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return the run as a JSON-ready dictionary."""
        return {
            "run_id": self.run_id,
            "problem": self.problem,
            "solver": self.solver,
            "protocol": self.protocol,
            "replicate": self.replicate,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class Plan:
    """Every run a campaign needs, with the facts that explain it.

    A plan is plain data: the store writes it to ``plan.json`` and
    :func:`~dispatchatlas.campaigns.store.read_plan` reads it back.

    Attributes:
        campaign: Campaign id.
        config_hash: Config hash of the campaign.
        seed: Campaign seed.
        replicates: Replicates of stochastic solvers.
        problems: The problems, in source order.
        solvers: The solvers, in campaign order.
        protocols: The protocols, in campaign order.
        runs: The runs, ordered by problem, replicate, solver, and protocol.
        exclusions: Pairs that do not run, and why.
        caveats: Pairings that run but need care in the analysis.
        replicate_timed: Whether a solver declared deterministic runs every
            replicate under a protocol that sets a time limit.
        seed_namespace: The campaign's seed namespace, or ``None``.
    """

    campaign: str
    config_hash: str
    seed: int
    replicates: int
    problems: tuple[PlannedProblem, ...]
    solvers: tuple[SolverEntry, ...]
    protocols: tuple[Protocol, ...]
    runs: tuple[PlannedRun, ...]
    exclusions: tuple[Exclusion, ...] = ()
    caveats: tuple[Caveat, ...] = ()
    replicate_timed: bool = False
    seed_namespace: str | None = None

    @cached_property
    def _runs(self) -> Mapping[str, PlannedRun]:
        return {run.run_id: run for run in self.runs}

    @cached_property
    def runs_digest(self) -> str:
        """SHA-256 hex digest of the canonical JSON of every run's id and seed, sorted by id.

        Two plans have the same digest exactly when they hold the same runs
        with the same seeds, in whatever order. A run id covers the config
        hash, the problem and its content, the solver and its parameters, the
        protocol, and the replicate; the seed is derived by the library that
        plans. The digest does not cover the solver entries: the metadata a
        solver declares can differ between two plans of one digest, and
        :meth:`Store.plan_change
        <dispatchatlas.campaigns.store.Store.plan_change>` names such a
        difference.
        """
        return digest(sorted([run.run_id, run.seed] for run in self.runs))

    def run(self, run_id: str) -> PlannedRun:
        """Return the run with the given id.

        Raises:
            KeyError: If the plan has no such run.
        """
        return self._runs[run_id]

    def has_run(self, run_id: str) -> bool:
        """Return whether the plan contains a run with the given id."""
        return run_id in self._runs

    def problem(self, problem_id: str) -> PlannedProblem:
        """Return the problem with the given id.

        Raises:
            KeyError: If the plan has no such problem.
        """
        for item in self.problems:
            if item.id == problem_id:
                return item
        raise KeyError(problem_id)

    def solver(self, name: str) -> SolverEntry:
        """Return the solver with the given name.

        Raises:
            KeyError: If the plan has no such solver.
        """
        for item in self.solvers:
            if item.name == name:
                return item
        raise KeyError(name)

    def protocol(self, name: str) -> Protocol:
        """Return the protocol with the given name.

        Raises:
            KeyError: If the plan has no such protocol.
        """
        for item in self.protocols:
            if item.name == name:
                return item
        raise KeyError(name)

    def replicates_of(self, solver: SolverEntry, protocol: Protocol | None = None) -> int:
        """Return how many replicates a solver runs under a protocol.

        The solver's own count when its entry carries one. Otherwise every
        replicate for a stochastic solver, and for one declared deterministic
        a single run, or every replicate when :attr:`replicate_timed` is set
        and the protocol sets a time limit.

        Args:
            solver: A solver of the plan.
            protocol: A protocol of the plan. ``None`` asks for the count
                under a protocol without a time limit.
        """
        return _replicates(solver, protocol, self.replicates, self.replicate_timed)

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as a JSON-ready document.

        Besides the plan's fields, the document lists every protocol's
        parities, the fair-comparison assessment, and :attr:`runs_digest`,
        derived from the plan for readers of the file.
        """
        assessment = fairness.assess(self)
        return {
            "schema": PLAN_SCHEMA,
            "campaign": self.campaign,
            "config_hash": self.config_hash,
            "runs_digest": self.runs_digest,
            "seed": self.seed,
            "seed_namespace": self.seed_namespace,
            "replicates": self.replicates,
            "replicate_timed": self.replicate_timed,
            "problems": [problem.to_dict() for problem in self.problems],
            "solvers": [solver.to_dict() for solver in self.solvers],
            "protocols": [
                {**protocol.to_dict(), "parities": list(fairness.parities(protocol.stop))}
                for protocol in self.protocols
            ],
            "runs": [run.to_dict() for run in self.runs],
            "exclusions": [item.to_dict() for item in self.exclusions],
            "caveats": [item.to_dict() for item in self.caveats],
            "fairness": assessment.to_dict(),
        }

    @classmethod
    def from_dict(cls, document: object) -> Plan:
        """Build a plan from :meth:`to_dict` output. Derived entries are ignored.

        A document without ``replicate_timed`` or ``seed_namespace`` reads as
        a plan that sets neither.

        Raises:
            SerializationError: If the document is not a plan of this schema
                version, or is malformed.
        """
        if not isinstance(document, Mapping) or document.get("schema") != PLAN_SCHEMA:
            raise SerializationError(f"expected a {PLAN_SCHEMA!r} document")
        replicate_timed = document.get("replicate_timed", False)
        namespace = document.get("seed_namespace")
        if not isinstance(replicate_timed, bool) or not isinstance(namespace, str | None):
            raise SerializationError(
                "malformed plan document: replicate_timed must be true or false and "
                "seed_namespace a string or null"
            )
        try:
            return cls(
                replicate_timed=replicate_timed,
                seed_namespace=namespace,
                campaign=str(document["campaign"]),
                config_hash=str(document["config_hash"]),
                seed=int(document["seed"]),
                replicates=int(document["replicates"]),
                problems=tuple(
                    PlannedProblem(
                        id=item["id"],
                        family=item["family"],
                        digest=item["digest"],
                        source=int(item["source"]),
                        tasks=int(item["tasks"]),
                        resources=int(item["resources"]),
                        objectives=tuple(item["objectives"]),
                        features=tuple(item["features"]),
                    )
                    for item in document["problems"]
                ),
                solvers=tuple(SolverEntry.from_dict(item) for item in document["solvers"]),
                protocols=tuple(
                    protocol_from_dict({"name": item["name"], "stop": item["stop"]})
                    for item in document["protocols"]
                ),
                runs=tuple(
                    PlannedRun(
                        run_id=item["run_id"],
                        problem=item["problem"],
                        solver=item["solver"],
                        protocol=item["protocol"],
                        replicate=int(item["replicate"]),
                        seed=int(item["seed"]),
                    )
                    for item in document["runs"]
                ),
                exclusions=tuple(Exclusion.from_dict(item) for item in document["exclusions"]),
                caveats=tuple(Caveat.from_dict(item) for item in document["caveats"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SerializationError(f"malformed plan document: {error!r}") from error


@dataclass(frozen=True)
class Prepared:
    """A plan together with what executing it needs.

    Attributes:
        campaign: The campaign.
        plan: Its plan.
        problems: The problems as solved, by id.
        registry: The solver registry the plan was checked against.
    """

    campaign: Campaign
    plan: Plan
    problems: Mapping[str, Problem]
    registry: SolverRegistry


def plan(
    campaign: Campaign,
    *,
    registry: SolverRegistry | None = None,
    families: FamilyRegistry | None = None,
) -> Plan:
    """Plan a campaign without running anything.

    Args:
        campaign: The campaign.
        registry: Solver registry. Defaults to the built-in solvers and plugins.
        families: Benchmark-family registry for family and catalog sources.
            Defaults to the built-in families and plugins.

    Returns:
        The verified plan.

    Raises:
        ConfigError: If a source cannot be loaded, a solver id or parameter
            is unknown, problem ids clash, or no run remains.
    """
    return prepare(campaign, registry=registry, families=families).plan


def prepare(
    campaign: Campaign,
    *,
    registry: SolverRegistry | None = None,
    families: FamilyRegistry | None = None,
) -> Prepared:
    """Plan a campaign and keep the loaded problems and the registry for execution.

    Takes the same arguments and raises the same errors as :func:`plan`.
    """
    solvers = registry if registry is not None else default_solvers()
    entries = tuple(_solver_entry(spec, solvers) for spec in campaign.solvers)
    loaded = _load_problems(campaign, families)
    problems = {problem.id: problem for _, problem in loaded}
    assessment = compatibility.assess(
        list(problems.values()),
        [(spec.name, solvers.info(spec.id)) for spec in campaign.solvers],
    )
    config_hash = campaign.config_hash
    counts = {
        (entry.name, protocol.name): _replicates(
            entry, protocol, campaign.replicates, campaign.replicate_timed
        )
        for entry in entries
        for protocol in campaign.protocols
    }
    planned_problems: list[PlannedProblem] = []
    runs: list[PlannedRun] = []
    for source, problem in loaded:
        problem_hash = problem_digest(problem)
        planned_problems.append(
            PlannedProblem(
                id=problem.id,
                family=problem.family,
                digest=problem_hash,
                source=source,
                tasks=len(problem.tasks),
                resources=len(problem.resources),
                objectives=problem.objectives,
                features=tuple(sorted(problem.features)),
            )
        )
        for replicate in range(max(counts.values())):
            seed = run_seed(campaign.seed, problem.id, replicate, campaign.seed_namespace)
            for spec, entry in zip(campaign.solvers, entries, strict=True):
                if assessment.excludes(problem.id, spec.name):
                    continue
                for protocol in campaign.protocols:
                    if replicate >= counts[entry.name, protocol.name]:
                        continue
                    identifier = run_id(
                        config_hash, problem.id, problem_hash, spec, protocol.name, replicate
                    )
                    runs.append(
                        PlannedRun(
                            identifier, problem.id, spec.name, protocol.name, replicate, seed
                        )
                    )
    result = Plan(
        campaign=campaign.id,
        config_hash=config_hash,
        seed=campaign.seed,
        replicates=campaign.replicates,
        problems=tuple(planned_problems),
        solvers=entries,
        protocols=campaign.protocols,
        runs=tuple(runs),
        exclusions=assessment.exclusions,
        caveats=assessment.caveats,
        replicate_timed=campaign.replicate_timed,
        seed_namespace=campaign.seed_namespace,
    )
    if not runs:
        raise ConfigError(
            f"campaign {campaign.id!r} plans no runs: every solver is excluded from every problem"
        )
    verdict = fairness.assess(result)
    if verdict.issues:
        raise ConfigError("the plan is not a fair comparison: " + "; ".join(verdict.issues))
    _report(result)
    for note in verdict.notes:
        _LOG.warning("campaign %s: %s", campaign.id, note)
    return Prepared(campaign, result, problems, solvers)


def check_compatibility(
    campaign: Campaign,
    *,
    registry: SolverRegistry | None = None,
    families: FamilyRegistry | None = None,
) -> Compatibility:
    """Load a campaign's problems and report which solvers each one excludes.

    Unlike :func:`plan`, it reports rather than fails when every pair is
    excluded.

    Args:
        campaign: The campaign.
        registry: Solver registry. Defaults to the built-in solvers and plugins.
        families: Benchmark-family registry. Defaults to the built-in families
            and plugins.

    Raises:
        ConfigError: If a source cannot be loaded or a solver id is unknown.
    """
    solvers = registry if registry is not None else default_solvers()
    infos = [(spec.name, solvers.info(spec.id)) for spec in campaign.solvers]
    problems = [problem for _, problem in _load_problems(campaign, families)]
    return compatibility.assess(problems, infos)


# ------------------------------------------------------------------- helpers


def _solver_entry(spec: SolverSpec, registry: SolverRegistry) -> SolverEntry:
    """Check a solver specification against the registry and describe it."""
    try:
        info = registry.info(spec.id)
        threads = getattr(registry.create(spec.id, **spec.arguments()), "threads", None)
    except ConfigError as error:
        raise ConfigError(f"solver {spec.name!r}: {error}") from error
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        threads = None
    return SolverEntry(
        name=spec.name,
        id=spec.id,
        label=spec.label,
        parameters=jsonable(spec.parameters),
        family=info.family,
        stochastic=info.stochastic,
        iterative=info.iterative,
        multi_objective=info.multi_objective,
        threads=threads,
        replicates=spec.replicates,
    )


def _replicates(
    solver: SolverEntry, protocol: Protocol | None, replicates: int, replicate_timed: bool
) -> int:
    """Return how many replicates a solver runs under a protocol.

    Args:
        solver: The solver.
        protocol: The protocol, or ``None`` for one without a time limit.
        replicates: The campaign's replicates.
        replicate_timed: Whether a solver declared deterministic runs them
            all under a time limit.
    """
    if solver.replicates is not None:
        return solver.replicates
    timed = protocol is not None and protocol.stop.time_limit is not None
    if solver.stochastic or (replicate_timed and timed):
        return replicates
    return 1


def _load_problems(
    campaign: Campaign, families: FamilyRegistry | None
) -> list[tuple[int, Problem]]:
    """Return every problem of the campaign with the index of its source."""
    registry = families if families is not None else default_families()
    loaded: list[tuple[int, Problem]] = []
    for index, source in enumerate(campaign.instances):
        try:
            problems = _read_source(source, campaign, registry)
            if campaign.objectives is not None:
                problems = [problem.with_objectives(*campaign.objectives) for problem in problems]
        except (DispatchAtlasError, OSError) as error:
            raise ConfigError(f"instances[{index}] ({_describe(source)}): {error}") from error
        loaded.extend((index, problem) for problem in problems)
    # Problem ids name directories of the result store, which may ignore case.
    seen: dict[str, int] = {}
    for index, problem in loaded:
        try:
            check_name(problem.id, "problem id")
        except ConfigError as error:
            raise ConfigError(f"instances[{index}]: {error}") from error
        key = problem.id.lower()
        if key in seen:
            raise ConfigError(
                f"problem id {problem.id!r} occurs more than once (instances[{seen[key]}] and "
                f"instances[{index}]); problem ids must be unique, ignoring case"
            )
        seen[key] = index
    return loaded


def _read_source(
    source: InstanceSource, campaign: Campaign, families: FamilyRegistry
) -> list[Problem]:
    if isinstance(source, FamilySource):
        generator = families.create(source.family, **source.arguments())
        return [generator.generate(seed) for seed in source.seeds]
    if isinstance(source, SuiteSource):
        directory = None if source.directory is None else campaign.resolve(source.directory)
        found = find_suite(source.suite)
        return [found.load(name, directory) for name in source.instances]
    if isinstance(source, CatalogSource):
        return build_catalog(source.catalog, seed=source.seed, registry=families)
    if isinstance(source, DirectorySource):
        return load_directory(campaign.resolve(source.directory), verify=True)
    path = campaign.resolve(source.file)
    if source.format is None and path.suffix.lower() == ".json":
        return [load_problem(path)]
    return [load_instance(path, format=source.format)]


def _describe(source: InstanceSource) -> str:
    if isinstance(source, FamilySource):
        return f"family {source.family!r}"
    if isinstance(source, SuiteSource):
        return f"suite {source.suite!r}"
    if isinstance(source, CatalogSource):
        return f"catalog {source.catalog!r}"
    if isinstance(source, DirectorySource):
        return f"directory {source.directory.as_posix()!r}"
    return f"file {source.file.as_posix()!r}"


def _report(result: Plan) -> None:
    """Log the plan's size, exclusions, and caveats."""
    _LOG.info(
        "campaign %s: %d runs over %d problems, %d solvers, %d protocols",
        result.campaign,
        len(result.runs),
        len(result.problems),
        len(result.solvers),
        len(result.protocols),
    )
    by_solver: dict[str, list[Exclusion]] = {}
    for item in result.exclusions:
        by_solver.setdefault(item.solver, []).append(item)
    for solver, items in by_solver.items():
        features = sorted({feature for item in items for feature in item.features})
        _LOG.warning(
            "solver %s is excluded from %d problem(s) because it does not model %s",
            solver,
            len(items),
            ", ".join(features),
        )
    for caveat in result.caveats:
        _LOG.info(
            "caveat for %s on %d problem(s): %s",
            caveat.solver,
            len(caveat.problems),
            caveat.message,
        )
