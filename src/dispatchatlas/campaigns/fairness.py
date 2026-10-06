"""Fair comparison: the same budget, problems, and seeds for every solver, verified.

Solvers are compared fairly when, under each protocol, every one of them runs
with identical stop criteria on the same problems with the same seeds. The
planner builds runs that way; :func:`assess` checks it on the finished plan
instead of assuming it, so the claim can also be re-checked from a stored
``plan.json``:

- every protocol bounds the run with a time, evaluation, or iteration limit;
- every run refers to one of the plan's protocols, so all runs under a
  protocol share its stop criteria;
- every solver has its full set of replicates under every protocol on every
  problem it is not excluded from (exclusions are recorded with their reason,
  see :mod:`~dispatchatlas.campaigns.compatibility`);
- all runs of one problem and replicate share one seed, so results are paired
  across solvers and protocols.

A protocol's **parities** name the budgets it equalises, read off its stop
criteria: ``equal-wall-clock`` (``time_limit``), ``equal-evaluations``
(``max_evaluations``), ``equal-iterations`` (``max_iterations``), and
``fixed-target`` (``target``). Wall-clock budgets depend on the speed and load
of the machine, and runs they stop cannot be replayed; an iteration means a
different amount of work in different solvers.

The assessment also carries **notes**: facts that break none of these rules
but that a reader of a comparison needs (:func:`notes`). A note never refuses
a plan. Both concern protocols with a time limit:

- solvers whose ``threads`` parameters differ are noted, because equal
  wall-clock budgets then buy unequal processor time. Each solver counts with
  the value its plan entry records, which is the one its instance holds, the
  default included; a solver without such a parameter counts as one thread;
- solvers declared deterministic that run once are noted when the campaign
  has more replicates, because the result of a run that the clock ends
  depends on the speed and load of the machine and one run does not show by
  how much. The campaign key ``replicate_timed`` gives them every replicate,
  and a solver whose entry states its own count is not noted.

A plan says what should run. :func:`assess_outcomes` compares the records of
the completed runs with it, protocol by protocol: for which reasons the runs
of the iterative solvers ended, since runs of one protocol that ended for
different reasons did not spend the same budget, and which solvers' runs took
more wall time than the protocol's time limit allows.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dispatchatlas.core.errors import ConfigError

if TYPE_CHECKING:
    from dispatchatlas.campaigns.config import Protocol
    from dispatchatlas.campaigns.planning import Plan
    from dispatchatlas.campaigns.records import RunRecord, SolverEntry
    from dispatchatlas.core.solver import StopCriteria

__all__ = [
    "EQUAL_EVALUATIONS",
    "EQUAL_ITERATIONS",
    "EQUAL_WALL_CLOCK",
    "FIXED_TARGET",
    "OVERRUN_TOLERANCE",
    "PARITIES",
    "Fairness",
    "Outcomes",
    "Overrun",
    "ProtocolOutcomes",
    "assess",
    "assess_outcomes",
    "notes",
    "parities",
]

EQUAL_WALL_CLOCK = "equal-wall-clock"
EQUAL_EVALUATIONS = "equal-evaluations"
EQUAL_ITERATIONS = "equal-iterations"
FIXED_TARGET = "fixed-target"
PARITIES = (EQUAL_WALL_CLOCK, EQUAL_EVALUATIONS, EQUAL_ITERATIONS, FIXED_TARGET)
"""Every parity name, in report order."""

OVERRUN_TOLERANCE = 0.05
"""Default share of a time limit by which a completed run may exceed it unreported."""

_SHOWN = 3
"""Examples an issue lists before summarising the rest."""


def parities(stop: StopCriteria) -> tuple[str, ...]:
    """Return the parities that stop criteria realise, in :data:`PARITIES` order."""
    found: list[str] = []
    if stop.time_limit is not None:
        found.append(EQUAL_WALL_CLOCK)
    if stop.max_evaluations is not None:
        found.append(EQUAL_EVALUATIONS)
    if stop.max_iterations is not None:
        found.append(EQUAL_ITERATIONS)
    if stop.target is not None:
        found.append(FIXED_TARGET)
    return tuple(found)


@dataclass(frozen=True)
class Fairness:
    """The fair-comparison assessment of a plan.

    Attributes:
        solvers: Names of the solvers compared.
        protocols: Parities of each protocol, by protocol name.
        paired: Whether every problem and replicate has one seed.
        issues: Every violation found. Empty for a fair plan.
        notes: Facts that violate nothing but bear on a comparison. They
            leave a plan fair.
    """

    solvers: tuple[str, ...]
    protocols: Mapping[str, tuple[str, ...]]
    paired: bool
    issues: tuple[str, ...]
    notes: tuple[str, ...] = ()

    @property
    def fair(self) -> bool:
        """Whether no violation was found."""
        return not self.issues

    @property
    def parities(self) -> tuple[str, ...]:
        """Every parity some protocol realises, in :data:`PARITIES` order."""
        found = {parity for items in self.protocols.values() for parity in items}
        return tuple(parity for parity in PARITIES if parity in found)

    def to_dict(self) -> dict[str, Any]:
        """Return the assessment as a JSON-ready dictionary."""
        return {
            "fair": self.fair,
            "solvers": list(self.solvers),
            "protocols": {name: list(items) for name, items in self.protocols.items()},
            "parities": list(self.parities),
            "seeds": "common random numbers per problem and replicate",
            "paired": self.paired,
            "issues": list(self.issues),
            "notes": list(self.notes),
        }


def assess(plan: Plan) -> Fairness:
    """Check that a plan compares its solvers fairly.

    Args:
        plan: A plan, freshly built or read from a store.

    Returns:
        The assessment. Its ``issues`` name every violation, and its
        ``notes`` what else bears on the comparison.
    """
    issues: list[str] = []
    for protocol in plan.protocols:
        if not protocol.stop.bounded:
            issues.append(
                f"protocol {protocol.name!r} sets no time, evaluation, or iteration limit"
            )

    excluded = {(item.problem, item.solver) for item in plan.exclusions}
    expected = {
        (problem.id, solver.name, protocol.name, replicate)
        for problem in plan.problems
        for solver in plan.solvers
        if (problem.id, solver.name) not in excluded
        for protocol in plan.protocols
        for replicate in range(plan.replicates_of(solver, protocol))
    }
    cells = Counter((run.problem, run.solver, run.protocol, run.replicate) for run in plan.runs)
    missing = sorted(expected - set(cells))
    if missing:
        issues.append(_listed("missing runs", missing))
    unexpected = sorted(set(cells) - expected)
    if unexpected:
        issues.append(
            _listed("runs outside the grid of problems, solvers, and protocols", unexpected)
        )
    repeated = sorted(cell for cell, count in cells.items() if count > 1)
    if repeated:
        issues.append(_listed("runs planned twice", repeated))
    identifiers = Counter(run.run_id for run in plan.runs)
    clashes = sorted(identifier for identifier, count in identifiers.items() if count > 1)
    if clashes:
        issues.append(_listed("run ids used twice", clashes))

    seeds: dict[tuple[str, int], set[int]] = {}
    for run in plan.runs:
        seeds.setdefault((run.problem, run.replicate), set()).add(run.seed)
    unpaired = sorted(key for key, values in seeds.items() if len(values) > 1)
    if unpaired:
        issues.append(_listed("problems and replicates with several seeds", unpaired))

    return Fairness(
        solvers=tuple(solver.name for solver in plan.solvers),
        protocols={protocol.name: parities(protocol.stop) for protocol in plan.protocols},
        paired=not unpaired,
        issues=tuple(issues),
        notes=tuple(note for protocol in plan.protocols for note in notes(plan, protocol)),
    )


def notes(
    plan: Plan, protocol: Protocol, solvers: Collection[str] | None = None
) -> tuple[str, ...]:
    """Return what bears on a comparison under one protocol without breaking a rule.

    Args:
        plan: The plan.
        protocol: One of its protocols.
        solvers: Names of the solvers compared. ``None`` compares every
            solver of the plan.

    Returns:
        One sentence, without its capital and full stop, for each fact: the
        compared solvers' thread counts differ under a time limit, and
        solvers declared deterministic run once under a time limit although
        the campaign has more replicates. Empty for a protocol without a time
        limit.
    """
    if protocol.stop.time_limit is None:
        return ()
    compared = [solver for solver in plan.solvers if solvers is None or solver.name in solvers]
    found: list[str] = []
    threads = {solver.name: _threads(solver) for solver in compared}
    if len(set(threads.values())) > 1:
        counts = ", ".join(f"{name}: {count}" for name, count in threads.items())
        found.append(
            f"protocol {protocol.name!r} sets a time limit and compares solvers whose thread "
            f"counts differ ({counts}), so its equal wall-clock budgets are unequal processor "
            "budgets"
        )
    single = [
        solver.name
        for solver in compared
        if not solver.stochastic
        and solver.replicates is None
        and plan.replicates_of(solver, protocol) < plan.replicates
    ]
    if single:
        found.append(
            f"protocol {protocol.name!r} sets a time limit and gives one run per problem to "
            f"the solvers declared deterministic ({', '.join(single)}): a run that the time "
            "limit ends depends on the speed and load of the machine, and one run does not "
            "show by how much; replicate_timed = true gives them every replicate"
        )
    return tuple(found)


@dataclass(frozen=True)
class Overrun:
    """The completed runs of one solver against the time limit of their protocol.

    Attributes:
        runs: Completed runs of the solver under the protocol.
        over: How many of them took more wall time than the time limit with
            the tolerance allows.
        worst: The longest wall time among those, as a multiple of the time
            limit.
    """

    runs: int
    over: int
    worst: float


@dataclass(frozen=True)
class ProtocolOutcomes:
    """How the completed runs of one protocol ended.

    Attributes:
        protocol: Name of the protocol.
        stop_reasons: For each iterative solver with a completed run, in plan
            order, how many of its runs ended for each stop reason. Solvers
            that are not iterative end when their algorithm does and are
            left out.
        overruns: For each solver with a completed run over the time limit,
            in plan order, the count and extent. Empty for a protocol
            without a time limit.
    """

    protocol: str
    stop_reasons: Mapping[str, Mapping[str, int]]
    overruns: Mapping[str, Overrun]

    @property
    def divergent(self) -> bool:
        """Whether the runs of the iterative solvers ended for more than one reason.

        Runs that ended for different reasons, such as one at the time limit
        and one for lack of improvement, did not spend the same budget, in
        one solver or across several. A reached target or a proof of
        optimality is a legitimate end, so a divergence qualifies a
        comparison and does not refuse it.
        """
        return len({reason for counts in self.stop_reasons.values() for reason in counts}) > 1

    def to_dict(self) -> dict[str, Any]:
        """Return the outcomes as a JSON-ready dictionary."""
        return {
            "protocol": self.protocol,
            "stop_reasons": {name: dict(counts) for name, counts in self.stop_reasons.items()},
            "divergent": self.divergent,
            "overruns": {
                name: {"runs": item.runs, "over": item.over, "worst": item.worst}
                for name, item in self.overruns.items()
            },
        }


@dataclass(frozen=True)
class Outcomes:
    """The outcomes of a campaign's completed runs, compared with its plan.

    Attributes:
        protocols: The outcomes under each protocol, in plan order.
        overrun_tolerance: Share of a time limit by which a run may exceed
            it without counting as over.
        records: How many records counted: those of runs of the plan.
    """

    protocols: tuple[ProtocolOutcomes, ...]
    overrun_tolerance: float
    records: int

    @property
    def divergent(self) -> tuple[str, ...]:
        """Names of the protocols whose stop reasons diverge."""
        return tuple(item.protocol for item in self.protocols if item.divergent)

    def protocol(self, name: str) -> ProtocolOutcomes:
        """Return the outcomes under the protocol with the given name.

        Raises:
            KeyError: If the plan has no such protocol.
        """
        for item in self.protocols:
            if item.protocol == name:
                return item
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """Return the assessment as a JSON-ready dictionary."""
        return {
            "records": self.records,
            "overrun_tolerance": self.overrun_tolerance,
            "divergent": list(self.divergent),
            "protocols": [item.to_dict() for item in self.protocols],
        }


def assess_outcomes(
    plan: Plan,
    records: Iterable[RunRecord],
    *,
    overrun_tolerance: float = OVERRUN_TOLERANCE,
) -> Outcomes:
    """Compare the outcomes of a campaign's completed runs with its plan.

    A plan can be fair and its runs still not comparable: under one
    protocol, a solver that stops for lack of improvement spends less than
    the budget another one uses to its time limit, and a solver that honours
    no deadline spends more. Neither shows in the plan.

    Args:
        plan: The plan.
        records: Records of completed runs. A record of a run that is not in
            the plan does not count.
        overrun_tolerance: Share of the time limit by which the wall time of
            a completed run may exceed it without counting as over.

    Returns:
        For each protocol, the stop reasons of the iterative solvers' runs
        and the solvers whose runs took more wall time than the time limit
        allows.

    Raises:
        ConfigError: If ``overrun_tolerance`` is not zero or a positive
            number.
    """
    tolerance = overrun_tolerance
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not tolerance >= 0:
        raise ConfigError(f"overrun_tolerance must be zero or a positive number, got {tolerance!r}")
    reasons: dict[tuple[str, str], Counter[str]] = {}
    walls: dict[tuple[str, str], list[float]] = {}
    counted = 0
    for record in records:
        if not plan.has_run(record.run_id):
            continue
        counted += 1
        run = plan.run(record.run_id)
        cell = (run.protocol, run.solver)
        reasons.setdefault(cell, Counter())[record.outcome.stop_reason.value] += 1
        walls.setdefault(cell, []).append(record.provenance.wall_seconds)
    protocols: list[ProtocolOutcomes] = []
    for protocol in plan.protocols:
        limit = protocol.stop.time_limit
        overruns: dict[str, Overrun] = {}
        for solver in plan.solvers:
            taken = walls.get((protocol.name, solver.name), [])
            over = [] if limit is None else [w for w in taken if w > limit * (1.0 + tolerance)]
            if over and limit is not None:
                overruns[solver.name] = Overrun(len(taken), len(over), max(over) / limit)
        protocols.append(
            ProtocolOutcomes(
                protocol=protocol.name,
                stop_reasons={
                    solver.name: dict(sorted(reasons[protocol.name, solver.name].items()))
                    for solver in plan.solvers
                    if solver.iterative and (protocol.name, solver.name) in reasons
                },
                overruns=overruns,
            )
        )
    return Outcomes(tuple(protocols), float(tolerance), counted)


def _threads(solver: SolverEntry) -> int:
    """Return the threads a solver of a plan may run.

    An entry that records no ``threads``, for a solver without the parameter
    or in a plan stored without the field, counts with the ``threads`` its
    parameters set, or 1.
    """
    if solver.threads is not None:
        return solver.threads
    value = solver.parameters.get("threads", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 1
    return value


def _listed(what: str, items: list[Any]) -> str:
    shown = ", ".join(str(item) for item in items[:_SHOWN])
    more = f" and {len(items) - _SHOWN} more" if len(items) > _SHOWN else ""
    return f"{len(items)} {what}: {shown}{more}"
