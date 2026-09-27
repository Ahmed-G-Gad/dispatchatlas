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
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dispatchatlas.campaigns.planning import Plan
    from dispatchatlas.core.solver import StopCriteria

__all__ = [
    "EQUAL_EVALUATIONS",
    "EQUAL_ITERATIONS",
    "EQUAL_WALL_CLOCK",
    "FIXED_TARGET",
    "PARITIES",
    "Fairness",
    "assess",
    "parities",
]

EQUAL_WALL_CLOCK = "equal-wall-clock"
EQUAL_EVALUATIONS = "equal-evaluations"
EQUAL_ITERATIONS = "equal-iterations"
FIXED_TARGET = "fixed-target"
PARITIES = (EQUAL_WALL_CLOCK, EQUAL_EVALUATIONS, EQUAL_ITERATIONS, FIXED_TARGET)
"""Every parity name, in report order."""

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
    """

    solvers: tuple[str, ...]
    protocols: Mapping[str, tuple[str, ...]]
    paired: bool
    issues: tuple[str, ...]

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
        }


def assess(plan: Plan) -> Fairness:
    """Check that a plan compares its solvers fairly.

    Args:
        plan: A plan, freshly built or read from a store.

    Returns:
        The assessment. Its ``issues`` name every violation.
    """
    issues: list[str] = []
    protocol_names = [protocol.name for protocol in plan.protocols]
    for protocol in plan.protocols:
        if not protocol.stop.bounded:
            issues.append(
                f"protocol {protocol.name!r} sets no time, evaluation, or iteration limit"
            )

    excluded = {(item.problem, item.solver) for item in plan.exclusions}
    expected = {
        (problem.id, solver.name, protocol, replicate)
        for problem in plan.problems
        for solver in plan.solvers
        if (problem.id, solver.name) not in excluded
        for protocol in protocol_names
        for replicate in range(plan.replicates_of(solver))
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
    )


def _listed(what: str, items: list[Any]) -> str:
    shown = ", ".join(str(item) for item in items[:_SHOWN])
    more = f" and {len(items) - _SHOWN} more" if len(items) > _SHOWN else ""
    return f"{len(items)} {what}: {shown}{more}"
