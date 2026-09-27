"""Place requests on an edge-fog-cloud continuum under two objectives.

Twelve devices' requests arrive over time on a platform of devices, edge, fog,
and cloud nodes. Every request can run on its own device or be offloaded,
which pays upload and download transfers across the network, so placement
trades response time against energy: the cloud is fast but draws the most
power, the device is slow and frugal.

Three multi-objective evolutionary algorithms approximate the Pareto front of
``mean_response_time`` and ``energy`` with the same budget and seed, and HEFT,
a single-objective list heuristic, adds one point for reference. The fronts
are compared by the hypervolume they dominate, computed exactly by
:func:`dispatchatlas.core.pareto.hypervolume` against one reference point
shared by every solver: the nadir of the union of all fronts, pushed out by
10 % of its range. The script finally shows where the extreme solutions place
the compute tasks.

Run it from the repository root::

    python examples/multi_objective_placement.py
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from dispatchatlas.benchmarks import generate
from dispatchatlas.core import Problem, Schedule, non_dominated
from dispatchatlas.core.pareto import hypervolume
from dispatchatlas.solvers import solve

OBJECTIVES = ("mean_response_time", "energy")
SOLVERS = ("nsga-ii", "sms-emoa", "moea-d")
BUDGET = 1000
SEED = 5


def placement(problem: Problem, schedule: Schedule) -> str:
    """Count the compute tasks of a schedule on each tier."""
    tiers: Counter[str] = Counter()
    for item in schedule.assignments:
        mode = problem.task(item.task).modes[item.mode]
        if mode.demands and mode.location is not None:
            tiers[problem.resource(mode.location).tier or "none"] += 1
    return ", ".join(f"{tier} {count}" for tier, count in sorted(tiers.items()))


def reference_point(fronts: Sequence[Sequence[tuple[float, ...]]]) -> tuple[float, ...]:
    """Return the nadir of the union of the fronts, pushed out by 10 % of its range."""
    points = [point for front in fronts for point in front]
    kept = [points[i] for i in non_dominated(points)]
    ideal = [min(point[k] for point in kept) for k in range(len(OBJECTIVES))]
    nadir = [max(point[k] for point in kept) for k in range(len(OBJECTIVES))]
    return tuple(high + 0.1 * (high - low) for low, high in zip(ideal, nadir, strict=True))


def main() -> None:
    """Approximate the fronts, score them, and describe the extreme placements."""
    problem = generate("continuum-offloading", seed=3, applications=12, objectives=OBJECTIVES)
    print(f"{problem.id}: {problem.size} tasks on {len(problem.resources)} nodes")
    print(f"objectives {' and '.join(OBJECTIVES)}, both minimised\n")

    results = {
        solver: solve(problem, solver, max_evaluations=BUDGET, seed=SEED) for solver in SOLVERS
    }
    heft = solve(problem, "heft")
    fronts = {
        solver: [point.values for point in result.front] for solver, result in results.items()
    }
    fronts["heft"] = [tuple(heft.objectives[name] for name in OBJECTIVES)]
    reference = reference_point(list(fronts.values()))

    print(f"{'solver':<10} {'points':>6} {'hypervolume':>12}")
    for solver, front in fronts.items():
        print(f"{solver:<10} {len(front):>6} {hypervolume(front, reference):>12.4f}")

    best = max(SOLVERS, key=lambda solver: hypervolume(fronts[solver], reference))
    ordered = sorted(results[best].front, key=lambda point: point.values)
    print(f"\nextreme solutions of the {best} front:")
    for label, point in (("fastest response", ordered[0]), ("least energy", ordered[-1])):
        response, energy = point.values
        assert point.schedule is not None
        where = placement(problem, point.schedule)
        print(f"  {label}: {response:.3f} s at {energy:.1f} J, compute tasks on {where}")


if __name__ == "__main__":
    main()
