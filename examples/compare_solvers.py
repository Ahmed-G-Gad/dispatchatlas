"""Compare solvers on one problem against its proven optimum.

The problem is a permutation flow shop of 7 jobs on 4 machines, small enough
for the ``enumeration`` solver to decode every job order and prove the optimal
makespan. Each heuristic then gets the same budget of evaluated schedules and
the same seeds: run ``k`` of every stochastic solver receives
``derive_seed(ROOT_SEED, "compare", k)``, so the runs are paired across
solvers. Deterministic solvers run once.

For each solver the script prints the best, mean, and worst makespan over its
runs, the relative gap of the mean to the optimum, and how many runs reached
the optimum.

Run it from the repository root::

    python examples/compare_solvers.py
"""

from __future__ import annotations

from statistics import fmean

from dispatchatlas.benchmarks import generate
from dispatchatlas.core import Status, derive_seed
from dispatchatlas.solvers import default_registry, solve

ROOT_SEED = 2024
"""Root of every run seed."""

RUNS = 10
"""Runs per stochastic solver."""

BUDGET = 300
"""Schedules each run may evaluate."""

SOLVERS = (
    "neh",
    "iterated-greedy",
    "tabu-search",
    "simulated-annealing",
    "genetic-algorithm",
    "particle-swarm-optimization",
    "random-search",
)


def main() -> None:
    """Prove the optimum, run every solver, and print one line per solver."""
    problem = generate("flow-shop", seed=4, jobs=7, machines=4)
    exact = solve(problem, "enumeration", max_tasks=problem.size)
    assert exact.status is Status.OPTIMAL
    assert exact.value is not None
    optimum = exact.value
    print(f"{problem.id}: optimal makespan {optimum:g}, proven over {exact.evaluations} job orders")
    print(f"every run below may evaluate {BUDGET} schedules\n")

    registry = default_registry()
    print(
        f"{'solver':<28} {'runs':>4} {'best':>6} {'mean':>7} {'worst':>6} {'gap':>7} {'optimal':>8}"
    )
    for solver in SOLVERS:
        runs = RUNS if registry.info(solver).stochastic else 1
        values = []
        for k in range(runs):
            seed = derive_seed(ROOT_SEED, "compare", k)
            result = solve(problem, solver, max_evaluations=BUDGET, seed=seed)
            assert result.value is not None
            values.append(result.value)
        mean = fmean(values)
        gap = 100.0 * (mean - optimum) / optimum
        hits = sum(value == optimum for value in values)
        print(
            f"{solver:<28} {runs:>4} {min(values):>6g} {mean:>7.1f} {max(values):>6g} "
            f"{gap:>6.2f}% {hits:>4}/{runs}"
        )


if __name__ == "__main__":
    main()
