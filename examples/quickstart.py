"""Generate a job shop, solve it with three solvers, and check every schedule.

The job shop is drawn from the ``job-shop`` benchmark family with a fixed seed,
so it is the same problem on every machine. Each solver gets the same
evaluation budget where it needs one, and every returned schedule is checked
against every constraint of the problem by :func:`dispatchatlas.core.check_schedule`,
which is independent of the schedule builder the solvers share.

Run it from the repository root::

    python examples/quickstart.py
"""

from __future__ import annotations

from dispatchatlas.benchmarks import generate
from dispatchatlas.core import check_schedule
from dispatchatlas.solvers import solve

SOLVERS = ("heft", "tabu-search", "simulated-annealing")
"""A list-scheduling heuristic and two local searches."""

BUDGET = 2000
"""Schedules each iterative solver may evaluate."""


def main() -> None:
    """Solve one job shop with each solver and print what each found."""
    problem = generate("job-shop", seed=1, jobs=6, machines=4)
    print(f"{problem.id}: {problem.size} operations on {len(problem.resources)} machines")
    for solver in SOLVERS:
        result = solve(problem, solver, max_evaluations=BUDGET, seed=3)
        assert result.schedule is not None
        violations = check_schedule(problem, result.schedule)
        verdict = "feasible" if not violations else f"{len(violations)} violations"
        print(
            f"{solver:<20} makespan {result.value:>6g}  "
            f"{result.evaluations:>5} evaluations  {result.stop_reason}  {verdict}"
        )


if __name__ == "__main__":
    main()
