# Extending

DispatchAtlas finds solvers and benchmark families in registries, and
objectives in a registry of their own. A new solver or family can live in
your own code, registered by a call, or in a separate package that
DispatchAtlas discovers through an entry point, so that the command line and
campaign files can use it by id like a built-in one.

## A solver

A solver is a class with a class-level `info: SolverInfo` and a method
`solve(problem, stop, seed)`. The repository's
[`examples/plugin/`](https://github.com/ahmed-g-gad/dispatchatlas/tree/main/examples/plugin)
directory holds a complete, installable plugin with one solver, a descent by
random insertion moves that restarts when it stops improving:

<!-- fmt: off -->
```python
--8<-- "examples/plugin/src/dispatchatlas_example_plugin/__init__.py"
```
<!-- fmt: on -->

The pieces, in the order a reader meets them:

- **Parameters** are dataclass fields with defaults. `__post_init__` rejects
  invalid values with `ValueError`, which the registry reports as a
  `ConfigError` naming the solver. The registry reads the parameters and their
  defaults from the fields.
- **Metadata**: `id` is kebab case and unique; `family` is one of the
  standard families, or a kebab-case family of your own; `summary` states the
  mechanism in one sentence; `references` holds bibliography keys, DOIs, or
  URLs. `stochastic`, `iterative`, and `multi_objective` default to true,
  true, and false. `supports` lists the problem features the solver models
  correctly, every feature by default, since the shared schedule builder
  handles them all; campaigns never pair the solver with a problem that has
  another. `requires` names the extras the solver needs.
- **The search.** `Search(problem, stop, seed, info)` checks the problem's
  features and the stop criteria, owns the random generator `search.rng`, and
  keeps the best schedule. Inside `with search:`, `search.next_iteration()`
  starts each iteration and `search.evaluate(order)` decodes and scores each
  candidate; either raises when a stop criterion fires, and leaving the `with`
  block ends the run there. `search.result()` builds the result, with exact
  counts and the trace. A solver written this way honours every stop
  criterion without code of its own.

A candidate is a task order, which the builder repairs to respect the
dependencies, and optionally a mode for each task; `modes=None` lets the
builder give each task its earliest-finishing mode. The building blocks are
documented in the [solver framework](api/solvers.md):
`dispatchatlas.solvers.operators` holds moves and crossovers on orders and
modes, `dispatchatlas.solvers.encoding` maps real vectors to orders and modes
for methods that search vectors, and `Search.submit` scores a schedule that a
constructive heuristic built task by task with the builder of
`search.evaluator.decoder`.

A solver that does not search over orders, such as one that calls an
external optimiser, implements `solve` itself. It must then honour every
criterion it is given, return a `SolveResult` whose `objectives` come from
`dispatchatlas.core.evaluate`, and set `status` from what it knows; check its
schedules with `check_schedule` in its tests.

### Registering it in your code

```python
from dispatchatlas.core import StopCriteria
from dispatchatlas.solvers import default_registry
from dispatchatlas_example_plugin import InsertionDescent

registry = default_registry()
registry.register(InsertionDescent)
solver = registry.create("insertion-descent", restart_after=50)
```

Pass such a registry to a campaign with `run(campaign, root, registry=registry)`.
The process backend sends each solver instance to a worker process, which
imports the solver's class by name, so define the class at the top level of a
module or of the script, never inside its `if __name__ == "__main__":` block.

### Publishing it as a plugin

A package publishes solvers through the `dispatchatlas.solvers` entry-point
group. Each entry point names a function that takes the registry:

```toml
--8<-- "examples/plugin/pyproject.toml"
```

Once the package is installed, `default_registry()` calls every such
function, so the solver appears in `dispatchatlas solvers list` and can be
named in campaign files:

```bash
pip install ./examples/plugin
dispatchatlas solvers show insertion-descent
dispatchatlas version                       # lists the installed plugins
```

A plugin that fails to load is skipped with a warning and registers none of
its solvers, so one broken package cannot disable the others.
`default_registry(plugins=False)` loads the built-in solvers alone.

### Testing it

The tests of the built-in solvers show what to check, and
`tests/test_examples.py` applies the same checks to the example plugin: that
each single limit (evaluations, iterations, time) ends a run with the right
stop reason and exact counts, that the schedule passes `check_schedule`, that
equal seeds give equal results, and that invalid parameters are refused. Add
tests of the method itself: small cases computed by hand, known optima of tiny
instances, and a comparison with `random-search` at an equal number of
evaluations.

## A benchmark family

A family is a dataclass whose fields are its parameters, with a class-level
`info: FamilyInfo` and a method `generate(seed)` that returns a problem.
Equal parameters and seed must give the same problem, so draw every random
number from a generator seeded with `seed`:

```python
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.benchmarks import FamilyInfo, FamilyRegistry
from dispatchatlas.core import Problem, Resource, Task


@dataclass(frozen=True)
class SingleMachine:
    """Jobs with release times on one machine.

    Attributes:
        jobs: Number of jobs.
        max_time: Largest processing time.
    """

    jobs: int = 10
    max_time: int = 20

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="single-machine",
        name="Single machine with release times",
        problem_class="single-machine",
        summary="Jobs with integer processing times and release times on one machine.",
        objectives=("total_completion_time",),
    )

    def __post_init__(self) -> None:
        for name in ("jobs", "max_time"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    def generate(self, seed: int = 0) -> Problem:
        """Return the problem for ``seed``."""
        rng = random.Random(seed)
        times = [rng.randint(1, self.max_time) for _ in range(self.jobs)]
        latest = sum(times) // 2
        tasks = tuple(
            Task.on(f"j{index}", "m", float(time), release=float(rng.randint(0, latest)))
            for index, time in enumerate(times)
        )
        return Problem(
            f"single-machine-{self.jobs}-{seed}",
            tasks,
            (Resource("m"),),
            objectives=self.info.objectives,
            family=self.info.id,
            metadata={"jobs": self.jobs, "max_time": self.max_time, "seed": seed},
        )


def register(registry: FamilyRegistry) -> None:
    """Add the family to ``registry``."""
    registry.register(SingleMachine)
```

Register it in your code with `default_registry().register(SingleMachine)`
from `dispatchatlas.benchmarks`, and pass that registry to a campaign as
`run(campaign, root, families=registry)`. A package publishes families
through the `dispatchatlas.benchmarks` entry-point group, exactly as solvers:

```toml
[project.entry-points."dispatchatlas.benchmarks"]
my_families = "my_package.families:register"
```

Name the problems after the family, its size, and the seed, and record the
parameters and the seed in the metadata, so that a problem says where it came
from. The builders of the built-in families, such as `flow_shop_problem` in
`dispatchatlas.benchmarks.shop` and `build_problem` in
`dispatchatlas.benchmarks.continuum`, turn plain data into problems with the
conventions of the library.

## Objectives

An objective is a named function of a problem and a schedule, with a sense
and a unit. Register it, and problems can declare it by name:

```python
from dispatchatlas.core import Objective, Problem, Schedule, register_objective


def max_tardiness(problem: Problem, schedule: Schedule) -> float:
    """Largest amount by which a task finishes after its due date."""
    ends = {item.task: item.end for item in schedule.assignments}
    return max(
        (
            max(0.0, ends[task.id] - task.due)
            for task in problem.tasks
            if task.due is not None and task.id in ends
        ),
        default=0.0,
    )


register_objective(
    Objective("max_tardiness", "min", "time", "Largest tardiness over the tasks.", max_tardiness)
)
```

Three rules make an objective work everywhere:

- **Accept partial schedules.** Insertion heuristics such as NEH score
  partial schedules while they build one, so the function must accept a
  schedule that leaves tasks out, and count only the tasks it holds.
- **Register it where it runs.** The registry belongs to the process. The
  worker processes of a campaign's process backend rebuild every problem, so
  the objective must be registered in them too. Every worker runs the
  registration hooks of the installed plugins when it starts, so an objective
  that a plugin's hook registers works in every backend and from
  `dispatchatlas campaign run`. An objective registered by a script instead
  must be registered at the top level of that script, or of a module it
  imports at its top level, since every worker imports the main module.
- **Know which solvers use it.** Search-based solvers optimise any
  registered objective, through its function. The exact solvers model only the
  objectives they list and refuse the others. `replace=True` re-registers a
  built-in name with a function of your own, and `unregister_objective`
  removes your registration again.

The analysis reads the sense of an objective from the registry, so the tables
and figures orient themselves without further settings.
