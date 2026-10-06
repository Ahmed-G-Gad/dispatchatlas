# Solvers

Every solver implements one contract: it has metadata, `info`, and a method
`solve(problem, stop, seed)` that returns a `SolveResult`. The library's own
solvers and those of plugins are interchangeable everywhere. The
[solver catalogue](../catalogue/solvers.md) describes every registered solver.

## Running a solver

`solve` runs a registered solver by id, with its stop criteria and parameters
as keyword arguments:

```python
from dispatchatlas.benchmarks import generate
from dispatchatlas.solvers import solve

problem = generate("rcpsp", seed=3, activities=30)
result = solve(problem, "tabu-search", max_evaluations=5000, seed=1, tenure=10)
```

The registry separates building a solver from running it, which a script that
runs one solver many times prefers:

```python
from dispatchatlas.core import StopCriteria
from dispatchatlas.solvers import default_registry

registry = default_registry()
solver = registry.create("tabu-search", tenure=10)
result = solver.solve(problem, StopCriteria(max_evaluations=5000), seed=1)
```

`registry.create` refuses an unknown id or parameter, and a parameter value
the solver's checks reject, with a `ConfigError` that names what is known.
`registry.info(id)` returns the metadata: the family, a one-sentence summary
of the mechanism, the parameters and their defaults, whether the solver is
stochastic, iterative, or multi-objective, the problem features it models, the
extras it needs, and its sources. From the shell:

```bash
dispatchatlas solvers list --family local-search
dispatchatlas solvers show tabu-search
dispatchatlas solve problem.json --solver tabu-search --set tenure=10 --max-evaluations 5000 --seed 1
```

A solver refuses a problem with a feature it does not model, such as setup
times for a solver that ignores them, with an `UnsupportedProblemError`
before it starts; `info.accepts(problem)` tells in advance.

## Solver families

<!-- dispatchatlas:solver-families -->

## Results

A `SolveResult` holds:

- `status`: what is known about the schedule. `optimal`: proven optimal for
  the primary objective. `feasible`: meets every constraint, optimality not
  proven. `violated`: a schedule was built but misses a hard deadline.
  `infeasible`: proven that no feasible schedule exists. `unknown`: no
  schedule found, and infeasibility not proven.
- `schedule`, and `objectives`: the value of every objective of the problem
  for that schedule. `value` is the primary objective's value.
- `stop_reason`: `completed` (the algorithm finished on its own),
  `time-limit`, `evaluation-limit`, `iteration-limit`, `target-reached`, or
  `stagnation` (no improvement for `patience` iterations).
- `elapsed`, `evaluations`, and `iterations`: the effort spent.
- `trace`: a point each time the best schedule improved, with the elapsed
  time, the evaluations and iteration so far, the value, and whether the
  schedule was feasible, plus a final point.
- `bound` and `gap`: a proven bound on the primary objective and the relative
  distance `|value - bound| / |value|`, for exact solvers.
- `front`: the non-dominated schedules of a multi-objective solver.
- `extras`: solver-specific facts, such as a backend's version and raw status.

## Stop criteria

`StopCriteria` holds five criteria; the first one met ends the run:

| Criterion | Stops the run |
| --- | --- |
| `time_limit` | after that many wall-clock seconds |
| `max_evaluations` | after that many evaluated schedules |
| `max_iterations` | after that many main-loop iterations |
| `target` | once a feasible schedule reaches that value of the primary objective or better |
| `patience` | after that many iterations without improvement |

A sixth field, `schedule`, ends no run: it states which clock a parameter
schedule follows, `"deterministic"` by default or `"binding"` (see
[Determinism and seeds](#determinism-and-seeds)).

**Iterative** solvers, those that run until a criterion fires, need at least
one of the three limits `time_limit`, `max_evaluations`, and `max_iterations`,
and raise `SolverError` without one. Each of them works with any single one of
the three. Solvers that finish on their own, such as the priority rules, NEH,
and the enumeration, need no criterion but honour those they are given.

What an iteration is depends on the solver, as its documentation states: one
move of a local search, one generation of a genetic algorithm, one update of a
swarm. Iteration limits are therefore not comparable across solvers, while
evaluation limits are.

## Budgets and exact evaluation counts

Order-based solvers, the large majority, run inside a `Search`, the run-time
that owns the random generator, the evaluator, the stop criteria, the best
schedule found, and the trace. Every candidate schedule a solver scores goes
through it, and counts as one **evaluation**:

- `Search.evaluate(order, modes)` decodes a task order into a schedule with
  the shared builder and scores it;
- `Search.submit(decoded)` scores a schedule a constructive heuristic built
  task by task;
- `Search.probe(order, modes)` scores a partial or tentative order, for
  example an insertion position, without making it the best schedule.

The evaluation limit is checked before each evaluation, so a run limited to
`N` evaluations scores exactly `N` schedules when the limit ends it, never
`N + 1`. The iteration limit is checked when an iteration starts, and the time
limit before every evaluation and iteration, and inside long computations that
evaluate nothing. The counts in the result are the search's own counts, and
the score of every candidate comes from the builder that produces the reported
schedule, so what a solver reports is what it evaluated.

Exact solvers count what they compute: `enumeration` counts each complete
schedule it decodes, `cp-sat` counts each improving solution the backend
reports, and `milp` counts the one schedule it reads back. Their time limit
covers building the model. See the
[exact solvers](../api/solvers-exact.md) for the guarantees of their statuses
and bounds.

The time limit is cooperative: a solver checks it itself. Campaigns add a hard
limit that kills a run that overruns (see [Campaigns](campaigns.md#execution)).

A solver counts its `elapsed` time and its time limit from the instant it
creates its clock: for an order-based solver, after it has built its
evaluator; for `cp-sat`, `milp`, and `restricted-neighbourhood-search`, after
the backend is imported. A caller that wants one origin for every solver sets
it with `clock_origin` of
[`dispatchatlas.core.clock`](../api/core.md), and the clocks created inside
the block start there. A campaign does so at the start of every attempt, so
in a campaign `elapsed` and the time limit cover everything a solver does
from that start on. What a solver does once in a process, the import of its
backend, a campaign has it do before that start, through the solver's
`warm_up` method (see [Extending](../extending.md#a-solver)).

Inside a `clock_origin` block a clock counts from the origin, not from its
own creation, and so does the time limit of a search created there. Code
that runs several searches in turn inside one block opens a block of its own
around each, `with clock_origin():`, to give each its whole limit.

## How candidates are compared

Solvers compare schedules by a lexicographic score: first the total overrun
of hard deadlines, then the primary objective in minimisation form. A schedule
that meets every deadline always beats one that does not
([details](model.md#how-solvers-compare-schedules)). The best schedule of a
run is the one with the smallest score; the result reports it with the status
`feasible`, or `violated` if even the best one overruns a deadline.

Multi-objective solvers keep, in addition, an archive of every non-dominated
feasible schedule they evaluate, over all the problem's objectives in
minimisation form, and return it as the front. Single-objective solvers
optimise the primary objective only; the result still reports every
objective's value.

## Determinism and seeds

Every random choice of a solver comes from the generators its search is
seeded with: `Search.rng`, a `random.Random(seed)`, and
`Search.numpy_rng()`, a `numpy.random.default_rng(seed)`. No solver uses global
random state. Hence:

- Equal problem, parameters, stop criteria, and seed give the same schedule,
  objective values, counts, and trace (apart from its elapsed times) on every
  run, as long as no time limit ends it. A run stopped by its time limit stops
  at a point that depends on the speed and load of the machine.
- Deterministic solvers ignore the seed.
- Solvers whose parameters follow a schedule over the run, such as a falling
  inertia weight or a shrinking population, read `Search.progress()`: the
  fraction of the binding limit spent. Which limits count is the `schedule`
  field of the stop criteria. With `"deterministic"`, the default, the time
  limit plays no part when an evaluation or iteration limit is set, so the
  schedule replays exactly, and it stops short of its end when the time limit
  ends the run first. With `"binding"`, every set limit counts: the schedule
  reaches its end at whichever limit ends the run, and under a time limit the
  run depends on the machine. With the time limit as the only limit, both
  follow the elapsed time. Such a solver records in the `extras` of its
  result `schedule_clock`, the limit that places the end of the run on the
  schedule (`evaluations`, `iterations`, or `time`), and `schedule_progress`,
  that place between 0 and 1.
- `cp-sat` is deterministic with its default single worker and without a time
  limit.

Draw the seeds of repeated runs from one root seed with `derive_seed`, so
that runs are independent and reproducible:

```python
from dispatchatlas.core import derive_seed

values = []
for k in range(10):
    seed = derive_seed(2024, "sa", k)
    values.append(solve(problem, "simulated-annealing", max_evaluations=5000, seed=seed).value)
```

Campaigns do this for you, and share the seeds across solvers so that their
results are paired (see [Campaigns](campaigns.md#seeds-and-paired-comparisons)).

Transcendental functions of the C library, such as `exp` and `log`, may differ
in their last bit between platforms, which can steer a stochastic search
elsewhere. Results therefore reproduce exactly on the same platform with the
same versions of Python and NumPy, and usually, but not always, elsewhere.

## Writing a solver

A new solver is a dataclass with `info` and `solve`. [Extending](../extending.md)
walks through one, and the
[contributing guide](https://github.com/ahmed-g-gad/dispatchatlas/blob/main/CONTRIBUTING.md#the-solver-contract)
lists the contract every built-in solver follows.
