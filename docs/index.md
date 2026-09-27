# DispatchAtlas

DispatchAtlas is a Python library for reproducible benchmarking of scheduling
algorithms. One data model covers machine scheduling, resource-constrained
project scheduling, and the placement of tasks across an edge-fog-cloud
continuum. Benchmark families generate problems from a seed, and published
suites such as Taillard's are rebuilt exactly. Every solver, from dispatching
rules to metaheuristics and exact models, runs under the same stop criteria
and counts its evaluations the same way, so budgets are comparable. Campaigns
run every solver on every problem with paired seeds, record each run, resume
after an interruption, and replay runs to check that they reproduce. The
analysis turns a campaign into tables and figures, with every p-value next to
an effect size and a confidence interval.

```python
from dispatchatlas.benchmarks import generate
from dispatchatlas.core import check_schedule
from dispatchatlas.solvers import solve

problem = generate("job-shop", seed=1, jobs=6, machines=4)
result = solve(problem, "tabu-search", max_evaluations=2000, seed=3)

print(result.status, result.value, result.evaluations, result.stop_reason)
print(check_schedule(problem, result.schedule))
```

```text
feasible 361.0 2000 evaluation-limit
()
```

## What the library provides

- [**Data model**](guide/model.md): tasks with alternative modes, resources of
  any capacity, dependencies with time lags and network transfers, release
  times, due dates, hard deadlines, setup times, and permutation schedules,
  with precise semantics and a feasibility checker that is independent of the
  solvers.
- [**Benchmarks**](guide/benchmarks.md): seeded benchmark families, published
  suites rebuilt offline or read from their files, instance catalogs written
  to disk with checksums, and descriptive instance features.
- [**Solvers**](guide/solvers.md): heuristics, local search, population and
  swarm methods, multi-objective algorithms, learning methods, and exact
  solvers behind one contract with exact budgets and seeded determinism.
- [**Campaigns**](guide/campaigns.md): experiments described in a file, with
  fair comparisons verified from the plan, paired seeds, hard time limits,
  resumption, and replay.
- [**Analysis**](guide/analyzing.md): rank tests, corrections, effect sizes,
  bootstrap intervals, quality indicators, tables, and figures, collected in a
  reproducible bundle.

## Where to go next

- [Getting started](getting-started.md) installs the library and walks through
  a first problem and a first campaign.
- The [solver catalogue](catalogue/solvers.md) and the
  [benchmark catalogue](catalogue/benchmarks.md) list everything that is
  registered, generated from the code.
- The [command line](cli.md) reference documents every command.
- [Extending](extending.md) shows how to add a solver, a benchmark family, or
  an objective, in a separate package if you like.
- The [API reference](api/index.md) documents every public module.
