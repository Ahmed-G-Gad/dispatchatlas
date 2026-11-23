# DispatchAtlas

[![CI](https://github.com/ahmed-g-gad/dispatchatlas/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmed-g-gad/dispatchatlas/actions/workflows/ci.yml)
[![Documentation](https://github.com/ahmed-g-gad/dispatchatlas/actions/workflows/docs.yml/badge.svg)](https://dispatchatlas.ahmedgad.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ahmed-g-gad/dispatchatlas/blob/main/LICENSE.txt)

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

Documentation: <https://dispatchatlas.ahmedgad.com>

## Installation

DispatchAtlas needs Python 3.12 or newer. Its only required dependency is
NumPy. It installs from this repository with [uv](https://docs.astral.sh/uv/):

```bash
# add it to a uv project, alone or with the exact solvers' backends
uv add "dispatchatlas @ git+https://github.com/ahmed-g-gad/dispatchatlas"
uv add "dispatchatlas[exact] @ git+https://github.com/ahmed-g-gad/dispatchatlas"

# install only the command line, or run it without installing it
uv tool install git+https://github.com/ahmed-g-gad/dispatchatlas
uvx --from git+https://github.com/ahmed-g-gad/dispatchatlas dispatchatlas --help
```

Optional extras add solver backends:

| Extra | Installs | Enables |
| --- | --- | --- |
| `exact` | OR-Tools, PuLP | the `cp-sat` and `milp` solvers (CBC backend) and `restricted-neighbourhood-search` |
| `highs` | highspy | the `highs` backend of `milp`, with `exact` |
| `scip` | PySCIPOpt | the `scip` backend of `milp`, with `exact` |
| `gurobi` | gurobipy | the `gurobi` backend of `milp`, with `exact`; needs a Gurobi licence beyond small models |
| `all` | `exact`, `highs`, `scip` | every open-source backend |

To work on the library itself, clone the repository and let uv create the
environment:

```bash
git clone https://github.com/ahmed-g-gad/dispatchatlas.git
cd dispatchatlas
uv sync
uv run pytest
```

## Quickstart in Python

Generate a job shop, solve it with tabu search under a budget of 2000
evaluated schedules, and check the schedule against every constraint:

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

`check_schedule` returns the violated constraints, none here. Equal inputs and
seed give the same schedule. The
[examples](https://github.com/ahmed-g-gad/dispatchatlas/tree/main/examples)
compare solvers against a proven optimum, run and analyse a campaign, and
place requests on a continuum under two objectives.

## Quickstart on the command line

The same problem from the shell, then a small campaign from the repository's
examples: run it, analyse it, and replay it.

```bash
dispatchatlas benchmarks generate job-shop --seed 1 --set jobs=6 --set machines=4 --output results/js.json
dispatchatlas solve results/js.json --solver tabu-search --max-evaluations 2000 --seed 3 --output results/tabu.json
dispatchatlas check results/js.json results/tabu.json

dispatchatlas campaign run examples/campaigns/quickstart.toml --output results
dispatchatlas analyze results/quickstart --output results/quickstart-analysis
dispatchatlas campaign replay results/quickstart
```

`dispatchatlas solvers list`, `dispatchatlas benchmarks list`, and
`dispatchatlas suites list` show what is available, and every command
explains its options with `--help`.

## Features

- **One data model.** Tasks run in one of several modes, each with its own
  duration and resource demands; resources have any capacity; dependencies
  carry time lags and data transfers priced by a network; tasks have release
  times, due dates, and hard deadlines; machines can have sequence-dependent
  setup times; problems can require a permutation schedule. Every problem is
  validated when it is built, and `check_schedule` checks any schedule
  against every constraint, independently of the solvers.
- **13 benchmark families**: flow, job, open, flexible job, and hybrid flow
  shops, and job shops drawn by Taillard's generator from seeds he did not
  publish; unrelated parallel machines with setups; single-mode and multi-mode
  resource-constrained projects; random task graphs on heterogeneous
  processors; and request offloading, workflows, and stream pipelines on an
  edge-fog-cloud continuum.
- **10 published suites**: Taillard's flow, job, and open shops, rebuilt
  offline from their published seeds, and the OR-Library job and flow shops,
  Brandimarte's flexible job shops, and PSPLIB J30 to J120, read from the files
  you download. Readers for the OR-Library, FJS, PSPLIB, WfCommons, and Pegasus
  DAX formats, and a streaming reader for the batch jobs of the Alibaba 2018
  cluster trace; instance catalogs written to disk with SHA-256 manifests;
  descriptive features of every instance.
- **76 solvers**: dispatching rules and list heuristics such as HEFT;
  constructive heuristics such as NEH and beam search; local search from hill
  climbing and simulated annealing to tabu search, iterated greedy, and ALNS;
  genetic, memetic, and ant colony algorithms, BRKGA, differential evolution,
  CMA-ES, and particle swarms; multi-objective algorithms including NSGA-II,
  NSGA-III, MOEA/D, and SMS-EMOA; learning methods and hyper-heuristics; exact
  CP-SAT, time-indexed MILP, and enumeration solvers; a matheuristic; and
  job-shop search families, each written once for random keys and job
  sequences. Each
  solver declares the problem features it models and the sources it follows.
- **9 objectives**: makespan, total completion time, total flow time, mean
  response time, total tardiness, maximum lateness, tardy tasks, energy, and
  cost. A plugin can register more.
- **Campaigns** described in TOML or JSON: every solver on every problem under
  every stopping protocol, paired seeds, worker processes with hard time
  limits and retries, one JSON record per run, resumption, and replay.
- **Statistics** for comparing algorithms over many problems: Friedman and
  Iman-Davenport tests, Nemenyi critical differences, Wilcoxon signed-rank
  tests with Holm, Hochberg, Shaffer, or Finner correction, Bayesian sign and
  signed-rank tests, effect sizes with bootstrap intervals, the power of the
  design, relative percentage deviation, hypervolume, IGD, and IGD+.
- **9 figure families**, from performance and data profiles to
  critical-difference diagrams, Pareto fronts, and Gantt charts, written as SVG
  and TikZ without a plotting library, and tables in Markdown, CSV, and LaTeX,
  collected in a byte-reproducible bundle with a manifest of SHA-256 digests.
- **Plugins**: other packages add solvers and benchmark families through
  entry points, and objectives through a registry.

## Citing

If you use DispatchAtlas in your research, please cite the version you used.
The citation metadata is in [CITATION.cff](https://github.com/ahmed-g-gad/dispatchatlas/blob/main/CITATION.cff); GitHub renders it as
APA and BibTeX under "Cite this repository", and the
[Citing page](https://dispatchatlas.ahmedgad.com/citing/) of the documentation
gives a BibTeX entry. Please also cite the methods you compare: the command
`dispatchatlas references --solver tabu-search --format bibtex` prints the
sources of a solver, and works the same for benchmark families and suites.

## Contributing

Contributions are welcome: see [CONTRIBUTING.md](https://github.com/ahmed-g-gad/dispatchatlas/blob/main/CONTRIBUTING.md) for the
development setup and conventions, and [CODE_OF_CONDUCT.md](https://github.com/ahmed-g-gad/dispatchatlas/blob/main/CODE_OF_CONDUCT.md)
for the community standards. Report security issues as described in
[SECURITY.md](https://github.com/ahmed-g-gad/dispatchatlas/blob/main/SECURITY.md).

## License

DispatchAtlas is released under the [MIT License](https://github.com/ahmed-g-gad/dispatchatlas/blob/main/LICENSE.txt).
