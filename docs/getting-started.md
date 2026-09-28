# Getting started

This page installs DispatchAtlas, solves a first problem from Python and from
the command line, and runs, analyses, and replays a first campaign.

## Installation

DispatchAtlas needs Python 3.12 or newer. Its only required dependency is
NumPy. It installs from its repository with [uv](https://docs.astral.sh/uv/).

=== "In a project"

    ```bash
    uv add "dispatchatlas @ git+https://github.com/ahmed-g-gad/dispatchatlas"
    uv add "dispatchatlas[exact] @ git+https://github.com/ahmed-g-gad/dispatchatlas"   # with the exact solvers' backends
    ```

=== "The command line"

    ```bash
    uv tool install git+https://github.com/ahmed-g-gad/dispatchatlas
    uvx --from git+https://github.com/ahmed-g-gad/dispatchatlas dispatchatlas --help   # without installing it
    ```

=== "From source"

    ```bash
    git clone https://github.com/ahmed-g-gad/dispatchatlas.git
    cd dispatchatlas
    uv sync                              # .venv with the package and the development tools
    uv run dispatchatlas --help
    ```

The optional extras add solver backends:

| Extra | Installs | Enables |
| --- | --- | --- |
| `exact` | OR-Tools, PuLP | the `cp-sat` and `milp` solvers (CBC backend) and `restricted-neighbourhood-search` |
| `highs` | highspy | the `highs` backend of `milp`, with `exact` |
| `scip` | PySCIPOpt | the `scip` backend of `milp`, with `exact` |
| `gurobi` | gurobipy | the `gurobi` backend of `milp`, with `exact`; needs a Gurobi licence beyond small models |
| `all` | `exact`, `highs`, `scip` | every open-source backend |

`dispatchatlas version` shows what is installed: the versions of the package,
Python, and NumPy, each optional backend with the extra that provides it, and
the installed plugins.

## A first problem in Python

A benchmark family generates a problem from a seed. Equal parameters and seed
give the same problem:

```python
from dispatchatlas.benchmarks import generate
from dispatchatlas.core import check_schedule
from dispatchatlas.solvers import solve

problem = generate("job-shop", seed=1, jobs=6, machines=4)
print(problem.summary())
```

```text
{'id': 'job-shop-6x4-1', 'family': 'job-shop', 'tasks': 24, 'resources': 4, 'dependencies': 18, 'modes': 24, 'groups': 6, 'objectives': ['makespan'], 'features': ['groups', 'precedence']}
```

Six jobs of four operations each: 24 tasks on 4 machines, with the order of
the operations of a job as 18 dependencies. `solve` runs a registered solver
by its id. Iterative solvers need a limit, here 2000 evaluated schedules:

```python
result = solve(problem, "tabu-search", max_evaluations=2000, seed=3)
print(result.status, result.value, result.evaluations, result.stop_reason)
print(check_schedule(problem, result.schedule))
print(result.schedule.assignments[:2])
```

```text
feasible 361.0 2000 evaluation-limit
()
(Assignment(task='j1-o0', mode=0, start=0.0, end=33.0), Assignment(task='j2-o0', mode=0, start=0.0, end=58.0))
```

The result holds the schedule, the value of every objective of the problem,
why the run stopped, and how many schedules it evaluated. `check_schedule`
lists the constraints a schedule breaks, none here; it does not share code
with the solvers, so it is an independent check of their output. Running the
same call again gives the same schedule.

Problems, schedules, and results are stored as JSON documents:

```python
from dispatchatlas.core import load_problem, save_problem, save_result

save_problem(problem, "js.json")
save_result(result, "tabu.json")
assert load_problem("js.json") == problem
```

## The same from the command line

```bash
dispatchatlas benchmarks generate job-shop --seed 1 --set jobs=6 --set machines=4 --output results/js.json
dispatchatlas solve results/js.json --solver tabu-search --max-evaluations 2000 --seed 3 --output results/tabu.json
dispatchatlas check results/js.json results/tabu.json
```

```text
wrote problem job-shop-6x4-1 (24 tasks, 4 resources) to results/js.json
solver       tabu-search (seed 3)
problem      job-shop-6x4-1
status       feasible
stop reason  evaluation-limit
objectives   makespan 361
effort       2000 evaluations, 100 iterations
time         0.225 s
check        every constraint holds
result       results/tabu.json
problem     job-shop-6x4-1: valid, 24 tasks, 4 resources, 18 dependencies
features    groups, precedence
objectives  makespan
schedule    results/tabu.json (result, 24 assignments)
verdict     feasible
values      makespan 361
```

Only the reported time differs between machines. `dispatchatlas solvers
list`, `dispatchatlas benchmarks list`, and `dispatchatlas suites list` show
what is available, and `--help` explains every command.

## A first campaign

A campaign compares solvers under equal conditions. This one, from the
repository's `examples/campaigns/quickstart.toml`, runs three solvers on three
generated flow shops, each run limited to 1000 evaluated schedules:

```toml
--8<-- "examples/campaigns/quickstart.toml"
```

`neh` is deterministic and runs once per problem; the two stochastic solvers
run five replicates each, with the same five seeds per problem. Count the runs
first, then run them:

```bash
dispatchatlas campaign estimate examples/campaigns/quickstart.toml
dispatchatlas campaign run examples/campaigns/quickstart.toml --output results
```

```text
campaign     quickstart (config hash 06cb9a987c5c)
runs         33 planned: 0 completed, 0 failed, 33 remaining
grid         3 problems, 3 solvers, 1 protocol, 0 excluded pairs
by protocol  evaluations-1000 33
by solver    neh 3, simulated-annealing 15, random-search 15
workers      1
duration     33 runs without a time limit, whose duration has no bound

campaign   quickstart, session 1
directory  results/quickstart
runs       33 planned: 33 completed, 0 failed, 0 pending
session    33 attempts: 33 succeeded, 0 failed, in 3.48 s
```

Every run leaves one JSON record under `results/quickstart/runs/`. Running the
command again resumes the campaign: completed runs are kept and only missing
ones run. Analyse the records:

```bash
dispatchatlas analyze results/quickstart --output results/quickstart-analysis
```

The bundle holds an `index.md` that lists every table and figure, the
limitations of the analysis, the runs as CSV, the tables in Markdown, CSV, and
LaTeX, the figures in SVG and TikZ, and a manifest with every file's digest.
The solver table of this campaign:

| Solver | Runs | Succeeded | Infeasible | Failed | Missing | Success rate | Problems | Average rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| simulated-annealing | 15 | 15 | 0 | 0 | 0 | 1 | 3 | 1.33333 |
| neh | 3 | 3 | 0 | 0 | 0 | 1 | 3 | 2.33333 |
| random-search | 15 | 15 | 0 | 0 | 0 | 1 | 3 | 2.33333 |

Three problems are too few for any pairwise test to reach significance, and
`limitations.md` says so rather than leaving it to the reader. Finally, check
that the runs reproduce:

```bash
dispatchatlas campaign replay results/quickstart
```

```text
campaign  quickstart (config hash 06cb9a987c5c)
verdicts  33 match, 0 mismatch, 0 skipped, 0 failed
```

## The same campaign from Python

The process backend starts its workers with the `spawn` method, which imports
the main module again in every worker, so a script that runs a campaign keeps
its work under `if __name__ == "__main__":`.

```python
from dispatchatlas.analysis import load_dataset, summarize, write_bundle
from dispatchatlas.campaigns import load_campaign, replay, run


def main() -> None:
    campaign = load_campaign("examples/campaigns/quickstart.toml")
    summary = run(campaign, "results")
    dataset = load_dataset(summary.directory)
    for row in summarize(dataset).solver_summaries:
        print(row.solver, row.succeeded, row.planned, round(row.average_rank, 2))
    write_bundle(summary.directory, "results/quickstart-analysis")
    print(replay(campaign, "results").ok)


if __name__ == "__main__":
    main()
```

```text
neh 3 3 2.33
simulated-annealing 15 15 1.33
random-search 15 15 2.33
True
```

## Next steps

- The [user guide](guide/model.md) explains the data model, benchmarks,
  solvers, campaigns, and analysis in depth.
- The examples in the repository's
  [`examples/`](https://github.com/ahmed-g-gad/dispatchatlas/tree/main/examples)
  directory compare solvers against a proven optimum, run and analyse a
  campaign from Python, and place requests on a continuum under two
  objectives.
