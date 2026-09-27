# Examples

Run the examples from the repository root, in the environment that `uv sync` creates. Each
script's docstring explains what it does, step by step.

| Script | Shows |
| --- | --- |
| `quickstart.py` | a generated job shop solved by three solvers, with every schedule checked against every constraint |
| `compare_solvers.py` | heuristics on a small flow shop against the optimum that `enumeration` proves, with paired seeds |
| `campaign_analysis.py` | a campaign run, analysed, compared with baselines, written as a bundle, and replayed, all through the Python API |
| `multi_objective_placement.py` | requests placed on an edge-fog-cloud continuum under response time and energy, with the fronts compared by hypervolume |

```bash
uv run python examples/quickstart.py
uv run python examples/campaign_analysis.py --output results/example
```

## Campaign files

`campaigns/` holds campaign files for the command line. The comment at the top of each file gives
its commands.

| File | Campaign |
| --- | --- |
| `quickstart.toml` | three solvers on three generated flow shops, the campaign of the getting-started guide |
| `reference.toml` | every key of the format, each one explained |
| `taillard-flow-shop.toml` | constructive, local-search, and population methods on the first ten Taillard flow shops, under equal evaluations and under equal wall-clock time |
| `continuum-placement.toml` | multi-objective algorithms against list heuristics on the continuum, analysed by hypervolume, IGD, and IGD+ |

`reference.toml` also reads `problem.json`, a problem document, and `instances/`, a directory of two
flexible job shops with its manifest.

## Plugin

`plugin/` is an installable package that adds the solver `insertion-descent` through the
`dispatchatlas.solvers` entry point. Its README shows how to install and use it.
