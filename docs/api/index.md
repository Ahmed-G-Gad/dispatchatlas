# API reference

The reference documents every public module of DispatchAtlas, generated from
the docstrings of the installed code. The package is organised in layers,
each depending only on packages above it in this list:

| Package | Holds |
| --- | --- |
| [`dispatchatlas.core`](core.md) | the data model, objectives, feasibility checking, seeds, serialization, and the solver contract; no third-party dependency |
| [`dispatchatlas.references`](references.md) | the bibliography of the implemented methods |
| [`dispatchatlas.benchmarks`](benchmarks.md) | benchmark families, published suites, file readers, catalogs, and instance features |
| [`dispatchatlas.solvers`](solvers.md) | the solver registry, the search run-time, the schedule builder, and every solver |
| [`dispatchatlas.processes`](processes.md) | pinned child processes for caller-defined calls, the rows file, process groups and snapshots, detached launch, and watching a run |
| [`dispatchatlas.campaigns`](campaigns.md) | campaign files, planning, execution, records, the check of a campaign directory, and replay |
| [`dispatchatlas.analysis`](analysis.md) | datasets, statistics, tables, figures, and bundles |
| [`dispatchatlas.cli`](references.md#dispatchatlas.cli) | the command line, a thin layer over the others |

Each package re-exports its main names, so `from dispatchatlas.core import
Problem` and `from dispatchatlas.solvers import solve` work; the reference
documents every name in the module that defines it. The solvers are grouped
by kind: [heuristics and local search](solvers-heuristics.md),
[population and swarm methods](solvers-population.md),
[multi-objective methods](solvers-multiobjective.md),
[learning methods](solvers-learning.md),
[exact and hybrid methods](solvers-exact.md), and
[job shop methods](solvers-shop.md). The analysis has separate
pages for its [statistics](analysis-stats.md) and its
[figures](analysis-figures.md).

Names that start with an underscore are private and may change without
notice. Errors the library raises on purpose derive from
`DispatchAtlasError`, and also from the matching built-in exception, such as
`ValueError` for `ConfigError` and `ModelError`, and `ImportError` for
`MissingDependencyError`.
