"""DispatchAtlas: reproducible benchmarking of scheduling algorithms.

The package is organised in layers:

- :mod:`dispatchatlas.core`: the data model, objectives, feasibility checking,
  seeds, serialization, and the solver contract.
- :mod:`dispatchatlas.benchmarks`: instance generators, published benchmark
  suites, and file-format readers.
- :mod:`dispatchatlas.solvers`: the solver registry and its heuristics,
  metaheuristics, exact methods, and learning methods.
- :mod:`dispatchatlas.processes`: pinned child processes for caller-defined
  calls, a resumable rows file, and process tools.
- :mod:`dispatchatlas.campaigns`: reproducible, resumable experiments.
- :mod:`dispatchatlas.analysis`: statistics, tables, figures, and reports.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dispatchatlas")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
