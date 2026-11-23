"""A synthetic noisy target with a known best configuration, for the tests of iterated racing.

The target has four parameters, one of each kind, and the best configuration is
:data:`BEST`. The cost of a run is the true cost of the configuration, scaled by a factor that
grows with the instance, so that the costs of different instances are not commensurable, plus
Gaussian noise that depends on the seed of the run and on the configuration. No solver runs.
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Sequence

from dispatchatlas.tuning import (
    Categorical,
    Configuration,
    Experiment,
    Integer,
    Ordinal,
    ParameterSpace,
    Real,
)

SPACE = ParameterSpace(
    [
        Real("x", 0.0, 1.0),
        Integer("n", 1, 20),
        Ordinal("k", [1, 2, 4, 8, 16]),
        Categorical("m", ["a", "b", "c"]),
    ]
)

BEST: Configuration = {"x": 0.3, "n": 7, "k": 4, "m": "b"}
"""The configuration of least true cost, 0."""

NOISE = 0.3
"""The standard deviation of the noise of a run on an instance of scale 1."""

_PENALTY = {"a": 0.5, "b": 0.0, "c": 1.0}
_ORDER = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}


def true_cost(configuration: Configuration) -> float:
    """Return the cost of a configuration without noise: 0 at :data:`BEST`, never negative."""
    x, n, k, m = (configuration[name] for name in ("x", "n", "k", "m"))
    assert isinstance(x, float)
    assert isinstance(n, int)
    assert isinstance(k, int)
    assert isinstance(m, str)
    return 10.0 * (x - 0.3) ** 2 + 0.05 * (n - 7) ** 2 + 0.3 * abs(_ORDER[k] - 2) + _PENALTY[m]


def noisy_cost(experiment: Experiment[int]) -> float:
    """Return the cost of one run on an instance that is an integer, its index in the set."""
    key = zlib.crc32(repr(sorted(experiment.configuration.items())).encode())
    noise = random.Random(experiment.seed * 1_000_003 + key).gauss(0.0, NOISE)
    return (1.0 + 0.5 * experiment.instance) * (1.0 + true_cost(experiment.configuration)) + noise


class Runner:
    """A ``run_batch`` of the noisy target that keeps every batch it was given."""

    def __init__(self) -> None:
        self.batches: list[list[Experiment[int]]] = []

    def __call__(self, experiments: Sequence[Experiment[int]]) -> list[float]:
        self.batches.append(list(experiments))
        return [noisy_cost(experiment) for experiment in experiments]

    @property
    def experiments(self) -> list[Experiment[int]]:
        """Every experiment run, in order."""
        return [experiment for batch in self.batches for experiment in batch]
