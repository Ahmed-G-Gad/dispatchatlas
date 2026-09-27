"""Bayesian sign and signed-rank tests over paired differences.

Both tests of Benavoli, Corani, Demšar and Zaffalon (2017) put a posterior on
the probability mass of three regions of a paired difference: below the
region of practical equivalence (the *rope*, ``[-rope, rope]``, bounds
included), inside it, and above it.

- The **sign test** counts the differences in each region. The posterior of
  the three masses is ``Dirichlet(n_below, n_rope + prior, n_above)``: the
  prior's pseudo-observations sit in the rope.
- The **signed-rank test** puts a Dirichlet process on the distribution of
  the differences, with a prior pseudo-observation at zero of weight
  ``prior``. Each posterior draw weighs the atoms ``0, d_1, ..., d_n`` by
  ``Dirichlet(prior, 1, ..., 1)`` weights ``w``, and a region's mass is the
  total weight ``w_i w_j`` of the ordered pairs whose Walsh average
  ``(z_i + z_j) / 2`` falls in it.

Each result reports two things per region. ``mean_*`` is the posterior
expected mass. ``probability_*`` is the posterior probability that the region
holds the largest of the three masses, the decision quantity of the authors'
reference implementation ``baycomp``, estimated from ``samples`` Monte Carlo
draws of ``numpy.random.default_rng(seed)``. For the sign test the expected
masses are exact; for the signed-rank test they are Monte Carlo averages of
the same draws.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dispatchatlas.analysis.stats.ranks import finite_or_infinite
from dispatchatlas.core.errors import ConfigError

__all__ = ["BayesianResult", "bayesian_sign_test", "bayesian_signed_rank_test"]


@dataclass(frozen=True, slots=True)
class BayesianResult:
    """Posterior facts about the three regions of a paired difference.

    Attributes:
        probability_below: Posterior probability that the region below
            ``-rope`` holds the largest mass.
        probability_rope: Same, for the rope.
        probability_above: Same, for the region above ``rope``.
        mean_below: Posterior expected mass below ``-rope``.
        mean_rope: Posterior expected mass of the rope.
        mean_above: Posterior expected mass above ``rope``.
        rope: Half-width of the rope.
        prior: Strength of the prior.
        pairs: Number of differences.
        samples: Monte Carlo draws.
        seed: Seed of the draws.
    """

    probability_below: float
    probability_rope: float
    probability_above: float
    mean_below: float
    mean_rope: float
    mean_above: float
    rope: float
    prior: float
    pairs: int
    samples: int
    seed: int


def bayesian_sign_test(
    differences: Iterable[float],
    *,
    rope: float = 0.0,
    prior: float = 1.0,
    samples: int = 2000,
    seed: int = 0,
) -> BayesianResult:
    """Return the Bayesian sign test of paired differences.

    Args:
        differences: One difference per matched pair; infinities are allowed.
        rope: Half-width of the region of practical equivalence, at least 0.
        prior: Pseudo-observations the prior places in the rope, positive.
        samples: Monte Carlo draws for the probabilities, positive.
        seed: Seed of the draws, at least 0.

    Returns:
        Exact posterior means and Monte Carlo probabilities.

    Raises:
        ConfigError: If there is no difference, a difference is NaN, or a
            parameter is out of range.
    """
    values = finite_or_infinite(differences, "differences")
    _check(values, rope, prior, samples, seed)
    below = sum(1 for value in values if value < -rope)
    above = sum(1 for value in values if value > rope)
    inside = len(values) - below - above
    concentration = (float(below), inside + prior, float(above))
    generator = np.random.default_rng(seed)
    draws = np.column_stack(
        [
            generator.gamma(shape, size=samples) if shape > 0.0 else np.zeros(samples)
            for shape in concentration
        ]
    )
    total = prior + len(values)
    means = (below / total, (inside + prior) / total, above / total)
    return _result(draws, means, rope, prior, len(values), samples, seed)


def bayesian_signed_rank_test(
    differences: Iterable[float],
    *,
    rope: float = 0.0,
    prior: float = 0.5,
    samples: int = 2000,
    seed: int = 0,
) -> BayesianResult:
    """Return the Bayesian signed-rank test of paired differences.

    The cost grows as ``samples * (n + 1)^2``.

    Args:
        differences: One finite difference per matched pair.
        rope: Half-width of the region of practical equivalence, at least 0.
        prior: Weight of the prior pseudo-observation at zero, positive.
        samples: Monte Carlo draws, positive.
        seed: Seed of the draws, at least 0.

    Returns:
        Monte Carlo posterior means and probabilities.

    Raises:
        ConfigError: If there is no difference, a difference is not finite,
            or a parameter is out of range.
    """
    values = finite_or_infinite(differences, "differences")
    _check(values, rope, prior, samples, seed)
    if not all(math.isfinite(value) for value in values):
        raise ConfigError("the Bayesian signed-rank test needs finite differences")
    atoms = np.array([0.0, *values])
    midpoints = (atoms[:, np.newaxis] + atoms[np.newaxis, :]) / 2.0
    below_pairs = (midpoints < -rope).astype(np.float64)
    above_pairs = (midpoints > rope).astype(np.float64)
    weights = np.ones(atoms.size)
    weights[0] = prior
    generator = np.random.default_rng(seed)
    draws = generator.dirichlet(weights, size=samples)
    mass_below = np.sum((draws @ below_pairs) * draws, axis=1)
    mass_above = np.sum((draws @ above_pairs) * draws, axis=1)
    masses = np.column_stack([mass_below, 1.0 - mass_below - mass_above, mass_above])
    means = (
        float(np.mean(masses[:, 0])),
        float(np.mean(masses[:, 1])),
        float(np.mean(masses[:, 2])),
    )
    return _result(masses, means, rope, prior, len(values), samples, seed)


def _check(values: list[float], rope: float, prior: float, samples: int, seed: int) -> None:
    if not values:
        raise ConfigError("a Bayesian test needs at least one difference")
    if not (math.isfinite(rope) and rope >= 0.0):
        raise ConfigError(f"the rope must be a finite number of at least 0, got {rope!r}")
    if not (math.isfinite(prior) and prior > 0.0):
        raise ConfigError(f"the prior strength must be positive, got {prior!r}")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ConfigError(f"samples must be a positive integer, got {samples!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"the seed must be a non-negative integer, got {seed!r}")


def _result(
    masses: NDArray[np.float64],
    means: tuple[float, float, float],
    rope: float,
    prior: float,
    pairs: int,
    samples: int,
    seed: int,
) -> BayesianResult:
    """Build the result from draws of the three region masses (columns, unnormalised)."""
    winners = np.bincount(np.argmax(masses, axis=1), minlength=3) / samples
    return BayesianResult(
        probability_below=float(winners[0]),
        probability_rope=float(winners[1]),
        probability_above=float(winners[2]),
        mean_below=means[0],
        mean_rope=means[1],
        mean_above=means[2],
        rope=float(rope),
        prior=float(prior),
        pairs=pairs,
        samples=samples,
        seed=seed,
    )
