"""The covariance matrix adaptation evolution strategy (CMA-ES)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.continuous.core import Ordered, Run
from dispatchatlas.solvers.continuous.encoded import EncodedSolver

__all__ = ["CmaEs", "Strategy"]


@dataclass(frozen=True, eq=False)
class Strategy:
    """The strategy constants of CMA-ES for one dimension and population size.

    The defaults of Hansen's tutorial (2016) for positive recombination
    weights, as in Hansen and Ostermeier (2001) with the rank-mu update of
    Hansen, Müller and Koumoutsakos (2003):

    - ``mu = floor(lambda / 2)`` parents with weights proportional to
      ``ln((lambda + 1) / 2) - ln i``, summing to 1;
    - ``mu_eff = 1 / sum(w_i^2)``;
    - ``c_sigma = (mu_eff + 2) / (n + mu_eff + 5)`` and
      ``d_sigma = 1 + 2 max(0, sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma``;
    - ``c_c = (4 + mu_eff / n) / (n + 4 + 2 mu_eff / n)``;
    - ``c_1 = 2 / ((n + 1.3)^2 + mu_eff)`` and
      ``c_mu = min(1 - c_1, 2 (mu_eff - 2 + 1 / mu_eff) / ((n + 2)^2 + mu_eff))``;
    - ``chi_n = sqrt(n) (1 - 1 / (4 n) + 1 / (21 n^2))``, the expected length of
      a standard normal vector.

    Attributes:
        dimension: ``n``.
        offspring: ``lambda``.
        parents: ``mu``.
        weights: Recombination weights of the ``mu`` best offspring.
        mu_eff: Variance-effective selection mass.
        c_sigma: Learning rate of the step-size path.
        d_sigma: Damping of the step-size update.
        c_c: Learning rate of the covariance path.
        c_1: Learning rate of the rank-one update.
        c_mu: Learning rate of the rank-mu update.
        chi_n: Expected length of an ``n``-dimensional standard normal vector.
    """

    dimension: int
    offspring: int
    parents: int
    weights: npt.NDArray[np.float64]
    mu_eff: float
    c_sigma: float
    d_sigma: float
    c_c: float
    c_1: float
    c_mu: float
    chi_n: float

    @classmethod
    def default(cls, dimension: int, offspring: int | None = None) -> Strategy:
        """Return the constants for ``dimension``; ``lambda`` defaults to ``4 + floor(3 ln n)``."""
        n = float(dimension)
        lam = offspring if offspring is not None else 4 + int(3 * math.log(n))
        mu = lam // 2
        raw = math.log((lam + 1) / 2) - np.log(np.arange(1, mu + 1))
        weights = raw / raw.sum()
        mu_eff = 1.0 / float(np.sum(weights**2))
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0.0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        return cls(dimension, lam, mu, weights, mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu, chi_n)


@dataclass(frozen=True)
class CmaEs(EncodedSolver):
    """CMA-ES, the evolution strategy of Hansen and Ostermeier (2001).

    Each generation samples ``lambda`` points ``x_k = m + sigma y_k`` with
    ``y_k ~ N(0, C)`` and ranks them by fitness. Then:

    1. the mean moves to the weighted mean of the ``mu`` best points,
       ``m' = m + sigma y_w`` with ``y_w = sum(w_i y_{i:lambda})``;
    2. the conjugate evolution path ``p_sigma`` accumulates
       ``C^{-1/2} y_w`` and the step size follows its length:
       ``sigma' = sigma exp((c_sigma / d_sigma) (|p_sigma| / chi_n - 1))``
       (cumulative step-size adaptation);
    3. the evolution path ``p_c`` accumulates ``y_w``, except while
       ``|p_sigma|`` is too long, the ``h_sigma`` stall of the tutorial;
    4. the covariance combines a rank-one update with ``p_c`` and the rank-mu
       update of Hansen, Müller and Koumoutsakos (2003) with the selected steps:
       ``C' = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + delta C) + c_mu sum(w_i y_i y_i^T)``,
       where ``delta = (1 - h_sigma) c_c (2 - c_c)``.

    The eigendecomposition ``C = B D^2 B^T`` that sampling needs is refreshed
    once the evaluations since the last one exceed
    ``lambda / (10 n (c_1 + c_mu))``. The constants are those of
    :class:`Strategy`.

    The first distribution is centred on the box, or on the best seed, and
    has, along coordinate ``i``, a standard deviation of ``step_size`` times
    the box width. On a strict box a sample's coordinates outside the box are
    set to the nearest bound, the boundary repair of Hansen (2016), and the
    repaired point is both evaluated and used in the updates, so the mean, a
    weighted mean of repaired points, stays inside the box. Random keys need
    no repair. The step size and the axis ratio ``max(D) / min(D)`` of the
    final distribution appear in the result's extras.

    Attributes:
        population_size: Offspring ``lambda`` per generation, at least 2.
            ``None`` uses ``4 + floor(3 ln n)``.
        step_size: Initial step size as a fraction of the box width.
    """

    population_size: int | None = None
    step_size: float = 0.3

    info: ClassVar[SolverInfo] = SolverInfo(
        id="cma-es",
        name="CMA-ES",
        family="population",
        summary="Samples random-key vectors from a multivariate normal distribution, moves "
        "its mean to the weighted mean of the best samples, and adapts its step size and "
        "covariance matrix from evolution paths and the selected steps.",
        references=(
            "hansen2001completely",
            "hansen2003reducing",
            "hansen2016cma",
            "bean1994genetic",
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.population_size is not None and self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not self.step_size > 0.0:
            raise ValueError("step_size must be positive")

    def optimize[F: Ordered](self, run: Run[F], rng: random.Random) -> None:
        """Adapt the search distribution until the run is exhausted."""
        box = run.box
        n = box.dimension
        strategy = Strategy.default(n, self.population_size)
        lam, mu, weights = strategy.offspring, strategy.parents, strategy.weights
        generator = np.random.default_rng(rng.getrandbits(64))
        lower, upper = np.array(box.lower), np.array(box.upper)
        widths = upper - lower
        if run.seeds:
            best = min(range(len(run.seeds)), key=lambda k: run.seeds[k][1])
            mean = np.array(run.seeds[best][0], dtype=float)
        else:
            mean = (lower + upper) / 2
        sigma = self.step_size
        basis = np.eye(n)
        scales = widths.copy()
        covariance = np.diag(widths**2)
        path_sigma = np.zeros(n)
        path_c = np.zeros(n)
        generation = 0
        decomposed = 0
        stall_limit = (1.4 + 2 / (n + 1)) * strategy.chi_n
        path_scale = math.sqrt(strategy.c_sigma * (2 - strategy.c_sigma) * strategy.mu_eff)
        c_scale = math.sqrt(strategy.c_c * (2 - strategy.c_c) * strategy.mu_eff)
        while True:
            run.next_iteration()
            steps = generator.standard_normal((lam, n)) @ (basis * scales).T
            points = mean + sigma * steps
            if box.strict:
                points = np.clip(points, lower, upper)
                steps = (points - mean) / sigma
            fitness = [run.evaluate(point.tolist()) for point in points]
            ranking = sorted(range(lam), key=lambda k: fitness[k])
            selected = steps[ranking[:mu]]
            step = weights @ selected
            mean = mean + sigma * step
            generation += 1
            whitened = basis @ ((basis.T @ step) / scales)
            path_sigma = (1 - strategy.c_sigma) * path_sigma + path_scale * whitened
            length = float(np.linalg.norm(path_sigma))
            correction = math.sqrt(1 - (1 - strategy.c_sigma) ** (2 * generation))
            stalled = length / correction >= stall_limit
            path_c = (1 - strategy.c_c) * path_c + (0.0 if stalled else c_scale) * step
            delta = strategy.c_c * (2 - strategy.c_c) if stalled else 0.0
            rank_mu = (selected * weights[:, None]).T @ selected
            covariance = (
                (1 - strategy.c_1 - strategy.c_mu) * covariance
                + strategy.c_1 * (np.outer(path_c, path_c) + delta * covariance)
                + strategy.c_mu * rank_mu
            )
            sigma *= math.exp((strategy.c_sigma / strategy.d_sigma) * (length / strategy.chi_n - 1))
            if (generation - decomposed) * lam > lam / (10 * n * (strategy.c_1 + strategy.c_mu)):
                decomposed = generation
                covariance = np.triu(covariance) + np.triu(covariance, 1).T
                eigenvalues, basis = np.linalg.eigh(covariance)
                scales = np.sqrt(np.maximum(eigenvalues, 1e-20))
            run.note("sigma", sigma)
            run.note("axis_ratio", float(scales.max() / scales.min()))
