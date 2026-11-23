"""Iterated racing: automatic configuration of a solver's parameters (López-Ibáñez et al. 2016).

Iterated racing repeats three steps: it samples new configurations from a distribution,
selects the best of them by racing, and updates the distribution to favour the best
configurations found so far (López-Ibáñez et al. 2016, p. 45). :class:`IteratedRacing`
is the elitist variant that irace uses by default (p. 49) over a
:class:`~dispatchatlas.tuning.space.ParameterSpace`, with its budget counted in runs.

One run of :meth:`IteratedRacing.tune`, with ``N_param`` the number of parameters:

1. It plans ``N_iter = floor(2 + log2(N_param))`` races (:func:`default_iterations`).
   Race ``j`` gets the share ``B_j = (B - B_used) / (N_iter - j + 1)`` of the budget
   that is left (:func:`iteration_budget`), and starts from
   ``N_j = floor((B_j + N_elite * e) / max(first_test + each_test * min(5, j),
   nm(T_new + e, each_test)))`` configurations (:func:`candidate_count`).
2. The first race starts from configurations that are uniform over the space. Later
   races start from the elite configurations of the race before and
   ``N_j - N_elite`` new ones, each drawn around a parent that is chosen from the
   elites with probability proportional to its rank (:func:`parent_probabilities`).
   Before drawing, each elite's sampling model narrows (:func:`narrow_model`): the
   standard deviation of a numerical parameter shrinks by a factor that grows with the
   number of new candidates, and a categorical parameter's probabilities move towards
   the parent's value. A new configuration inherits its parent's model.
3. The race (:func:`~dispatchatlas.tuning.racing.race`) is elitist. It runs on
   ``T_new`` new instances followed by the ``e`` instances the elites have costs on,
   in a shuffled order. The elites are not discarded before the new candidates have
   been evaluated on all of them, their costs there are not run again, and a race
   ends after ``T_max`` consecutive tests that discard nobody.
4. Its best ``min(N_surv, N_min)`` survivors become the elites. When two of the
   candidates of a race are the same configuration, which means the sampling models
   have converged, the elites that produced them take a soft restart
   (:func:`restart_model`) and the new candidates are drawn again, once per race.
5. The run ends when the budget is spent or a race would have no more candidates than
   elites. The elites of the last race are the result.

Every experiment is one run, performed by the caller's ``run_batch``, and every
record is returned, so that a study can archive its tuning trace. Every random draw
comes from a generator derived with
:func:`~dispatchatlas.core.seeds.derive_seed` from the seed of the run, so the same
seed and the same costs give the same configurations.

Not implemented are the conditional parameters (p. 47), the initial configurations
(p. 49), and the forbidden configurations (pp. 49-50) of the irace package, its paired
t-test as an alternative test (p. 47), and its parallel evaluation (p. 49); the
caller's ``run_batch`` decides how a batch of experiments runs.
"""

from __future__ import annotations

import bisect
import itertools
import math
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import numpy as np

from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.seeds import derive_seed
from dispatchatlas.tuning.racing import Block, Entry, RaceResult, Record, RunBatch, race
from dispatchatlas.tuning.space import (
    Categorical,
    Configuration,
    ParameterSpace,
    SamplingModel,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "IteratedRacing",
    "TuningResult",
    "candidate_count",
    "default_iterations",
    "default_survivors",
    "iteration_budget",
    "narrow_model",
    "parent_probabilities",
    "restart_model",
]

_CATEGORICAL_CEILING = 0.2
"""The largest probability of a level after the update is ``0.2 ** (1 / N_param)``, which keeps
categorical parameters from converging too fast in elitist racing (López-Ibáñez et al. 2016,
p. 49, equation 6)."""

_RESTART_KEEP = 0.9
"""The share of a probability that the soft restart keeps; the rest comes from the largest one
(López-Ibáñez et al. 2016, p. 48)."""


def default_iterations(parameters: int) -> int:
    """Return the default number of races, ``floor(2 + log2(parameters))``.

    The motivation is to give a larger space more iterations, with at least two so that
    the search can intensify (López-Ibáñez et al. 2016, p. 46).

    Args:
        parameters: The number of parameters, at least 1.

    Raises:
        ConfigError: If ``parameters`` is not a positive integer.
    """
    _positive(parameters, "parameters")
    return math.floor(2.0 + math.log2(parameters))


def default_survivors(parameters: int) -> int:
    """Return the default minimum number of survivors, ``floor(2 + log2(parameters))``.

    The paper names this setting ``N_min`` and prints no default for it (López-Ibáñez
    et al. 2016, p. 47 and Table A.1, p. 55), so the value is the library's choice: the
    default number of races of :func:`default_iterations`. It bounds the number of
    elites that a race hands on, and ends a race once that few configurations remain.

    Args:
        parameters: The number of parameters, at least 1.

    Raises:
        ConfigError: If ``parameters`` is not a positive integer.
    """
    return default_iterations(parameters)


def iteration_budget(remaining: int, iterations: int, iteration: int) -> int:
    """Return the runs that race ``iteration`` may spend, the share it takes of those left.

    The share is ``remaining / (iterations - iteration + 1)`` rounded down: the runs left
    divided equally among the races still to come. This is
    ``B_j = (B - B_used) / (N_iter - j + 1)`` of López-Ibáñez et al. (2016, p. 46), a
    whole number of runs here. The runs that a race leaves unspent go to the next ones.

    Args:
        remaining: The runs left, ``B - B_used``, not negative.
        iterations: The planned number of races, ``N_iter``, at least ``iteration``.
        iteration: The race, ``j``, from 1.

    Raises:
        ConfigError: If an argument is out of range.
    """
    _positive(iteration, "iteration")
    _positive(iterations, "iterations")
    if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 0:
        raise ConfigError(f"remaining must be a non-negative integer, got {remaining!r}")
    if iterations < iteration:
        raise ConfigError(f"iteration {iteration} lies beyond the {iterations} planned")
    return remaining // (iterations - iteration + 1)


def candidate_count(
    budget: int,
    *,
    first_test: int,
    each_test: int,
    iteration: int,
    elites: int = 0,
    seen: int = 0,
    new_instances: int = 1,
) -> int:
    """Return the number of configurations of a race, ``N_j``.

    With ``e`` the instances the elites have costs on (``seen``), the number is
    ``floor((B_j + N_elite * e) / max(mu + T_each * min(5, j), nm(T_new + e, T_each)))``
    (López-Ibáñez et al. 2016, equation 5, p. 49), where ``mu`` is ``first_test`` (p. 47),
    ``nm(x, d)`` is the smallest multiple of ``d`` not below ``x``, and the elites'
    costs on ``e`` instances cost the race nothing. The number falls with the race
    because later races need more evaluations to tell similar configurations apart
    (p. 47), and the term ``min(5, j)`` stops the fall after the fifth. In the first
    race, with no elites and no instances seen, it reduces to
    ``floor(B_1 / (first_test + each_test))`` (pp. 46-47). The paper writes the elites of the
    race as ``N_j^elite``; here they are the elites that enter it.

    Args:
        budget: The runs of the race, ``B_j``.
        first_test: Instances before the first test, ``T_first`` (``mu``).
        each_test: Instances between tests, ``T_each``.
        iteration: The race, ``j``, from 1.
        elites: The elites that enter the race, ``N_elite``.
        seen: The instances the elites have costs on, ``e``.
        new_instances: New instances in front of the old ones, ``T_new``.

    Raises:
        ConfigError: If an argument is out of range.
    """
    for name, value, least in (
        ("budget", budget, 0),
        ("first_test", first_test, 1),
        ("each_test", each_test, 1),
        ("iteration", iteration, 1),
        ("elites", elites, 0),
        ("seen", seen, 0),
        ("new_instances", new_instances, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < least:
            raise ConfigError(f"{name} must be an integer of at least {least}, got {value!r}")
    multiple = -(-(new_instances + seen) // each_test) * each_test
    needed = max(first_test + each_test * min(5, iteration), multiple)
    return (budget + elites * seen) // needed


def parent_probabilities(elites: int) -> tuple[float, ...]:
    """Return the probability of choosing each elite as the parent of a new configuration.

    The elite of rank ``r`` among ``n`` elites has probability
    ``(n - r + 1) / (n (n + 1) / 2)``, proportional to ``n - r + 1``, so that higher
    ranked elites are more likely (López-Ibáñez et al. 2016, equation 1, p. 47). The
    result is in rank order, the best first.

    Args:
        elites: The number of elites, at least 1.

    Raises:
        ConfigError: If ``elites`` is not a positive integer.
    """
    _positive(elites, "elites")
    total = elites * (elites + 1) / 2.0
    return tuple((elites - rank + 1) / total for rank in range(1, elites + 1))


def narrow_model(
    space: ParameterSpace,
    model: SamplingModel,
    parent: Configuration,
    *,
    iteration: int,
    iterations: int,
    new: int,
) -> SamplingModel:
    """Return an elite's sampling model after the update that precedes sampling in race ``j``.

    The update applies, in order, to the model of an elite ``parent``:

    - the standard deviation of every numerical parameter is multiplied by
      ``(1 / new) ** (1 / N_param)`` (equation 2, p. 47), so that the volume of the
      sampling region shrinks by a constant factor that is larger the more new
      candidates are drawn;
    - the probability of every level of a categorical parameter is multiplied by
      ``1 - (j - 1) / N_iter``, and the level that the parent has gains
      ``(j - 1) / N_iter`` (equations 3 and 4, p. 48);
    - then no probability stays above ``0.2 ** (1 / N_param)`` (equation 6, p. 49),
      which is the elitist variant's guard against too fast a convergence, and the
      probabilities are normalised.

    Args:
        space: The parameter space.
        model: The elite's model before the update.
        parent: The elite's configuration.
        iteration: The race, ``j``, at least 2 (the first race samples uniformly).
        iterations: The planned races, ``N_iter``, at least ``iteration``.
        new: The number of new configurations of the race, ``N_new``, at least 1.

    Raises:
        ConfigError: If an argument is out of range, or the model or the parent lacks
            a parameter.
    """
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 2:
        raise ConfigError(f"narrowing needs the iteration to be at least 2, got {iteration!r}")
    _positive(new, "new")
    if iterations < iteration:
        raise ConfigError(f"iteration {iteration} lies beyond the {iterations} planned")
    parameters = len(space)
    factor = (1.0 / new) ** (1.0 / parameters)
    toward = (iteration - 1) / iterations
    ceiling = _CATEGORICAL_CEILING ** (1.0 / parameters)
    spreads = {name: spread * factor for name, spread in model.spreads.items()}
    weights: dict[str, tuple[float, ...]] = {}
    for parameter in space.parameters:
        if not isinstance(parameter, Categorical):
            continue
        name = parameter.name
        if name not in model.weights or name not in parent:
            raise ConfigError(f"the model or the parent lacks the parameter {name!r}")
        moved = [weight * (1.0 - toward) for weight in model.weights[name]]
        moved[parameter.index(parent[name])] += toward
        capped = [min(weight, ceiling) for weight in moved]
        total = math.fsum(capped)
        weights[name] = tuple(weight / total for weight in capped)
    return SamplingModel(spreads, weights)


def restart_model(space: ParameterSpace, model: SamplingModel, *, new: int) -> SamplingModel:
    """Return an elite's sampling model after a soft restart (López-Ibáñez et al. 2016, p. 48).

    A soft restart partly re-initialises the model of an elite whose new configurations
    came out the same as others:

    - each probability ``p`` of a categorical parameter becomes
      ``0.9 p + 0.1 max(P)``, for ``max(P)`` the largest probability of the parameter,
      and the probabilities are normalised;
    - the standard deviation of a numerical parameter, an ordinal one in indices,
      goes back two races, with the value it had in the second race as its ceiling:
      ``min(s * new ** (2 / N_param), (range / 2) * (1 / new) ** (1 / N_param))``,
      where ``range`` is ``high - low``, or the number of levels minus one for an ordinal
      parameter.

    Args:
        space: The parameter space.
        model: The elite's model after :func:`narrow_model` in this race.
        new: The number of new configurations of the race, ``N_new``, at least 1.

    Raises:
        ConfigError: If ``new`` is not positive or the model lacks a parameter.
    """
    _positive(new, "new")
    parameters = len(space)
    spreads: dict[str, float] = {}
    weights: dict[str, tuple[float, ...]] = {}
    for parameter in space.parameters:
        name = parameter.name
        if isinstance(parameter, Categorical):
            if name not in model.weights:
                raise ConfigError(f"the model lacks the parameter {name!r}")
            old = model.weights[name]
            mixed = [_RESTART_KEEP * p + (1.0 - _RESTART_KEEP) * max(old) for p in old]
            total = math.fsum(mixed)
            weights[name] = tuple(p / total for p in mixed)
        else:
            if name not in model.spreads:
                raise ConfigError(f"the model lacks the parameter {name!r}")
            back = model.spreads[name] * new ** (2.0 / parameters)
            second = parameter.spread * (1.0 / new) ** (1.0 / parameters)
            spreads[name] = min(back, second)
    return SamplingModel(spreads, weights)


@dataclass(frozen=True, slots=True)
class TuningResult[I]:
    """The outcome of a tuning run.

    Attributes:
        elites: The elite configurations of the last race, the best first.
        runs: The number of experiments run, which never exceeds the budget.
        records: Every experiment the run performed, with its cost, in the order they
            ran: the archive of the tuning trace.
        races: The result of every race, in order. The records of a race, its stop
            reason, and the rank sums of its survivors are there.
    """

    elites: tuple[Configuration, ...]
    runs: int
    records: tuple[Record[I], ...]
    races: tuple[RaceResult[I], ...]

    @property
    def best(self) -> Configuration:
        """The best elite configuration, the first of ``elites``."""
        return self.elites[0]


@dataclass(frozen=True, slots=True)
class IteratedRacing:
    """The elitist iterated racing of irace, with a budget of runs and a seed.

    Calling :meth:`tune` configures a solver on a set of instances; the module
    description gives the algorithm. The settings other than ``space``, ``budget`` and
    ``seed`` are those of irace as the paper prints them, each with the page, or the
    library's own choice where the paper prints none.

    Attributes:
        space: The parameter space.
        budget: The most experiments to run, ``B`` (López-Ibáñez et al. 2016, p. 46), at
            least 1. An experiment is the application of a configuration to an instance.
        seed: The seed from which every random draw is derived, not negative.
        first_test: Instances evaluated before the first test, ``T_first``; the default
            is 5 (pp. 49, 53), and at least 2.
        each_test: Instances between two tests, ``T_each``; the default is 1 (p. 47).
        alpha: The level of the Friedman test and of Conover's comparisons; the default
            is 0.05 (p. 47).
        new_instances: New instances at the front of every race after the first,
            ``T_new``; the default is 1 (pp. 49, 53).
        idle_tests: A race ends after this many consecutive tests that discard nobody,
            ``T_max``; the default is 2 (p. 49).
        min_survivors: The fewest survivors, ``N_min``. A race ends once this few
            remain, and at most this many elites go on. ``None``, the default, takes
            :func:`default_survivors`, the library's choice, since the paper prints no
            default.
        iterations: The planned number of races, ``N_iter``; ``None``, the default,
            takes :func:`default_iterations` (p. 46).
        sample_instances: Whether the instances come in a random order, a new one for each
            pass through the set (the ``sampleInstances`` option, p. 56). The paper does not
            print its default; the library's choice is ``True``. With ``False`` they come
            in the order given, which lets a caller arrange them in blocks of one instance
            per class, as the paper advises for classes (p. 47).
    """

    space: ParameterSpace
    budget: int
    _: KW_ONLY
    seed: int
    first_test: int = 5
    each_test: int = 1
    alpha: float = 0.05
    new_instances: int = 1
    idle_tests: int = 2
    min_survivors: int | None = None
    iterations: int | None = None
    sample_instances: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.space, ParameterSpace):
            raise ConfigError(f"space must be a ParameterSpace, got {self.space!r}")
        for name, value, least in (
            ("budget", self.budget, 1),
            ("seed", self.seed, 0),
            ("first_test", self.first_test, 2),
            ("each_test", self.each_test, 1),
            ("new_instances", self.new_instances, 1),
            ("idle_tests", self.idle_tests, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                raise ConfigError(f"{name} must be an integer of at least {least}, got {value!r}")
        for name, optional in (
            ("min_survivors", self.min_survivors),
            ("iterations", self.iterations),
        ):
            if optional is not None and (
                isinstance(optional, bool) or not isinstance(optional, int) or optional < 1
            ):
                raise ConfigError(
                    f"{name} must be None or an integer of at least 1, got {optional!r}"
                )
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise ConfigError(f"alpha must be a number, got {self.alpha!r}")
        if not 0.0 < self.alpha < 1.0:
            raise ConfigError(f"alpha must lie strictly between 0 and 1, got {self.alpha!r}")
        if not isinstance(self.sample_instances, bool):
            raise ConfigError(
                f"sample_instances must be true or false, got {self.sample_instances!r}"
            )

    def tune[I](self, instances: Sequence[I], run_batch: RunBatch[I]) -> TuningResult[I]:
        """Search the space for the configurations that cost least on the instances.

        Each call starts afresh and draws only from generators derived from ``seed``, so
        the same ``seed``, instances, and costs from ``run_batch`` give the same
        configurations, records, and result. Every block of instances gets a seed of its
        own, which every configuration shares on it, and the same instance on a later pass
        gets a new seed (López-Ibáñez et al. 2016, p. 56). ``run_batch`` gets the
        experiments of one step of a race at a time, which is the batch to run in
        parallel.

        Args:
            instances: The tuning instances, at least one. They are handed to
                ``run_batch`` as they are.
            run_batch: The function that runs experiments and returns their costs, smaller
                being better. It must return one cost per experiment, in order, and no NaN.

        Returns:
            The elite configurations of the last race, with every record.

        Raises:
            ConfigError: If there is no instance, or the budget affords the first race no
                more than ``N_min`` configurations, which a race could not narrow down;
                the paper warns that a budget that is too small gives a configuration no
                better than a random one (p. 55).
        """
        if len(instances) == 0:
            raise ConfigError("tuning needs at least one instance")
        parameters = len(self.space)
        planned = self.iterations if self.iterations is not None else default_iterations(parameters)
        keep = (
            self.min_survivors if self.min_survivors is not None else default_survivors(parameters)
        )
        stream = _Stream(instances, self.seed, self.sample_instances)
        blocks: dict[int, tuple[int, Block[I]]] = {}
        elites: list[_Candidate] = []
        races: list[RaceResult[I]] = []
        used = fresh = iteration = 0
        while True:
            iteration += 1
            planned = max(planned, iteration)
            budget = iteration_budget(self.budget - used, planned, iteration)
            old = sorted({seed for c in elites for seed in c.costs}, key=lambda s: blocks[s][0])
            count = candidate_count(
                budget,
                first_test=self.first_test,
                each_test=self.each_test,
                iteration=iteration,
                elites=len(elites),
                seen=len(old),
                new_instances=self.new_instances,
            )
            if iteration == 1 and count <= keep:
                raise ConfigError(
                    f"a budget of {self.budget} runs affords the first race {count} "
                    f"configurations, no more than the {keep} survivors; raise the budget"
                )
            if iteration > 1 and count <= len(elites):
                break
            new = count - len(elites)
            rng = np.random.default_rng(derive_seed(self.seed, "tuning:sample", iteration))
            children = self._sample(rng, elites, new, iteration, planned)
            shuffle = np.random.default_rng(derive_seed(self.seed, "tuning:order", iteration))
            fresh_blocks = [stream.block(fresh + i) for i in range(self.new_instances)]
            tail = [stream.block(fresh + self.new_instances + i) for i in range(budget // 2 + 1)]
            for offset, block in enumerate([*fresh_blocks, *tail]):
                blocks[block.seed] = (fresh + offset, block)
            old_blocks = [blocks[old[i]][1] for i in shuffle.permutation(len(old)).tolist()]
            pool = [*elites, *children]
            result = race(
                [child.configuration for child in children],
                [*fresh_blocks, *old_blocks, *tail],
                run_batch,
                budget=budget,
                first_test=self.first_test,
                each_test=self.each_test,
                alpha=self.alpha,
                elites=[Entry(c.configuration, c.costs) for c in elites],
                min_survivors=keep,
                idle_tests=self.idle_tests,
            )
            for candidate, entry in zip(pool, result.entries, strict=True):
                candidate.costs = dict(entry.costs)
            races.append(result)
            used += result.runs
            known = set(old)
            fresh += sum(1 for block in result.blocks if block.seed not in known)
            elites = [pool[k] for k in result.ranking[:keep]]
            if result.runs == 0 or used >= self.budget:
                break
        return TuningResult(
            elites=tuple(c.configuration for c in elites),
            runs=used,
            records=tuple(record for result in races for record in result.records),
            races=tuple(races),
        )

    def _sample(
        self,
        rng: np.random.Generator,
        elites: list[_Candidate],
        new: int,
        iteration: int,
        planned: int,
    ) -> list[_Candidate]:
        """Return the new candidates of a race, with the soft restart applied once."""
        if iteration == 1:
            model = self.space.initial_model()
            return [_Candidate(config, model, {}, -1) for config in self.space.sample(rng, new)]
        for elite in elites:
            elite.model = narrow_model(
                self.space,
                elite.model,
                elite.configuration,
                iteration=iteration,
                iterations=planned,
                new=new,
            )
        children = self._draw(rng, elites, new)
        converged = {
            child.parent
            for child in children
            for other in (*elites, *children)
            if child is not other
            and self.space.distance(child.configuration, other.configuration) == 0.0
        }
        if not converged:
            return children
        for parent in converged:
            elites[parent].model = restart_model(self.space, elites[parent].model, new=new)
        return self._draw(rng, elites, new)

    def _draw(
        self, rng: np.random.Generator, elites: list[_Candidate], new: int
    ) -> list[_Candidate]:
        """Draw ``new`` configurations, each around an elite chosen by rank."""
        cumulative = list(itertools.accumulate(parent_probabilities(len(elites))))
        children: list[_Candidate] = []
        for _ in range(new):
            parent = min(bisect.bisect_right(cumulative, rng.random()), len(elites) - 1)
            model = elites[parent].model
            config = self.space.sample_around(rng, elites[parent].configuration, model)
            children.append(_Candidate(config, model, {}, parent))
        return children


@dataclass(slots=True)
class _Candidate:
    """A configuration in the run, with the model its children inherit and its costs by seed."""

    configuration: Configuration
    model: SamplingModel
    costs: dict[int, float]
    parent: int


class _Stream[I]:
    """The blocks of a run, by position: each pass through the instances is one order.

    The order of a pass is a random permutation when ``shuffle`` is true and the given
    order otherwise. The block at a position has a seed of its own, so an instance that
    comes round again runs under a new seed.
    """

    def __init__(self, instances: Sequence[I], seed: int, shuffle: bool) -> None:
        self._instances = instances
        self._seed = seed
        self._shuffle = shuffle
        self._orders: dict[int, list[int]] = {}

    def block(self, position: int) -> Block[I]:
        count = len(self._instances)
        turn, offset = divmod(position, count)
        if turn not in self._orders:
            if self._shuffle:
                generator = np.random.default_rng(derive_seed(self._seed, "tuning:instances", turn))
                self._orders[turn] = generator.permutation(count).tolist()
            else:
                self._orders[turn] = list(range(count))
        index = self._orders[turn][offset]
        return Block(self._instances[index], derive_seed(self._seed, "tuning:run", position))


def _positive(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")
