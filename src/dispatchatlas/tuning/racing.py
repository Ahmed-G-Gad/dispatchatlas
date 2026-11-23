"""Racing: elimination of configurations by the Friedman test and Conover's comparisons.

A race starts with a finite set of candidate configurations and evaluates them on
one instance after another. After the first ``first_test`` instances, and then
after every ``each_test`` instances, a statistical test discards the candidates that
perform worse than another one, and the race goes on with the survivors until few
remain, the budget is spent, or the instances end (López-Ibáñez et al. 2016,
pp. 46-47; Birattari et al. 2002). The defaults are those of irace: five instances
before the first test (p. 49), then one test per instance (p. 47).

The test is that of F-Race (López-Ibáñez et al. 2016, p. 47). The Friedman test ranks
the surviving candidates within each instance, with tied costs sharing the mean of
their ranks, and tests whether the sums of the ranks differ. When it rejects, each
candidate is compared with the best one by Conover's post-hoc test (Conover 1999):
with ``b`` instances, ``k`` candidates, rank sums ``R``, and ``A`` the sum of the
squares of all ranks, a candidate ``j`` differs from the best ``i`` when

    ``|R_j - R_i| > t * sqrt(2 (b A - sum(R^2)) / ((b - 1) (k - 1)))``

for the ``1 - alpha / 2`` quantile ``t`` of the Student t distribution with
``(b - 1) (k - 1)`` degrees of freedom. :func:`compare_with_best` evaluates the
comparison through :func:`~dispatchatlas.analysis.stats.distributions.t_sf`, as a
p-value. The tests are a selection heuristic: the level ``alpha`` is that of each
comparison, with no correction for the number of comparisons or of tests
(López-Ibáñez et al. 2016, p. 47). Because the ranks are taken within an instance, a
raw cost such as a makespan serves as the cost of an experiment.

The race never starts a solver. The caller supplies ``run_batch``, which takes the
experiments of one step and returns their costs, so the caller decides where and how
they run. One experiment is one run, and the budget of a race counts runs.

An elitist race (López-Ibáñez et al. 2016, p. 49) starts from elite configurations
that carry the costs they already have on earlier instances. Those costs are not
run again, and an elite is not discarded before the instances it already has costs
on are used up, that is, before the new candidates have been evaluated on as many
instances as the elites were; after that, any configuration may be discarded.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from dispatchatlas.analysis.stats.distributions import t_sf
from dispatchatlas.analysis.stats.nonparametric import friedman_test
from dispatchatlas.analysis.stats.ranks import oriented_ranks
from dispatchatlas.core._frozen import FrozenMap
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.tuning.space import Configuration

__all__ = [
    "Block",
    "Comparison",
    "Entry",
    "Experiment",
    "RaceResult",
    "Record",
    "RunBatch",
    "StopReason",
    "compare_with_best",
    "race",
]


@dataclass(frozen=True, slots=True)
class Block[I]:
    """One instance with the seed that every configuration runs it under.

    Running all configurations on an instance under the same seed reduces the
    variance of their comparison (López-Ibáñez et al. 2016, p. 56). Costs of an
    :class:`Entry` are keyed by the seed, so the blocks of a race need different seeds.

    Attributes:
        instance: The instance, whatever the caller's ``run_batch`` takes.
        seed: The seed of the runs on this instance.
    """

    instance: I
    seed: int


@dataclass(frozen=True, slots=True)
class Experiment[I]:
    """One run for the caller to perform: a configuration applied to an instance.

    Attributes:
        configuration: The configuration to run.
        instance: The instance to run it on.
        seed: The seed of the run.
    """

    configuration: Configuration
    instance: I
    seed: int


type RunBatch[I] = Callable[[Sequence[Experiment[I]]], Sequence[float]]
"""The function that runs experiments: it takes a sequence of experiments and returns
their costs, one per experiment and in the same order. Smaller is better; a failed run
may cost ``math.inf``."""


@dataclass(frozen=True, slots=True)
class Record[I]:
    """An experiment and its cost.

    Attributes:
        configuration: The configuration that was run.
        instance: The instance it was run on.
        seed: The seed of the run.
        cost: The cost the run returned.
    """

    configuration: Configuration
    instance: I
    seed: int
    cost: float


@dataclass(frozen=True, slots=True)
class Entry:
    """A configuration with the costs it already has.

    Attributes:
        configuration: The configuration.
        costs: Its cost on each block it has been run on, by the block's seed.
            An elite that enters a race carries the costs of the earlier races.
    """

    configuration: Configuration
    costs: Mapping[int, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "costs", FrozenMap(self.costs))


type StopReason = Literal["survivors", "budget", "idle", "instances"]
"""Why a race ended.

``"survivors"``: at most ``min_survivors`` configurations were left, or there was one
to start with. ``"budget"``: the budget did not cover the next instance for every
survivor. ``"idle"``: ``idle_tests`` consecutive tests discarded nobody, after the
elites lost their protection. ``"instances"``: the instances ran out.
"""


@dataclass(frozen=True, slots=True)
class Comparison:
    """The Friedman test of some candidates, and Conover's comparisons with the best.

    Attributes:
        rank_sums: The sum over the blocks of each candidate's rank, 1 for the
            lowest cost, tied costs sharing the mean rank.
        statistic: The tie-corrected Friedman statistic.
        p_value: The p-value of the Friedman test, from a chi-square distribution
            with ``k - 1`` degrees of freedom.
        best: The index of the candidate with the smallest rank sum, the first of
            them when several share it.
        df: The degrees of freedom of the comparisons, ``(b - 1) (k - 1)``.
        scale: The standard error of the difference of two rank sums,
            ``sqrt(2 (b A - sum(R^2)) / df)``; zero when every block ranks the
            candidates alike.
        p_values: The two-sided p-value of each candidate against the best, from the
            Student t tail of ``(R_j - R_best) / scale``; 1 for the best itself, and
            0 for any difference when ``scale`` is zero.
        discarded: The indices of the candidates whose p-value is below ``alpha``,
            when the Friedman test rejects at ``alpha``, and none otherwise.
    """

    rank_sums: tuple[float, ...]
    statistic: float
    p_value: float
    best: int
    df: int
    scale: float
    p_values: tuple[float, ...]
    discarded: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RaceResult[I]:
    """The outcome of a race.

    The entries of a race are numbered from 0: the elites in the order given, then
    the candidates.

    Attributes:
        ranking: The numbers of the surviving entries, the best first: by the sum of
            their ranks over the blocks the race evaluated, which every survivor has
            costs on, and an earlier entry first among equal sums, so an elite comes
            before a candidate.
        rank_sums: The rank sum of each survivor, in the order of ``ranking``; all 0
            when no block was evaluated.
        entries: Every entry after the race, eliminated ones included, with its costs
            from before the race and from the race.
        blocks: The blocks the race evaluated, in order.
        records: The experiments the race ran, in the order they ran.
        stop: Why the race ended.
    """

    ranking: tuple[int, ...]
    rank_sums: tuple[float, ...]
    entries: tuple[Entry, ...]
    blocks: tuple[Block[I], ...]
    records: tuple[Record[I], ...]
    stop: StopReason

    @property
    def runs(self) -> int:
        """The number of experiments the race ran, which never exceeds its budget."""
        return len(self.records)


def compare_with_best(costs: Sequence[Sequence[float]], alpha: float = 0.05) -> Comparison:
    """Return the Friedman test of the candidates and Conover's comparisons with the best.

    The costs of each block are ranked from the lowest, tied costs sharing the mean of
    their ranks (see :func:`~dispatchatlas.analysis.stats.ranks.oriented_ranks`). The
    Friedman test is :func:`~dispatchatlas.analysis.stats.nonparametric.friedman_test`.
    When it rejects at ``alpha``, every candidate is compared with the one that has the
    smallest rank sum, and those whose two-sided p-value is below ``alpha`` are
    discarded (Conover 1999; López-Ibáñez et al. 2016, p. 47). There is no correction
    for multiple comparisons, as in irace.

    Args:
        costs: One row per block and one column per candidate, at least two of each,
            every row of the same length. Costs may be infinite, not NaN.
        alpha: The level of the Friedman test and of each comparison, strictly
            between 0 and 1.

    Raises:
        ConfigError: If the costs are not such a table, or ``alpha`` is out of range.
    """
    _check_alpha(alpha)
    blocks = len(costs)
    groups = len(costs[0]) if blocks else 0
    if blocks < 2 or groups < 2:
        raise ConfigError("the comparison needs at least two blocks and two candidates")
    if any(len(row) != groups for row in costs):
        raise ConfigError("every block must hold the cost of every candidate")
    friedman = friedman_test([{str(k): row[k] for k in range(groups)} for row in costs])
    ranks = [oriented_ranks(row, "min") for row in costs]
    rank_sums = tuple(math.fsum(row[k] for row in ranks) for k in range(groups))
    square_sum = math.fsum(rank * rank for row in ranks for rank in row)
    df = (blocks - 1) * (groups - 1)
    residual = blocks * square_sum - math.fsum(total * total for total in rank_sums)
    scale = math.sqrt(2.0 * max(residual, 0.0) / df)
    best = min(range(groups), key=lambda k: (rank_sums[k], k))
    p_values: list[float] = []
    for k in range(groups):
        gap = rank_sums[k] - rank_sums[best]
        if gap <= 0.0:
            p_values.append(1.0)
        elif scale == 0.0:
            p_values.append(0.0)
        else:
            p_values.append(min(1.0, 2.0 * t_sf(gap / scale, df)))
    rejected = friedman.p_value < alpha
    return Comparison(
        rank_sums=rank_sums,
        statistic=friedman.statistic,
        p_value=friedman.p_value,
        best=best,
        df=df,
        scale=scale,
        p_values=tuple(p_values),
        discarded=tuple(k for k in range(groups) if rejected and p_values[k] < alpha),
    )


def race[I](
    candidates: Sequence[Configuration],
    instances: Sequence[Block[I]],
    run_batch: RunBatch[I],
    *,
    budget: int,
    first_test: int = 5,
    each_test: int = 1,
    alpha: float = 0.05,
    elites: Sequence[Entry] = (),
    min_survivors: int = 1,
    idle_tests: int | None = None,
) -> RaceResult[I]:
    """Race configurations on instances and eliminate by the Friedman and Conover tests.

    The race takes the blocks in order. At each step it asks ``run_batch`` for the
    experiments of the next blocks that the surviving entries have no cost for, in a
    single call: the first ``first_test`` blocks, and then ``each_test`` blocks at a
    time, in the order block by block and entry by entry. After each step it runs
    :func:`compare_with_best` on the survivors over all the blocks so far, and drops
    those it discards (López-Ibáñez et al. 2016, p. 46-47).

    The race counts runs. It starts a block only when the budget that is left covers
    the experiments of every survivor on it (p. 47), so ``budget`` is never exceeded;
    the blocks of a step that the budget covers are run, and the race then ends
    without a test unless the step is complete.

    Elites (p. 49) are entries that arrive with costs. Their costs are used as they
    are, and are not run again. The elites cannot be discarded by a test before the
    last block on which any elite has a cost, because up to there the candidates have
    not been evaluated on as many blocks as the elites; if the caller puts the new
    blocks first and the elites' blocks after them, this is the protection of
    ``T_new + e`` blocks that the paper gives. From that block on, any entry may be
    discarded, and ``idle_tests`` consecutive tests that discard nobody end the race.

    At the end the survivors are ranked by their rank sums over the blocks the race
    evaluated, tied costs sharing the mean of their ranks, and the entries are returned
    with all their costs.

    Args:
        candidates: The new configurations to race.
        instances: The blocks, in the order the race takes them. Their seeds must
            differ, since costs are kept by seed.
        run_batch: The function that runs experiments and returns their costs.
        budget: The most runs the race may start, not negative.
        first_test: The blocks evaluated before the first test, at least 2, because the
            Friedman test needs two blocks. The default is 5 (López-Ibáñez et al. 2016,
            p. 49).
        each_test: The blocks between two tests after the first, at least 1; the default
            is 1 (p. 47).
        alpha: The level of the tests; the default is 0.05 (p. 47).
        elites: Entries that carry costs from earlier races.
        min_survivors: The race ends when at most this many entries remain after a test
            (p. 47), at least 1.
        idle_tests: End the race after this many consecutive tests that discard nobody,
            counted once the elites are no longer protected (p. 49); ``None`` for no such
            limit. The default of irace is 2.

    Returns:
        The result, with the survivors ranked.

    Raises:
        ConfigError: If an argument is out of range, there is no entry, two blocks share
            a seed, or ``run_batch`` returns a wrong number of costs or a NaN.
    """
    _check_race_arguments(budget, first_test, each_test, alpha, min_survivors, idle_tests)
    entries = [*elites, *(Entry(candidate, {}) for candidate in candidates)]
    if not entries:
        raise ConfigError("a race needs at least one configuration")
    seeds = [block.seed for block in instances]
    if len(set(seeds)) != len(seeds):
        raise ConfigError("the blocks of a race need different seeds")
    position_of = {seed: position + 1 for position, seed in enumerate(seeds)}
    protected = max(
        (position_of[seed] for entry in elites for seed in entry.costs if seed in position_of),
        default=0,
    )
    known = [dict(entry.costs) for entry in entries]
    alive = list(range(len(entries)))
    records: list[Record[I]] = []
    spent = done = quiet = 0
    next_test = first_test
    stop: StopReason | None = "survivors" if len(alive) < 2 else None
    while stop is None:
        if done == len(instances):
            stop = "instances"
            break
        end = min(next_test, len(instances))
        experiments: list[Experiment[I]] = []
        owners: list[int] = []
        affordable = done
        for position in range(done, end):
            block = instances[position]
            missing = [k for k in alive if block.seed not in known[k]]
            if spent + len(experiments) + len(missing) > budget:
                break
            for k in missing:
                experiments.append(Experiment(entries[k].configuration, block.instance, block.seed))
                owners.append(k)
            affordable = position + 1
        if experiments:
            for k, experiment, cost in zip(
                owners, experiments, _run(run_batch, experiments), strict=True
            ):
                known[k][experiment.seed] = cost
                records.append(
                    Record(experiment.configuration, experiment.instance, experiment.seed, cost)
                )
            spent += len(experiments)
        done = affordable
        if done < end:
            stop = "budget"
        elif done < next_test:
            stop = "instances"
        else:
            table = [[known[k][instances[p].seed] for k in alive] for p in range(done)]
            comparison = compare_with_best(table, alpha)
            dropped = {alive[i] for i in comparison.discarded}
            if done < protected:
                dropped -= set(range(len(elites)))
            alive = [k for k in alive if k not in dropped]
            next_test += each_test
            if len(alive) <= min_survivors:
                stop = "survivors"
            elif done >= protected:
                quiet = 0 if dropped else quiet + 1
                if idle_tests is not None and quiet >= idle_tests:
                    stop = "idle"
    sums = _rank_sums([[known[k][instances[p].seed] for k in alive] for p in range(done)], alive)
    order = sorted(range(len(alive)), key=lambda i: (sums[i], alive[i]))
    return RaceResult(
        ranking=tuple(alive[i] for i in order),
        rank_sums=tuple(sums[i] for i in order),
        entries=tuple(Entry(entry.configuration, known[k]) for k, entry in enumerate(entries)),
        blocks=tuple(instances[:done]),
        records=tuple(records),
        stop=stop,
    )


def _rank_sums(table: Sequence[Sequence[float]], alive: Sequence[int]) -> list[float]:
    """Return the rank sum of each column of a table of blocks by survivors, 0 without blocks."""
    sums = [0.0] * len(alive)
    for row in table:
        for i, rank in enumerate(oriented_ranks(row, "min")):
            sums[i] += rank
    return sums


def _run[I](run_batch: RunBatch[I], experiments: Sequence[Experiment[I]]) -> list[float]:
    """Run one batch and return its costs, refusing a wrong count or a NaN."""
    costs = run_batch(experiments)
    if len(costs) != len(experiments):
        raise ConfigError(
            f"run_batch returned {len(costs)} costs for {len(experiments)} experiments"
        )
    values: list[float] = []
    for cost in costs:
        try:
            value = float(cost)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"run_batch must return numbers, got {cost!r}") from error
        if math.isnan(value):
            raise ConfigError("run_batch returned a NaN cost")
        values.append(value)
    return values


def _check_alpha(alpha: float) -> None:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 < alpha < 1.0:
        raise ConfigError(f"alpha must lie strictly between 0 and 1, got {alpha!r}")


def _check_race_arguments(
    budget: int,
    first_test: int,
    each_test: int,
    alpha: float,
    min_survivors: int,
    idle_tests: int | None,
) -> None:
    for name, value, least in (
        ("budget", budget, 0),
        ("first_test", first_test, 2),
        ("each_test", each_test, 1),
        ("min_survivors", min_survivors, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < least:
            raise ConfigError(f"{name} must be an integer of at least {least}, got {value!r}")
    if idle_tests is not None and (
        isinstance(idle_tests, bool) or not isinstance(idle_tests, int) or idle_tests < 1
    ):
        raise ConfigError(
            f"idle_tests must be None or an integer of at least 1, got {idle_tests!r}"
        )
    _check_alpha(alpha)
