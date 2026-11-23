"""Racing: the Friedman test and Conover's comparisons by hand, and races worked out step by step.

The comparisons are checked against arithmetic done by hand, which the comments show, and the
Student t tail is checked against the finite sum for an even number of degrees of freedom
(Abramowitz and Stegun, 26.7.4), which does not use the incomplete beta function of the library.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import pytest

from dispatchatlas.analysis.stats.distributions import t_sf
from dispatchatlas.analysis.stats.nonparametric import friedman_test
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.tuning import (
    Block,
    Entry,
    Experiment,
    RaceResult,
    compare_with_best,
    race,
)


def t_two_sided_even(t: float, nu: int) -> float:
    """Return ``P(|T| > t)`` for an even number of degrees of freedom.

    ``P(|T| < t) = sin(a) (1 + 1/2 cos(a)^2 + (1 3)/(2 4) cos(a)^4 + ...)`` with the terms up to
    ``cos(a)^(nu - 2)``, for ``tan(a) = t / sqrt(nu)``.
    """
    assert nu % 2 == 0
    cos2 = nu / (nu + t * t)
    term = total = 1.0
    for j in range(1, nu // 2):
        term *= (2 * j - 1) / (2 * j) * cos2
        total += term
    return 1.0 - t / math.sqrt(nu + t * t) * total


class Table:
    """A ``run_batch`` that reads costs from a function of the configuration's name and the seed."""

    def __init__(self, cost: Callable[[str, int], float]) -> None:
        self.cost = cost
        self.batches: list[list[tuple[str, int]]] = []

    def __call__(self, experiments: Sequence[Experiment[int]]) -> list[float]:
        pairs = [(str(e.configuration["name"]), e.seed) for e in experiments]
        self.batches.append(pairs)
        return [self.cost(name, seed) for name, seed in pairs]


def blocks(count: int) -> list[Block[int]]:
    """Blocks with seeds 0, 1, ... and the instances 0, 10, 20, ..."""
    return [Block(10 * seed, seed) for seed in range(count)]


def named(*names: str) -> list[dict[str, str]]:
    return [{"name": name} for name in names]


# ------------------------------------------------------- Friedman and Conover


def test_the_comparison_follows_the_friedman_and_conover_arithmetic() -> None:
    # Candidates A, B, C on six blocks. In three blocks the costs rank them A < B < C, and in
    # three B < A < C, so the ranks are (1, 2, 3) three times and (2, 1, 3) three times.
    costs = [(1.0, 2.0, 3.0)] * 3 + [(2.0, 1.0, 3.0)] * 3
    comparison = compare_with_best(costs)
    # Rank sums: A 3 + 6 = 9, B 6 + 3 = 9, C 18.
    assert comparison.rank_sums == (9.0, 9.0, 18.0)
    # Friedman: Q = 12 / (6 3 4) (81 + 81 + 324) - 3 6 4 = 81 - 72 = 9, whose chi-square tail
    # with 2 degrees of freedom is exp(-9 / 2).
    assert comparison.statistic == pytest.approx(9.0)
    assert comparison.p_value == pytest.approx(math.exp(-4.5))
    # Conover: the sum of squared ranks is A = 6 (1 + 4 + 9) = 84, and
    # b A - sum(R^2) = 504 - (81 + 81 + 324) = 18, with (6 - 1)(3 - 1) = 10 degrees of freedom,
    # so a difference of rank sums has the standard error sqrt(2 18 / 10) = sqrt(3.6).
    assert comparison.df == 10
    assert comparison.scale == pytest.approx(math.sqrt(3.6))
    # The best is A, which shares the least rank sum with B and comes first. B does not differ
    # from it, and C differs by 9, which is 9 / sqrt(3.6) = 4.743 standard errors.
    assert comparison.best == 0
    assert comparison.p_values[:2] == (1.0, 1.0)
    p_c = t_two_sided_even(9.0 / math.sqrt(3.6), 10)
    assert comparison.p_values[2] == pytest.approx(p_c, rel=1e-9)
    assert comparison.p_values[2] == pytest.approx(2.0 * t_sf(9.0 / math.sqrt(3.6), 10), rel=1e-12)
    assert 0.0005 < p_c < 0.0015
    assert comparison.discarded == (2,)


def test_conover_discards_only_when_the_friedman_test_rejects() -> None:
    # Ranks (1, 2, 3), (1, 3, 2), (1, 2, 3), (2, 1, 3): rank sums (5, 8, 11) and Q = 4.5, whose
    # tail exp(-2.25) = 0.105 does not reject at 0.05. A comparison of C with the best alone would:
    # b A - sum(R^2) = 4 56 - 210 = 14, df = 6, so the standard error is sqrt(28 / 6) and the
    # difference of 6 is 2.777 of them, two-sided p = 0.032.
    costs = [(1.0, 2.0, 3.0), (1.0, 3.0, 2.0), (1.0, 2.0, 3.0), (2.0, 1.0, 3.0)]
    comparison = compare_with_best(costs)
    assert comparison.rank_sums == (5.0, 8.0, 11.0)
    assert comparison.statistic == pytest.approx(4.5)
    assert comparison.p_value == pytest.approx(math.exp(-2.25))
    assert comparison.scale == pytest.approx(math.sqrt(28.0 / 6.0))
    assert comparison.p_values[2] == pytest.approx(
        t_two_sided_even(6.0 / math.sqrt(28.0 / 6.0), 6), rel=1e-9
    )
    assert comparison.p_values[2] < 0.05 < comparison.p_value
    assert comparison.discarded == ()
    # A looser level lets the Friedman test reject, and then C goes.
    assert compare_with_best(costs, alpha=0.11).discarded == (2,)


def test_blocks_that_rank_the_candidates_alike_discard_every_other_candidate() -> None:
    # Ranks (1, 2, 3) in four blocks: sums (4, 8, 12), Q = 8, tail exp(-4) = 0.018. The ranks
    # agree completely, so b A - sum(R^2) = 224 - 224 = 0 and a difference is infinitely many
    # standard errors: p = 0 for each candidate with a larger sum.
    comparison = compare_with_best([(1.0, 2.0, 3.0)] * 4)
    assert comparison.rank_sums == (4.0, 8.0, 12.0)
    assert comparison.statistic == pytest.approx(8.0)
    assert comparison.p_value == pytest.approx(math.exp(-4.0))
    assert comparison.scale == 0.0
    assert comparison.p_values == (1.0, 0.0, 0.0)
    assert comparison.discarded == (1, 2)


def test_tied_costs_share_the_mean_of_their_ranks() -> None:
    # Block 1: costs (1, 1, 3), ranks (1.5, 1.5, 3). Block 2: (2, 1, 3), ranks (2, 1, 3).
    # Block 3: (1, 2, 2), ranks (1, 2.5, 2.5). Rank sums (4.5, 5, 8.5), which total 18.
    comparison = compare_with_best([(1.0, 1.0, 3.0), (2.0, 1.0, 3.0), (1.0, 2.0, 2.0)])
    assert comparison.rank_sums == (4.5, 5.0, 8.5)
    # Two tie groups of two: sum(t^3 - t) = 12, so Q = (12 / 36 (20.25 + 25 + 72.25) - 36) /
    # (1 - 12 / (3 3 8)) = (39.1667 - 36) / 0.83333 = 3.8, with tail exp(-1.9).
    assert comparison.statistic == pytest.approx(3.8)
    assert comparison.p_value == pytest.approx(math.exp(-1.9))
    # The squares of the ranks sum to 13.5 + 14 + 13.5 = 41, so b A - sum(R^2) = 123 - 117.5 =
    # 5.5, with 4 degrees of freedom.
    assert comparison.df == 4
    assert comparison.scale == pytest.approx(math.sqrt(2.0 * 5.5 / 4.0))
    friedman = friedman_test(
        [
            {"a": 1.0, "b": 1.0, "c": 3.0},
            {"a": 2.0, "b": 1.0, "c": 3.0},
            {"a": 1.0, "b": 2.0, "c": 2.0},
        ]
    )
    assert comparison.statistic == friedman.statistic
    assert comparison.p_value == friedman.p_value


def test_two_candidates_reduce_to_the_t_test_of_the_signs() -> None:
    # A is better than B in 7 of 8 blocks. Rank sums A 7 + 2 = 9 and B 14 + 1 = 15, so the
    # difference is 6. Here b A - sum(R^2) = 8 40 - (81 + 225) = 14 with 7 degrees of freedom,
    # so the standard error is sqrt(2 14 / 7) = 2, and the difference is 3 of them. This is the
    # one-sample t statistic of the within-block rank differences, +1 seven times and -1 once:
    # mean 0.75, variance (7 0.0625 + 3.0625) / 7 = 0.5, t = 0.75 / sqrt(0.5 / 8) = 3.
    comparison = compare_with_best([(1.0, 2.0)] * 7 + [(2.0, 1.0)])
    assert comparison.rank_sums == (9.0, 15.0)
    assert comparison.scale == 2.0
    assert comparison.df == 7
    differences = [1.0] * 7 + [-1.0]
    mean = sum(differences) / 8
    variance = sum((d - mean) ** 2 for d in differences) / 7
    statistic = mean / math.sqrt(variance / 8)
    assert statistic == pytest.approx(3.0)
    assert comparison.p_values[1] == pytest.approx(2.0 * t_sf(statistic, 7), rel=1e-12)
    assert 0.019 < comparison.p_values[1] < 0.021
    assert comparison.discarded == (1,)


def test_an_infinite_cost_ranks_last() -> None:
    comparison = compare_with_best([(1.0, 2.0, math.inf), (2.0, 1.0, math.inf)] * 3)
    assert comparison.rank_sums == (9.0, 9.0, 18.0)
    assert comparison.discarded == (2,)


@pytest.mark.parametrize(
    "costs",
    [
        [],
        [(1.0, 2.0)],
        [(1.0,), (2.0,)],
        [(1.0, 2.0), (1.0, 2.0, 3.0)],
        [(1.0, 2.0), (1.0, math.nan)],
    ],
)
def test_the_comparison_refuses_tables_that_are_not_blocks_by_candidates(
    costs: list[tuple[float, ...]],
) -> None:
    with pytest.raises(ConfigError):
        compare_with_best(costs)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, True, math.nan])
def test_the_comparison_refuses_a_level_outside_the_open_unit_interval(alpha: float) -> None:
    with pytest.raises(ConfigError, match="alpha"):
        compare_with_best([(1.0, 2.0)] * 3, alpha=alpha)


# -------------------------------------------------------------- a race by hand

# The cost of A, B and C on the block with each seed. Blocks 0-2 rank A < B < C, blocks 3-5
# B < A < C, and after that A and B alternate. C stays last.
STEPS = {
    **dict.fromkeys((0, 1, 2, 6, 8, 10, 11), (1.0, 2.0, 3.0)),
    **dict.fromkeys((3, 4, 5, 7, 9), (2.0, 1.0, 3.0)),
}


def steps(name: str, seed: int) -> float:
    return STEPS[seed]["ABC".index(name)]


def test_a_race_runs_the_steps_the_test_and_the_budget_give() -> None:
    table = Table(steps)
    result = race(
        named("A", "B", "C"),
        blocks(12),
        table,
        budget=26,
        first_test=6,
        each_test=1,
        min_survivors=1,
    )
    # Six blocks of three candidates are one batch of 18, in the order block by block and
    # candidate by candidate. At the sixth block the ranks are (1, 2, 3) x 3 and (2, 1, 3) x 3: the
    # first example above, in which C alone is discarded. Each later step runs one block for A and
    # B, 2 runs: 20, 22, 24, 26. The next block would cost 2 more than the 26 allowed.
    assert [len(batch) for batch in table.batches] == [18, 2, 2, 2, 2]
    assert table.batches[0] == [(name, seed) for seed in range(6) for name in "ABC"]
    assert table.batches[1:] == [
        [("A", 6), ("B", 6)],
        [("A", 7), ("B", 7)],
        [("A", 8), ("B", 8)],
        [("A", 9), ("B", 9)],
    ]
    assert result.runs == 26 == len(result.records)
    assert result.stop == "budget"
    assert len(result.blocks) == 10
    assert result.blocks == tuple(blocks(10))
    # A and B alternate, so no later test discards either, and the 10 blocks give each the rank
    # sum 3 1 + 3 2 + 1 + 2 + 1 + 2 = 15 against 3 2 + 3 1 + 2 + 1 + 2 + 1 = 15: a tie, which
    # the lower number wins. C was discarded and takes no part.
    assert result.ranking == (0, 1)
    assert result.rank_sums == (15.0, 15.0)
    costs = {entry.configuration["name"]: entry.costs for entry in result.entries}
    assert sorted(costs["A"]) == sorted(costs["B"]) == list(range(10))
    assert sorted(costs["C"]) == list(range(6))
    assert costs["C"][5] == 3.0
    first = result.records[0]
    assert (first.configuration, first.instance, first.seed, first.cost) == (
        {"name": "A"},
        0,
        0,
        1.0,
    )
    assert result.records[3].instance == 10  # block 1 of A: instance 10, seed 1
    # The discarding at the sixth block is that of the comparison.
    assert compare_with_best([STEPS[seed] for seed in range(6)]).discarded == (2,)


@pytest.mark.parametrize("survivors", [2, 3])
def test_a_race_ends_once_few_enough_entries_survive_a_test(survivors: int) -> None:
    table = Table(steps)
    result = race(
        named("A", "B", "C"), blocks(12), table, budget=1000, first_test=6, min_survivors=survivors
    )
    # The first test, after 6 blocks, discards C and leaves A and B, which are no more than the
    # entries a race may end with. The ranks are those among A and B: (1, 2) three times and
    # (2, 1) three times.
    assert result.stop == "survivors"
    assert result.runs == 18
    assert len(result.blocks) == 6
    assert result.ranking == (0, 1)
    assert result.rank_sums == (9.0, 9.0)
    # A race always runs up to its first test, even with fewer entries than it may end with.
    small = race(
        named("A", "B"), blocks(12), Table(steps), budget=1000, first_test=6, min_survivors=2
    )
    assert small.stop == "survivors"
    assert small.runs == 12


def test_a_race_ranks_survivors_by_their_rank_sums_with_elites_first_among_equals() -> None:
    # Three configurations with the same costs everywhere tie in every block.
    elite = Entry({"name": "E"}, {})
    result = race(
        named("A", "B"), blocks(4), Table(lambda name, seed: 5.0), budget=100, elites=[elite]
    )
    assert result.ranking == (0, 1, 2)
    # Mean rank 2 in each of 4 blocks.
    assert result.rank_sums == (8.0, 8.0, 8.0)
    assert result.stop == "instances"  # the four blocks end before the first test, after five
    assert result.runs == 12


# --------------------------------------------------------------- elitist races


def protected_race(*, protect: bool) -> tuple[Table, RaceResult[int]]:
    """Race an elite E, which is worst, against N1 and N2 on the blocks 0-5.

    Block 0 is new to every entry, and E has costs on blocks 1-4, so it is protected until the
    race has evaluated the new entries on 5 blocks.
    """
    table = Table(lambda name, seed: {"E": 9.0, "N1": 1.0, "N2": 2.0}[name])
    known = dict.fromkeys((1, 2, 3, 4), 9.0) if protect else {}
    result = race(
        named("N1", "N2") if protect else named("E", "N1", "N2"),
        blocks(6),
        table,
        budget=100,
        first_test=4,
        min_survivors=1,
        elites=[Entry({"name": "E"}, known)] if protect else [],
    )
    return table, result


def test_an_elite_survives_the_tests_until_the_new_entries_have_seen_its_blocks() -> None:
    table, result = protected_race(protect=True)
    # The entries are E (0), N1 (1) and N2 (2). The first test, after 4 blocks, sees ranks
    # (3, 1, 2) in every block, the sums 12, 4, 8, and Q = 8 with p = exp(-4) = 0.018: perfect
    # agreement, so every entry but N1 differs from the best. E is the worst of all, yet it
    # stays, because the new entries have seen 4 blocks and E has costs on blocks up to the 5th.
    # N2 goes. The second test, after block 4 (the 5th), compares E and N1: ranks (2, 1) in five
    # blocks, Q = 12 / 30 (100 + 25) - 45 = 5 with p = 0.025. Now the new entries have seen
    # 5 blocks, and E is discarded.
    assert result.ranking == (1,)
    assert result.stop == "survivors"
    assert len(result.blocks) == 5
    # Block 0: E, N1, N2 (E has no cost there); blocks 1-3: N1 and N2 (E's costs are known): 6;
    # block 4: N1 alone. The elite's costs are never run again.
    assert [len(batch) for batch in table.batches] == [9, 1]
    assert result.runs == 10
    assert [name for name, seed in table.batches[0] if name == "E"] == ["E"]
    assert sorted(result.entries[0].costs) == [0, 1, 2, 3, 4]
    assert result.entries[0].costs[1] == 9.0


def test_without_the_protection_the_same_entry_is_discarded_at_the_first_test() -> None:
    table, result = protected_race(protect=False)
    # E has no costs, so the first test, after 4 blocks and 12 runs, discards E and N2.
    assert result.ranking == (1,)
    assert result.stop == "survivors"
    assert len(result.blocks) == 4
    assert result.runs == 12
    assert [len(batch) for batch in table.batches] == [12]


def test_the_idle_limit_counts_the_tests_after_the_elites_lose_their_protection() -> None:
    # Everything ties, so no test discards anybody. E has costs on blocks 1-3, so it is protected
    # while fewer than 4 blocks are done. With a test after every block from the second, those
    # at 2 and 3 blocks do not count towards the limit of 2, and those at 4 and 5 do.
    table = Table(lambda name, seed: 5.0)
    result = race(
        named("N"),
        blocks(10),
        table,
        budget=100,
        first_test=2,
        idle_tests=2,
        elites=[Entry({"name": "E"}, {1: 5.0, 2: 5.0, 3: 5.0})],
    )
    assert result.stop == "idle"
    assert len(result.blocks) == 5
    unlimited = race(
        named("N"),
        blocks(10),
        Table(lambda name, seed: 5.0),
        budget=100,
        first_test=2,
        elites=[Entry({"name": "E"}, {1: 5.0, 2: 5.0, 3: 5.0})],
    )
    assert unlimited.stop == "instances"
    assert len(unlimited.blocks) == 10


def test_a_discard_resets_the_count_of_idle_tests() -> None:
    # Four candidates: D is worst everywhere, C next, in the first five blocks, so the first
    # test (5 blocks) discards C and D; with the tied A and B left, nothing is discarded and the
    # race stops after two idle tests.
    cost = {"A": 1.0, "B": 1.0, "C": 3.0, "D": 4.0}
    result = race(
        named("A", "B", "C", "D"),
        blocks(20),
        Table(lambda name, seed: cost[name]),
        budget=1000,
        idle_tests=2,
    )
    assert result.ranking == (0, 1)
    assert result.stop == "idle"
    assert len(result.blocks) == 7  # tests after 5 (discards), 6 and 7 blocks (nobody)


# ------------------------------------------------------------ stops and budget


def test_a_single_entry_needs_no_race() -> None:
    table = Table(lambda name, seed: 1.0)
    result = race(named("A"), blocks(5), table, budget=10)
    assert (result.runs, result.stop, result.ranking, table.batches) == (0, "survivors", (0,), [])
    assert result.rank_sums == (0.0,)
    assert result.blocks == ()


def test_a_race_ends_when_the_instances_run_out_before_the_first_test() -> None:
    table = Table(steps)
    result = race(named("A", "B", "C"), blocks(3), table, budget=100, first_test=5)
    assert result.stop == "instances"
    assert result.runs == 9
    assert len(result.blocks) == 3
    assert result.ranking == (0, 1, 2)
    assert result.rank_sums == (3.0, 6.0, 9.0)


@pytest.mark.parametrize("budget", range(0, 60))
def test_a_race_never_runs_more_than_its_budget(budget: int) -> None:
    table = Table(steps)
    result = race(named("A", "B", "C"), blocks(12), table, budget=budget, first_test=6)
    assert result.runs <= budget
    assert result.runs == len(result.records) == sum(len(batch) for batch in table.batches)
    survivors = len(result.ranking)
    if result.stop == "budget":
        # The race stopped because the next block would cost every survivor one run more.
        assert budget - result.runs < survivors
    # Every survivor has a cost on each block the race evaluated.
    for k in result.ranking:
        assert set(range(len(result.blocks))) <= set(result.entries[k].costs)
    # Each batch is made of whole blocks: every seed in it comes with the same candidates.
    for batch in table.batches:
        names_by_seed: dict[int, list[str]] = {}
        for name, seed in batch:
            names_by_seed.setdefault(seed, []).append(name)
        assert len({tuple(names) for names in names_by_seed.values()}) == 1


def test_a_race_that_cannot_afford_the_first_test_ends_without_one() -> None:
    table = Table(steps)
    result = race(named("A", "B", "C"), blocks(12), table, budget=17, first_test=6)
    # Five whole blocks cost 15; the sixth would cost 18 in all.
    assert result.runs == 15
    assert result.stop == "budget"
    assert len(result.blocks) == 5
    assert result.ranking == (0, 1, 2)
    assert result.rank_sums == (7.0, 8.0, 15.0)  # A 3 + 2 2, B 6 + 2, C 15: blocks 0-4
    empty = race(named("A", "B", "C"), blocks(12), Table(steps), budget=0)
    assert (empty.runs, empty.stop, empty.rank_sums, empty.blocks) == (
        0,
        "budget",
        (0.0, 0.0, 0.0),
        (),
    )


def test_a_test_every_few_blocks_runs_the_blocks_of_a_step_in_one_batch() -> None:
    table = Table(steps)
    result = race(named("A", "B", "C"), blocks(12), table, budget=1000, first_test=2, each_test=3)
    # Tests come after 2, 5, 8, and 11 blocks. The first step is 2 blocks of 3 candidates, and
    # the second 3 blocks of 3: at 5 blocks the rank sums are (7, 8, 15), Q = 7.6 with tail
    # exp(-3.8) = 0.022, and C is 8 / sqrt(3) = 4.6 standard errors from A, so C goes. Two steps
    # of 3 blocks of 2 follow, with no discard at 8 blocks (rank sums 12 and 12) or 11 (16, 17).
    # The twelfth block comes alone, because the instances end before the test at 14.
    assert [len(batch) for batch in table.batches] == [6, 9, 6, 6, 2]
    assert len(result.blocks) == 12
    assert result.stop == "instances"
    assert result.ranking == (0, 1)


# ---------------------------------------------------------- arguments, contract


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"budget": -1}, "budget"),
        ({"budget": 1.5}, "budget"),
        ({"first_test": 1}, "first_test"),
        ({"first_test": True}, "first_test"),
        ({"each_test": 0}, "each_test"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
        ({"min_survivors": 0}, "min_survivors"),
        ({"idle_tests": 0}, "idle_tests"),
    ],
)
def test_a_race_refuses_arguments_out_of_range(overrides: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {"budget": 10, **overrides}
    with pytest.raises(ConfigError, match=message):
        race(named("A", "B"), blocks(3), Table(steps), **arguments)  # type: ignore[arg-type]


def test_a_race_needs_an_entry_and_blocks_with_different_seeds() -> None:
    with pytest.raises(ConfigError, match="at least one configuration"):
        race([], blocks(3), Table(steps), budget=10)
    with pytest.raises(ConfigError, match="different seeds"):
        race(named("A", "B"), [Block(0, 1), Block(1, 1)], Table(steps), budget=10)


def test_the_costs_must_come_back_one_per_experiment_and_not_nan() -> None:
    def short(experiments: Sequence[Experiment[int]]) -> list[float]:
        return [1.0]

    def nan(experiments: Sequence[Experiment[int]]) -> list[float]:
        return [math.nan] * len(experiments)

    def words(experiments: Sequence[Experiment[int]]) -> list[float]:
        return ["a"] * len(experiments)  # type: ignore[list-item]

    for bad, message in ((short, "1 costs for"), (nan, "NaN"), (words, "numbers")):
        with pytest.raises(ConfigError, match=message):
            race(named("A", "B"), blocks(6), bad, budget=100)


def test_a_failed_run_may_cost_infinity() -> None:
    cost = {"good": 1.0, "other": 2.0, "failed": math.inf}

    def run(name: str, seed: int) -> float:
        # good and other alternate as the best, so only the failed run differs
        if name == "other" and seed % 2 == 0:
            return 0.5
        return cost[name]

    result = race(
        named("good", "other", "failed"), blocks(20), Table(run), budget=1000, first_test=6
    )
    assert 2 not in result.ranking
    assert set(result.ranking) == {0, 1}


def test_an_entry_keeps_its_costs_read_only() -> None:
    costs = {1: 2.0}
    entry = Entry({"name": "A"}, costs)
    costs[3] = 4.0
    assert dict(entry.costs) == {1: 2.0}
    with pytest.raises(TypeError):
        entry.costs[1] = 0.0  # type: ignore[index]
