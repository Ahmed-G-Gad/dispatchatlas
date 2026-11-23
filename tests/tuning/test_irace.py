"""Iterated racing: the published formulas, the sampling models, and runs on a noisy target.

The formulas of López-Ibáñez et al. (2016) are checked against arithmetic done by hand, which the
comments show. The runs use the synthetic target of ``targets.py``, whose best configuration is
known and whose costs are noisy; no solver runs, and each run takes a fraction of a second.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
import pytest

from dispatchatlas.core.errors import ConfigError
from dispatchatlas.tuning import (
    Categorical,
    Entry,
    Experiment,
    IteratedRacing,
    Ordinal,
    ParameterSpace,
    RaceResult,
    Real,
    SamplingModel,
    TuningResult,
    Value,
    candidate_count,
    default_iterations,
    default_survivors,
    iteration_budget,
    narrow_model,
    parent_probabilities,
    restart_model,
)
from dispatchatlas.tuning import irace as irace_module
from dispatchatlas.tuning.irace import _Candidate
from tests.tuning.targets import BEST, SPACE, Runner, noisy_cost, true_cost

INSTANCES = list(range(10))
"""The tuning instances of the noisy target: its scale grows with the index."""

KEEP = default_survivors(len(SPACE))
PLANNED = default_iterations(len(SPACE))


def tune(budget: int, seed: int, **settings: object) -> tuple[TuningResult[int], Runner]:
    runner = Runner()
    result = IteratedRacing(SPACE, budget, seed=seed, **settings).tune(INSTANCES, runner)  # type: ignore[arg-type]
    return result, runner


def elites_of(previous: RaceResult[int]) -> list[int]:
    """The entries of a race that go on to the next one: its best ``N_min`` survivors."""
    return list(previous.ranking[:KEEP])


# ----------------------------------------------------------------- the formulas


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [(1, 2), (2, 3), (3, 3), (4, 4), (5, 4), (7, 4), (8, 5), (11, 5), (15, 5), (16, 6), (32, 7)],
)
def test_the_default_number_of_races_is_two_plus_the_binary_logarithm_rounded_down(
    parameters: int, expected: int
) -> None:
    # López-Ibáñez et al. (2016, p. 46): N_iter = floor(2 + log2 N_param).
    assert default_iterations(parameters) == expected == math.floor(2 + math.log2(parameters))
    # The paper prints no default for N_min, and the library takes the same number.
    assert default_survivors(parameters) == expected


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "4"])
def test_the_defaults_need_a_positive_number_of_parameters(bad: object) -> None:
    with pytest.raises(ConfigError):
        default_iterations(bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        default_survivors(bad)  # type: ignore[arg-type]


def test_each_race_gets_its_share_of_the_runs_that_are_left() -> None:
    # B_j = (B - B_used) / (N_iter - j + 1) (p. 46), rounded down to whole runs.
    assert iteration_budget(1000, 4, 1) == 250
    assert iteration_budget(750, 4, 2) == 250
    assert iteration_budget(700, 4, 2) == 233
    assert iteration_budget(100, 4, 4) == 100
    assert iteration_budget(0, 4, 3) == 0
    for arguments in ((100, 4, 5), (-1, 4, 1), (100, 4, 0), (100, 0, 1), (True, 4, 1), (1.5, 4, 1)):
        with pytest.raises(ConfigError):
            iteration_budget(*arguments)


def test_the_number_of_candidates_follows_equation_five() -> None:
    # mu = T_first = 5 and T_each = 1. N_j = floor((B_j + N_elite e) / max(mu + T_each min(5, j),
    # nm(T_new + e, T_each))), with the elites' costs on e instances free (p. 49).
    first = {"first_test": 5, "each_test": 1}
    assert candidate_count(250, iteration=1, **first) == 250 // 6 == 41
    # Race 2: 4 elites on 6 instances: (200 + 24) / max(5 + 2, 1 + 6) = 224 / 7.
    assert candidate_count(200, iteration=2, elites=4, seen=6, **first) == 32
    # With 12 instances seen the instances bind: (200 + 48) / max(7, 13) = 19.
    assert candidate_count(200, iteration=2, elites=4, seen=12, **first) == 248 // 13 == 19
    # Two new instances in front: nm(2 + 12, 1) = 14, so (200 + 48) / 14 = 17.
    assert candidate_count(200, iteration=2, elites=4, seen=12, new_instances=2, **first) == 17
    # A test every 3 instances: nm(1 + 6, 3) = 9 against 5 + 3 2 = 11: 224 / 11 = 20.
    assert candidate_count(200, first_test=5, each_test=3, iteration=2, elites=4, seen=6) == 20
    # The divisor stops growing at the fifth race: 5 + min(5, j) is 10 from there on.
    assert [candidate_count(300, iteration=j, **first) for j in (1, 2, 3, 4, 5, 6, 9)] == [
        50,
        42,
        37,
        33,
        30,
        30,
        30,
    ]
    for bad in ({"budget": -1}, {"iteration": 0}, {"elites": -1}, {"each_test": 0}):
        arguments = {"budget": 10, "first_test": 5, "each_test": 1, "iteration": 1, **bad}
        with pytest.raises(ConfigError):
            candidate_count(**arguments)


def test_parents_are_weighted_by_rank() -> None:
    # p_z = (N - r_z + 1) / (N (N + 1) / 2) (p. 47), the best first.
    assert parent_probabilities(4) == pytest.approx((0.4, 0.3, 0.2, 0.1))
    assert parent_probabilities(3) == pytest.approx((3 / 6, 2 / 6, 1 / 6))
    assert parent_probabilities(1) == (1.0,)
    assert sum(parent_probabilities(7)) == pytest.approx(1.0)
    with pytest.raises(ConfigError):
        parent_probabilities(0)


def small_space() -> ParameterSpace:
    return ParameterSpace([Real("x", 0.0, 1.0), Categorical("c", ["a", "b", "c"])])


def test_narrowing_shrinks_the_spreads_and_moves_the_categorical_weights_to_the_parent() -> None:
    space = small_space()  # N_param = 2
    parent: dict[str, Value] = {"x": 0.4, "c": "b"}
    # Race 2 of 4 with 16 new candidates: the spread is multiplied by (1 / 16) ** (1 / 2) = 1 / 4
    # (equation 2, p. 47), and every probability by 1 - 1 / 4, the parent's level gaining 1 / 4
    # (equations 3 and 4, p. 48): (1/3 3/4, 1/3 3/4 + 1/4, 1/3 3/4) = (1/4, 1/2, 1/4). The cap
    # 0.2 ** (1 / 2) = 0.4472 (equation 6, p. 49) cuts the 1/2, and the sum is 1/2 + 0.4472.
    first = narrow_model(space, space.initial_model(), parent, iteration=2, iterations=4, new=16)
    assert first.spreads["x"] == pytest.approx(0.125)
    cap = math.sqrt(0.2)
    total = 0.5 + cap
    assert first.weights["c"] == pytest.approx((0.25 / total, cap / total, 0.25 / total))
    assert sum(first.weights["c"]) == pytest.approx(1.0)
    # Race 3 with 4 new candidates: a factor (1 / 4) ** (1 / 2) = 1 / 2 on the spread, and the
    # weight 2 / 4 of the parent's level: the weights (1/4, cap, 1/4) / total become
    # (1/8, cap / 2 + 1/2, 1/8) / total, capped again and normalised.
    second = narrow_model(space, first, parent, iteration=3, iterations=4, new=4)
    assert second.spreads["x"] == pytest.approx(0.0625)
    moved = [0.25 / total * 0.5, cap / total * 0.5 + 0.5, 0.25 / total * 0.5]
    capped = [min(weight, cap) for weight in moved]
    assert second.weights["c"] == pytest.approx(tuple(w / sum(capped) for w in capped))
    assert second.weights["c"][1] > first.weights["c"][1]


def test_narrowing_refuses_what_the_formulas_do_not_define() -> None:
    space = small_space()
    model = space.initial_model()
    parent: dict[str, Value] = {"x": 0.4, "c": "b"}
    for arguments in (
        {"iteration": 1, "iterations": 4, "new": 4},  # the first race samples uniformly
        {"iteration": 2, "iterations": 1, "new": 4},
        {"iteration": 2, "iterations": 4, "new": 0},
    ):
        with pytest.raises(ConfigError):
            narrow_model(space, model, parent, **arguments)
    with pytest.raises(ConfigError, match="lacks the parameter 'c'"):
        narrow_model(space, model, {"x": 0.4}, iteration=2, iterations=4, new=4)
    with pytest.raises(ConfigError, match="lacks the parameter 'c'"):
        narrow_model(space, SamplingModel({"x": 0.5}, {}), parent, iteration=2, iterations=4, new=4)


def test_a_soft_restart_brings_the_spread_back_two_races_and_flattens_the_weights() -> None:
    space = ParameterSpace(
        [Real("x", 0.0, 1.0), Ordinal("o", [1, 2, 3, 4, 5]), Categorical("c", ["a", "b", "c"])]
    )  # N_param = 3
    # 16 new candidates. A spread goes back two races, s 16 ** (2 / 3) = 6.35 s, but no higher
    # than the spread of the second race: (range / 2) (1 / 16) ** (1 / 3) = (range / 2) 0.397. The
    # range of the real parameter is 1 and that of the ordinal one 5 - 1 = 4 (p. 48).
    small, large = 0.01, 0.2
    model = SamplingModel({"x": small, "o": large}, {"c": (0.25, 0.5, 0.25)})
    restarted = restart_model(space, model, new=16)
    ceiling = (1.0 / 16.0) ** (1.0 / 3.0)
    assert restarted.spreads["x"] == pytest.approx(small * 16 ** (2.0 / 3.0))
    assert restarted.spreads["x"] < 0.5 * ceiling
    assert restarted.spreads["o"] == pytest.approx(2.0 * ceiling)  # 0.2 * 6.35 = 1.27 is above it
    # Each probability p becomes 0.9 p + 0.1 max(P) = (0.275, 0.5, 0.275), and the sum is 1.05.
    assert restarted.weights["c"] == pytest.approx((0.275 / 1.05, 0.5 / 1.05, 0.275 / 1.05))
    with pytest.raises(ConfigError):
        restart_model(space, model, new=0)
    with pytest.raises(ConfigError, match="lacks the parameter 'o'"):
        restart_model(space, SamplingModel({"x": 0.1}, {"c": (0.5, 0.25, 0.25)}), new=4)


# -------------------------------------------------- sampling inside the tuner


def test_new_candidates_repeating_each_other_restart_the_elite_that_produced_them_once() -> None:
    # Two categorical parameters of two levels give 4 configurations, so 8 new candidates and
    # an elite repeat each other, whatever is drawn.
    space = ParameterSpace([Categorical("p", ["a", "b"]), Categorical("q", ["c", "d"])])
    tuner = IteratedRacing(space, 200, seed=1)
    elite = _Candidate({"p": "a", "q": "c"}, space.initial_model(), {}, -1)
    children = tuner._sample(np.random.default_rng(0), [elite], 8, 2, 3)
    assert len(children) == 8
    narrowed = narrow_model(
        space, space.initial_model(), elite.configuration, iteration=2, iterations=3, new=8
    )
    assert elite.model == restart_model(space, narrowed, new=8)
    assert elite.model != narrowed
    # The new candidates were drawn again from the restarted model, and inherit it.
    assert all(child.model is elite.model and child.parent == 0 for child in children)


def test_distinct_new_candidates_leave_the_model_narrowed_and_not_restarted() -> None:
    space = ParameterSpace([Real("x", 0.0, 1.0), Real("y", 0.0, 1.0)])
    tuner = IteratedRacing(space, 200, seed=1)
    elite = _Candidate({"x": 0.5, "y": 0.5}, space.initial_model(), {}, -1)
    children = tuner._sample(np.random.default_rng(0), [elite], 3, 2, 3)
    narrowed = narrow_model(
        space, space.initial_model(), elite.configuration, iteration=2, iterations=3, new=3
    )
    assert elite.model == narrowed
    assert len(children) == 3


def test_the_first_candidates_are_uniform_with_the_initial_model() -> None:
    tuner = IteratedRacing(SPACE, 200, seed=1)
    children = tuner._sample(np.random.default_rng(0), [], 5, 1, 4)
    assert len(children) == 5
    assert all(child.parent == -1 and child.model == SPACE.initial_model() for child in children)
    assert all(child.costs == {} for child in children)
    assert len({child.configuration for child in children}) == 5


def test_children_choose_their_parents_in_proportion_to_the_ranks() -> None:
    space = ParameterSpace([Real("x", 0.0, 1.0), Real("y", 0.0, 1.0)])
    tuner = IteratedRacing(space, 200, seed=1)
    elites = [
        _Candidate({"x": value, "y": value}, space.initial_model(), {}, -1)
        for value in (0.1, 0.5, 0.9)
    ]
    children = tuner._draw(np.random.default_rng(2), elites, 6000)
    counts = Counter(child.parent for child in children)
    # Probabilities (3, 2, 1) / 6, so 3000, 2000 and 1000 of 6000 (standard deviations 39, 37, 29).
    assert abs(counts[0] - 3000) < 150
    assert abs(counts[1] - 2000) < 150
    assert abs(counts[2] - 1000) < 150
    assert all(child.model is elites[child.parent].model for child in children)


# ------------------------------------------------------------------ arguments


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"budget": 0}, "budget"),
        ({"budget": 12.5}, "budget"),
        ({"seed": -1}, "seed"),
        ({"first_test": 1}, "first_test"),
        ({"each_test": 0}, "each_test"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
        ({"alpha": "0.05"}, "alpha"),
        ({"new_instances": 0}, "new_instances"),
        ({"idle_tests": 0}, "idle_tests"),
        ({"min_survivors": 0}, "min_survivors"),
        ({"iterations": 0}, "iterations"),
        ({"sample_instances": 1}, "sample_instances"),
        ({"space": [Real("x", 0, 1)]}, "ParameterSpace"),
    ],
)
def test_iterated_racing_refuses_settings_out_of_range(
    changes: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {"space": SPACE, "budget": 500, "seed": 0, **changes}
    with pytest.raises(ConfigError, match=message):
        IteratedRacing(**arguments)  # type: ignore[arg-type]


def test_the_seed_is_a_keyword_and_the_published_defaults_are_the_settings() -> None:
    with pytest.raises(TypeError):
        IteratedRacing(SPACE, 500, 1)  # type: ignore[call-arg]
    tuner = IteratedRacing(SPACE, 500, seed=1)
    # First test after 5 instances, then every instance, 5 % level, 1 new instance per race,
    # and at most 2 tests in a row that discard nobody: pp. 47 and 49.
    assert (tuner.first_test, tuner.each_test, tuner.alpha) == (5, 1, 0.05)
    assert (tuner.new_instances, tuner.idle_tests) == (1, 2)
    assert (tuner.min_survivors, tuner.iterations) == (None, None)
    assert tuner.sample_instances is True


def test_tuning_needs_instances_and_a_budget_for_a_first_race() -> None:
    with pytest.raises(ConfigError, match="at least one instance"):
        IteratedRacing(SPACE, 500, seed=0).tune([], Runner())
    # 4 parameters: 4 races, so 25 runs for the first, which afford 25 // 6 = 4 candidates, no
    # more than the 4 survivors a race keeps.
    with pytest.raises(ConfigError, match="raise the budget"):
        IteratedRacing(SPACE, 100, seed=0).tune(INSTANCES, Runner())
    result, _ = tune(120, 0)  # 30 // 6 = 5 candidates
    assert len(result.races[0].entries) == 5


# ------------------------------------------------------- budget and structure


@pytest.mark.parametrize("budget", [120, 137, 250, 333, 800, 1000])
@pytest.mark.parametrize("seed", [0, 1])
def test_the_runs_never_exceed_the_budget(budget: int, seed: int) -> None:
    result, runner = tune(budget, seed)
    assert 0 < result.runs <= budget
    assert result.runs == len(result.records) == len(runner.experiments)
    assert result.runs == sum(race.runs for race in result.races)
    assert [(r.configuration, r.instance, r.seed) for r in result.records] == [
        (e.configuration, e.instance, e.seed) for e in runner.experiments
    ]
    assert all(
        record.cost == noisy_cost(Experiment(record.configuration, record.instance, record.seed))
        for record in result.records
    )


def test_a_race_never_spends_more_than_its_share_and_the_run_follows_the_formulas() -> None:
    budget = 960
    result, runner = tune(budget, 3)
    planned, used = PLANNED, 0
    previous: RaceResult[int] | None = None
    for number, race in enumerate(result.races, start=1):
        planned = max(planned, number)
        share = iteration_budget(budget - used, planned, number)
        if previous is None:
            carried: list[int] = []
            seen = 0
        else:
            carried = elites_of(previous)
            seen = len(set().union(*(previous.entries[k].costs for k in carried)))
        expected = candidate_count(
            share,
            first_test=5,
            each_test=1,
            iteration=number,
            elites=len(carried),
            seen=seen,
            new_instances=1,
        )
        assert len(race.entries) == expected
        assert race.runs <= share
        if previous is not None:
            # The elites come first, in rank order, with the configurations of the race before.
            for position, k in enumerate(carried):
                assert race.entries[position].configuration is previous.entries[k].configuration
        used += race.runs
        previous = race
    assert used == result.runs
    # 960 runs over 4 races: the first gets 240, and 240 // 6 = 40 configurations race on 5
    # instances in one batch of 200.
    assert len(result.races[0].entries) == 40
    assert len(runner.batches[0]) == 200
    # The run ended because the budget was spent or another race would hold no new candidate.
    assert previous is not None
    carried = elites_of(previous)
    seen = len(set().union(*(previous.entries[k].costs for k in carried)))
    after = candidate_count(
        iteration_budget(budget - used, max(planned, len(result.races) + 1), len(result.races) + 1),
        first_test=5,
        each_test=1,
        iteration=len(result.races) + 1,
        elites=len(carried),
        seen=seen,
    )
    assert used == budget or previous.runs == 0 or after <= len(carried)


def test_the_planned_number_of_races_grows_while_budget_remains() -> None:
    result, _ = tune(600, 0, iterations=1)
    # One race planned takes the whole budget, and what it leaves runs further races.
    assert len(result.races) >= 2
    assert result.runs <= 600
    sparse, _ = tune(600, 0, iterations=2)
    assert len(sparse.races[0].entries) == (600 // 2) // 6


def test_a_race_goes_on_only_while_more_than_n_min_entries_remain_after_a_test() -> None:
    stops: Counter[str] = Counter()
    for seed in range(6):
        result, _ = tune(800, seed)
        for race in result.races:
            stops[race.stop] += 1
            if race.stop == "survivors":
                assert len(race.ranking) <= KEEP
            elif len(race.blocks) >= 5:
                # The race passed a test, and there were more than N_min entries after it.
                assert len(race.ranking) > KEEP
    # The runs end races in each way: by the number of survivors, by T_max, and by the budget.
    assert stops["survivors"] > 0
    assert stops["idle"] > 0
    assert stops["budget"] > 0


def test_a_race_that_runs_nothing_ends_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def nothing(candidates, instances, run_batch, *, elites, **settings):
        calls.append(len(candidates))
        assert len(calls) < 20, "the run did not end"
        entries = (*elites, *(Entry(candidate, {}) for candidate in candidates))
        ranking = tuple(range(len(entries)))
        return RaceResult(ranking, (0.0,) * len(entries), entries, (), (), "budget")

    monkeypatch.setattr(irace_module, "race", nothing)
    result = IteratedRacing(SPACE, 500, seed=0).tune(INSTANCES, Runner())
    assert (len(calls), result.runs, len(result.races)) == (1, 0, 1)
    assert len(result.elites) == KEEP


def test_elites_are_kept_until_the_new_candidates_saw_their_instances_and_not_run_again() -> None:
    eliminated_elites = 0
    for seed in range(6):
        result, _ = tune(800, seed)
        earlier: set[int] = set()
        previous = None
        for number, race in enumerate(result.races):
            if previous is not None:
                carried = elites_of(previous)
                old = set().union(*(previous.entries[k].costs for k in carried))
                protected = 1 + len(old)  # T_new + e blocks
                # One new instance goes first: a block no earlier race ran.
                assert race.blocks
                assert race.blocks[0].seed not in earlier
                for position, k in enumerate(carried):
                    entry = race.entries[position]
                    ran = {r.seed for r in race.records if r.configuration is entry.configuration}
                    # An elite's costs from earlier races are used, not run again.
                    assert not ran & set(previous.entries[k].costs)
                    assert set(previous.entries[k].costs) <= set(entry.costs)
                    if position not in race.ranking:
                        eliminated_elites += 1
                        assert len(race.blocks) >= protected, (seed, number)
            earlier |= {record.seed for record in race.records}
            previous = race
    assert eliminated_elites > 0  # the check above is not vacuous


def test_the_instances_of_the_elites_follow_the_new_one_in_a_random_order() -> None:
    out_of_order = 0
    for seed in range(4):
        result, _ = tune(800, seed)
        # The blocks of a run are taken from the stream in order, so the order in which they first
        # appear in a race is the order of their positions in the stream.
        position: dict[int, int] = {}
        previous = None
        for race in result.races:
            if previous is not None:
                old = set().union(*(previous.entries[k].costs for k in elites_of(previous)))
                presented = [position[block.seed] for block in race.blocks[1:] if block.seed in old]
                # The old instances come right after the new one.
                assert {block.seed for block in race.blocks[1 : 1 + len(presented)]} <= old
                assert len(presented) == len(set(presented)) >= min(4, len(old))
                out_of_order += presented != sorted(presented)
            for block in race.blocks:
                position.setdefault(block.seed, len(position))
            previous = race
    assert out_of_order > 0


def test_every_block_has_one_seed_for_all_configurations_and_a_new_seed_on_a_later_pass() -> None:
    # Four instances, and a first race of 5 or more instances, so one instance comes round again.
    runner = Runner()
    result = IteratedRacing(SPACE, 800, seed=4).tune([0, 1, 2, 3], runner)
    instance_of_seed: dict[int, int] = {}
    seeds_by_instance: dict[int, set[int]] = {}
    for record in result.records:
        assert instance_of_seed.setdefault(record.seed, record.instance) == record.instance
        seeds_by_instance.setdefault(record.instance, set()).add(record.seed)
    blocks = [block for race in result.races for block in race.blocks]
    assert set(instance_of_seed) <= {block.seed for block in blocks}
    assert all(len(seeds) > 1 for seeds in seeds_by_instance.values())
    # The first pass is a permutation of the four instances, and the fifth block repeats one of
    # them under a seed of its own.
    first = result.races[0].blocks[:5]
    assert {block.instance for block in first[:4]} == {0, 1, 2, 3}
    assert len({block.seed for block in first}) == 5
    assert first[4].instance in {block.instance for block in first[:4]}


def test_instances_come_in_the_given_order_when_they_are_not_sampled() -> None:
    result, _ = tune(800, 0, sample_instances=False)
    assert [block.instance for block in result.races[0].blocks[:5]] == list(range(5))
    shuffled, _ = tune(800, 0)
    first = [block.instance for block in shuffled.races[0].blocks[:5]]
    assert set(first) <= set(INSTANCES)
    assert first != list(range(5))


def test_instances_are_handed_over_as_they_are_and_need_not_be_hashable() -> None:
    instances = [{"id": index} for index in INSTANCES]

    def run_batch(experiments: Sequence[Experiment[dict[str, int]]]) -> list[float]:
        return [
            noisy_cost(Experiment(e.configuration, e.instance["id"], e.seed)) for e in experiments
        ]

    result = IteratedRacing(SPACE, 400, seed=2).tune(instances, run_batch)
    assert {record.instance["id"] for record in result.records} <= set(INSTANCES)
    assert all(any(record.instance is item for item in instances) for record in result.records[:50])


def test_the_settings_reach_the_races() -> None:
    result, runner = tune(800, 0, first_test=3, min_survivors=2)
    assert len(runner.batches[0]) == len(result.races[0].entries) * 3
    assert len(result.elites) <= 2
    longer, _ = tune(800, 0, new_instances=2)
    earlier: set[int] = set()
    for race in longer.races:
        if earlier:
            assert {block.seed for block in race.blocks[:2]}.isdisjoint(earlier)
        earlier |= {record.seed for record in race.records}


# ------------------------------------------------------------------ the replay


def test_the_same_seed_and_costs_give_the_same_configurations() -> None:
    first, first_runner = tune(800, 7)
    again, again_runner = tune(800, 7)
    assert again == first
    assert again.elites == first.elites
    assert again.records == first.records
    assert again_runner.experiments == first_runner.experiments
    other, _ = tune(800, 8)
    assert other.records != first.records
    assert other.elites != first.elites


def test_a_result_is_read_only_and_names_its_best() -> None:
    result, _ = tune(400, 5)
    assert result.best is result.elites[0]
    assert isinstance(result.elites, tuple)
    for configuration in result.elites:
        assert set(configuration) == set(SPACE.names)
        with pytest.raises(TypeError):
            configuration["x"] = 0.0  # type: ignore[index]


# ------------------------------------------------------------- the noisy target


def random_search(seed: int, budget: int) -> float:
    """The true cost of the best of random configurations that spend ``budget`` runs.

    Each configuration runs on five instances, and the cost of a run is divided by the scale of
    its instance, so that the comparison between configurations is as fair as it can be.
    """
    rng = np.random.default_rng(seed)
    best, value = None, math.inf
    for configuration in SPACE.sample(rng, budget // 5):
        total = 0.0
        for block in range(5):
            instance = block % len(INSTANCES)
            run = Experiment(configuration, instance, 1000 * seed + block)
            total += noisy_cost(run) / (1.0 + 0.5 * instance)
        if total < value:
            best, value = configuration, total
    assert best is not None
    return true_cost(best)


def test_the_target_has_the_known_best_configuration() -> None:
    assert true_cost(BEST) == 0.0
    rng = np.random.default_rng(0)
    assert all(true_cost(c) > 0.0 for c in SPACE.sample(rng, 300) if c != BEST)


def test_iterated_racing_finds_the_best_configuration_of_a_noisy_target() -> None:
    budget = 800  # 160 configurations on 5 instances, for a space of 4 parameters
    tuned, baseline, right_category = [], [], 0
    for seed in range(8):
        result, _ = tune(budget, seed)
        best = result.best
        tuned.append(true_cost(best))
        baseline.append(random_search(seed, budget))
        right_category += best["m"] == "b"
        # A best configuration costing 0.7 is among the best 1 % of the uniformly random ones
        # (about 0.46) or only a little worse; the noise makes the neighbouring ordinal level of
        # the best one, which costs 0.3 more, hard to rule out.
        assert true_cost(best) < 0.7
    assert right_category >= 7
    # Random search with the same runs finds configurations several times worse, on average.
    assert sum(tuned) / len(tuned) < 0.5 * sum(baseline) / len(baseline)
    assert sum(tuned) / len(tuned) < 0.15
