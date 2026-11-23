"""The steady-state genetic algorithm runs one control logic on both representations, in step."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from dispatchatlas.core import (
    Problem,
    SolveResult,
    StopCriteria,
    StopReason,
    UnsupportedProblemError,
    derive_seed,
)
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.serialization import canonical_json, schedule_to_dict
from dispatchatlas.core.solver import SOLVER_FAMILIES
from dispatchatlas.references import bibliography
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.registry import SolverRegistry
from dispatchatlas.solvers.scheduling import Decoded
from dispatchatlas.solvers.shop.decoding import BUILDERS, BuilderName, JobSequenceDecoder
from dispatchatlas.solvers.shop.family import REPRESENTATIONS
from dispatchatlas.solvers.shop.genetic import ShopSteadyStateGa, _Population
from dispatchatlas.solvers.shop.instance import JOB_SHOP_FEATURES, JobShop
from dispatchatlas.solvers.shop.representation import (
    KeyMove,
    KeyRepresentation,
    Representation,
    SequenceCrossover,
    SequenceMove,
    SequenceRepresentation,
)
from dispatchatlas.solvers.shop.run import (
    CONTROL_STREAM,
    OPERATOR_STREAM,
    Evaluated,
    ShopRun,
    schedule_digest,
)
from dispatchatlas.solvers.shop.sampling import ShopRandomSampling
from tests.helpers import chain_problem, random_problem, small_job_shop
from tests.solvers.shop.contract import check_contract
from tests.solvers.shop.families import PAIRED_BUILDERS, Recording, one_job, untimed
from tests.solvers.shop.instances import ta01

# The matched pairs of moves: one key redrawn with the insert, two keys perturbed with the swap.
MATCHED: list[tuple[KeyMove, SequenceMove]] = [("redraw", "insert"), ("perturb", "swap")]

# A step of two perturbed keys within the range that calibrate_step gives on Taillard's shops.
STEP = 0.6

# A population small enough for a few dozen evaluations to breed many children from it.
POPULATION = 6


def _ta01() -> Problem:
    return ta01().problem


PROBLEMS = [pytest.param(_ta01, id="ta01"), pytest.param(small_job_shop, id="3x3")]


def _arms(builder: BuilderName = "active", **parameters: Any) -> list[ShopSteadyStateGa]:
    """Return the solver on random keys and on job sequences, alike in every other parameter."""
    return [
        ShopSteadyStateGa(representation, builder, **parameters)
        for representation in REPRESENTATIONS
    ]


def _serialised(problem: Problem, result: Any) -> str:
    assert result.schedule is not None
    return canonical_json(schedule_to_dict(result.schedule, problem.id))


def _steps(evaluations: int, population: int = POPULATION) -> int:
    """Return the iterations a run begins under an evaluation limit.

    Each one evaluates a child, and the last is begun and refused by the budget, unless the
    limit falls inside the initial population, where none begins.
    """
    return max(evaluations - population + 1, 0)


# --------------------------------------------------------- the replacement rule


def _member(label: str, value: float, violation: float = 0.0) -> Evaluated[str]:
    """Return a stand-in for an evaluated member: a label for its genotype, and its score."""
    return Evaluated(
        label, np.empty(0, dtype=np.intp), Score(violation, value), Decoded((), (), (), ())
    )


def _population(*values: float) -> _Population[str]:
    """Return a population whose members a, b, c, ... have these values, a the oldest."""
    return _Population([_member(chr(ord("a") + slot), v) for slot, v in enumerate(values)])


def _labels(population: _Population[str]) -> str:
    return "".join(member.genotype for member in population.members)


def test_a_child_replaces_the_worst_member_when_it_is_no_worse() -> None:
    population = _population(5, 9, 7, 3)
    assert population.worst() == 1
    assert population.admit(_member("e", 8))
    assert _labels(population) == "aecd"
    # Equal to the worst member, which is now e, it is no worse and replaces it.
    assert population.admit(_member("f", 8))
    assert _labels(population) == "afcd"
    # Worse than the worst member, it is dropped.
    assert not population.admit(_member("g", 9))
    assert _labels(population) == "afcd"
    assert [member.score.value for member in population.members] == [5, 8, 7, 3]


def test_among_equal_worst_members_the_oldest_is_replaced() -> None:
    population = _population(4, 6, 6, 2)
    # b and c tie at 6, and b is older.
    assert population.worst() == 1
    assert population.admit(_member("e", 6))
    assert _labels(population) == "aecd"
    # Now c is older than e, though e sits in the lower slot: age decides, not the slot.
    assert population.worst() == 2
    assert population.admit(_member("f", 6))
    assert _labels(population) == "aefd"
    assert population.worst() == 1
    assert population.admit(_member("g", 6))
    assert _labels(population) == "agfd"
    assert population.worst() == 2


def test_a_dropped_child_leaves_the_ages_as_they_were() -> None:
    population = _population(4, 6, 6, 2)
    assert not population.admit(_member("e", 7))
    assert _labels(population) == "abcd"
    assert population.worst() == 1
    assert population.admit(_member("f", 6))
    assert _labels(population) == "afcd"


def test_equal_members_are_replaced_in_the_order_they_came() -> None:
    population = _population(6, 6, 6, 6)
    order = []
    for label in "efghijkl":
        order.append(population.worst())
        assert population.admit(_member(label, 6))
    # The initial members are older the earlier they come, and each child is the youngest.
    assert order == [0, 1, 2, 3, 0, 1, 2, 3]


def test_an_overrun_of_a_deadline_is_worse_than_any_value() -> None:
    population = _Population([_member("a", 5), _member("b", 0, violation=1.0), _member("c", 7)])
    assert population.worst() == 1
    assert not population.admit(_member("d", 0, violation=2.0))
    assert population.admit(_member("e", 0, violation=1.0))
    assert _labels(population) == "aec"
    # A smaller overrun is better than e's, however large its value, but still worse than none.
    assert population.admit(_member("f", 99, violation=0.5))
    assert _labels(population) == "afc"
    assert population.worst() == 1


def test_a_tournament_selects_the_fittest_and_among_equals_the_lowest_slot() -> None:
    population = _population(5, 3, 3, 8, 3)
    # Both tournaments have the same contestants, so both parents are the winner.
    assert population.parents([0, 3], [0, 3]) == (0, 0)
    assert population.parents([1, 2, 4], [1, 2, 4]) == (1, 1)
    assert population.parents([4, 2], [4, 2]) == (2, 2)
    assert population.parents([3, 0, 3], [3, 0, 3]) == (0, 0)


def test_the_fitter_winner_is_the_first_parent_and_equal_ones_come_by_slot() -> None:
    population = _population(5, 3, 3, 8, 3)
    # Winners 0 (5) and 2 (3): the second tournament's winner is fitter, so it comes first.
    assert population.parents([0, 3], [4, 2]) == (2, 0)
    assert population.parents([4, 2], [0, 3]) == (2, 0)
    # Winners 4 and 1 both have 3: the lower slot is first, whichever tournament it won.
    assert population.parents([4], [1]) == (1, 4)
    assert population.parents([1], [4]) == (1, 4)
    # A member that wins both tournaments is both parents.
    assert population.parents([3, 3], [3]) == (3, 3)
    # A contestant may be drawn twice.
    assert population.parents([0, 0], [3, 3]) == (0, 3)


# ------------------------------------------------------------- common start


@pytest.mark.parametrize("builder", PAIRED_BUILDERS)
@pytest.mark.parametrize("make", PROBLEMS)
def test_both_arms_start_from_the_same_population(
    make: Callable[[], Problem], builder: BuilderName, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = make()
    shop = JobShop.from_problem(problem)
    recording = Recording(monkeypatch)
    for solver in _arms(builder, population_size=POPULATION):
        solver.solve(problem, StopCriteria(max_evaluations=40), seed=7)
    keys_run, sequence_run = recording.runs()
    # The population is the first draws of the operator stream, decoded, in either representation.
    operators = np.random.default_rng(derive_seed(7, OPERATOR_STREAM))
    decoder = JobSequenceDecoder(shop, builder)
    drawn = [
        decoder.decode(SequenceRepresentation(shop).initial(operators)) for _ in range(POPULATION)
    ]
    for run in (keys_run, sequence_run):
        members = run[:POPULATION]
        assert [e.digest for e in members] == [schedule_digest(d) for d in drawn]
        assert [(e.decoded.sequence, e.decoded.starts, e.decoded.ends) for e in members] == [
            (d.sequence, d.starts, d.ends) for d in drawn
        ]
        # Members are evaluated as solutions of their own, children as moves of a parent.
        assert all(e.parent is None for e in members)
        assert all(e.parent is not None for e in run[POPULATION:])


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_both_arms_evaluate_the_same_schedule_first(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = Recording(monkeypatch)
    shop = JobShop.from_problem(small_job_shop())
    for builder in PAIRED_BUILDERS:
        # The first schedule is the first draw of the operator stream, decoded.
        first = JobSequenceDecoder(shop, builder).decode(
            SequenceRepresentation(shop).initial(
                np.random.default_rng(derive_seed(seed, OPERATOR_STREAM))
            )
        )
        keys, sequence = (
            solver.solve(shop.problem, StopCriteria(max_evaluations=30), seed=seed)
            for solver in _arms(builder, population_size=POPULATION)
        )
        keys_run, sequence_run = recording.runs()[-2:]
        assert keys_run[0].digest == sequence_run[0].digest == schedule_digest(first)
        assert keys_run[0].parent is sequence_run[0].parent is None
        assert keys.extras["first_schedule_digest"] == sequence.extras["first_schedule_digest"]
        assert keys.extras["first_schedule_digest"] == schedule_digest(first).hex()


def test_without_crossover_or_mutation_the_operator_stream_is_drawn_for_the_population_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The tournaments, the replacement and the observer draw nothing from the operator stream.
    recording = Recording(monkeypatch)
    shop = JobShop.from_problem(small_job_shop())
    for solver in _arms(
        "semi-active", population_size=POPULATION, crossover_rate=0.0, mutation_rate=0.0
    ):
        solver.solve(shop.problem, StopCriteria(max_evaluations=50), seed=6)
    expected = np.random.default_rng(derive_seed(6, OPERATOR_STREAM))
    expected.random(POPULATION * shop.size)
    assert len(recording.finished) == 2
    for run in recording.finished:
        assert run.operators.bit_generator.state == expected.bit_generator.state


# ------------------------------------------------------------------ budget


@pytest.mark.parametrize(
    "evaluations", [1, POPULATION - 1, POPULATION, POPULATION + 1, 3 * POPULATION]
)
def test_every_genetic_run_ends_at_its_evaluation_limit(evaluations: int) -> None:
    for builder in PAIRED_BUILDERS:
        for key_move, sequence_move in MATCHED:
            results = [
                solver.solve(small_job_shop(), StopCriteria(max_evaluations=evaluations), seed=9)
                for solver in _arms(
                    builder,
                    population_size=POPULATION,
                    key_move=key_move,
                    key_step=STEP,
                    sequence_move=sequence_move,
                )
            ]
            for result in results:
                assert result.evaluations == evaluations
                assert result.stop_reason is StopReason.EVALUATION_LIMIT
                assert result.extras["repairs"] == 0
                assert result.extras["observed_evaluations"] == evaluations
                assert result.extras["observed_moves"] == max(evaluations - POPULATION, 0)
                assert ("replacement_share" in result.extras) is (evaluations > POPULATION)


def test_the_genetic_algorithm_runs_under_an_iteration_or_a_time_limit_alone() -> None:
    for solver in _arms("active", population_size=POPULATION):
        by_iterations = solver.solve(small_job_shop(), StopCriteria(max_iterations=6), seed=2)
        assert by_iterations.stop_reason is StopReason.ITERATION_LIMIT
        # The initial population is not an iteration: each of the six iterations breeds a child.
        assert by_iterations.iterations == 6
        assert by_iterations.evaluations == POPULATION + 6
        by_time = solver.solve(small_job_shop(), StopCriteria(time_limit=0.05), seed=2)
        assert by_time.stop_reason is StopReason.TIME_LIMIT
        assert by_time.evaluations >= 1


# ------------------------------------------------------- control stream


@pytest.mark.parametrize("evaluations", [POPULATION - 2, POPULATION, 60])
@pytest.mark.parametrize("contests", [1, 2, 5])
@pytest.mark.parametrize(
    ("crossover_rate", "mutation_rate"),
    [pytest.param(0.0, 0.0, id="never"), (0.5, 0.5), pytest.param(1.0, 1.0, id="always")],
)
def test_the_control_stream_is_drawn_in_step_whatever_the_outcomes(
    crossover_rate: float,
    mutation_rate: float,
    contests: int,
    evaluations: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = Recording(monkeypatch)
    for builder in PAIRED_BUILDERS:
        for solver in _arms(
            builder,
            population_size=POPULATION,
            tournament_size=contests,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
        ):
            solver.solve(small_job_shop(), StopCriteria(max_evaluations=evaluations), seed=5)
    expected = np.random.default_rng(derive_seed(5, CONTROL_STREAM))
    # One call of two tournaments' contestants and two coins for every iteration begun.
    expected.random(_steps(evaluations) * (2 * contests + 2))
    assert len(recording.finished) == 2 * len(PAIRED_BUILDERS)
    for run in recording.finished:
        assert run.control.bit_generator.state == expected.bit_generator.state


# ----------------------------------------------------------------- breeding


@dataclass
class Birth:
    """What a run handed to ``evaluate``: a genotype and its parent, then its evaluation.

    The evaluation stays ``None`` when the budget refused the genotype.
    """

    genotype: Any
    parent: Evaluated[Any] | None
    evaluated: Evaluated[Any] | None = None


@dataclass(frozen=True)
class Crossing:
    """One call of ``recombine``: the parents it got, its inheritance, and the child it made."""

    first: Any
    second: Any
    inheritance: float
    child: Any


@dataclass(frozen=True)
class Mutation:
    """One call of ``move``: the genotype it got, and the one it made."""

    genotype: Any
    moved: Any


class Breeding:
    """Keeps what every run hands to ``evaluate``, and every call of its operators.

    The wrappers read what they are given and return the operators' results unchanged.

    Args:
        monkeypatch: The test's monkeypatch, which undoes the wrappers after it.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.births: dict[ShopRun[Any], list[Birth]] = {}
        self.crossings: dict[Representation[Any], list[Crossing]] = {}
        self.mutations: dict[Representation[Any], list[Mutation]] = {}
        self.finished: list[ShopRun[Any]] = []
        evaluate, finish = ShopRun.evaluate, ShopRun.finish

        def recorded_evaluate(
            run: ShopRun[Any], genotype: Any, parent: Evaluated[Any] | None = None
        ) -> Evaluated[Any]:
            birth = Birth(genotype, parent)
            self.births.setdefault(run, []).append(birth)
            birth.evaluated = evaluate(run, genotype, parent)
            return birth.evaluated

        def recorded_finish(run: ShopRun[Any]) -> None:
            self.finished.append(run)
            finish(run)

        monkeypatch.setattr(ShopRun, "evaluate", recorded_evaluate)
        monkeypatch.setattr(ShopRun, "finish", recorded_finish)
        for representation_class in (KeyRepresentation, SequenceRepresentation):
            self._record(monkeypatch, representation_class)

    def _record(self, monkeypatch: pytest.MonkeyPatch, cls: type[Representation[Any]]) -> None:
        recombine, move = cls.recombine, cls.move

        def recorded_recombine(
            representation: Representation[Any],
            first: Any,
            second: Any,
            rng: np.random.Generator,
            inheritance: float,
        ) -> Any:
            child = recombine(representation, first, second, rng, inheritance)
            self.crossings.setdefault(representation, []).append(
                Crossing(first, second, inheritance, child)
            )
            return child

        def recorded_move(
            representation: Representation[Any], genotype: Any, rng: np.random.Generator
        ) -> Any:
            moved = move(representation, genotype, rng)
            self.mutations.setdefault(representation, []).append(Mutation(genotype, moved))
            return moved

        monkeypatch.setattr(cls, "recombine", recorded_recombine)
        monkeypatch.setattr(cls, "move", recorded_move)


@dataclass
class Replay:
    """What a replay of a run's control stream found, and how often a tie decided a step."""

    children: int = 0
    entered: int = 0
    tied_worst: int = 0
    tied_parents: int = 0
    same_parent: int = 0
    flipped: int = 0


def _replay(
    solver: ShopSteadyStateGa, breeding: Breeding, result: SolveResult, seed: int
) -> Replay:
    """Replay the control stream of the run that finished last against its recorded evaluations.

    The replay is written from the rules, not from the solver's code: a naive population of
    the evaluations the run made, its tournaments read from the control numbers, and a linear
    scan for the worst member. Every iteration must have had the parents the replay names,
    handed to the operators in the order the replay says, with the operators called exactly
    when the control numbers say, and every child that the replay admits must be the child the
    run admitted: the replacement share of the result proves it.
    """
    run = breeding.finished[-1]
    births = breeding.births[run]
    crossings = iter(breeding.crossings.get(run.representation, []))
    mutations = iter(breeding.mutations.get(run.representation, []))
    size, contests = solver.population_size, solver.tournament_size
    members = [b.evaluated for b in births[:size] if b.evaluated is not None]
    assert len(members) == size
    assert all(b.parent is None for b in births[:size])
    # The birth number of each slot's member: the lower, the older.
    born = list(range(size))
    entries = size
    control = np.random.default_rng(derive_seed(seed, CONTROL_STREAM))
    replay = Replay()

    def fitness(slot: int) -> tuple[float, int]:
        return members[slot].score.value, slot

    def staleness(slot: int) -> tuple[float, int]:
        return members[slot].score.value, -born[slot]

    for birth in births[size:]:
        draws = control.random(2 * contests + 2).tolist()
        slots = [int(u * size) for u in draws[: 2 * contests]]
        winners = min(slots[:contests], key=fitness), min(slots[contests:], key=fitness)
        first, second = sorted(winners, key=fitness)
        replay.same_parent += first == second
        replay.tied_parents += first != second and fitness(first)[0] == fitness(second)[0]
        replay.flipped += (first, second) != winners
        parent = members[first]
        assert birth.parent is parent
        genotype = parent.genotype
        if draws[-2] < solver.crossover_rate:
            crossing = next(crossings)
            assert crossing.first is parent.genotype
            assert crossing.second is members[second].genotype
            assert crossing.inheritance == solver.inheritance
            genotype = crossing.child
        if draws[-1] < solver.mutation_rate:
            mutation = next(mutations)
            assert mutation.genotype is genotype
            genotype = mutation.moved
        assert birth.genotype is genotype
        child = birth.evaluated
        if child is None:
            # The iteration the budget refused: the last one, and the only one without a child.
            assert birth is births[-1]
            break
        replay.children += 1
        worst = max(range(size), key=staleness)
        values = [member.score.value for member in members]
        replay.tied_worst += values.count(values[worst]) > 1
        if child.score.value <= values[worst]:
            members[worst], born[worst] = child, entries
            entries += 1
            replay.entered += 1
    assert next(crossings, None) is None
    assert next(mutations, None) is None
    assert run.control.bit_generator.state == control.bit_generator.state
    if replay.children:
        assert result.extras["replacement_share"] == replay.entered / replay.children
    else:
        assert "replacement_share" not in result.extras
    return replay


BREEDING = [
    pytest.param({"crossover_rate": 0.9, "mutation_rate": 0.2, "inheritance": 0.7}, id="default"),
    pytest.param({"crossover_rate": 1.0, "mutation_rate": 1.0, "inheritance": 0.5}, id="always"),
    pytest.param({"crossover_rate": 0.0, "mutation_rate": 1.0}, id="mutation-only"),
    pytest.param(
        {"crossover_rate": 1.0, "mutation_rate": 0.0, "inheritance": 0.9}, id="crossover-only"
    ),
    pytest.param({"crossover_rate": 0.0, "mutation_rate": 0.0}, id="clones"),
]


@pytest.mark.parametrize("contests", [1, 2, 4])
@pytest.mark.parametrize("parameters", BREEDING)
@pytest.mark.parametrize("builder", PAIRED_BUILDERS)
def test_every_child_is_bred_from_the_parents_the_control_stream_names(
    builder: BuilderName,
    parameters: dict[str, float],
    contests: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    breeding = Breeding(monkeypatch)
    evaluations = 90
    total = Replay()
    for seed in (3, 4):
        for solver in _arms(
            builder,
            population_size=POPULATION,
            tournament_size=contests,
            key_move="perturb",
            key_step=STEP,
            sequence_move="swap",
            **parameters,
        ):
            result = solver.solve(
                small_job_shop(), StopCriteria(max_evaluations=evaluations), seed=seed
            )
            replay = _replay(solver, breeding, result, seed)
            assert replay.children == evaluations - POPULATION
            total.tied_worst += replay.tied_worst
            total.tied_parents += replay.tied_parents
            total.same_parent += replay.same_parent
            total.flipped += replay.flipped
    # The replay has power: ties, repeated members and parents out of order all came up.
    assert total.tied_worst > 0
    assert total.flipped > 0
    if contests > 1:
        assert total.tied_parents > 0
    assert total.same_parent > 0 or contests == 1


@pytest.mark.parametrize("contests", [2, 3])
def test_every_child_on_ta01_is_bred_from_the_parents_the_control_stream_names(
    contests: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    breeding = Breeding(monkeypatch)
    for solver in _arms(
        "semi-active",
        population_size=10,
        tournament_size=contests,
        key_move="perturb",
        key_step=STEP,
    ):
        result = solver.solve(_ta01(), StopCriteria(max_evaluations=200), seed=11)
        replay = _replay(solver, breeding, result, 11)
        assert replay.children == 190
        assert replay.flipped > 0


def test_a_run_that_ends_in_the_population_replays_without_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    breeding = Breeding(monkeypatch)
    for evaluations in (POPULATION, POPULATION + 1):
        for solver in _arms("active", population_size=POPULATION):
            result = solver.solve(
                small_job_shop(), StopCriteria(max_evaluations=evaluations), seed=2
            )
            replay = _replay(solver, breeding, result, 2)
            assert replay.children == evaluations - POPULATION


@pytest.mark.parametrize("inheritance", [0.0, 1.0])
def test_a_child_takes_everything_from_the_first_parent_or_from_the_second_at_the_bounds(
    inheritance: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    breeding = Breeding(monkeypatch)
    for solver in _arms(
        "semi-active",
        population_size=POPULATION,
        crossover_rate=1.0,
        mutation_rate=0.0,
        inheritance=inheritance,
    ):
        result = solver.solve(_ta01(), StopCriteria(max_evaluations=80), seed=15)
        crossings = breeding.crossings[breeding.finished[-1].representation]
        assert len(crossings) == 80 - POPULATION + 1
        for crossing in crossings:
            taken = crossing.first if inheritance == 1.0 else crossing.second
            assert np.array_equal(crossing.child, taken)
        if inheritance == 1.0:
            # Every child is a copy of its first parent, the fitter of its parents.
            assert result.extras["unchanged_move_share"] == 1.0


@pytest.mark.parametrize("sequence_crossover", ["ppx", "jox"])
@pytest.mark.parametrize(("key_move", "sequence_move"), MATCHED)
def test_a_mutation_is_one_move_of_the_arms_tuned_kind(
    key_move: KeyMove,
    sequence_move: SequenceMove,
    sequence_crossover: SequenceCrossover,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    breeding = Breeding(monkeypatch)
    evaluations = 60
    for solver in _arms(
        "semi-active",
        population_size=POPULATION,
        crossover_rate=0.0,
        mutation_rate=1.0,
        key_move=key_move,
        key_step=STEP,
        sequence_move=sequence_move,
        sequence_crossover=sequence_crossover,
    ):
        solver.solve(small_job_shop(), StopCriteria(max_evaluations=evaluations), seed=8)
    keys_run, sequence_run = breeding.finished
    keys, sequence = keys_run.representation, sequence_run.representation
    assert isinstance(keys, KeyRepresentation)
    assert (keys.key_move, keys.key_step) == (key_move, STEP)
    assert isinstance(sequence, SequenceRepresentation)
    assert (sequence.move_kind, sequence.crossover) == (sequence_move, sequence_crossover)
    # Every iteration begun mutates its child once, and the keys that change are the move's.
    moves = breeding.mutations[keys]
    assert len(moves) == evaluations - POPULATION + 1
    assert {int(np.count_nonzero(m.moved != m.genotype)) for m in moves} == {keys.key_count}
    assert keys.key_count == (1 if key_move == "redraw" else 2)
    moves = breeding.mutations[sequence]
    assert len(moves) == evaluations - POPULATION + 1
    for move in moves:
        assert sorted(move.moved) == sorted(move.genotype)
        assert not np.array_equal(move.moved, move.genotype)
        if sequence_move == "swap":
            assert np.count_nonzero(move.moved != move.genotype) == 2
        else:
            assert _is_insert(move.genotype, move.moved)


def _is_insert(before: Any, after: Any) -> bool:
    """Return whether ``after`` is ``before`` with one entry moved to another position."""
    jobs, wanted = list(before), list(after)
    for source in range(len(jobs)):
        for target in range(len(jobs)):
            moved = jobs.copy()
            moved.insert(target, moved.pop(source))
            if moved == wanted:
                return True
    return False


# ------------------------------------------------------------------ results


def test_the_genetic_algorithm_replays_from_its_seed() -> None:
    problem = _ta01()
    stop = StopCriteria(max_evaluations=150)
    for solver in _arms("semi-active", population_size=10):
        first, second, other = (solver.solve(problem, stop, seed=s) for s in (13, 13, 14))
        assert _serialised(problem, first) == _serialised(problem, second)
        assert [(p.evaluations, p.iteration, p.value) for p in first.trace] == [
            (p.evaluations, p.iteration, p.value) for p in second.trace
        ]
        assert untimed(first.extras) == untimed(second.extras)
        assert first.extras["first_schedule_digest"] != other.extras["first_schedule_digest"]


@pytest.mark.parametrize("sequence_crossover", ["ppx", "jox"])
@pytest.mark.parametrize(("key_move", "sequence_move"), [*MATCHED, ("redraw", "mixed")])
@pytest.mark.parametrize("builder", BUILDERS)
def test_the_genetic_algorithm_passes_the_contract(
    builder: BuilderName,
    key_move: KeyMove,
    sequence_move: SequenceMove,
    sequence_crossover: SequenceCrossover,
) -> None:
    for representation in REPRESENTATIONS:
        for problem in (small_job_shop(), chain_problem()):
            result = check_contract(
                ShopSteadyStateGa,
                problem,
                representation=representation,
                builder=builder,
                population_size=8,
                key_move=key_move,
                key_step=STEP,
                sequence_move=sequence_move,
                sequence_crossover=sequence_crossover,
            )
            assert result.extras["repairs"] == 0
            # The observer sees the population and every child, as a move of its first parent.
            assert result.extras["observed_evaluations"] == 40
            assert result.extras["observed_moves"] == 40 - 8
            share = result.extras["replacement_share"]
            assert isinstance(share, float)
            assert 0.0 < share <= 1.0


@pytest.mark.parametrize("builder", BUILDERS)
def test_the_genetic_algorithm_passes_the_contract_on_ta01(builder: BuilderName) -> None:
    for representation in REPRESENTATIONS:
        result = check_contract(
            ShopSteadyStateGa,
            _ta01(),
            representation=representation,
            builder=builder,
            population_size=8,
        )
        assert result.extras["repairs"] == 0
        assert result.extras["observed_moves"] == 40 - 8


def test_the_genetic_algorithm_beats_random_sampling_at_equal_evaluations_on_ta01() -> None:
    problem = _ta01()
    stop = StopCriteria(max_evaluations=1000)
    for representation in REPRESENTATIONS:
        sampled = ShopRandomSampling(representation, "semi-active").solve(problem, stop, seed=7)
        evolved = ShopSteadyStateGa(representation, "semi-active").solve(problem, stop, seed=7)
        assert evolved.value is not None
        assert sampled.value is not None
        assert evolved.value < sampled.value


# ---------------------------------------------------------------- edge cases


@pytest.mark.parametrize("operations", [1, 2, 3])
def test_the_genetic_algorithm_evaluates_the_single_sequence_of_one_job_and_stops(
    operations: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = one_job(operations)
    recording = Recording(monkeypatch)
    for builder in BUILDERS:
        # Every draw would call for a move, which a job sequence of one job has none of.
        for solver in _arms(
            builder, population_size=POPULATION, crossover_rate=1.0, mutation_rate=1.0
        ):
            result = solver.solve(problem, StopCriteria(max_evaluations=30), seed=3)
            assert result.evaluations == 1
            assert result.stop_reason is StopReason.COMPLETED
            assert result.value == sum(step + 2 for step in range(operations))
            assert result.extras["repairs"] == 0
            assert "replacement_share" not in result.extras
    assert len(recording.finished) == 2 * len(BUILDERS)


def test_the_genetic_algorithm_finishes_every_run_it_ends_in_the_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = Recording(monkeypatch)
    for solver in _arms("active", population_size=POPULATION):
        result = solver.solve(small_job_shop(), StopCriteria(max_evaluations=3), seed=1)
        assert result.evaluations == 3
        assert result.extras["first_schedule_digest"]
    assert len(recording.finished) == 2


def test_a_population_may_be_smaller_than_a_tournament() -> None:
    for solver in _arms("semi-active", population_size=2, tournament_size=7):
        result = solver.solve(small_job_shop(), StopCriteria(max_evaluations=40), seed=4)
        assert result.evaluations == 40
        assert result.stop_reason is StopReason.EVALUATION_LIMIT


def test_a_run_stops_when_its_first_member_reaches_the_target() -> None:
    for solver in _arms("active", population_size=POPULATION):
        result = solver.solve(
            small_job_shop(), StopCriteria(max_evaluations=200, target=1e9), seed=1
        )
        assert result.stop_reason is StopReason.TARGET_REACHED
        assert result.evaluations == 1
        assert "replacement_share" not in result.extras


def test_a_run_stops_when_a_child_reaches_the_target() -> None:
    for solver in _arms("semi-active", population_size=POPULATION):
        # A target the initial population misses by one, so that only a child can reach it.
        for seed in range(30):
            population = solver.solve(
                small_job_shop(), StopCriteria(max_evaluations=POPULATION), seed=seed
            )
            assert population.value is not None
            if population.value >= 13:
                break
        else:
            pytest.fail("no seed gives an initial population that misses a makespan of 12")
        target = population.value - 1
        result = solver.solve(
            small_job_shop(), StopCriteria(max_evaluations=5000, target=target), seed=seed
        )
        assert result.stop_reason is StopReason.TARGET_REACHED
        assert result.value is not None
        assert result.value <= target
        assert POPULATION < result.evaluations < 5000
        # The search ended inside a child, and still wrote its measures.
        assert "first_schedule_digest" in result.extras
        assert result.extras["observed_evaluations"] == result.evaluations


def test_the_genetic_algorithm_refuses_a_problem_that_is_not_a_job_shop() -> None:
    problem = random_problem(2, tasks=8, max_modes=2, pools=1)
    for solver in _arms("active"):
        with pytest.raises(UnsupportedProblemError):
            solver.solve(problem, StopCriteria(max_evaluations=10), seed=0)


# -------------------------------------------------------------- the solver


def test_the_genetic_algorithm_declares_what_it_models_and_cites_the_bibliography() -> None:
    info = ShopSteadyStateGa.info
    assert info.id == "shop-steady-state-ga"
    assert info.family in SOLVER_FAMILIES
    assert info.shape == "job-shop"
    assert info.supports == JOB_SHOP_FEATURES
    assert info.accepts(small_job_shop())
    assert not info.accepts(random_problem(2, tasks=8, max_modes=2, pools=1))
    assert info.summary.endswith(".")
    assert info.references
    assert set(info.references) <= set(bibliography())


def test_the_genetic_algorithm_registers_and_is_created_by_its_id() -> None:
    registry = SolverRegistry()
    registry.register(ShopSteadyStateGa)
    solver = registry.create("shop-steady-state-ga", representation="keys", population_size=5)
    assert solver == ShopSteadyStateGa("keys", population_size=5)
    assert set(registry.info("shop-steady-state-ga").parameters) == {
        "representation",
        "builder",
        "population_size",
        "tournament_size",
        "crossover_rate",
        "mutation_rate",
        "inheritance",
        "key_move",
        "key_step",
        "sequence_move",
        "sequence_crossover",
    }
    with pytest.raises(ConfigError, match="population_size"):
        registry.create("shop-steady-state-ga", population_size=1)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"representation": "permutation"}, "unknown representation"),
        ({"builder": "giffler"}, "unknown builder"),
        ({"population_size": 1}, "population_size"),
        ({"population_size": 0}, "population_size"),
        ({"population_size": True}, "population_size"),
        ({"population_size": 3.0}, "population_size"),
        ({"tournament_size": 0}, "tournament_size"),
        ({"tournament_size": True}, "tournament_size"),
        ({"tournament_size": 2.0}, "tournament_size"),
        ({"crossover_rate": -0.1}, "crossover_rate"),
        ({"crossover_rate": 1.1}, "crossover_rate"),
        ({"crossover_rate": math.nan}, "crossover_rate"),
        ({"crossover_rate": True}, "crossover_rate"),
        ({"crossover_rate": "0.5"}, "crossover_rate"),
        ({"mutation_rate": -0.1}, "mutation_rate"),
        ({"mutation_rate": 1.1}, "mutation_rate"),
        ({"mutation_rate": math.nan}, "mutation_rate"),
        ({"inheritance": -0.1}, "inheritance"),
        ({"inheritance": 1.1}, "inheritance"),
        ({"inheritance": math.nan}, "inheritance"),
        ({"key_move": "shift"}, "unknown key move"),
        ({"key_step": 0.0}, "key_step"),
        ({"key_step": math.inf}, "key_step"),
        ({"key_step": True}, "key_step"),
        ({"key_step": "0.5"}, "key_step"),
        ({"sequence_move": "reverse"}, "unknown sequence move"),
        ({"sequence_crossover": "ox"}, "unknown crossover"),
    ],
)
def test_the_genetic_algorithm_refuses_bad_parameters(
    parameters: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ShopSteadyStateGa(**parameters)


def test_the_parameters_of_the_other_representation_are_checked_too() -> None:
    with pytest.raises(ValueError, match="key_step"):
        ShopSteadyStateGa("sequence", key_step=-1.0)
    with pytest.raises(ValueError, match="sequence move"):
        ShopSteadyStateGa("keys", sequence_move="reverse")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown crossover"):
        ShopSteadyStateGa("keys", sequence_crossover="ox")  # type: ignore[arg-type]


def test_the_bounds_of_the_rates_are_accepted() -> None:
    for rate in (0, 1, 0.0, 1.0):
        ShopSteadyStateGa(crossover_rate=rate, mutation_rate=rate, inheritance=rate)
    ShopSteadyStateGa(population_size=2, tournament_size=1)
