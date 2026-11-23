"""The particle swarm runs one control logic on both representations, with two update rules."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from dispatchatlas.core import Problem, StopCriteria, StopReason, derive_seed
from dispatchatlas.core.serialization import canonical_json, schedule_to_dict
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.shop.decoding import BUILDERS, BuilderName
from dispatchatlas.solvers.shop.family import REPRESENTATIONS, RepresentationName
from dispatchatlas.solvers.shop.instance import JobShop
from dispatchatlas.solvers.shop.representation import SequenceMove, SequenceRepresentation, reflect
from dispatchatlas.solvers.shop.run import CONTROL_STREAM, OPERATOR_STREAM, Evaluated, ShopRun
from dispatchatlas.solvers.shop.sampling import ShopRandomSampling
from dispatchatlas.solvers.shop.swarm import (
    TOPOLOGIES,
    ShopParticleSwarm,
    Topology,
    _ConstrictionUpdate,
    _DiscreteUpdate,
    _Memory,
    constriction_coefficient,
)
from tests.helpers import chain_problem, small_job_shop
from tests.solvers.shop.contract import check_contract
from tests.solvers.shop.families import PAIRED_BUILDERS, one_job, untimed
from tests.solvers.shop.instances import random_job_shop, ta01


def _ta01() -> Problem:
    return ta01().problem


def _random_shop() -> Problem:
    return random_job_shop(3, jobs=4, machines=3)


PROBLEMS = [pytest.param(_ta01, id="ta01"), pytest.param(small_job_shop, id="3x3")]
SMALL = [pytest.param(small_job_shop, id="3x3"), pytest.param(_random_shop, id="4x3")]


def _arms(builder: BuilderName = "active", **parameters: Any) -> list[ShopParticleSwarm]:
    """Return the solver on random keys and on job sequences, alike in every other parameter."""
    return [
        ShopParticleSwarm(representation, builder, **parameters)
        for representation in REPRESENTATIONS
    ]


def _serialised(problem: Problem, result: Any) -> str:
    assert result.schedule is not None
    return canonical_json(schedule_to_dict(result.schedule, problem.id))


# ------------------------------------------------------------------ the spy


@dataclass(frozen=True)
class Step:
    """One evaluation of a run: the genotype it took, its value, its digest, its parent's digest."""

    genotype: NDArray[Any]
    value: float
    digest: bytes
    parent: bytes | None


class Spy:
    """Keeps what a swarm evaluates, what its update rule is given, and the runs that finish.

    The wrappers read what they are given and change nothing in it.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.steps: dict[ShopRun[Any], list[Step]] = {}
        self.finished: list[ShopRun[Any]] = []
        self.calls: list[tuple[int, NDArray[Any], NDArray[Any]]] = []
        evaluate, finish = ShopRun.evaluate, ShopRun.finish

        def spied_evaluate(
            run: ShopRun[Any], genotype: Any, parent: Evaluated[Any] | None = None
        ) -> Evaluated[Any]:
            evaluated = evaluate(run, genotype, parent)
            self.steps.setdefault(run, []).append(
                Step(
                    np.array(genotype),
                    evaluated.score.value,
                    evaluated.digest,
                    None if parent is None else parent.digest,
                )
            )
            return evaluated

        def spied_finish(run: ShopRun[Any]) -> None:
            self.finished.append(run)
            finish(run)

        monkeypatch.setattr(ShopRun, "evaluate", spied_evaluate)
        monkeypatch.setattr(ShopRun, "finish", spied_finish)
        for rule in (_ConstrictionUpdate, _DiscreteUpdate):
            monkeypatch.setattr(rule, "update", self._spied_update(rule.update))

    def _spied_update(self, update: Callable[..., Any]) -> Callable[..., Any]:
        def spied(
            rule: Any,
            particle: int,
            position: NDArray[Any],
            own: NDArray[Any],
            leader: NDArray[Any],
            rng: np.random.Generator,
        ) -> Any:
            self.calls.append((particle, np.array(own), np.array(leader)))
            return update(rule, particle, position, own, leader, rng)

        return spied

    def runs(self) -> list[list[Step]]:
        """Return the steps of each run, the runs in the order of their first evaluation."""
        return list(self.steps.values())


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> Spy:
    return Spy(monkeypatch)


# --------------------------------------------------------- an independent model


class Oracle:
    """The swarm's memory stated again, by brute force, from the values it evaluates.

    The personal best of a particle holds the best value it has had, and the stamp of
    the evaluation that set it, a clock of its own. The leader of a particle is
    found by taking the least value in its neighbourhood, and among the particles
    that hold it the greatest stamp.
    """

    def __init__(self, topology: Topology, values: list[float]) -> None:
        self.topology = topology
        self.size = len(values)
        self.values = list(values)
        self.stamps = list(range(1, self.size + 1))
        self.clock = self.size
        self.ties_kept = 0
        self.leaders_by_stamp = 0
        self.local_leaders = 0

    def leader(self, particle: int) -> int:
        if self.topology == "global":
            neighbours = list(range(self.size))
        else:
            neighbours = sorted({(particle - 1) % self.size, particle, (particle + 1) % self.size})
        best = min(self.values[other] for other in neighbours)
        held = [other for other in neighbours if self.values[other] == best]
        if len(held) > 1:
            self.leaders_by_stamp += 1
        leader = max(held, key=lambda other: self.stamps[other])
        if best > min(self.values):
            self.local_leaders += 1
        return leader

    def record(self, particle: int, value: float) -> None:
        if value <= self.values[particle]:
            if value == self.values[particle]:
                self.ties_kept += 1
            self.values[particle] = value
            self.clock += 1
            self.stamps[particle] = self.clock


def _replay_sequences(
    shop: JobShop, steps: list[Step], seed: int, parameters: dict[str, Any]
) -> Oracle:
    """Replay a run on job sequences from its values, and check each position it evaluated."""
    size = parameters["swarm_size"]
    rng = np.random.default_rng(derive_seed(seed, OPERATOR_STREAM))
    representation = SequenceRepresentation(shop, parameters["sequence_move"])
    positions = [representation.initial(rng) for _ in range(size)]
    bests = list(positions)
    oracle = Oracle(parameters["topology"], [step.value for step in steps[:size]])
    for number, step in enumerate(steps):
        if number < size:
            assert np.array_equal(step.genotype, positions[number])
            continue
        particle = (number - size) % size
        leader = oracle.leader(particle)
        coins = rng.random(3).tolist()
        moved = positions[particle]
        if coins[0] < parameters["move_probability"]:
            moved = representation.move(moved, rng)
        if coins[1] < parameters["personal_probability"]:
            moved = representation.recombine(bests[particle], moved, rng, parameters["inheritance"])
        if coins[2] < parameters["leader_probability"]:
            moved = representation.recombine(bests[leader], moved, rng, parameters["inheritance"])
        assert np.array_equal(step.genotype, moved), f"evaluation {number}"
        positions[particle] = moved
        if step.value <= oracle.values[particle]:
            bests[particle] = moved
        oracle.record(particle, step.value)
    return oracle


def _replay_keys(
    shop: JobShop, steps: list[Step], seed: int, parameters: dict[str, Any]
) -> tuple[Oracle, int]:
    """Replay a run on keys, key by key in plain floats, and count the keys that were folded."""
    size, keys = parameters["swarm_size"], shop.size
    rng = np.random.default_rng(derive_seed(seed, OPERATOR_STREAM))
    cognitive, social = parameters["cognitive"], parameters["social"]
    phi = cognitive + social
    chi = 2.0 / (phi - 2.0 + math.sqrt(phi * phi - 4.0 * phi))
    positions = [rng.random(keys).tolist() for _ in range(size)]
    bests = [list(position) for position in positions]
    velocities = [[0.0] * keys for _ in range(size)]
    oracle = Oracle(parameters["topology"], [step.value for step in steps[:size]])
    folded = 0
    for number, step in enumerate(steps):
        if number < size:
            assert np.array_equal(step.genotype, positions[number])
            continue
        particle = (number - size) % size
        leader = oracle.leader(particle)
        draws = rng.random((2, keys)).tolist()
        moved = []
        for key in range(keys):
            x, own, lead = positions[particle][key], bests[particle][key], bests[leader][key]
            velocity = chi * (
                velocities[particle][key]
                + cognitive * draws[0][key] * (own - x)
                + social * draws[1][key] * (lead - x)
            )
            target = x + velocity
            if 0.0 <= target <= 1.0:
                new = target
            else:
                folded += 1
                remainder = abs(target) % 2.0
                new = remainder if remainder <= 1.0 else 2.0 - remainder
                if math.floor(target) % 2 == 1:
                    velocity = -velocity
            velocities[particle][key] = velocity
            moved.append(new)
        assert np.allclose(step.genotype, moved, rtol=0.0, atol=1e-12), f"evaluation {number}"
        positions[particle] = moved
        if step.value <= oracle.values[particle]:
            bests[particle] = moved
        oracle.record(particle, step.value)
    return oracle, folded


# ------------------------------------------------------------- common start


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("builder", PAIRED_BUILDERS)
@pytest.mark.parametrize("make", PROBLEMS)
def test_both_swarm_arms_start_from_the_same_swarm(
    make: Callable[[], Problem], builder: BuilderName, topology: Topology, spy: Spy
) -> None:
    problem = make()
    shop = JobShop.from_problem(problem)
    size = 8
    keys, sequence = (
        solver.solve(problem, StopCriteria(max_evaluations=40), seed=7)
        for solver in _arms(builder, swarm_size=size, topology=topology)
    )
    keys_run, sequence_run = spy.runs()
    # The first schedule, which is the first particle's, and the whole swarm are equal.
    assert keys_run[0].digest == sequence_run[0].digest
    assert keys.extras["first_schedule_digest"] == sequence.extras["first_schedule_digest"]
    assert keys.extras["first_schedule_digest"] == keys_run[0].digest.hex()
    assert [s.digest for s in keys_run[:size]] == [s.digest for s in sequence_run[:size]]
    # The swarm is the first draws of the operator stream, in either representation.
    operators = np.random.default_rng(derive_seed(7, OPERATOR_STREAM))
    sorter = SequenceRepresentation(shop)
    for particle in range(size):
        draw = operators.random(shop.size)
        assert np.array_equal(keys_run[particle].genotype, draw)
        assert np.array_equal(sequence_run[particle].genotype, sorter.from_keys(draw))
        assert keys_run[particle].parent is sequence_run[particle].parent is None
    assert keys_run[size].parent is not None
    assert sequence_run[size].parent is not None


@pytest.mark.parametrize("evaluations", [5, 8, 13])
@pytest.mark.parametrize("make", PROBLEMS)
def test_the_swarm_is_drawn_first_and_the_control_stream_is_never_drawn(
    make: Callable[[], Problem], evaluations: int, spy: Spy
) -> None:
    problem = make()
    shop = JobShop.from_problem(problem)
    size, seed = 8, 5
    # The refused evaluation still follows an update: the draws of that one are made too.
    updates = max(0, evaluations - size + 1)
    quiet: dict[str, Any] = {
        "move_probability": 0.0,
        "personal_probability": 0.0,
        "leader_probability": 0.0,
    }
    for solver in (
        ShopParticleSwarm("keys", "semi-active", swarm_size=size),
        ShopParticleSwarm("sequence", "semi-active", swarm_size=size, **quiet),
    ):
        solver.solve(problem, StopCriteria(max_evaluations=evaluations), seed=seed)
        run = spy.finished[-1]
        expected = np.random.default_rng(derive_seed(seed, OPERATOR_STREAM))
        for _ in range(size):
            expected.random(shop.size)
        for _ in range(updates):
            if solver.representation == "keys":
                expected.random((2, shop.size))
            else:
                expected.random(3)
        assert run.operators.bit_generator.state == expected.bit_generator.state
        fresh = np.random.default_rng(derive_seed(seed, CONTROL_STREAM))
        assert run.control.bit_generator.state == fresh.bit_generator.state


# ------------------------------------------------------------------ memory


def _evaluated(label: int, value: float) -> Evaluated[int]:
    return Evaluated(label, np.empty(0, dtype=np.intp), Score(0.0, value), cast("Any", None))


def test_the_global_leader_is_the_best_personal_best_and_the_most_recent_among_equals() -> None:
    memory = _Memory[int]("global")
    for label, value in enumerate([5.0, 3.0, 3.0, 7.0, 3.0]):
        memory.begin(_evaluated(label, value))
    # Three particles hold 3, and the last of them was set last.
    assert [memory.leader_of(particle) for particle in range(5)] == [4] * 5
    # A new position that is no worse than the best replaces it, and becomes the leader.
    memory.record(1, _evaluated(10, 3.0))
    assert memory.best[1].genotype == 10
    assert memory.leader_of(0) == 1
    # A new position that is worse than the particle's best is its position and no more.
    memory.record(4, _evaluated(11, 4.0))
    assert (memory.current[4].genotype, memory.best[4].genotype) == (11, 4)
    assert memory.leader_of(2) == 1
    # A particle that improves on its own best without reaching the swarm's best does not lead.
    memory.record(0, _evaluated(12, 4.0))
    assert memory.best[0].genotype == 12
    assert memory.leader_of(3) == 1
    # A better one leads, and ties with itself keep it leading.
    memory.record(3, _evaluated(13, 2.0))
    assert memory.leader_of(0) == 3
    memory.record(3, _evaluated(14, 2.0))
    assert (memory.best[3].genotype, memory.leader_of(0)) == (14, 3)


def test_the_ring_leader_is_the_best_of_a_particle_and_its_two_neighbours() -> None:
    memory = _Memory[int]("ring")
    for label, value in enumerate([5.0, 3.0, 3.0, 7.0, 3.0]):
        memory.begin(_evaluated(label, value))
    # Neighbourhoods: {4, 0, 1}, {0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 0}; ties go to the
    # personal best set last, and the first and the last particle are neighbours.
    assert [memory.leader_of(particle) for particle in range(5)] == [4, 2, 2, 4, 4]
    memory.record(1, _evaluated(10, 3.0))
    assert [memory.leader_of(particle) for particle in range(5)] == [1, 1, 1, 4, 4]
    memory.record(2, _evaluated(11, 9.0))
    assert memory.best[2].genotype == 2
    assert [memory.leader_of(particle) for particle in range(5)] == [1, 1, 1, 4, 4]
    memory.record(0, _evaluated(12, 1.0))
    assert [memory.leader_of(particle) for particle in range(5)] == [0, 0, 1, 4, 0]


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("size", [1, 2, 3])
def test_a_ring_of_fewer_than_four_particles_leads_by_the_same_rule(
    topology: Topology, size: int
) -> None:
    memory = _Memory[int](topology)
    for label in range(size):
        memory.begin(_evaluated(label, 4.0))
    # Every neighbourhood holds the whole swarm: the most recent of equals leads.
    assert [memory.leader_of(particle) for particle in range(size)] == [size - 1] * size
    memory.record(0, _evaluated(7, 2.0))
    assert [memory.leader_of(particle) for particle in range(size)] == [0] * size


# ----------------------------------------------------------- the personal and leader bests


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("make", SMALL)
def test_personal_and_leader_bests_never_worsen(
    make: Callable[[], Problem], representation: RepresentationName, topology: Topology, spy: Spy
) -> None:
    problem = make()
    size, evaluations = 6, 6 + 6 * 12
    ShopParticleSwarm(representation, "semi-active", swarm_size=size, topology=topology).solve(
        problem, StopCriteria(max_evaluations=evaluations), seed=4
    )
    (steps,) = spy.runs()
    assert len(steps) == evaluations
    value_of = {step.genotype.tobytes(): step.value for step in steps}
    # The update of evaluation `size + k` is call `k`, and it reads what the memory held
    # after the evaluations before it.
    assert len(spy.calls) == evaluations - size + 1
    previous: dict[int, float] = {}
    leaders: list[float] = []
    for call, (particle, own, leader) in enumerate(spy.calls):
        before = steps[: size + call]
        own_values = [step.value for step in before[particle::size]]
        assert value_of[own.tobytes()] == min(own_values)
        assert value_of[own.tobytes()] <= previous.get(particle, math.inf)
        previous[particle] = value_of[own.tobytes()]
        # The leader's best is no worse than the particle's own, and is the best in its
        # neighbourhood: the whole swarm, or the particle and its two neighbours.
        if topology == "global":
            neighbours = list(range(size))
        else:
            neighbours = sorted({(particle - 1) % size, particle, (particle + 1) % size})
        held = [min(step.value for step in before[other::size]) for other in neighbours]
        assert value_of[leader.tobytes()] == min(held)
        assert value_of[leader.tobytes()] <= value_of[own.tobytes()]
        leaders.append(value_of[leader.tobytes()])
    if topology == "global":
        # The swarm's best never worsens, and is the best the run has evaluated.
        assert leaders == sorted(leaders, reverse=True)
        assert leaders[-1] == min(step.value for step in steps)


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("make", SMALL)
@pytest.mark.parametrize("seed", [1, 2])
def test_the_job_sequence_swarm_replays_from_an_independent_model(
    make: Callable[[], Problem], seed: int, topology: Topology, spy: Spy
) -> None:
    problem = make()
    shop = JobShop.from_problem(problem)
    parameters: dict[str, Any] = {
        "swarm_size": 10,
        "topology": topology,
        "move_probability": 0.6,
        "personal_probability": 0.7,
        "leader_probability": 0.8,
        "inheritance": 0.6,
        "sequence_move": "swap",
    }
    ShopParticleSwarm("sequence", "semi-active", **parameters).solve(
        problem, StopCriteria(max_evaluations=10 + 10 * 12), seed=seed
    )
    (steps,) = spy.runs()
    oracle = _replay_sequences(shop, steps, seed, parameters)
    # The model was tested: ties decided personal bests and leaders in these runs.
    assert oracle.ties_kept > 0
    assert oracle.leaders_by_stamp > 0
    assert (oracle.local_leaders > 0) is (topology == "ring")


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("make", SMALL)
@pytest.mark.parametrize("seed", [1, 2])
def test_the_key_swarm_replays_from_an_independent_model(
    make: Callable[[], Problem], seed: int, topology: Topology, spy: Spy
) -> None:
    problem = make()
    shop = JobShop.from_problem(problem)
    parameters: dict[str, Any] = {
        "swarm_size": 10,
        "topology": topology,
        "cognitive": 2.1,
        "social": 2.05,
    }
    ShopParticleSwarm("keys", "semi-active", **parameters).solve(
        problem, StopCriteria(max_evaluations=10 + 10 * 12), seed=seed
    )
    (steps,) = spy.runs()
    oracle, folded = _replay_keys(shop, steps, seed, parameters)
    # The model was tested: keys left [0, 1] and were folded, and ties decided leaders.
    assert folded > 10
    assert oracle.ties_kept > 0
    assert oracle.leaders_by_stamp > 0
    assert (oracle.local_leaders > 0) is (topology == "ring")


# ---------------------------------------------------------- the job sequence update


SEED = 9
SWARM = 6


def _first_updates(
    parameters: dict[str, Any], spy: Spy, updates: int = 1
) -> tuple[list[Step], JobShop]:
    """Run the job-sequence swarm on the 3x3 shop for its swarm and ``updates`` updates."""
    problem = small_job_shop()
    ShopParticleSwarm("sequence", "semi-active", swarm_size=SWARM, **parameters).solve(
        problem, StopCriteria(max_evaluations=SWARM + updates), seed=SEED
    )
    return spy.runs()[-1], JobShop.from_problem(problem)


def _swarm_draws(
    shop: JobShop, move: SequenceMove = "insert"
) -> tuple[SequenceRepresentation, list[Any], np.random.Generator]:
    """Draw the swarm of the run above from a generator of our own, and leave it ready."""
    rng = np.random.default_rng(derive_seed(SEED, OPERATOR_STREAM))
    representation = SequenceRepresentation(shop, move)
    return representation, [representation.initial(rng) for _ in range(SWARM)], rng


def _most_recent_best(steps: list[Step]) -> int:
    """Return the particle that leads the swarm after its evaluations: the last of the best."""
    values = [step.value for step in steps[:SWARM]]
    return max(index for index, value in enumerate(values) if value == min(values))


def test_an_update_that_applies_no_operation_leaves_the_position_where_it_was(spy: Spy) -> None:
    zero = {"move_probability": 0.0, "personal_probability": 0.0, "leader_probability": 0.0}
    steps, _ = _first_updates(zero, spy, updates=2 * SWARM)
    # The swarm and two updates of each particle, which repeat its position and its schedule.
    for particle in range(SWARM):
        positions = [steps[number].genotype for number in range(particle, len(steps), SWARM)]
        assert len(positions) == 3
        assert all(np.array_equal(positions[0], other) for other in positions)
    assert steps[SWARM].digest == steps[0].digest
    assert steps[SWARM].parent == steps[0].digest


@pytest.mark.parametrize("kind", ["insert", "swap"])
def test_a_move_alone_changes_the_position_by_one_move_of_its_kind(kind: str, spy: Spy) -> None:
    parameters = {
        "move_probability": 1.0,
        "personal_probability": 0.0,
        "leader_probability": 0.0,
        "sequence_move": kind,
    }
    steps, shop = _first_updates(parameters, spy)
    representation, initial, rng = _swarm_draws(shop, cast("SequenceMove", kind))
    # The three numbers come first, and then the move draws its own.
    rng.random(3)
    assert np.array_equal(steps[SWARM].genotype, representation.move(initial[0], rng))
    assert not np.array_equal(steps[SWARM].genotype, initial[0])


def test_the_crossover_with_the_leaders_best_takes_its_genes_from_that_best(spy: Spy) -> None:
    # With an inheritance of 1 the child is its first parent, the leader's best, and with 0 it
    # is the particle: the best is the first parent.
    base = {"move_probability": 0.0, "personal_probability": 0.0, "leader_probability": 1.0}
    pulled, shop = _first_updates({**base, "inheritance": 1.0}, spy, updates=SWARM)
    kept, _ = _first_updates({**base, "inheritance": 0.0}, spy, updates=SWARM)
    _, initial, _ = _swarm_draws(shop)
    best = _most_recent_best(pulled)
    assert _most_recent_best(kept) == best
    for particle in range(SWARM):
        # Each takes the best of the swarm: the first the best of the swarm, and the next the
        # one before it, whose new best ties with it and leads, being the most recent.
        assert np.array_equal(pulled[SWARM + particle].genotype, initial[best])
        assert np.array_equal(kept[SWARM + particle].genotype, initial[particle])


def test_the_crossover_with_the_personal_best_takes_its_genes_from_that_best(spy: Spy) -> None:
    # After the move the particle is not at its best: an inheritance of 1 brings it back.
    base = {"move_probability": 1.0, "personal_probability": 1.0, "leader_probability": 0.0}
    pulled, shop = _first_updates({**base, "inheritance": 1.0}, spy, updates=SWARM)
    kept, _ = _first_updates({**base, "inheritance": 0.0}, spy, updates=SWARM)
    _, initial, _ = _swarm_draws(shop)
    for particle in range(SWARM):
        assert np.array_equal(pulled[SWARM + particle].genotype, initial[particle])
        assert not np.array_equal(kept[SWARM + particle].genotype, initial[particle])


def test_the_three_operations_come_in_the_order_of_the_chapters_update(spy: Spy) -> None:
    # One update with all three: a move, then the personal best, then the leader's best.
    parameters = {
        "move_probability": 1.0,
        "personal_probability": 1.0,
        "leader_probability": 1.0,
        "inheritance": 0.7,
        "sequence_move": "insert",
    }
    steps, shop = _first_updates(parameters, spy)
    representation, initial, rng = _swarm_draws(shop)
    best = _most_recent_best(steps)
    rng.random(3)
    moved = representation.move(initial[0], rng)
    crossed = representation.recombine(initial[0], moved, rng, 0.7)
    crossed = representation.recombine(initial[best], crossed, rng, 0.7)
    assert np.array_equal(steps[SWARM].genotype, crossed)


# ------------------------------------------------------------------ the key update


def _constriction(size: int = 2, keys: int = 6, **parameters: float) -> _ConstrictionUpdate:
    return _ConstrictionUpdate(
        size, keys, parameters.get("cognitive", 2.05), parameters.get("social", 2.05)
    )


def test_the_constriction_coefficient_follows_the_total_acceleration() -> None:
    assert constriction_coefficient(4.1) == pytest.approx(0.7298437881283576, rel=1e-14)
    assert constriction_coefficient(5.0) == pytest.approx((3.0 - math.sqrt(5.0)) / 2.0, rel=1e-14)
    values = [constriction_coefficient(phi) for phi in (4.0001, 4.1, 4.5, 5.0, 8.0, 100.0)]
    assert all(0.0 < value < 1.0 for value in values)
    assert values == sorted(values, reverse=True)
    assert constriction_coefficient(4.0 + 1e-12) == pytest.approx(1.0, abs=1e-5)
    # The coefficient solves the equation of the constriction: chi = 2 / |2 - phi - sqrt(...)|.
    for phi in (4.1, 6.0):
        assert constriction_coefficient(phi) == pytest.approx(
            2.0 / abs(2.0 - phi - math.sqrt(phi * phi - 4.0 * phi)), rel=1e-14
        )


@pytest.mark.parametrize("phi", [4.0, 3.9, 0.0, -1.0, math.nan, math.inf])
def test_the_constriction_coefficient_refuses_a_total_acceleration_of_4_or_less(
    phi: float,
) -> None:
    with pytest.raises(ValueError, match="above 4"):
        constriction_coefficient(phi)


def test_the_default_accelerations_give_the_published_coefficient() -> None:
    update = _constriction()
    assert update.constriction == constriction_coefficient(4.1)
    assert update.constriction == pytest.approx(0.72984, abs=1e-5)
    assert (update.cognitive, update.social) == (2.05, 2.05)


def test_a_velocity_update_follows_the_constriction_form_with_two_draws() -> None:
    update = _constriction(keys=6, cognitive=2.2, social=2.9)
    position = np.array([0.45, 0.5, 0.55, 0.5, 0.48, 0.52])
    own = np.array([0.4, 0.55, 0.5, 0.5, 0.45, 0.6])
    leader = np.array([0.6, 0.45, 0.4, 0.55, 0.5, 0.5])
    update.velocity[1] = [0.01, -0.02, 0.03, 0.0, 0.02, -0.01]
    rng = np.random.default_rng(5)
    moved = update.update(1, position, own, leader, rng)
    # The draws are two arrays of one number per key: the cognitive one first.
    draws = np.random.default_rng(5).random((2, 6))
    chi = constriction_coefficient(5.1)
    velocity = chi * (
        np.array([0.01, -0.02, 0.03, 0.0, 0.02, -0.01])
        + 2.2 * draws[0] * (own - position)
        + 2.9 * draws[1] * (leader - position)
    )
    assert np.allclose(update.velocity[1], velocity, rtol=1e-14, atol=0.0)
    assert np.allclose(moved, position + velocity, rtol=1e-14, atol=0.0)
    after = np.random.default_rng(5)
    after.random((2, 6))
    assert rng.bit_generator.state == after.bit_generator.state
    # The other particle's velocity is untouched, and nothing was folded.
    assert not update.velocity[0].any()
    assert update.measures() == {"reflected_share": 0.0}


FOLDS = [
    # (position, step, folded position, velocity turns)
    pytest.param(0.9, 0.4, 0.7, True, id="above-1-one-fold"),
    pytest.param(0.1, -0.3, 0.2, True, id="below-0-one-fold"),
    pytest.param(0.95, 1.3, 0.25, False, id="two-folds-from-above"),
    pytest.param(0.5, -1.8, 0.7, False, id="two-folds-from-below"),
    pytest.param(0.5, 2.4, 0.9, False, id="two-folds-from-the-middle"),
    pytest.param(0.5, 2.8, 0.7, True, id="three-folds"),
    pytest.param(0.5, 0.2, 0.7, False, id="inside"),
    pytest.param(0.5, 0.5, 1.0, False, id="on-the-upper-bound"),
    pytest.param(0.5, -0.5, 0.0, False, id="on-the-lower-bound"),
]


@pytest.mark.parametrize(("start", "step", "folded", "turns"), FOLDS)
def test_a_key_outside_the_range_is_reflected_and_its_velocity_turns_with_each_fold(
    start: float, step: float, folded: float, turns: bool
) -> None:
    update = _constriction(size=1, keys=1)
    # A coefficient of 1 and no pull make the step the velocity itself.
    update.constriction = 1.0
    position = np.array([start])
    update.velocity[0] = [step]
    moved = update.update(0, position, position, position, np.random.default_rng(1))
    assert moved == pytest.approx([folded], abs=1e-12)
    assert 0.0 <= moved[0] <= 1.0
    assert update.velocity[0, 0] == pytest.approx(-step if turns else step, abs=1e-15)
    # The velocity keeps its size: a key leaves the bound as fast as it reached it.
    assert abs(update.velocity[0, 0]) == pytest.approx(abs(step), abs=1e-15)
    outside = not 0.0 <= start + step <= 1.0
    assert update.reflected == int(outside)
    assert update.measures() == {"reflected_share": float(outside)}


def test_the_keys_of_one_update_are_folded_each_by_its_own_travel() -> None:
    update = _constriction(size=1, keys=5)
    update.constriction = 1.0
    position = np.array([0.9, 0.1, 0.5, 0.95, 0.5])
    steps = [0.4, -0.3, 0.2, 1.3, -1.8]
    update.velocity[0] = steps
    moved = update.update(0, position, position, position, np.random.default_rng(1))
    assert moved == pytest.approx([0.7, 0.2, 0.7, 0.25, 0.7], abs=1e-12)
    assert update.velocity[0] == pytest.approx([-0.4, 0.3, 0.2, 1.3, -1.8], abs=1e-15)
    assert update.measures() == {"reflected_share": 4 / 5}
    # The position is left to the representation's own fold: reflect agrees with it.
    assert np.allclose(moved, reflect(position + np.array(steps)), rtol=0.0, atol=1e-15)


def test_the_share_of_reflected_keys_counts_every_key_of_every_update() -> None:
    update = _constriction(size=1, keys=4)
    update.constriction = 1.0
    assert update.measures() == {}
    position = np.array([0.9, 0.5, 0.5, 0.5])
    update.velocity[0] = [0.4, 0.0, 0.0, 0.0]
    update.update(0, position, position, position, np.random.default_rng(1))
    update.velocity[0] = [0.0, 0.0, 0.0, 0.0]
    update.update(0, position, position, position, np.random.default_rng(1))
    assert update.measures() == {"reflected_share": 1 / 8}


def test_the_key_swarm_evaluates_every_particle_it_folds_and_keeps_its_keys_in_range(
    spy: Spy,
) -> None:
    evaluations, size = 6 + 6 * 20, 6
    result = ShopParticleSwarm(
        "keys", "semi-active", swarm_size=size, cognitive=2.1, social=2.05
    ).solve(small_job_shop(), StopCriteria(max_evaluations=evaluations), seed=2)
    (steps,) = spy.runs()
    share = result.extras["reflected_share"]
    assert isinstance(share, float)
    assert share > 0.0
    # Every update was evaluated, whether or not its keys left the range: the evaluations are
    # the swarm and then one for each update but the one the budget refused.
    assert len(steps) == result.evaluations == evaluations
    assert len(spy.calls) == evaluations - size + 1
    for step in steps:
        assert (step.genotype >= 0.0).all()
        assert (step.genotype <= 1.0).all()


def test_a_single_key_particle_that_leads_itself_does_not_move() -> None:
    # Without velocity, a particle that is its own best and leader has nothing to follow.
    result = ShopParticleSwarm("keys", "semi-active", swarm_size=1).solve(
        small_job_shop(), StopCriteria(max_evaluations=25), seed=3
    )
    assert result.evaluations == 25
    assert result.extras["revisit_share"] == 24 / 25
    assert result.extras["unchanged_move_share"] == 1.0
    assert result.extras["reflected_share"] == 0.0


# ------------------------------------------------------------------ budget


@pytest.mark.parametrize("evaluations", [1, 2, 7, 8, 9, 16, 17, 100])
def test_every_swarm_run_ends_at_its_evaluation_limit(evaluations: int, spy: Spy) -> None:
    size = 8
    for builder in PAIRED_BUILDERS:
        for topology in TOPOLOGIES:
            for solver in _arms(builder, swarm_size=size, topology=topology):
                result = solver.solve(
                    small_job_shop(), StopCriteria(max_evaluations=evaluations), seed=9
                )
                assert result.evaluations == evaluations
                assert result.stop_reason is StopReason.EVALUATION_LIMIT
                assert result.extras["repairs"] == 0
                # The swarm is one iteration of evaluations ahead of the iterations it starts.
                assert result.iterations == (
                    0 if evaluations < size else 1 + (evaluations - size) // size
                )
                assert len(spy.runs()[-1]) == evaluations


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_the_swarm_runs_under_an_iteration_or_a_time_limit_alone(
    representation: RepresentationName,
) -> None:
    size = 5
    solver = ShopParticleSwarm(representation, "active", swarm_size=size)
    by_iterations = solver.solve(small_job_shop(), StopCriteria(max_iterations=4), seed=2)
    assert by_iterations.stop_reason is StopReason.ITERATION_LIMIT
    assert by_iterations.iterations == 4
    assert by_iterations.evaluations == size + 4 * size
    by_time = solver.solve(small_job_shop(), StopCriteria(time_limit=0.05), seed=2)
    assert by_time.stop_reason is StopReason.TIME_LIMIT
    assert by_time.evaluations >= 1


# ----------------------------------------------------------------- results


@pytest.mark.parametrize("sequence_move", ["insert", "swap", "mixed"])
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("builder", BUILDERS)
def test_the_swarm_passes_the_contract(
    builder: BuilderName, topology: Topology, sequence_move: SequenceMove
) -> None:
    size = 8
    for representation in REPRESENTATIONS:
        for problem in (small_job_shop(), chain_problem()):
            result = check_contract(
                ShopParticleSwarm,
                problem,
                representation=representation,
                builder=builder,
                swarm_size=size,
                topology=topology,
                sequence_move=sequence_move,
            )
            assert result.extras["repairs"] == 0
            # The observer sees the swarm and then every update as a move of its particle.
            assert result.extras["observed_evaluations"] == 40
            assert result.extras["observed_moves"] == 40 - size
            assert ("reflected_share" in result.extras) is (representation == "keys")


def test_the_swarm_with_the_default_parameters_passes_the_contract() -> None:
    for representation in REPRESENTATIONS:
        check_contract(ShopParticleSwarm, small_job_shop(), representation=representation)


def test_the_swarm_replays_from_its_seed() -> None:
    problem = _ta01()
    stop = StopCriteria(max_evaluations=150)
    for solver in _arms("semi-active", swarm_size=10, topology="ring"):
        first, second, other = (solver.solve(problem, stop, seed=s) for s in (13, 13, 14))
        assert _serialised(problem, first) == _serialised(problem, second)
        assert [(p.evaluations, p.iteration, p.value) for p in first.trace] == [
            (p.evaluations, p.iteration, p.value) for p in second.trace
        ]
        assert untimed(first.extras) == untimed(second.extras)
        assert first.extras["first_schedule_digest"] != other.extras["first_schedule_digest"]


def test_the_swarm_beats_random_sampling_at_equal_evaluations_on_ta01() -> None:
    problem = _ta01()
    stop = StopCriteria(max_evaluations=1000)
    for representation in REPRESENTATIONS:
        sampled = ShopRandomSampling(representation, "semi-active").solve(problem, stop, seed=7)
        swarm = ShopParticleSwarm(representation, "semi-active").solve(problem, stop, seed=7)
        assert swarm.value is not None
        assert sampled.value is not None
        assert swarm.value < sampled.value


@pytest.mark.parametrize("operations", [1, 2, 3])
def test_the_swarm_evaluates_the_single_sequence_of_one_job_and_stops(
    operations: int, spy: Spy
) -> None:
    problem = one_job(operations)
    shop = JobShop.from_problem(problem)
    size = 5
    for builder in BUILDERS:
        for solver in _arms(builder, swarm_size=size):
            result = solver.solve(problem, StopCriteria(max_evaluations=30), seed=3)
            assert result.evaluations == 1
            assert result.stop_reason is StopReason.COMPLETED
            assert result.value == sum(step + 2 for step in range(operations))
            assert result.extras["repairs"] == 0
            run = spy.finished[-1]
            # The whole swarm was drawn, and no update drew a number or made a move.
            expected = np.random.default_rng(derive_seed(3, OPERATOR_STREAM))
            for _ in range(size):
                expected.random(shop.size)
            assert run.operators.bit_generator.state == expected.bit_generator.state
    assert len(spy.finished) == 2 * len(BUILDERS)
    assert spy.calls == []


# ------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"representation": "permutation"}, "unknown representation"),
        ({"builder": "giffler"}, "unknown builder"),
        ({"swarm_size": 0}, "swarm_size"),
        ({"swarm_size": -3}, "swarm_size"),
        ({"swarm_size": True}, "swarm_size"),
        ({"swarm_size": 5.0}, "swarm_size"),
        ({"topology": "star"}, "unknown topology"),
        ({"cognitive": -1.0}, "cognitive"),
        ({"cognitive": math.nan}, "cognitive"),
        ({"cognitive": True}, "cognitive"),
        ({"social": -0.5}, "social"),
        ({"social": math.inf}, "social"),
        ({"social": "2.0"}, "social"),
        ({"cognitive": 2.0, "social": 2.0}, "exceed 4"),
        ({"cognitive": 0.0, "social": 4.0}, "exceed 4"),
        ({"cognitive": 1e308, "social": 1e308}, "exceed 4"),
        ({"move_probability": -0.1}, "move_probability"),
        ({"move_probability": 1.1}, "move_probability"),
        ({"personal_probability": math.nan}, "personal_probability"),
        ({"leader_probability": 2}, "leader_probability"),
        ({"leader_probability": False}, "leader_probability"),
        ({"inheritance": -0.2}, "inheritance"),
        ({"inheritance": "0.5"}, "inheritance"),
        ({"sequence_move": "reverse"}, "unknown sequence move"),
    ],
)
def test_the_swarm_refuses_bad_parameters(parameters: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ShopParticleSwarm(**parameters)


def test_the_parameters_of_both_arms_are_checked_whichever_arm_runs() -> None:
    with pytest.raises(ValueError, match="move_probability"):
        ShopParticleSwarm("keys", move_probability=2.0)
    with pytest.raises(ValueError, match="exceed 4"):
        ShopParticleSwarm("sequence", cognitive=1.0, social=1.0)


def test_the_probabilities_zero_and_one_and_the_bounds_of_the_range_are_accepted() -> None:
    ShopParticleSwarm(
        move_probability=0.0, personal_probability=1.0, leader_probability=0.0, inheritance=1.0
    )
    ShopParticleSwarm(inheritance=0.0, swarm_size=1, cognitive=4.5, social=0.0)
    assert ShopParticleSwarm().info.id == "shop-particle-swarm"
    assert ShopParticleSwarm().info.references[:2] == ("bratton2007defining", "pan2007discrete")


def test_the_module_names_a_global_and_a_ring_topology() -> None:
    assert TOPOLOGIES == ("global", "ring")
