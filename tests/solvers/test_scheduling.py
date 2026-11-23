import random

import pytest

from dispatchatlas.core import check_schedule, evaluate
from dispatchatlas.core.model import Dependency, Mode, Problem, Resource, Task
from dispatchatlas.solvers.evaluation import Evaluator
from dispatchatlas.solvers.operators import random_modes, random_order
from dispatchatlas.solvers.scheduling import Decoder, precedence_feasible
from tests.helpers import chain_problem, random_problem

FEATURE_SETS = [
    {},
    {"setups": True},
    {"network": True},
    {"setups": True, "network": True, "deadlines": True},
    {"pools": 2, "max_modes": 1},
]


@pytest.mark.parametrize("options", FEATURE_SETS)
@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("insertion", [True, False])
def test_decoded_schedules_are_feasible(options, seed, insertion):
    problem = random_problem(seed, **options)
    decoder = Decoder(problem, insertion=insertion)
    rng = random.Random(seed)
    for _ in range(5):
        order = random_order(problem, rng)
        modes = random_modes(problem, rng) if rng.random() < 0.5 else None
        schedule = decoder.schedule(order, modes)
        violations = [v for v in check_schedule(problem, schedule) if v.code != "deadline"]
        assert violations == []


@pytest.mark.parametrize("seed", range(8))
def test_fast_objectives_match_the_registry(seed):
    problem = random_problem(seed, network=True).with_objectives(
        "makespan",
        "total_completion_time",
        "total_flow_time",
        "mean_response_time",
        "total_tardiness",
        "max_lateness",
        "tardy_tasks",
        "energy",
        "cost",
    )
    evaluator = Evaluator(problem)
    decoded = evaluator.decoder.decode(random_order(problem, random.Random(seed)))
    fast = evaluator.values(decoded)
    reference = evaluate(problem, decoded.schedule(problem))
    assert fast == pytest.approx(reference)


def test_precedence_repair_keeps_priorities_where_possible():
    problem = chain_problem()
    assert precedence_feasible(problem, [1, 2, 0]) == [2, 0, 1]


def test_decode_counts_repairs():
    problem = chain_problem()
    decoder = Decoder(problem)
    assert decoder.repairs == 0
    decoder.decode([0, 1, 2])
    assert decoder.repairs == 1
    # An order that needs repair counts once, as an order that does not.
    assert decoder.decode([1, 2, 0]).sequence == (2, 0, 1)
    assert decoder.repairs == 2
    decoder.schedule([2, 0, 1], [0, 0, 0])
    assert decoder.repairs == 3
    # Tasks a builder places are never repaired.
    builder = decoder.builder()
    for task in (0, 1, 2):
        builder.place(task, 0)
    assert builder.decoded().sequence == (0, 1, 2)
    assert decoder.repairs == 3
    # Each decoder counts its own decodes, and the evaluator's decoder counts evaluations.
    assert Decoder(problem).repairs == 0
    evaluator = Evaluator(problem)
    for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2], [0, 2, 1]):
        evaluator.evaluate(order)
    assert evaluator.decoder.repairs == evaluator.evaluations == 4


def test_insertion_fills_an_idle_gap():
    problem = Problem(
        "gap",
        tasks=(
            Task.on("a", "m1", 2),
            Task.on("b", "m1", 3, release=5),
            Task.on("c", "m1", 2),
        ),
        resources=(Resource("m1"),),
    )
    decoded = Decoder(problem).decode([0, 1, 2])
    assert decoded.starts == (0.0, 5.0, 2.0)
    appended = Decoder(problem, insertion=False).decode([0, 1, 2])
    assert appended.starts == (0.0, 5.0, 8.0)


def test_mode_choice_prefers_earliest_finish():
    problem = Problem(
        "modes",
        tasks=(
            Task.on("a", "m1", 5),
            Task("b", (Mode(4, {"m1": 1}), Mode(6, {"m2": 1}))),
        ),
        resources=(Resource("m1"), Resource("m2")),
    )
    decoded = Decoder(problem).decode([0, 1])
    assert decoded.modes == (0, 1)
    assert decoded.ends[1] == 6.0


def test_cumulative_profile_packs_tasks():
    problem = Problem(
        "pool",
        tasks=tuple(Task.on(f"t{i}", "p", 2, demand=1) for i in range(5)),
        resources=(Resource("p", capacity=2),),
    )
    decoded = Decoder(problem).decode(range(5))
    assert sorted(decoded.starts) == [0.0, 0.0, 2.0, 2.0, 4.0]


def test_permutation_problems_decode_to_permutation_schedules():
    rng = random.Random(3)
    jobs, machines = 6, 4
    tasks = []
    dependencies = []

    for j in range(jobs):
        for m in range(machines):
            tasks.append(Task.on(f"j{j}m{m}", f"m{m}", rng.randint(1, 9), group=f"j{j}"))
            if m:
                dependencies.append(Dependency(f"j{j}m{m - 1}", f"j{j}m{m}"))
    problem = Problem(
        "pfsp",
        tasks=tuple(tasks),
        resources=tuple(Resource(f"m{m}") for m in range(machines)),
        dependencies=tuple(dependencies),
        permutation=True,
    )
    decoder = Decoder(problem)
    for _ in range(20):
        schedule = decoder.schedule(random_order(problem, rng))
        assert check_schedule(problem, schedule) == ()
