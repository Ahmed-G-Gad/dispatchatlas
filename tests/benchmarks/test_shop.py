from __future__ import annotations

import math
import random
from collections import Counter
from statistics import fmean

import pytest

from dispatchatlas.benchmarks import default_registry
from dispatchatlas.benchmarks.shop import (
    FlexibleJobShop,
    FlowShop,
    HybridFlowShop,
    JobShop,
    OpenShop,
    flexible_job_shop_problem,
    flow_shop_problem,
    hybrid_flow_shop_problem,
    job_shop_problem,
    open_shop_problem,
)
from dispatchatlas.core import ConfigError, ModelError
from dispatchatlas.core.model import Problem

REGISTRY = default_registry(plugins=False)


def _chains(problem: Problem) -> dict[str, list[str]]:
    """Return each job's tasks in dependency order, checking they form one chain."""
    successors = {d.before: d.after for d in problem.dependencies}
    assert len(successors) == len(problem.dependencies), "a task has two successors"
    chains = {}
    for group, members in problem.groups.items():
        ids = [problem.tasks[i].id for i in members]
        first = [task for task in ids if all(successors.get(o) != task for o in ids)]
        assert len(first) == 1, f"job {group} is not one chain"
        chain = [first[0]]
        while chain[-1] in successors:
            chain.append(successors[chain[-1]])
        assert sorted(chain) == sorted(ids)
        chains[group] = chain
    return chains


def _machine(problem: Problem, task_id: str, mode: int = 0) -> str:
    return next(iter(problem.task(task_id).modes[mode].demands))


# ------------------------------------------------------------------ flow shop


def test_flow_shop_reproduces_the_documented_draws():
    problem = FlowShop(jobs=7, machines=4).generate(13)
    rng = random.Random(13)
    expected = [[rng.randint(1, 99) for _ in range(7)] for _ in range(4)]
    for job, chain in enumerate(_chains(problem).values()):
        assert [problem.task(t).modes[0].duration for t in chain] == [
            expected[k][job] for k in range(4)
        ]
        assert [_machine(problem, t) for t in chain] == ["m0", "m1", "m2", "m3"]


def test_flow_shop_structure():
    problem = FlowShop(jobs=12, machines=5, min_time=10, max_time=20).generate(1)
    assert problem.size == 60
    assert problem.permutation
    assert [r.id for r in problem.resources] == ["m0", "m1", "m2", "m3", "m4"]
    assert list(problem.groups) == [f"j{j:02d}" for j in range(12)]
    assert problem.task("j07-m3").group == "j07"
    assert all(10 <= task.modes[0].duration <= 20 for task in problem.tasks)
    assert len(problem.dependencies) == 12 * 4
    assert problem.features == {"precedence", "groups", "permutation"}


def test_general_flow_shop_drops_the_permutation_constraint():
    problem = FlowShop(permutation=False).generate(2)
    assert not problem.permutation
    assert problem.id == "flow-shop-20x5-noperm-2"


def test_flow_shop_due_dates_and_weights():
    problem = FlowShop(jobs=10, machines=4, due_date_tightness=1.3, weighted=True).generate(4)
    weights = []
    for chain in _chains(problem).values():
        tasks = [problem.task(t) for t in chain]
        total = sum(task.modes[0].duration for task in tasks)
        assert tasks[-1].due == math.floor(1.3 * total + 1e-9)
        assert all(task.due is None for task in tasks[:-1])
        assert len({task.weight for task in tasks}) == 1
        weights.append(tasks[0].weight)
    assert weights == [4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0]
    assert problem.id == "flow-shop-10x4-due1.3-w-4"


@pytest.mark.parametrize(
    ("jobs", "counts"),
    [
        (1, {2.0: 1}),
        (3, {4.0: 1, 2.0: 1, 1.0: 1}),
        (5, {4.0: 1, 2.0: 3, 1.0: 1}),
        (20, {4.0: 4, 2.0: 12, 1.0: 4}),
    ],
)
def test_weights_follow_the_twenty_sixty_twenty_split(jobs, counts):
    problem = JobShop(jobs=jobs, machines=2, weighted=True).generate(0)
    weights = Counter(problem.tasks[members[0]].weight for members in problem.groups.values())
    assert weights == counts


def test_tardiness_objectives_need_due_dates():
    with pytest.raises(ConfigError, match="due_date_tightness"):
        REGISTRY.create("flow-shop", objectives=("total_tardiness",))
    problem = REGISTRY.generate(
        "flow-shop", 0, due_date_tightness=1.5, objectives=("total_tardiness",)
    )
    assert problem.objectives == ("total_tardiness",)


# ------------------------------------------------------------------ job shop


def test_job_shop_reproduces_taillards_draw_order():
    problem = JobShop(jobs=4, machines=5).generate(21)
    rng = random.Random(21)
    times = [[rng.randint(1, 99) for _ in range(5)] for _ in range(4)]
    for job, chain in enumerate(_chains(problem).values()):
        order = list(range(5))
        for i in range(5):
            k = rng.randint(i, 4)
            order[i], order[k] = order[k], order[i]
        assert [_machine(problem, t) for t in chain] == [f"m{k}" for k in order]
        assert [problem.task(t).modes[0].duration for t in chain] == times[job]


@pytest.mark.parametrize("seed", range(5))
def test_every_job_visits_every_machine_once(seed):
    problem = JobShop(jobs=6, machines=7).generate(seed)
    for chain in _chains(problem).values():
        assert sorted(_machine(problem, t) for t in chain) == [f"m{k}" for k in range(7)]
    assert all(len(task.modes) == 1 for task in problem.tasks)


def test_job_shop_due_dates():
    problem = JobShop(jobs=5, machines=3, due_date_tightness=1.6).generate(8)
    for chain in _chains(problem).values():
        total = sum(problem.task(t).modes[0].duration for t in chain)
        assert problem.task(chain[-1]).due == math.floor(1.6 * total + 1e-9)


# ------------------------------------------------------------------ open shop


def test_open_shop_structure():
    problem = OpenShop(jobs=4, machines=3).generate(6)
    assert problem.dependencies == ()
    assert [r.id for r in problem.resources] == [
        "m0",
        "m1",
        "m2",
        "job-0",
        "job-1",
        "job-2",
        "job-3",
    ]
    assert {r.kind for r in problem.resources} == {"machine", "job"}
    for j in range(4):
        tasks = [problem.task(f"j{j}-m{k}") for k in range(3)]
        for k, task in enumerate(tasks):
            assert dict(task.modes[0].demands) == {f"m{k}": 1.0, f"job-{j}": 1.0}
            assert task.modes[0].location == f"m{k}"
    assert problem.features == {"multi-resource", "groups"}


def test_open_shop_assigns_times_through_the_machine_permutation():
    problem = OpenShop(jobs=3, machines=4).generate(9)
    rng = random.Random(9)
    times = [[rng.randint(1, 99) for _ in range(4)] for _ in range(3)]
    for j in range(3):
        order = list(range(4))
        for i in range(4):
            k = rng.randint(i, 3)
            order[i], order[k] = order[k], order[i]
        for i, machine in enumerate(order):
            assert problem.task(f"j{j}-m{machine}").modes[0].duration == times[j][i]


def test_open_shop_refuses_due_date_objectives():
    with pytest.raises(ConfigError, match="due dates"):
        REGISTRY.create("open-shop", objectives=("max_lateness",))


# ------------------------------------------------------------------ flexible job shop


@pytest.mark.parametrize("seed", range(6))
def test_flexible_job_shop_structure(seed):
    generator = FlexibleJobShop(
        jobs=6, machines=5, min_operations=2, max_operations=4, flexibility=3, max_time=9
    )
    problem = generator.generate(seed)
    for chain in _chains(problem).values():
        assert 2 <= len(chain) <= 4
        for task_id in chain:
            modes = problem.task(task_id).modes
            machines = [next(iter(mode.demands)) for mode in modes]
            assert 1 <= len(modes) <= 3
            assert machines == sorted(set(machines))
            assert all(1 <= mode.duration <= 9 for mode in modes)


def test_flexible_job_shop_reproduces_the_documented_draws():
    problem = FlexibleJobShop(jobs=2, machines=4, flexibility=2).generate(3)
    rng = random.Random(3)
    for chain in _chains(problem).values():
        assert len(chain) == rng.randint(5, 7)
        for task_id in chain:
            eligible = sorted(rng.sample(range(4), rng.randint(1, 2)))
            expected = [(f"m{k}", float(rng.randint(1, 20))) for k in eligible]
            modes = problem.task(task_id).modes
            assert [(next(iter(m.demands)), m.duration) for m in modes] == expected


def test_flexible_due_dates_use_the_mean_time_over_machines():
    problem = FlexibleJobShop(due_date_tightness=1.5).generate(2)
    for chain in _chains(problem).values():
        work = sum(fmean(m.duration for m in problem.task(t).modes) for t in chain)
        assert problem.task(chain[-1]).due == math.floor(1.5 * work + 1e-9)


def test_flexibility_cannot_exceed_the_machines():
    with pytest.raises(ConfigError, match="flexibility"):
        REGISTRY.create("flexible-job-shop", machines=2, flexibility=3)


# ------------------------------------------------------------------ hybrid flow shop


@pytest.mark.parametrize("unrelated", [False, True])
def test_hybrid_flow_shop_structure(unrelated):
    problem = HybridFlowShop(
        jobs=6, stages=4, min_machines=2, max_machines=3, unrelated=unrelated
    ).generate(5)
    rng = random.Random(5)
    widths = [rng.randint(2, 3) for _ in range(4)]
    assert [r.id for r in problem.resources] == [
        f"s{s}-m{k}" for s, width in enumerate(widths) for k in range(width)
    ]
    for chain in _chains(problem).values():
        assert len(chain) == 4
        for stage, task_id in enumerate(chain):
            modes = problem.task(task_id).modes
            assert [next(iter(m.demands)) for m in modes] == [
                f"s{stage}-m{k}" for k in range(widths[stage])
            ]
            if not unrelated:
                assert len({m.duration for m in modes}) == 1
    durations = {m.duration for t in problem.tasks for m in t.modes}
    assert len(durations) > 1


def test_unrelated_stages_draw_one_time_per_machine():
    problem = HybridFlowShop(
        jobs=1, stages=1, min_machines=3, max_machines=3, unrelated=True
    ).generate(0)
    rng = random.Random(0)
    rng.randint(3, 3)
    assert [m.duration for m in problem.tasks[0].modes] == [rng.randint(1, 99) for _ in range(3)]


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    ("family", "parameters", "message"),
    [
        ("flow-shop", {"min_time": 0}, "min_time"),
        ("flow-shop", {"min_time": 50, "max_time": 10}, "cannot exceed"),
        ("flow-shop", {"permutation": "yes"}, "permutation"),
        ("flow-shop", {"due_date_tightness": 0}, "positive"),
        ("flow-shop", {"due_date_tightness": "tight"}, "number"),
        ("job-shop", {"weighted": 1}, "weighted"),
        ("hybrid-flow-shop", {"min_machines": 3, "max_machines": 2}, "cannot exceed"),
        ("flexible-job-shop", {"min_operations": 0}, "min_operations"),
    ],
)
def test_invalid_parameters_are_refused(family, parameters, message):
    with pytest.raises(ConfigError, match=message):
        REGISTRY.create(family, **parameters)


def test_integer_tightness_is_stored_as_a_float():
    generator = FlowShop(due_date_tightness=2)
    assert generator.due_date_tightness == 2.0
    assert isinstance(generator.due_date_tightness, float)


# ------------------------------------------------------------------ builders


def test_flow_shop_builder_names_tasks_by_job_and_machine():
    problem = flow_shop_problem([[3, 2], [1, 4], [2, 0]], problem_id="toy", due_dates=[9, 9, 9])
    assert [t.id for t in problem.tasks] == ["j0-m0", "j0-m1", "j1-m0", "j1-m1", "j2-m0", "j2-m1"]
    assert problem.task("j2-m1").modes[0].duration == 0.0
    assert problem.task("j1-m1").due == 9.0
    assert problem.task("j1-m0").due is None
    assert problem.family is None


def test_ids_are_padded_so_they_sort_in_index_order():
    problem = flow_shop_problem([[1.0] * 11] * 12)
    groups = list(problem.groups)
    assert groups == sorted(groups) == [f"j{j:02d}" for j in range(12)]
    machines = [r.id for r in problem.resources]
    assert machines == sorted(machines)
    assert machines[-1] == "m10"


def test_job_shop_builder_follows_the_routes():
    problem = job_shop_problem([[(1, 3.0), (0, 2.0)], [(0, 4.0)]], machines=3)
    assert [r.id for r in problem.resources] == ["m0", "m1", "m2"]
    assert [t.id for t in problem.tasks] == ["j0-o0", "j0-o1", "j1-o0"]
    assert _machine(problem, "j0-o0") == "m1"
    assert [(d.before, d.after) for d in problem.dependencies] == [("j0-o0", "j0-o1")]


def test_flexible_builder_gives_one_mode_per_machine():
    problem = flexible_job_shop_problem([[[(0, 3.0), (2, 5.0)], [(1, 1.0)]]])
    modes = problem.task("j0-o0").modes
    assert [(dict(m.demands), m.duration) for m in modes] == [
        ({"m0": 1.0}, 3.0),
        ({"m2": 1.0}, 5.0),
    ]


def test_hybrid_builder_names_machines_by_stage():
    problem = hybrid_flow_shop_problem([[[2.0, 3.0], [4.0]], [[1.0, 1.0], [2.0]]])
    assert [r.id for r in problem.resources] == ["s0-m0", "s0-m1", "s1-m0"]
    assert [t.id for t in problem.tasks] == ["j0-s0", "j0-s1", "j1-s0", "j1-s1"]


def test_open_shop_builder_holds_machine_and_job():
    problem = open_shop_problem([[1.0, 2.0]])
    assert dict(problem.task("j0-m1").modes[0].demands) == {"m1": 1.0, "job-0": 1.0}


@pytest.mark.parametrize(
    "build",
    [
        lambda: flow_shop_problem([]),
        lambda: flow_shop_problem([[1.0, 2.0], [1.0]]),
        lambda: flow_shop_problem([[1.0]], due_dates=[1.0, 2.0]),
        lambda: flow_shop_problem([[1.0]], weights=[]),
        lambda: flow_shop_problem([[-1.0]]),
        lambda: job_shop_problem([[(3, 1.0)]], machines=2),
        lambda: job_shop_problem([[]]),
        lambda: flexible_job_shop_problem([[[]]]),
        lambda: flexible_job_shop_problem([[[(1.5, 2.0)]]]),  # type: ignore[list-item]
        lambda: hybrid_flow_shop_problem([[[1.0], [2.0]], [[1.0]]]),
        lambda: open_shop_problem([[1.0], [1.0, 2.0]]),
    ],
)
def test_builders_refuse_malformed_data(build):
    with pytest.raises(ModelError):
        build()
