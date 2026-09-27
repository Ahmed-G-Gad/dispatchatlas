import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dispatchatlas.core import (
    ConfigError,
    Dependency,
    Mode,
    Problem,
    Resource,
    SolveResult,
    StopCriteria,
    StopReason,
    Task,
    UnsupportedProblemError,
    check_schedule,
)
from dispatchatlas.solvers import default_registry
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.learning.common import (
    Improvement,
    due_dates,
    horizon,
)
from dispatchatlas.solvers.learning.hyper_heuristic import (
    GpHyperHeuristic,
    characterise,
    decision_situations,
    list_order,
    pick,
)
from dispatchatlas.solvers.learning.policy_gradient import (
    LinearPolicy,
    sample,
    softmax,
    task_features,
)
from dispatchatlas.solvers.learning.temporal_difference import (
    RULES,
    ActionValues,
    Agent,
    DispatchRules,
    TemporalDifferenceDispatching,
    greedy_action,
    observe,
    td_target,
)
from dispatchatlas.solvers.learning.trees import (
    FUNCTIONS,
    INITIAL_DEPTHS,
    TERMINALS,
    crossover,
    depth,
    evaluate,
    grow,
    mutate,
    protected_division,
    ramped_half_and_half,
    replace,
    size,
    subtree,
    task_depths,
    terminal_columns,
    text,
)
from dispatchatlas.solvers.scheduling import Decoder
from dispatchatlas.solvers.search import Search
from tests.helpers import random_problem
from tests.solvers import test_contract

REGISTRY = default_registry(plugins=False)

LEARNERS = [
    "q-learning-dispatching",
    "sarsa-dispatching",
    "reinforce-priority-policy",
    "gp-hyper-heuristic",
]
TD_AGENTS = ["q-learning-dispatching", "sarsa-dispatching"]
GREEDY_POLICIES = [*TD_AGENTS, "reinforce-priority-policy"]
"""Learners that evaluate the greedy policy of their untrained start first."""


class FixedRandom(random.Random):
    """A generator whose ``random()`` always returns the same value."""

    def __init__(self, value: float) -> None:
        super().__init__(0)
        self.value = value

    def random(self) -> float:
        return self.value


class FirstChoice(FixedRandom):
    """A generator that always explores, and always explores the first rule."""

    def __init__(self) -> None:
        super().__init__(0.0)

    def randrange(self, start, stop=None, step=1):
        return 0 if stop is None else start


def contended() -> Problem:
    """One machine and three tasks; the late-released one makes the order matter.

    The long task first gives makespan 6. Both unit tasks first strand the
    machine until the release at 3 and give makespan 8, the schedule of the
    shortest-processing-time rule and of the input order.
    """
    return Problem(
        "contended",
        tasks=(
            Task.on("short", "m", 1),
            Task.on("late", "m", 1, release=3),
            Task.on("long", "m", 4),
        ),
        resources=(Resource("m"),),
    )


def tardiness_problem(seed: int, jobs: int = 20, machines: int = 3) -> Problem:
    """Unrelated parallel machines with tight due dates and weighted tardiness."""
    rng = random.Random(seed)
    tasks = []
    for j in range(jobs):
        modes = tuple(Mode(rng.randint(1, 20), {f"m{m}": 1}) for m in range(machines))
        work = sum(mode.duration for mode in modes) / machines
        due = rng.uniform(0.1, 0.6) * work * jobs / machines
        tasks.append(Task(f"j{j}", modes, due=round(due), weight=float(rng.randint(1, 5))))
    return Problem(
        f"tardiness-{seed}",
        tasks=tuple(tasks),
        resources=tuple(Resource(f"m{m}") for m in range(machines)),
        objectives=("total_tardiness",),
    )


def flow_shop(jobs: int = 4, machines: int = 3) -> Problem:
    rng = random.Random(jobs)
    tasks, dependencies = [], []
    for j in range(jobs):
        for m in range(machines):
            tasks.append(Task.on(f"j{j}m{m}", f"m{m}", rng.randint(1, 9), group=f"j{j}"))
            if m:
                dependencies.append(Dependency(f"j{j}m{m - 1}", f"j{j}m{m}"))
    return Problem(
        "flow-shop",
        tuple(tasks),
        tuple(Resource(f"m{m}") for m in range(machines)),
        tuple(dependencies),
        permutation=True,
    )


def value_of(result: SolveResult) -> float:
    assert result.value is not None
    return result.value


def untrained_value(solver_id: str, problem: Problem) -> float:
    """The objective of the first schedule a learner evaluates: its untrained start."""
    stop = StopCriteria(max_evaluations=1)
    return value_of(REGISTRY.create(solver_id).solve(problem, stop, seed=0))


# ------------------------------------------------------------------ registration


def test_every_learner_is_registered_and_covered_by_the_contract():
    families = {solver_id: REGISTRY.info(solver_id).family for solver_id in LEARNERS}
    assert families == {
        "q-learning-dispatching": "learning",
        "sarsa-dispatching": "learning",
        "reinforce-priority-policy": "learning",
        "gp-hyper-heuristic": "hyper-heuristic",
    }
    assert set(LEARNERS) <= set(test_contract.ITERATIVE)


def test_one_agent_class_is_registered_under_both_targets():
    for solver_id, target in (
        ("q-learning-dispatching", "q-learning"),
        ("sarsa-dispatching", "sarsa"),
    ):
        solver = REGISTRY.create(solver_id, learning_rate=0.25)
        assert isinstance(solver, TemporalDifferenceDispatching)
        assert solver.target == target
        assert solver.info.id == solver_id
        assert "target" not in REGISTRY.info(solver_id).parameters
    assert TemporalDifferenceDispatching(target="sarsa").info.id == "sarsa-dispatching"


@pytest.mark.parametrize(
    ("solver_id", "parameters"),
    [
        ("q-learning-dispatching", {"learning_rate": 0.0}),
        ("q-learning-dispatching", {"learning_rate": 1.5}),
        ("sarsa-dispatching", {"discount": -0.1}),
        ("sarsa-dispatching", {"exploration": 1.1}),
        ("q-learning-dispatching", {"stage_buckets": 0}),
        ("q-learning-dispatching", {"target": "sarsa"}),
        ("reinforce-priority-policy", {"learning_rate": 0.0}),
        ("reinforce-priority-policy", {"temperature": 0.0}),
        ("reinforce-priority-policy", {"temperature": math.inf}),
        ("gp-hyper-heuristic", {"population": 0}),
        ("gp-hyper-heuristic", {"max_depth": 2}),
        ("gp-hyper-heuristic", {"crossover_rate": 0.9, "mutation_rate": 0.2}),
        ("gp-hyper-heuristic", {"mutation_rate": -0.1}),
        ("gp-hyper-heuristic", {"tournament_size": 0}),
        ("gp-hyper-heuristic", {"elites": 24}),
        ("gp-hyper-heuristic", {"offspring_multiplier": 0}),
        ("gp-hyper-heuristic", {"neighbours": 0}),
        ("gp-hyper-heuristic", {"situations": 0}),
        ("gp-hyper-heuristic", {"situation_size": 1}),
    ],
)
def test_invalid_parameters_are_refused(solver_id, parameters):
    with pytest.raises(ConfigError):
        REGISTRY.create(solver_id, **parameters)


def test_an_unknown_target_is_refused():
    with pytest.raises(ValueError, match="target"):
        TemporalDifferenceDispatching(target="expected-sarsa")  # type: ignore[arg-type]


def test_the_rule_agents_refuse_permutation_problems():
    problem = flow_shop()
    for solver_id in TD_AGENTS:
        with pytest.raises(UnsupportedProblemError):
            REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=5))
    for solver_id in ("reinforce-priority-policy", "gp-hyper-heuristic"):
        result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=30))
        assert result.schedule is not None
        assert check_schedule(problem, result.schedule) == ()


# ------------------------------------------------------------------ shared pieces


def test_due_dates_take_the_earlier_of_due_and_deadline_and_default_to_the_horizon():
    problem = Problem(
        "dues",
        tasks=(
            Task.on("a", "m", 2, due=9.0, deadline=7.0),
            Task.on("b", "m", 3, due=5.0),
            Task.on("c", "m", 1, release=4.0),
            Task.on("d", "m", 1, group="g"),
            Task.on("e", "m", 1, group="g", due=6.0),
        ),
        resources=(Resource("m"),),
    )
    assert horizon(problem) == 4.0 + 8.0
    assert due_dates(problem) == [7.0, 5.0, 12.0, 6.0, 6.0]


def test_the_reward_measures_improvement_over_the_reference_without_units():
    problem = contended()
    search = Search(
        problem, StopCriteria(max_evaluations=5), 0, REGISTRY.info("gp-hyper-heuristic")
    )
    reference = search.assess([0, 1, 2])  # makespan 8
    reward = Improvement.over(reference)
    assert reward(reference.score) == 0.0
    assert reward(Score(0.0, 6.0)) == pytest.approx(2.0 / 8.0)
    assert reward(Score(4.0, 8.0)) == pytest.approx(-4.0 / 8.0)


def test_budget_progress_follows_the_deterministic_limits_first():
    info = REGISTRY.info("gp-hyper-heuristic")
    problem = contended()
    search = Search(problem, StopCriteria(max_evaluations=10, max_iterations=4), 0, info)
    search.evaluate([0, 1, 2])
    assert search.progress(deterministic=True) == pytest.approx(0.1)
    search.next_iteration()
    assert search.progress(deterministic=True) == pytest.approx(0.25)
    timed = Search(problem, StopCriteria(time_limit=1000.0, max_evaluations=10), 0, info)
    assert timed.progress(deterministic=True) == 0.0
    clock = Search(problem, StopCriteria(time_limit=1000.0), 0, info)
    assert 0.0 <= clock.progress(deterministic=True) < 0.01


# ------------------------------------------------------------------ temporal difference


def test_states_bucket_progress_and_the_eligible_set():
    assert observe(0, 10, 1, 4) == (0, 0)
    assert observe(3, 10, 2, 4) == (1, 1)
    assert observe(9, 10, 7, 4) == (3, 2)
    assert observe(5, 10, 3, 1) == (0, 2)


def test_greedy_action_is_the_first_argmax():
    assert greedy_action([0.0, 0.0, 0.0, 0.0]) == 0
    assert greedy_action([0.1, 0.5, 0.5, 0.2]) == 1
    assert greedy_action([-1.0, -0.5, -2.0, -0.7]) == 1


def test_td_targets_differ_as_defined_on_a_hand_made_transition():
    next_values = [1.0, 3.0, 0.0, 2.0]
    explored = 0  # the agent explores the first rule in the next state
    assert td_target("q-learning", next_values, explored, 0.9) == pytest.approx(2.7)
    assert td_target("sarsa", next_values, explored, 0.9) == pytest.approx(0.9)
    assert td_target("sarsa", next_values, 1, 0.9) == pytest.approx(2.7)
    values = ActionValues(len(RULES))
    values.row((0, 0))[:] = [0.2, 0.0, 0.1, 0.0]
    assert values.update((0, 0), 2, 2.7, 0.5) is True  # 0.1 + 0.5 * (2.7 - 0.1) = 1.4
    assert values.get((0, 0)) == pytest.approx((0.2, 0.0, 1.4, 0.0))
    assert values.update((0, 0), 1, 0.9, 0.5) is False  # 0.45 stays below 1.4
    assert values.get((1, 1)) == (0.0,) * len(RULES)
    assert len(values) == 1


@pytest.mark.parametrize("target", ["q-learning", "sarsa"])
def test_the_greedy_policy_applies_the_argmax_rule_of_every_state(target):
    problem = random_problem(3, tasks=12, max_modes=2, pools=1)
    decoder = Decoder(problem)
    agent = Agent(problem, decoder, target=target, learning_rate=0.5, discount=0.9, stage_buckets=4)
    rng = random.Random(5)
    for _ in range(30):
        episode = agent.episode(rng, 0.3)
        agent.finish(episode, rng.uniform(-1.0, 1.0))
    assert any(
        any(value != 0.0 for value in agent.values.get(state)) for state, _ in episode.decisions
    )
    greedy = agent.greedy()
    assert greedy.decisions
    for state, rule in greedy.decisions:
        assert rule == greedy_action(agent.values.get(state))


def test_an_untrained_agent_applies_the_first_rule_everywhere():
    problem = random_problem(4, tasks=10, max_modes=1, pools=0)
    agent = Agent(
        problem, Decoder(problem), target="sarsa", learning_rate=0.5, discount=0.9, stage_buckets=4
    )
    assert {rule for _, rule in agent.greedy().decisions} == {0}
    assert len(agent.values) == 0


def test_q_learning_and_sarsa_update_the_same_walk_differently():
    problem = two_task_problem()
    learned = {}
    for target in ("q-learning", "sarsa"):
        agent = Agent(
            problem,
            Decoder(problem),
            target=target,
            learning_rate=0.5,
            discount=0.9,
            stage_buckets=4,
        )
        for state in ((0, 1), (2, 0)):
            agent.values.row(state)[:] = [0.0, 1.0, 0.5, 0.25]
        episode = agent.episode(FirstChoice(), 1.0)
        assert episode.decisions == (((0, 1), 0), ((2, 0), 0))
        learned[target] = agent.values.get((0, 1))
    # The one bootstrapped transition: Q-learning moves the first decision toward
    # 0.9 * max Q(next) = 0.9, SARSA toward 0.9 * Q(next, first rule) = 0.
    assert learned["q-learning"] == pytest.approx((0.45, 1.0, 0.5, 0.25))
    assert learned["sarsa"] == pytest.approx((0.0, 1.0, 0.5, 0.25))


def test_the_rules_pick_by_duration_due_date_and_dynamic_slack():
    problem = Problem(
        "rules",
        tasks=(
            Task.on("a", "m", 2, due=9.0),
            Task.on("b", "m", 1, due=3.0),
            Task.on("c", "m", 3, release=4.0, due=8.0),
        ),
        resources=(Resource("m"),),
    )
    rules = DispatchRules(problem)
    builder = Decoder(problem).builder()
    eligible = [0, 1, 2]
    picks = [rules.choose(rule, eligible, builder)[0] for rule in range(len(RULES))]
    # Shortest b, longest c, earliest due b; slack a 9-2, b 3-1, c 8-7.
    assert picks == [1, 2, 1, 2]
    builder.place(2, 0, 4.0)
    # With c on [4, 7], b (slack 3 - 1) comes before a (slack 9 - 2) at time 0.
    assert rules.choose(3, [0, 1], builder) == (1, 0, 0.0)
    builder.place(1, 0, 0.0)
    # a fills the idle gap between b and c.
    assert rules.choose(3, [0], builder) == (0, 0, 1.0)


def test_the_slack_rule_counts_the_path_after_the_task():
    problem = Problem(
        "path",
        tasks=(
            Task.on("a", "m", 2, due=12.0),
            Task.on("b", "m", 1, due=6.0),
            Task.on("tail", "n", 7),
        ),
        resources=(Resource("m"), Resource("n")),
        dependencies=(Dependency("a", "tail"),),
    )
    # Slack of a: 12 - 2 - 7 = 3; of b: 6 - 1 = 5.
    task, _, _ = DispatchRules(problem).choose(3, [0, 1], Decoder(problem).builder())
    assert task == 0


@pytest.mark.parametrize("target", ["q-learning", "sarsa"])
def test_episodes_decode_to_the_schedule_the_agent_built(target):
    problem = random_problem(7, tasks=15, max_modes=3, pools=1, network=True)
    decoder = Decoder(problem)
    agent = Agent(problem, decoder, target=target, learning_rate=0.5, discount=0.9, stage_buckets=4)
    rng = random.Random(1)
    for _ in range(5):
        episode = agent.episode(rng, 0.5)
        decoded = decoder.decode(episode.sequence, episode.modes)
        assert decoded.sequence == episode.sequence
        assert decoded.modes == episode.modes
        assert decoded.starts == episode.starts


@pytest.mark.parametrize("solver_id", TD_AGENTS)
def test_without_exploration_the_seed_does_not_matter(solver_id):
    problem = random_problem(8, tasks=20, max_modes=2, pools=1)
    stop = StopCriteria(max_evaluations=60)
    first = REGISTRY.create(solver_id, exploration=0.0).solve(problem, stop, seed=1)
    second = REGISTRY.create(solver_id, exploration=0.0).solve(problem, stop, seed=2)
    assert first.schedule == second.schedule
    assert first.extras == second.extras


def test_exploration_falls_linearly_with_the_budget_spent(monkeypatch):
    rates: list[float] = []
    episode = Agent.episode

    def recorded(self, rng, epsilon):
        rates.append(epsilon)
        return episode(self, rng, epsilon)

    monkeypatch.setattr(Agent, "episode", recorded)
    problem = random_problem(17, tasks=12, max_modes=2, pools=1)
    solver = REGISTRY.create("q-learning-dispatching")
    solver.solve(problem, StopCriteria(max_iterations=4), seed=0)
    assert rates == pytest.approx([0.3, 0.225, 0.15, 0.075])
    rates.clear()
    solver.solve(problem, StopCriteria(max_evaluations=30, time_limit=100.0), seed=0)
    assert rates[0] == pytest.approx(0.3 * (1.0 - 1.0 / 30.0))  # after the reference
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] < 0.03


class CountingRandom(random.Random):
    """A generator that counts its uniform draws and its index draws.

    It defines ``getrandbits`` so that ``randrange`` keeps drawing bits rather
    than calling ``random``, as it does for plain generators.
    """

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.draws = 0
        self.indices = 0

    def random(self) -> float:
        self.draws += 1
        return super().random()

    def getrandbits(self, k: int) -> int:
        return super().getrandbits(k)

    def randrange(self, start, stop=None, step=1):
        self.indices += 1
        return super().randrange(start, stop, step)


def test_episodes_draw_one_uniform_number_per_step_and_an_index_per_exploration():
    problem = random_problem(18, tasks=14, max_modes=2, pools=1)
    agent = Agent(
        problem, Decoder(problem), target="sarsa", learning_rate=0.5, discount=0.9, stage_buckets=4
    )
    rng = CountingRandom(3)
    agent.episode(rng, 0.0)
    assert (rng.draws, rng.indices) == (problem.size, 0)
    rng = CountingRandom(3)
    agent.episode(rng, 1.0)
    assert (rng.draws, rng.indices) == (problem.size, problem.size)


# ------------------------------------------------------------------ policy gradient


def two_task_problem() -> Problem:
    """Two independent tasks, 1 and 3 long, on separate machines."""
    return Problem(
        "two",
        tasks=(Task.on("quick", "m1", 1), Task.on("slow", "m2", 3)),
        resources=(Resource("m1"), Resource("m2")),
    )


def test_features_are_scaled_columns_with_a_horizon_slack():
    features = task_features(two_task_problem())
    # Horizon 4: slack 4 - 1 = 3 and 4 - 3 = 1, scaled by 3.
    assert features[0] == pytest.approx((1.0, 1.0 / 3.0, 0.0, 1.0, 0.0))
    assert features[1] == pytest.approx((1.0, 1.0, 0.0, 1.0 / 3.0, 0.0))


def test_softmax_and_inverse_transform_sampling():
    probabilities = softmax([1.0, 2.0, 3.0])
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities == pytest.approx(softmax([1001.0, 1002.0, 1003.0]))
    assert probabilities[2] / probabilities[1] == pytest.approx(math.e)
    assert sample([0.2, 0.3, 0.5], FixedRandom(0.1)) == 0
    assert sample([0.2, 0.3, 0.5], FixedRandom(0.2)) == 1
    assert sample([0.2, 0.3, 0.5], FixedRandom(0.999)) == 2
    assert sample([0.0, 1.0], FixedRandom(0.0)) == 1


@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_the_policy_gradient_step_on_a_two_action_toy(temperature):
    problem = two_task_problem()
    policy = LinearPolicy(problem, temperature)
    order, gradient = policy.rollout(FixedRandom(0.1))  # uniform policy: draws "quick"
    assert order == [0, 1]
    # First step: phi(quick) - (phi(quick) + phi(slow)) / 2; the second step has
    # one candidate and adds nothing.
    expected = [0.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 0.0]
    assert gradient == pytest.approx([value / temperature for value in expected])
    policy.step(gradient, 0.2 * 1.0)  # learning rate 0.2, advantage 1
    assert policy.weights == pytest.approx([0.2 * value / temperature for value in expected])
    quick, slow = policy.priorities()
    assert softmax([quick / temperature, slow / temperature])[0] > 0.5
    assert policy.greedy() == [0, 1]
    policy.step(gradient, -1.0)  # a negative advantage reverses the preference
    assert policy.greedy() == [1, 0]


def test_the_untrained_greedy_policy_is_the_topological_order():
    problem = random_problem(9, tasks=15, max_modes=1, pools=0)
    policy = LinearPolicy(problem, 1.0)
    assert policy.greedy() == list(problem.topological_order)
    order, _ = policy.rollout(random.Random(3))
    assert sorted(order) == list(range(problem.size))
    placed: set[int] = set()
    for task in order:
        assert set(problem.predecessors[task]) <= placed
        placed.add(task)
    rng = CountingRandom(3)
    policy.rollout(rng)
    assert (rng.draws, rng.indices) == (problem.size, 0)


def test_updates_scale_the_gradient_by_the_reward_above_the_running_mean(monkeypatch):
    rewards: list[float] = []
    scales: list[float] = []
    reward = Improvement.__call__
    step = LinearPolicy.step

    def recorded_reward(self, score):
        rewards.append(reward(self, score))
        return rewards[-1]

    def recorded_step(self, gradient, scale):
        scales.append(scale)
        step(self, gradient, scale)

    monkeypatch.setattr(Improvement, "__call__", recorded_reward)
    monkeypatch.setattr(LinearPolicy, "step", recorded_step)
    problem = random_problem(19, tasks=12, max_modes=2, pools=1)
    solver = REGISTRY.create("reinforce-priority-policy", learning_rate=0.5)
    solver.solve(problem, StopCriteria(max_iterations=6), seed=1)
    assert len(rewards) == len(scales) == 6
    for episode, scale in enumerate(scales):
        baseline = sum(rewards[:episode]) / episode if episode else 0.0
        assert scale == pytest.approx(0.5 * (rewards[episode] - baseline))


# ------------------------------------------------------------------ genetic programming


def test_division_is_protected():
    result = protected_division(np.array([3.0, 3.0, -2.0]), np.array([0.0, 2.0, -0.0]))
    assert result.tolist() == [1.0, 1.5, 1.0]
    problem = contended()
    columns = terminal_columns(problem)
    assert evaluate(("DIV", "DUR", "SUCC"), columns).tolist() == [1.0, 1.0, 1.0]
    assert evaluate(("DIV", "SUCC", "DUR"), columns).tolist() == [0.0, 0.0, 0.0]


def test_the_primitives_compute_their_operations():
    problem = contended()
    columns = terminal_columns(problem)
    assert columns["DUR"].tolist() == [1.0, 1.0, 4.0]
    assert columns["REL"].tolist() == [0.0, 3.0, 0.0]
    # Horizon 3 + 6 = 9, the due date of tasks without one.
    assert columns["SLACK"].tolist() == [9.0 - 0.0 - 1.0, 9.0 - 3.0 - 1.0, 9.0 - 0.0 - 4.0]
    cases = {
        "ADD": [1.0, 4.0, 4.0],
        "SUB": [1.0, -2.0, 4.0],
        "MUL": [0.0, 3.0, 0.0],
        "MAX": [1.0, 3.0, 4.0],
        "MIN": [0.0, 1.0, 0.0],
    }
    for name, expected in cases.items():
        assert evaluate((name, "DUR", "REL"), columns).tolist() == expected
    assert set(FUNCTIONS) == {"ADD", "SUB", "MUL", "DIV", "MAX", "MIN"}
    assert TERMINALS == ("DUR", "SUCC", "REL", "SLACK", "DEPTH")


def diamond(shortcut_first: bool) -> Problem:
    """a -> b -> c -> d with a shortcut a -> d, declared first or last."""
    chain = [Dependency("a", "b"), Dependency("b", "c"), Dependency("c", "d")]
    shortcut = [Dependency("a", "d")]
    return Problem(
        "diamond",
        tasks=tuple(Task.on(name, "m", 1) for name in "abcd"),
        resources=(Resource("m"),),
        dependencies=tuple(shortcut + chain if shortcut_first else chain + shortcut),
    )


@pytest.mark.parametrize("shortcut_first", [True, False])
def test_depth_is_the_longest_chain_despite_shortcuts(shortcut_first):
    problem = diamond(shortcut_first)
    assert task_depths(problem) == [0, 1, 2, 3]
    branching = Problem(
        "branching",
        tasks=tuple(Task.on(name, "m", 1) for name in "abcde"),
        resources=(Resource("m"),),
        dependencies=(
            Dependency("a", "e"),
            Dependency("a", "b"),
            Dependency("a", "c"),
            Dependency("b", "d"),
            Dependency("c", "d"),
            Dependency("d", "e"),
        ),
    )
    assert task_depths(branching) == [0, 1, 1, 2, 3]
    assert terminal_columns(problem)["DEPTH"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_ramped_half_and_half_and_the_variation_operators_keep_valid_trees():
    rng = random.Random(4)
    for _ in range(200):
        tree = ramped_half_and_half(rng)
        assert 1 <= depth(tree) <= max(INITIAL_DEPTHS)
        child = crossover(tree, ramped_half_and_half(rng), rng)
        mutant = mutate(child, rng)
        for item in (child, mutant):
            assert evaluate(item, terminal_columns(contended())).shape == (3,)
    full = grow(3, True, random.Random(1))
    assert depth(full) == 3
    assert size(full) == 7


def test_subtrees_are_addressed_in_pre_order():
    tree = ("ADD", ("MUL", "DUR", "REL"), "SLACK")
    assert [subtree(tree, position) for position in range(size(tree))] == [
        tree,
        ("MUL", "DUR", "REL"),
        "DUR",
        "REL",
        "SLACK",
    ]
    assert replace(tree, 3, "DEPTH") == ("ADD", ("MUL", "DUR", "DEPTH"), "SLACK")
    assert replace(tree, 0, "SUCC") == "SUCC"
    assert text(tree) == "ADD(MUL(DUR,REL),SLACK)"


def test_near_ties_go_to_the_shortest_then_the_lowest_task():
    durations = [3.0, 1.0, 1.0, 2.0]
    assert pick([0, 1, 2, 3], [5.0, 5.0 - 1e-12, 5.0, 4.0], durations) == 1
    assert pick([0, 2, 3], [5.0, 0.0, 5.0 - 1e-6, 0.0], durations) == 0
    assert pick([0, 3], [-math.inf, 0.0, 0.0, -math.inf], durations) == 3


def test_list_orders_follow_priorities_as_far_as_precedence_allows():
    problem = diamond(True)
    durations = [1.0] * 4
    assert list_order(problem, [0.0, 1.0, 2.0, 3.0], durations) == [0, 1, 2, 3]
    unrelated = contended()
    assert list_order(unrelated, [1.0, 2.0, 3.0], [1.0, 1.0, 4.0]) == [2, 1, 0]


def test_characterisations_record_the_reference_rank_of_each_pick():
    durations = [4.0, 1.0, 2.0, 3.0]
    situations = [(0, 1, 2), (0, 3), (1, 2, 3)]
    # By duration: (1, 2, 0), (3, 0), (1, 2, 3).
    assert characterise([9.0, 0.0, 5.0, 1.0], situations, durations) == (2, 1, 1)
    assert characterise([0.0, 9.0, 5.0, 1.0], situations, durations) == (0, 0, 0)


def test_decision_situations_hold_two_to_the_largest_size_distinct_tasks():
    situations = decision_situations(10, 50, 4, random.Random(2))
    assert len(situations) == 50
    for situation in situations:
        assert 2 <= len(situation) <= 4
        assert list(situation) == sorted(set(situation))
    assert {len(item) for item in situations} == {2, 3, 4}
    assert decision_situations(1, 3, 7, random.Random(2)) == [(0,)] * 3


def test_the_hyper_heuristic_reports_the_rule_of_its_best_schedule():
    problem = random_problem(10, tasks=15, max_modes=2, pools=1)
    result = GpHyperHeuristic().solve(problem, StopCriteria(max_evaluations=80), seed=3)
    rule = result.extras["rule"]
    assert isinstance(rule, str)
    assert rule
    assert set(rule.replace("(", " ").replace(")", " ").replace(",", " ").split()) <= set(
        TERMINALS + FUNCTIONS
    )


def test_every_rule_applied_is_one_evaluation():
    problem = random_problem(11, tasks=12, max_modes=1, pools=0)
    settings = GpHyperHeuristic(population=10, elites=2)
    first = settings.solve(problem, StopCriteria(max_iterations=1), seed=1)
    assert first.iterations == 1
    assert first.evaluations == 10 + 8  # the population, then one generation
    second = settings.solve(problem, StopCriteria(max_iterations=2), seed=1)
    assert second.evaluations == 10 + 2 * 8


# ------------------------------------------------------------------ learning evidence


@pytest.mark.parametrize("solver_id", GREEDY_POLICIES)
def test_the_learned_greedy_policy_beats_the_untrained_one(solver_id):
    problem = contended()
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=200), seed=0)
    assert untrained_value(solver_id, problem) == 8.0
    assert result.value == 6.0
    assert result.extras["policy_value"] == 6.0


def test_the_hyper_heuristic_finds_the_optimum_of_the_contended_problem():
    result = GpHyperHeuristic().solve(contended(), StopCriteria(max_evaluations=100), seed=0)
    assert result.value == 6.0


@pytest.mark.parametrize("solver_id", LEARNERS)
def test_no_learner_ends_worse_than_its_start(solver_id):
    problem = random_problem(12, tasks=25, max_modes=2, pools=1)
    start_budget = 24 if solver_id == "gp-hyper-heuristic" else 1  # the random population
    start = REGISTRY.create(solver_id).solve(
        problem, StopCriteria(max_evaluations=start_budget), seed=5
    )
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=300), seed=5)
    assert value_of(result) <= value_of(start)
    assert result.trace[0].value == start.trace[0].value


@pytest.mark.parametrize("solver_id", LEARNERS)
def test_beats_random_sampling_at_equal_evaluations(solver_id):
    problem = tardiness_problem(2)
    stop = StopCriteria(max_evaluations=150)
    baseline = REGISTRY.create("random-search").solve(problem, stop, seed=3)
    result = REGISTRY.create(solver_id).solve(problem, stop, seed=3)
    assert result.evaluations == 150
    assert value_of(result) < value_of(baseline)


# ------------------------------------------------------------------ budgets and replay


@pytest.mark.parametrize("solver_id", LEARNERS)
def test_stops_at_the_time_limit_despite_a_huge_iteration_limit(solver_id):
    problem = random_problem(13, tasks=40, max_modes=2, pools=1)
    stop = StopCriteria(time_limit=0.5, max_iterations=10**7)
    result = REGISTRY.create(solver_id).solve(problem, stop, seed=2)
    assert result.stop_reason is StopReason.TIME_LIMIT
    assert result.elapsed <= 1.5
    assert result.iterations < 10**7


@pytest.mark.parametrize("solver_id", LEARNERS)
def test_stops_exactly_at_the_evaluation_limit(solver_id):
    problem = random_problem(14, tasks=20, max_modes=2, pools=1)
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=40), seed=2)
    assert result.evaluations == 40
    assert result.stop_reason is StopReason.EVALUATION_LIMIT


@pytest.mark.parametrize("solver_id", LEARNERS)
def test_counts_one_iteration_per_episode_or_generation(solver_id):
    problem = random_problem(15, tasks=20, max_modes=2, pools=1)
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_iterations=5), seed=2)
    assert result.iterations == 5
    assert result.stop_reason is StopReason.ITERATION_LIMIT


@pytest.mark.parametrize("solver_id", LEARNERS)
def test_replays_identically_from_a_seed(solver_id):
    problem = random_problem(16, tasks=30, max_modes=2, pools=1)
    stop = StopCriteria(max_evaluations=150)
    runs = [REGISTRY.create(solver_id).solve(problem, stop, seed=seed) for seed in (4, 4, 5, 6)]
    first, again = runs[0], runs[1]
    assert first.schedule == again.schedule
    assert first.objectives == again.objectives
    assert first.extras == again.extras
    assert [point.value for point in first.trace] == [point.value for point in again.trace]
    assert len({run.schedule for run in runs}) > 1


def test_importing_the_library_loads_no_heavy_dependency():
    script = (
        "import sys\n"
        "import dispatchatlas\n"
        "import dispatchatlas.solvers.learning\n"
        "from dispatchatlas.solvers import default_registry\n"
        "default_registry(plugins=False)\n"
        "heavy = ('torch', 'sklearn', 'ortools', 'pulp', 'matplotlib')\n"
        "print(','.join(name for name in heavy if name in sys.modules))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert completed.stdout.strip() == ""
