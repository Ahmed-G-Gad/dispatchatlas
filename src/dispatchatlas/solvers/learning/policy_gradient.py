"""A linear softmax priority policy trained with REINFORCE.

Instead of applying a fixed priority rule, this solver learns one. A task's
priority is a weighted sum of five features, and the policy builds a task order
by drawing, at each step, one eligible task (all its predecessors already
drawn) with probability given by the softmax of the eligible tasks' priorities
divided by ``temperature``. The weights are trained with the REINFORCE
estimator of Williams (1992):

1. **Features** (:func:`task_features`): a constant 1, the shortest duration
   over the task's modes, the release time, the slack, and the number of direct
   successors. The slack is the due date from
   :func:`~dispatchatlas.solvers.learning.common.due_dates` minus the release
   time minus the shortest duration; tasks without a due date use the problem's
   horizon, the latest release plus the sum of the shortest durations. Every
   column except the constant is divided by its largest absolute value over
   the tasks, so the weights are comparable. The constant shifts every
   priority by the same amount, so it changes neither the probabilities nor
   the greedy order, and its gradient is zero.
2. **Rollout.** An episode draws one order with one uniform number per step
   (inverse-transform sampling) and sums, over the steps, the gradient of the
   log-probability of each choice with respect to the weights,
   ``(phi(chosen) - E[phi]) / temperature``, where ``E[phi]`` is the mean
   feature vector of the eligible tasks under the policy.
3. **Update.** The order is evaluated. Its reward is the
   :class:`~dispatchatlas.solvers.learning.common.Improvement` of its
   schedule over the reference schedule, and the weights move by
   ``learning_rate * (reward - baseline)`` times the summed gradient. The
   baseline is the mean reward of the earlier episodes, zero before the first,
   the reinforcement comparison that Williams (1992) discusses.
4. **Greedy policy.** The greedy policy takes the eligible task of highest
   priority, the lowest index on ties. With the initial all-zero weights it
   yields the problem's topological order, whose schedule is evaluated first
   and is the reference. After every update the greedy order is evaluated when
   it differs from the last one evaluated.

Orders are evaluated as they are drawn: the schedule builder places each task
at its earliest start in its earliest-finishing mode, and in a permutation
problem it keeps each job's tasks together, jobs in the order they first
appear. One episode is one iteration.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.learning.common import Improvement, due_dates, policy_value
from dispatchatlas.solvers.priorities import min_durations
from dispatchatlas.solvers.scheduling import precedence_feasible
from dispatchatlas.solvers.search import Candidate, Search

__all__ = [
    "FEATURES",
    "LinearPolicy",
    "ReinforcePriorityPolicy",
    "sample",
    "softmax",
    "task_features",
]

FEATURES: tuple[str, ...] = ("bias", "duration", "release", "slack", "successors")
"""Names of the policy's features, in weight order."""


def task_features(problem: Problem) -> list[tuple[float, ...]]:
    """Return the feature vector of every task, in the order of :data:`FEATURES`.

    Every column except the constant is divided by its largest absolute value
    over the tasks. A column of zeros stays zero.
    """
    durations = min_durations(problem)
    releases = [task.release for task in problem.tasks]
    slacks = [
        due - release - duration
        for due, release, duration in zip(due_dates(problem), releases, durations, strict=True)
    ]
    successors = [float(len(items)) for items in problem.successors]
    columns = [_scaled(column) for column in (durations, releases, slacks, successors)]
    return [(1.0, *values) for values in zip(*columns, strict=True)]


def _scaled(values: Sequence[float]) -> list[float]:
    scale = max((abs(value) for value in values), default=0.0)
    if scale == 0.0:
        return [0.0] * len(values)
    return [value / scale for value in values]


def softmax(scores: Sequence[float]) -> list[float]:
    """Return the softmax of ``scores``, computed after subtracting the largest one."""
    ceiling = max(scores)
    exponentials = [math.exp(score - ceiling) for score in scores]
    total = math.fsum(exponentials)
    return [value / total for value in exponentials]


def sample(probabilities: Sequence[float], rng: random.Random) -> int:
    """Return an index drawn with the given probabilities, from one uniform number.

    The first index whose cumulative probability exceeds the uniform draw is
    returned, the last one when rounding leaves the total below the draw.
    """
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold < cumulative:
            return index
    return len(probabilities) - 1


class LinearPolicy:
    """A softmax policy over eligible tasks with linear priorities.

    Args:
        problem: The problem.
        temperature: Divisor of the priorities inside the softmax.

    Attributes:
        features: Feature vector of every task, from :func:`task_features`.
        weights: One weight per feature, initially zero.
    """

    def __init__(self, problem: Problem, temperature: float) -> None:
        self.problem = problem
        self.temperature = temperature
        self.features = task_features(problem)
        self.weights = [0.0] * len(FEATURES)

    def priorities(self) -> list[float]:
        """Return the priority of every task, its weighted feature sum rounded once."""
        weights = self.weights
        return [
            math.fsum(weight * value for weight, value in zip(weights, row, strict=True))
            for row in self.features
        ]

    def rollout(
        self, rng: random.Random, check: Callable[[], None] | None = None
    ) -> tuple[list[int], list[float]]:
        """Draw a precedence-feasible order and the gradient of its log-probability.

        Args:
            rng: Source of the one uniform number drawn per step.
            check: Called before every step, to stop at the time limit.

        Returns:
            The order and, per weight, the sum over the steps of
            ``(phi(chosen) - E[phi]) / temperature``.
        """
        problem = self.problem
        features = self.features
        temperature = self.temperature
        priorities = self.priorities()
        waiting = [len(before) for before in problem.predecessors]
        eligible = [task for task, count in enumerate(waiting) if count == 0]
        order: list[int] = []
        gradient = [0.0] * len(FEATURES)
        while eligible:
            if check is not None:
                check()
            candidates = sorted(eligible)
            probabilities = softmax([priorities[task] / temperature for task in candidates])
            chosen = candidates[sample(probabilities, rng)]
            total = math.fsum(probabilities)
            for dimension in range(len(FEATURES)):
                weighted = math.fsum(
                    probability * features[task][dimension]
                    for probability, task in zip(probabilities, candidates, strict=True)
                )
                expected = weighted / total
                gradient[dimension] += (features[chosen][dimension] - expected) / temperature
            order.append(chosen)
            eligible.remove(chosen)
            for following in problem.successors[chosen]:
                waiting[following] -= 1
                if waiting[following] == 0:
                    eligible.append(following)
        return order, gradient

    def greedy(self) -> list[int]:
        """Return the order of the greedy policy: highest priority first, lowest index on ties."""
        priorities = self.priorities()
        ranked = sorted(range(self.problem.size), key=lambda task: (-priorities[task], task))
        return precedence_feasible(self.problem, ranked)

    def step(self, gradient: Sequence[float], scale: float) -> None:
        """Add ``scale`` times ``gradient`` to the weights."""
        for dimension, component in enumerate(gradient):
            self.weights[dimension] += scale * component


@dataclass(frozen=True)
class ReinforcePriorityPolicy:
    """A linear priority rule learned by the REINFORCE policy gradient (Williams 1992).

    The policy draws task orders from a softmax over linear priorities of five
    task features, and each schedule's reward moves the weights along the
    gradient of the drawn order's log-probability. The module docstring gives
    the details.

    Attributes:
        learning_rate: Step size of the weight updates, positive.
        temperature: Divisor of the priorities inside the softmax, positive.
            Larger values make the draws more uniform.
    """

    learning_rate: float = 0.2
    temperature: float = 1.0

    info: ClassVar[SolverInfo] = SolverInfo(
        id="reinforce-priority-policy",
        name="REINFORCE priority policy",
        family="learning",
        summary="Draws task orders from a softmax over linear priorities of five task "
        "features and moves the weights along the REINFORCE gradient of each order's "
        "log-probability, scaled by its schedule's reward minus the running mean reward.",
        references=("williams1992simple", "kolisch1996serial"),
    )

    def __post_init__(self) -> None:
        if not (self.learning_rate > 0.0 and math.isfinite(self.learning_rate)):
            raise ValueError("learning_rate must be a positive finite number")
        if not (self.temperature > 0.0 and math.isfinite(self.temperature)):
            raise ValueError("temperature must be a positive finite number")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Train the policy within the stop criteria and return the best schedule built.

        The result extras hold ``policy_value``, the objective of the latest
        greedy-policy schedule evaluated, and the final weights, as
        ``weight_<feature>``.
        """
        search = Search(problem, stop, seed, self.info)
        policy = LinearPolicy(problem, self.temperature)
        latest: Candidate | None = None
        with search:
            greedy = policy.greedy()
            latest = search.assess(greedy)
            reward = Improvement.over(latest)
            baseline = 0.0
            episodes = 0
            rng = search.rng
            while True:
                search.next_iteration()
                order, gradient = policy.rollout(rng, search.check)
                value = reward(search.evaluate(order))
                policy.step(gradient, self.learning_rate * (value - baseline))
                episodes += 1
                baseline += (value - baseline) / episodes
                candidate = policy.greedy()
                if candidate != greedy:
                    greedy = candidate
                    latest = search.assess(greedy)
        extras: dict[str, float] = {"policy_value": policy_value(search, latest)}
        for name, weight in zip(FEATURES, policy.weights, strict=True):
            extras[f"weight_{name}"] = weight
        return search.result(extras=extras)
