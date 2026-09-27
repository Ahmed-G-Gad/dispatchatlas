"""Tabular temporal-difference agents that learn which dispatching rule to apply.

Aydin and Oztemel (2000) train an agent that chooses, at each scheduling
decision, the priority rule that suits the current state of the shop, and
Zhang et al. (2011) learn choices among rule-based actions with SARSA. The
agents here learn such a choice on the problem being solved, by tabular
temporal-difference control:

- **Construction.** An episode builds one schedule a task at a time with the
  shared :class:`~dispatchatlas.solvers.scheduling.Builder`. At each step the
  agent observes a state and chooses a rule, the rule picks the next task among
  the eligible ones, whose predecessors are all placed, and the builder places
  it at its earliest start in its earliest-finishing mode. The placement order
  and modes are then evaluated, which rebuilds the same schedule.
- **Rules** (:data:`RULES`, in action order): shortest processing time,
  longest processing time, earliest due date, and minimum slack. Durations are
  the shortest over a task's modes and due dates come from
  :func:`~dispatchatlas.solvers.learning.common.due_dates`. The slack of an
  eligible task is its due date minus its earliest finish time in the partial
  schedule minus the longest path after it, so the slack rule reacts to the
  schedule built so far. Every rule breaks ties by the lowest task index.
- **States** (:func:`observe`): the fraction of tasks already placed, cut into
  ``stage_buckets`` equal bins, and the number of eligible tasks: one, two, or
  three or more.
- **Learning.** The agent chooses rules epsilon-greedily: a uniformly random
  rule with probability ``epsilon``, else the rule of largest action value, the
  first among ties. A decision followed by another earns no reward, and its
  value moves by ``learning_rate`` toward the discounted value of the next
  state: the largest action value there for Q-learning (Watkins and Dayan
  1992), the value of the rule the agent chooses there for SARSA (Rummery and
  Niranjan 1994; Sutton and Barto 2018). The last decision of an episode moves
  toward the episode's
  reward, the :class:`~dispatchatlas.solvers.learning.common.Improvement` of
  its schedule over the reference schedule.
- **Budget.** Action values start at zero, so the untrained greedy policy
  applies the first rule in every state. Its schedule is evaluated first and is
  the reference. The exploration rate falls linearly from ``exploration`` to
  zero as the budget is spent. After an episode that changes the greedy rule of
  some state, the greedy policy's schedule is evaluated too, unless it equals
  the last one evaluated. One episode is one iteration.

The rules interleave jobs freely, which a permutation problem forbids, so the
agents do not accept permutation problems.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Literal

from dispatchatlas.core.model import FEATURES, Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.learning.common import Improvement, due_dates, policy_value
from dispatchatlas.solvers.priorities import longest_tails, min_durations
from dispatchatlas.solvers.registry import SolverRegistry
from dispatchatlas.solvers.scheduling import Builder, Decoded, Decoder
from dispatchatlas.solvers.search import Candidate, Search

__all__ = [
    "INFOS",
    "READY_BUCKETS",
    "RULES",
    "ActionValues",
    "Agent",
    "DispatchRules",
    "Episode",
    "State",
    "Target",
    "TemporalDifferenceDispatching",
    "greedy_action",
    "observe",
    "register",
    "td_target",
]

type Target = Literal["q-learning", "sarsa"]
"""The bootstrap target of the value updates."""

type State = tuple[int, int]
"""A construction state: ``(stage bucket, eligible-set bucket)``."""

RULES: tuple[str, ...] = (
    "shortest-processing-time",
    "longest-processing-time",
    "earliest-due-date",
    "minimum-slack",
)
"""The dispatching rules the agents choose among, in action order."""

READY_BUCKETS = 3
"""Number of eligible-set buckets of a state: one, two, and three or more tasks."""

_MINIMUM_SLACK = RULES.index("minimum-slack")


def observe(placed: int, total: int, eligible: int, stage_buckets: int) -> State:
    """Return the state before a construction step.

    Args:
        placed: Tasks already placed.
        total: Tasks of the problem, at least one.
        eligible: Tasks eligible now, at least one.
        stage_buckets: Number of equal bins of the placed fraction.

    Returns:
        ``(min(placed * stage_buckets // total, stage_buckets - 1),
        min(eligible - 1, READY_BUCKETS - 1))``.
    """
    stage = min(placed * stage_buckets // total, stage_buckets - 1)
    return stage, min(eligible - 1, READY_BUCKETS - 1)


def greedy_action(values: Sequence[float]) -> int:
    """Return the action of largest value, the lowest index among ties."""
    best = 0
    for action in range(1, len(values)):
        if values[action] > values[best]:
            best = action
    return best


def td_target(target: Target, values: Sequence[float], action: int, discount: float) -> float:
    """Return the discounted value that a zero-reward transition moves toward.

    Args:
        target: ``"q-learning"`` uses the largest action value of the next
            state, whatever the agent does there. ``"sarsa"`` uses the value of
            the action the agent takes there.
        values: Action values of the next state.
        action: The action the agent takes in the next state.
        discount: Discount factor.

    Returns:
        ``discount * max(values)`` or ``discount * values[action]``.
    """
    if target == "q-learning":
        return discount * max(values)
    return discount * values[action]


class ActionValues:
    """The action values of the visited states, zero until updated.

    Args:
        actions: Number of actions of every state.
    """

    def __init__(self, actions: int) -> None:
        self.actions = actions
        self._rows: dict[State, list[float]] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def row(self, state: State) -> list[float]:
        """Return the live values of ``state``, adding a zero row on its first visit."""
        row = self._rows.get(state)
        if row is None:
            row = self._rows[state] = [0.0] * self.actions
        return row

    def get(self, state: State) -> tuple[float, ...]:
        """Return the values of ``state`` without adding a row."""
        row = self._rows.get(state)
        return (0.0,) * self.actions if row is None else tuple(row)

    def update(self, state: State, action: int, target: float, rate: float) -> bool:
        """Move one value toward ``target`` by the fraction ``rate`` of the difference.

        Returns:
            Whether the greedy action of ``state`` changed.
        """
        row = self.row(state)
        before = greedy_action(row)
        row[action] += rate * (target - row[action])
        return greedy_action(row) != before


class DispatchRules:
    """The rules of :data:`RULES`, applied to the eligible tasks of a partial schedule.

    Args:
        problem: The problem.
    """

    def __init__(self, problem: Problem) -> None:
        durations = min_durations(problem)
        dues = due_dates(problem)
        self._static: tuple[list[float], ...] = (durations, [-d for d in durations], dues)
        self._dues = dues
        self._after = [
            tail - duration
            for tail, duration in zip(longest_tails(problem), durations, strict=True)
        ]

    def choose(
        self, rule: int, eligible: Sequence[int], builder: Builder
    ) -> tuple[int, int, float]:
        """Return the task that ``rule`` picks among ``eligible``, with its mode and start.

        The mode is the task's earliest-finishing mode in the partial schedule
        of ``builder``, and the start its earliest start in that mode.
        """
        if rule == _MINIMUM_SLACK:
            options: list[tuple[float, int, int, float]] = []
            for task in eligible:
                start, finish, mode = builder.best(task)
                options.append((self._dues[task] - finish - self._after[task], task, mode, start))
            _, task, mode, start = min(options)
            return task, mode, start
        keys = self._static[rule]
        task = min(eligible, key=lambda item: (keys[item], item))
        start, _, mode = builder.best(task)
        return task, mode, start


@dataclass(frozen=True, slots=True)
class Episode:
    """One schedule built by an agent.

    Attributes:
        schedule: The schedule, as the agent's builder placed it.
        decisions: The state and the chosen rule of every construction step.
        changed: Whether a value update during the construction changed the
            greedy rule of some state.
    """

    schedule: Decoded
    decisions: tuple[tuple[State, int], ...]
    changed: bool = False

    @property
    def sequence(self) -> tuple[int, ...]:
        """Task indices in placement order."""
        return self.schedule.sequence

    @property
    def modes(self) -> tuple[int, ...]:
        """Mode index of every task."""
        return self.schedule.modes

    @property
    def starts(self) -> tuple[float, ...]:
        """Start time of every task."""
        return self.schedule.starts


class Agent:
    """The action values, the rules, and the construction loop of one run.

    Args:
        problem: The problem.
        decoder: The decoder of the run, whose builder places the tasks.
        target: The bootstrap target of the value updates.
        learning_rate: Step size of the value updates.
        discount: Discount of the next state's value.
        stage_buckets: Number of progress bins of a state.
        check: Called before every construction step. The solver passes
            :meth:`~dispatchatlas.solvers.search.Search.check`, which stops a
            construction that runs past the time limit.

    Attributes:
        values: The learned action values.
    """

    def __init__(
        self,
        problem: Problem,
        decoder: Decoder,
        *,
        target: Target,
        learning_rate: float,
        discount: float,
        stage_buckets: int,
        check: Callable[[], None] | None = None,
    ) -> None:
        self.problem = problem
        self.decoder = decoder
        self.target = target
        self.learning_rate = learning_rate
        self.discount = discount
        self.stage_buckets = stage_buckets
        self.values = ActionValues(len(RULES))
        self.rules = DispatchRules(problem)
        self._check = check

    def episode(self, rng: random.Random, epsilon: float) -> Episode:
        """Build a schedule with epsilon-greedy rule choices, updating values on the way.

        At every step the agent draws one uniform number, and a rule index when
        it explores. The value of the last decision is left for :meth:`finish`,
        since the episode's reward is known only once its schedule is evaluated.
        """
        return self._build(rng, epsilon)

    def greedy(self) -> Episode:
        """Build the schedule of the greedy policy, without learning or random draws."""
        return self._build(None, 0.0)

    def finish(self, episode: Episode, reward: float) -> bool:
        """Move the value of the episode's last decision toward its reward.

        Returns:
            Whether the greedy rule of that decision's state changed.
        """
        state, rule = episode.decisions[-1]
        return self.values.update(state, rule, reward, self.learning_rate)

    def _build(self, rng: random.Random | None, epsilon: float) -> Episode:
        problem = self.problem
        builder = self.decoder.builder()
        waiting = [len(before) for before in problem.predecessors]
        eligible = [task for task, count in enumerate(waiting) if count == 0]
        decisions: list[tuple[State, int]] = []
        changed = False
        while eligible:
            if self._check is not None:
                self._check()
            state = observe(len(builder.sequence), problem.size, len(eligible), self.stage_buckets)
            if rng is None:
                rule = greedy_action(self.values.get(state))
            else:
                values = self.values.row(state)
                if decisions and self.target == "q-learning":
                    changed |= self._bootstrap(decisions[-1], values, greedy_action(values))
                explore = rng.random() < epsilon
                rule = rng.randrange(len(RULES)) if explore else greedy_action(values)
                if decisions and self.target == "sarsa":
                    changed |= self._bootstrap(decisions[-1], values, rule)
            task, mode, start = self.rules.choose(rule, eligible, builder)
            builder.place(task, mode, start)
            eligible.remove(task)
            for following in problem.successors[task]:
                waiting[following] -= 1
                if waiting[following] == 0:
                    eligible.append(following)
            decisions.append((state, rule))
        return Episode(builder.decoded(), tuple(decisions), changed)

    def _bootstrap(self, decision: tuple[State, int], values: Sequence[float], rule: int) -> bool:
        """Update a decision toward the value of the state that followed it."""
        state, taken = decision
        target = td_target(self.target, values, rule, self.discount)
        return self.values.update(state, taken, target, self.learning_rate)


@dataclass(frozen=True)
class TemporalDifferenceDispatching:
    """A tabular temporal-difference agent that learns which dispatching rule to apply.

    Episodes build schedules in which one of four dispatching rules picks each
    next task, and the rewards of those schedules teach the agent which rule to
    apply in each construction state. ``target`` selects the registered solver:
    ``q-learning-dispatching`` or ``sarsa-dispatching``. The module docstring
    gives the details.

    Attributes:
        target: ``"q-learning"`` bootstraps from the largest action value of
            the next state, ``"sarsa"`` from the value of the rule chosen there.
        learning_rate: Step size of the value updates, in ``(0, 1]``.
        discount: Discount of the next state's value, in ``[0, 1]``.
        exploration: Exploration rate at the start of the run, in ``[0, 1]``.
            It falls linearly to zero as the budget is spent.
        stage_buckets: Number of equal bins of the placed fraction in a state,
            at least 1.
    """

    target: Target = "q-learning"
    learning_rate: float = 0.5
    discount: float = 0.9
    exploration: float = 0.3
    stage_buckets: int = 4

    def __post_init__(self) -> None:
        if self.target not in ("q-learning", "sarsa"):
            raise ValueError("target must be 'q-learning' or 'sarsa'")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in (0, 1]")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must lie in [0, 1]")
        if not 0.0 <= self.exploration <= 1.0:
            raise ValueError("exploration must lie in [0, 1]")
        if self.stage_buckets < 1:
            raise ValueError("stage_buckets must be at least 1")

    @property
    def info(self) -> SolverInfo:
        """Metadata of the solver that ``target`` selects."""
        return INFOS[self.target]

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Learn rule choices within the stop criteria and return the best schedule built.

        The result extras hold ``states``, the number of states visited, and
        ``policy_value``, the objective of the latest greedy-policy schedule
        evaluated.
        """
        search = Search(problem, stop, seed, self.info)
        agent = Agent(
            problem,
            search.evaluator.decoder,
            target=self.target,
            learning_rate=self.learning_rate,
            discount=self.discount,
            stage_buckets=self.stage_buckets,
            check=search.check,
        )
        latest: Candidate | None = None
        with search:
            greedy = agent.greedy()
            latest = search.submit(greedy.schedule)
            reward = Improvement.over(latest)
            rng = search.rng
            while True:
                epsilon = self.exploration * (1.0 - search.progress())
                search.next_iteration()
                episode = agent.episode(rng, epsilon)
                score = search.submit(episode.schedule).score
                if agent.finish(episode, reward(score)) or episode.changed:
                    candidate = agent.greedy()
                    if (candidate.sequence, candidate.modes) != (greedy.sequence, greedy.modes):
                        greedy = candidate
                        latest = search.submit(greedy.schedule)
        return search.result(
            extras={"states": len(agent.values), "policy_value": policy_value(search, latest)}
        )


_SUPPORTS = frozenset(FEATURES) - {"permutation"}

_PARAMETERS = {
    item.name: item.default
    for item in dataclasses.fields(TemporalDifferenceDispatching)
    if item.name != "target"
}

INFOS: dict[Target, SolverInfo] = {
    "q-learning": SolverInfo(
        id="q-learning-dispatching",
        name="Q-learning dispatching",
        family="learning",
        summary="Learns which of four dispatching rules should pick the next task in each "
        "coarse state of the construction by one-step Q-learning, bootstrapping from the "
        "best action value of the next state, and builds schedules with epsilon-greedy "
        "rule choices.",
        references=("watkins1992qlearning", "aydin2000dynamic", "panwalkar1977survey"),
        supports=_SUPPORTS,
        parameters=_PARAMETERS,
    ),
    "sarsa": SolverInfo(
        id="sarsa-dispatching",
        name="SARSA dispatching",
        family="learning",
        summary="Learns which of four dispatching rules should pick the next task in each "
        "coarse state of the construction by SARSA, bootstrapping from the value of the "
        "rule chosen in the next state, and builds schedules with epsilon-greedy rule "
        "choices.",
        references=(
            "rummery1994online",
            "sutton2018reinforcement",
            "aydin2000dynamic",
            "zhang2011semiconductor",
            "panwalkar1977survey",
        ),
        supports=_SUPPORTS,
        parameters=_PARAMETERS,
    ),
}
"""Metadata of the two registered agents, by target."""


def register(registry: SolverRegistry) -> None:
    """Add ``q-learning-dispatching`` and ``sarsa-dispatching`` to ``registry``."""
    for target, info in INFOS.items():
        registry.register(partial(TemporalDifferenceDispatching, target), info)
