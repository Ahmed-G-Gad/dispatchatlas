"""Single-solution walks that differ only in their acceptance rule.

Each method here repeats one step: draw a random neighbour of the current
solution (an insert, swap, or mode move, see
:mod:`~dispatchatlas.solvers.trajectory.space`), evaluate it, and let the rule
decide whether it becomes the current solution. One iteration is one step. Every
walk starts from a random precedence-feasible sequence.

=============================  =======================================================
Method                         A candidate becomes the current solution when
=============================  =======================================================
hill climbing                  it is strictly better (a restart ends a run of failures)
simulated annealing            it is no worse, or with probability ``exp(-Δ/T)``
threshold accepting            it is less than ``T`` worse
great deluge                   its cost is below the water level
record-to-record travel        its cost is below the best cost found plus a deviation
late acceptance hill climbing  it is no worse than the current solution, or than the
                               current solution ``L`` steps earlier
=============================  =======================================================

``Δ`` is how much worse the candidate is (see
:func:`~dispatchatlas.solvers.trajectory.space.worsening`), so no rule accepts a
larger deadline overrun and every rule accepts a smaller one. Simulated
annealing, threshold accepting, the great deluge, and record-to-record travel
measure their tolerances in units of the **move scale**: the mean worsening of
:data:`CALIBRATION_SAMPLES` random neighbours of the starting solution. That
makes their defaults independent of the objective's units.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import ClassVar, Protocol

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.trajectory.space import (
    SequenceSpace,
    Solution,
    move_scale,
    worsening,
)

__all__ = [
    "CALIBRATION_SAMPLES",
    "AcceptanceRule",
    "GreatDeluge",
    "HillClimbing",
    "Improvement",
    "LateAcceptance",
    "LateAcceptanceHillClimbing",
    "Metropolis",
    "RecordDeviation",
    "RecordToRecordTravel",
    "SimulatedAnnealing",
    "Threshold",
    "ThresholdAccepting",
    "WaterLevel",
    "walk",
]

CALIBRATION_SAMPLES = 20
"""Random neighbours of the starting solution evaluated to measure the move scale."""


class AcceptanceRule(Protocol):
    """Decides whether a candidate replaces the current solution of a walk."""

    def accept(self, candidate: Score, current: Score, rng: random.Random) -> bool:
        """Return whether ``candidate`` replaces ``current``."""
        ...

    def update(self, current: Score, *, accepted: bool) -> None:
        """Advance the rule's state after one decision.

        Args:
            current: The current solution's score after the decision.
            accepted: Whether the candidate was accepted.
        """
        ...


@dataclass(slots=True)
class Improvement:
    """Accept strictly better candidates only."""

    def accept(self, candidate: Score, current: Score, rng: random.Random) -> bool:
        """Return whether ``candidate`` is strictly better."""
        del rng
        return candidate < current

    def update(self, current: Score, *, accepted: bool) -> None:
        """Keep no state."""
        del current, accepted


@dataclass(slots=True)
class Metropolis:
    """The Metropolis rule of simulated annealing with geometric cooling.

    A candidate ``Δ`` worse is accepted with probability ``exp(-Δ/T)``, a
    candidate no worse always. After every ``epoch`` decisions the temperature
    is multiplied by ``cooling``.

    Attributes:
        temperature: The temperature ``T``.
        cooling: Factor applied to ``T`` after each epoch.
        epoch: Decisions per temperature.
        decisions: Decisions taken so far.
    """

    temperature: float
    cooling: float
    epoch: int
    decisions: int = 0

    def accept(self, candidate: Score, current: Score, rng: random.Random) -> bool:
        """Apply the Metropolis test."""
        delta = worsening(candidate, current)
        if delta <= 0.0:
            return True
        if math.isinf(delta) or self.temperature <= 0.0:
            return False
        return rng.random() < math.exp(-delta / self.temperature)

    def update(self, current: Score, *, accepted: bool) -> None:
        """Count the decision and cool at the end of an epoch."""
        del current, accepted
        self.decisions += 1
        if self.decisions % self.epoch == 0:
            self.temperature *= self.cooling


@dataclass(slots=True)
class Threshold:
    """The threshold-accepting rule of Dueck and Scheuer (1990).

    A candidate is accepted when it is less than ``threshold`` worse. After
    every ``epoch`` decisions the threshold is multiplied by ``cooling``.

    Attributes:
        threshold: The threshold ``T``.
        cooling: Factor applied to ``T`` after each epoch.
        epoch: Decisions per threshold value.
        decisions: Decisions taken so far.
    """

    threshold: float
    cooling: float
    epoch: int
    decisions: int = 0

    def accept(self, candidate: Score, current: Score, rng: random.Random) -> bool:
        """Accept a candidate less than the threshold worse."""
        del rng
        return worsening(candidate, current) < self.threshold

    def update(self, current: Score, *, accepted: bool) -> None:
        """Count the decision and lower the threshold at the end of an epoch."""
        del current, accepted
        self.decisions += 1
        if self.decisions % self.epoch == 0:
            self.threshold *= self.cooling


@dataclass(slots=True)
class WaterLevel:
    """The great-deluge rule of Dueck (1993), for minimisation.

    A candidate is accepted when its objective is below the water level, and
    the level drops by ``rain`` every time a candidate is accepted. Between
    candidates with different deadline overruns the smaller overrun wins.

    Attributes:
        level: The water level.
        rain: Drop of the level per accepted candidate.
    """

    level: float
    rain: float

    def accept(self, candidate: Score, current: Score, rng: random.Random) -> bool:
        """Accept a candidate below the level."""
        del rng
        delta = worsening(candidate, current)
        if math.isinf(delta):
            return delta < 0.0
        return candidate.value < self.level

    def update(self, current: Score, *, accepted: bool) -> None:
        """Lower the level after an acceptance."""
        del current
        if accepted:
            self.level -= self.rain


@dataclass(slots=True)
class RecordDeviation:
    """The record-to-record travel rule of Dueck (1993), for minimisation.

    A candidate is accepted when its objective is below the record, the best
    objective the walk has reached, plus a fixed ``deviation``. Between
    candidates with different deadline overruns the smaller overrun wins.

    Attributes:
        record: The best score the walk has reached.
        deviation: The allowed excess over the record.
    """

    record: Score
    deviation: float

    def accept(self, candidate: Score, current: Score, rng: random.Random) -> bool:
        """Accept a candidate within the deviation of the record."""
        del rng
        delta = worsening(candidate, current)
        if math.isinf(delta):
            return delta < 0.0
        return candidate.value < self.record.value + self.deviation

    def update(self, current: Score, *, accepted: bool) -> None:
        """Raise the record when the current solution beats it."""
        del accepted
        self.record = min(self.record, current)


@dataclass(slots=True)
class LateAcceptance:
    """The late-acceptance rule of Burke and Bykov (2017).

    At iteration ``i`` a candidate is accepted when it is no worse than the
    current solution or than entry ``i mod L`` of a history of ``L`` costs.
    That entry then records the current solution's cost. The history starts
    filled with the starting cost.

    Attributes:
        history: The last ``L`` current costs.
        iteration: Decisions taken so far.
    """

    history: list[Score]
    iteration: int = field(default=0)

    def accept(self, candidate: Score, current: Score, rng: random.Random) -> bool:
        """Compare with the current cost and the cost ``L`` steps back."""
        del rng
        return candidate <= current or candidate <= self.history[self.iteration % len(self.history)]

    def update(self, current: Score, *, accepted: bool) -> None:
        """Record the current cost in the history."""
        del accepted
        self.history[self.iteration % len(self.history)] = current
        self.iteration += 1


def walk(space: SequenceSpace, current: Solution, rule: AcceptanceRule) -> None:
    """Move from ``current`` through random neighbours until a stop criterion fires.

    Returns only when the space has no move; otherwise the search's
    :class:`~dispatchatlas.solvers.search.Exhausted` ends the walk.
    """
    search = space.search
    while True:
        search.next_iteration()
        step = space.neighbour(current)
        if step is None:
            return
        candidate = step[0]
        accepted = rule.accept(candidate.score, current.score, search.rng)
        if accepted:
            current = candidate
        rule.update(current.score, accepted=accepted)


def _walk_info(solver_id: str, name: str, summary: str, *references: str) -> SolverInfo:
    return SolverInfo(
        id=solver_id,
        name=name,
        family="local-search",
        summary=summary,
        references=references,
    )


def _check_fraction(name: str, value: float, *, closed: bool = False) -> None:
    upper_ok = value <= 1.0 if closed else value < 1.0
    if not (value > 0.0 and upper_ok):
        interval = "(0, 1]" if closed else "(0, 1)"
        raise ValueError(f"{name} must lie in {interval}, got {value!r}")


def _check_epoch(epoch: int | None) -> None:
    if epoch is not None and epoch < 1:
        raise ValueError(f"epoch must be at least 1, got {epoch!r}")


def _calibrated_start(search: Search) -> tuple[SequenceSpace, Solution, float]:
    """Return the space, a random starting solution, and the move scale around it."""
    space = SequenceSpace(search)
    current = space.random_start()
    scale = move_scale(space.deltas(current, CALIBRATION_SAMPLES))
    return space, current, scale


@dataclass(frozen=True)
class HillClimbing:
    """First-improvement hill climbing with random restarts.

    Each iteration evaluates one random neighbour and moves to it when it is
    strictly better. After ``restart_after`` consecutive neighbours fail to
    improve, the current solution is treated as a local optimum and the climb
    restarts from a new random sequence, which takes one iteration.

    Attributes:
        restart_after: Consecutive non-improving neighbours before a restart.
    """

    restart_after: int = 100

    info: ClassVar[SolverInfo] = _walk_info(
        "hill-climbing",
        "Hill climbing",
        "Moves to each sampled insert, swap, or mode neighbour that improves the "
        "current solution and restarts from a random sequence after a run of "
        "non-improving samples.",
        "hoos2004stochastic",
    )

    def __post_init__(self) -> None:
        if self.restart_after < 1:
            raise ValueError("restart_after must be at least 1")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Climb and restart until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            space = SequenceSpace(search)
            current = space.random_start()
            rule = Improvement()
            failures = 0
            while space.movable:
                search.next_iteration()
                if failures >= self.restart_after:
                    current, failures = space.random_start(), 0
                    continue
                step = space.neighbour(current)
                if step is None:  # pragma: no cover - a movable space always has a move
                    break
                if rule.accept(step[0].score, current.score, search.rng):
                    current, failures = step[0], 0
                else:
                    failures += 1
        return search.result()


@dataclass(frozen=True)
class SimulatedAnnealing:
    """Simulated annealing (Kirkpatrick, Gelatt and Vecchi 1983).

    A random neighbour ``Δ`` worse than the current solution is accepted with
    probability ``exp(-Δ/T)``. The starting temperature makes a move of the
    move scale acceptable with probability ``initial_acceptance``, that is
    ``T0 = scale / ln(1 / initial_acceptance)``, calibrating the start on the
    instance as in the experimental study of Johnson et al. (1989). The
    temperature is multiplied by ``cooling`` after every ``epoch`` moves.

    Attributes:
        initial_acceptance: Probability of accepting a move of the move scale at
            the start.
        cooling: Geometric cooling factor per epoch.
        epoch: Moves per temperature. Defaults to the number of sequence items.
    """

    initial_acceptance: float = 0.5
    cooling: float = 0.95
    epoch: int | None = None

    info: ClassVar[SolverInfo] = _walk_info(
        "simulated-annealing",
        "Simulated annealing",
        "Accepts a worse random neighbour with probability exp(-Δ/T), where T starts "
        "at a target acceptance rate of sampled worsening moves and cools geometrically.",
        "kirkpatrick1983optimization",
        "johnson1989optimization",
    )

    def __post_init__(self) -> None:
        _check_fraction("initial_acceptance", self.initial_acceptance)
        _check_fraction("cooling", self.cooling, closed=True)
        _check_epoch(self.epoch)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Anneal until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            space, current, scale = _calibrated_start(search)
            temperature = scale / math.log(1.0 / self.initial_acceptance)
            rule = Metropolis(temperature, self.cooling, self.epoch or space.items.count)
            walk(space, current, rule)
        return search.result()


@dataclass(frozen=True)
class ThresholdAccepting:
    """Threshold accepting (Dueck and Scheuer 1990).

    A random neighbour is accepted when it is less than the threshold worse
    than the current solution. The threshold starts at ``initial_threshold``
    move scales and is multiplied by ``cooling`` after every ``epoch`` moves.

    Attributes:
        initial_threshold: Starting threshold, in move scales.
        cooling: Geometric lowering factor per epoch.
        epoch: Moves per threshold value. Defaults to the number of sequence items.
    """

    initial_threshold: float = 1.0
    cooling: float = 0.95
    epoch: int | None = None

    info: ClassVar[SolverInfo] = _walk_info(
        "threshold-accepting",
        "Threshold accepting",
        "Accepts any random neighbour less than a threshold worse than the current "
        "solution and lowers the threshold geometrically.",
        "dueck1990threshold",
    )

    def __post_init__(self) -> None:
        if not self.initial_threshold >= 0.0:
            raise ValueError("initial_threshold must be non-negative")
        _check_fraction("cooling", self.cooling, closed=True)
        _check_epoch(self.epoch)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Walk under a falling threshold until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            space, current, scale = _calibrated_start(search)
            rule = Threshold(
                self.initial_threshold * scale, self.cooling, self.epoch or space.items.count
            )
            walk(space, current, rule)
        return search.result()


@dataclass(frozen=True)
class GreatDeluge:
    """The great deluge algorithm (Dueck 1993).

    A random neighbour is accepted when its objective is below the water level.
    The level starts ``initial_margin`` move scales above the starting
    objective and drops by ``rain_speed`` move scales at every acceptance, so
    the walk is squeezed towards ever better solutions.

    Attributes:
        initial_margin: Starting level above the starting objective, in move scales.
        rain_speed: Drop of the level per accepted move, in move scales.
    """

    initial_margin: float = 1.0
    rain_speed: float = 0.01

    info: ClassVar[SolverInfo] = _walk_info(
        "great-deluge",
        "Great deluge",
        "Accepts any random neighbour whose cost is below a water level that drops "
        "by a fixed amount at each accepted move.",
        "dueck1993new",
    )

    def __post_init__(self) -> None:
        if not self.initial_margin >= 0.0:
            raise ValueError("initial_margin must be non-negative")
        if not self.rain_speed > 0.0:
            raise ValueError("rain_speed must be positive")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Walk under a falling water level until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            space, current, scale = _calibrated_start(search)
            rule = WaterLevel(
                current.score.value + self.initial_margin * scale, self.rain_speed * scale
            )
            walk(space, current, rule)
        return search.result()


@dataclass(frozen=True)
class RecordToRecordTravel:
    """Record-to-record travel (Dueck 1993).

    A random neighbour is accepted when its objective is below the record, the
    best objective the walk has reached, plus a fixed deviation of
    ``deviation`` move scales.

    Attributes:
        deviation: Allowed excess over the record, in move scales.
    """

    deviation: float = 0.2

    info: ClassVar[SolverInfo] = _walk_info(
        "record-to-record-travel",
        "Record-to-record travel",
        "Accepts any random neighbour whose cost is below the best cost the walk has "
        "reached plus a fixed deviation.",
        "dueck1993new",
    )

    def __post_init__(self) -> None:
        if not self.deviation > 0.0:
            raise ValueError("deviation must be positive")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Travel within the deviation of the record until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            space, current, scale = _calibrated_start(search)
            walk(space, current, RecordDeviation(current.score, self.deviation * scale))
        return search.result()


@dataclass(frozen=True)
class LateAcceptanceHillClimbing:
    """Late acceptance hill climbing (Burke and Bykov 2017).

    A random neighbour is accepted when it is no worse than the current
    solution or than the current solution ``history`` iterations earlier.

    Attributes:
        history: Length ``L`` of the cost history.
    """

    history: int = 50

    info: ClassVar[SolverInfo] = _walk_info(
        "late-acceptance-hill-climbing",
        "Late acceptance hill climbing",
        "Accepts a random neighbour that is no worse than the current solution or "
        "than the current solution a fixed number of iterations earlier.",
        "burke2017late",
    )

    def __post_init__(self) -> None:
        if self.history < 1:
            raise ValueError("history must be at least 1")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Walk with a delayed acceptance test until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            space = SequenceSpace(search)
            current = space.random_start()
            walk(space, current, LateAcceptance([current.score] * self.history))
        return search.result()
