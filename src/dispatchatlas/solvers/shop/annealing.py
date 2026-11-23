"""Simulated annealing over random keys or job sequences, a job-shop search family.

:class:`ShopAnnealing` is simulated annealing (Kirkpatrick, Gelatt and Vecchi
1983, doi:10.1126/science.220.4598.671) written once against
:class:`~dispatchatlas.solvers.shop.representation.Representation`: the walk, the
calibration of its temperatures, the temperature schedule and the acceptance test
are the same code on random keys and on job sequences, and only the move differs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.shop.decoding import BuilderName
from dispatchatlas.solvers.shop.family import (
    RepresentationName,
    check_family,
    check_moves,
    make_representation,
)
from dispatchatlas.solvers.shop.instance import JOB_SHOP_FEATURES, JobShop
from dispatchatlas.solvers.shop.observers import TrajectoryObserver
from dispatchatlas.solvers.shop.representation import KeyMove, SequenceMove
from dispatchatlas.solvers.shop.run import ShopRun
from dispatchatlas.solvers.trajectory.space import move_scale, worsening

if TYPE_CHECKING:
    from dispatchatlas.solvers.shop.run import Evaluated

__all__ = ["ShopAnnealing"]


@dataclass(frozen=True)
class ShopAnnealing:
    """Simulated annealing on random keys or job sequences, with calibrated geometric cooling.

    **The walk.** The run's first evaluation is a random solution, which
    becomes the current solution. Every iteration then evaluates one move of the
    current solution, with the current solution as its parent, and applies the
    Metropolis test: a move ``Δ`` worse than the current solution
    (:func:`~dispatchatlas.solvers.trajectory.space.worsening`) becomes the
    current solution when a uniform number drawn from the control stream lies
    below ``exp(-Δ / T)``, so a move that does not worsen always does. The
    number is drawn at every iteration, whether or not the move worsens, so the
    runs on the two representations read the same number at the same iteration.

    **Moves.** A move of random keys redraws one key (``key_move="redraw"``) or
    perturbs two keys by normal steps of standard deviation ``key_step``,
    reflected into [0, 1] (``"perturb"``); a move of a job sequence is one
    insert or one swap (``sequence_move``). Each key move is matched to a
    sequence move: the redrawn key to the insert, and the two perturbed keys to
    the swap, for a ``key_step`` that
    :func:`~dispatchatlas.solvers.shop.distance.calibrate_step` sets so that
    their median distance is the swap's. The defaults are the redrawn key and
    the insert, a pair without a step to set. ``sequence_move="mixed"``, a swap
    or an insert with probability one half each, is accepted, but no key move
    is matched to it.

    **Temperatures.** After its first evaluation the run calibrates, inside
    :meth:`~dispatchatlas.solvers.shop.run.ShopRun.calibration`: it draws
    ``calibration_samples`` random solutions, all of them before any move, so
    that both representations draw the same ones, and then evaluates each of
    them and one move of it. These ``2 * calibration_samples`` evaluations count
    toward the budget, as many on both representations, and the observer does
    not see them. The move scale ``s`` is the mean worsening of the calibration
    moves that worsen their solution
    (:func:`~dispatchatlas.solvers.trajectory.space.move_scale`, which gives the
    mean size of the improving moves when none worsens, and 0 when none changes
    the objective). The temperatures at the start and at the end of the walk
    accept a move of the scale with the two target probabilities::

        T0 = s / ln(1 / initial_acceptance)
        Tf = s / ln(1 / final_acceptance)

    Setting the start by a target probability of accepting a move of the scale
    is the rule of
    :class:`~dispatchatlas.solvers.trajectory.acceptance.SimulatedAnnealing`,
    after the experimental study of Johnson et al. (1989), which measures its
    scale on moves of its starting solution; the end follows the same rule. The
    scale is measured at random solutions only, so the share of moves the walk
    accepts later, at better solutions, is measured (``acceptance_share``) and
    not imposed. When ``s`` is 0, both temperatures are 0 and the walk accepts
    only the moves that do not worsen.

    **Schedule.** The temperature falls geometrically from T0, when the
    calibration ends, to Tf, at the end of the run: ``T = T0 * (Tf / T0) ** p``.
    After the evaluation of each move, ``p = (q - q0) / (1 - q0)``, where ``q``
    is :meth:`~dispatchatlas.solvers.search.Search.progress` with
    ``deterministic=True`` and ``q0`` its value when the calibration ended.
    Under an evaluation limit ``q`` counts evaluations, so the temperature of an
    evaluation follows from its number alone, a time limit never changes it,
    and a run that ends at its evaluation limit decides its last move at Tf.

    A job shop of one job has a single job sequence, which the run evaluates
    before it stops, without calibration or walk.

    Besides the measures of the run and of its
    :class:`~dispatchatlas.solvers.shop.observers.TrajectoryObserver`, which sees
    the first evaluation and every move, the extras of the result hold
    ``initial_temperature`` and ``final_temperature``, T0 and Tf, once the
    calibration ends, and, once a move has been decided, ``last_temperature``,
    the temperature of the last decision, and ``acceptance_share``, the share of
    the decided moves that were accepted.

    Attributes:
        representation: ``"keys"`` or ``"sequence"``
            (:data:`~dispatchatlas.solvers.shop.family.REPRESENTATIONS`).
        builder: The schedule builder, one of
            :data:`~dispatchatlas.solvers.shop.decoding.BUILDERS`.
        initial_acceptance: The probability of accepting a move of the scale at
            the start of the walk, in (0, 1).
        final_acceptance: The probability of accepting a move of the scale at
            the end of the walk, in (0, ``initial_acceptance``].
        calibration_samples: How many random solutions the calibration draws and
            moves once each, at least 1.
        key_move: The move of random keys, ``"redraw"`` or ``"perturb"``.
        key_step: The standard deviation of a key perturbation, positive and
            finite. Only ``"perturb"`` reads it.
        sequence_move: The move of job sequences, ``"insert"``, ``"swap"`` or
            ``"mixed"``.
        distance_every: The observer measures the phenotype distance of the first
            move and of every ``distance_every``-th one after it, an integer of at
            least 1. Its other measures count every evaluation, and no decision of
            the walk depends on it.
    """

    representation: RepresentationName = "sequence"
    builder: BuilderName = "active"
    initial_acceptance: float = 0.5
    final_acceptance: float = 0.01
    calibration_samples: int = 20
    key_move: KeyMove = "redraw"
    key_step: float = 0.1
    sequence_move: SequenceMove = "insert"
    distance_every: int = 1

    info: ClassVar[SolverInfo] = SolverInfo(
        id="shop-simulated-annealing",
        name="Job-shop simulated annealing",
        family="local-search",
        summary="Accepts a worse random move of random keys or of a job sequence with "
        "probability exp(-Δ/T), where T falls geometrically over the budget between "
        "temperatures set by target acceptance probabilities of the mean worsening of "
        "moves of random solutions.",
        references=(
            "kirkpatrick1983optimization",
            "johnson1989optimization",
            "bean1994genetic",
            "bierwirth1995generalized",
        ),
        supports=JOB_SHOP_FEATURES,
        shape="job-shop",
    )

    def __post_init__(self) -> None:
        check_family(self.representation, self.builder, self.distance_every)
        check_moves(self.key_move, self.key_step, self.sequence_move)
        if not 0.0 < self.initial_acceptance < 1.0:
            raise ValueError(
                f"initial_acceptance must lie in (0, 1), got {self.initial_acceptance!r}"
            )
        if not 0.0 < self.final_acceptance <= self.initial_acceptance:
            raise ValueError(
                "final_acceptance must lie in (0, initial_acceptance], got "
                f"{self.final_acceptance!r}"
            )
        samples = self.calibration_samples
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError(
                f"calibration_samples must be an integer of at least 1, got {samples!r}"
            )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Calibrate the temperatures, then anneal until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        shop = JobShop.from_problem(problem)
        representation = make_representation(
            shop,
            self.representation,
            key_move=self.key_move,
            key_step=self.key_step,
            sequence_move=self.sequence_move,
        )
        observer = TrajectoryObserver(shop, self.distance_every)
        run = ShopRun(search, shop, self.builder, representation, observer)
        with search:
            current = run.evaluate(representation.initial(run.operators))
            if shop.jobs > 1:
                initial, final = self._calibrate(run)
                self._walk(run, current, initial, final)
        run.finish()
        return search.result()

    def _calibrate[G](self, run: ShopRun[G]) -> tuple[float, float]:
        """Return T0 and Tf from moves of random solutions, and record them in the extras."""
        representation, operators = run.representation, run.operators
        deltas: list[float] = []
        with run.calibration():
            # Every solution is drawn before any move, so both representations draw the same ones.
            solutions = [representation.initial(operators) for _ in range(self.calibration_samples)]
            for genotype in solutions:
                before = run.evaluate(genotype)
                after = run.evaluate(representation.move(genotype, operators), parent=before)
                deltas.append(worsening(after.score, before.score))
        scale = move_scale(deltas)
        initial = scale / math.log(1.0 / self.initial_acceptance)
        final = scale / math.log(1.0 / self.final_acceptance)
        run.search.extras["initial_temperature"] = initial
        run.search.extras["final_temperature"] = final
        return initial, final

    def _walk[G](
        self, run: ShopRun[G], current: Evaluated[G], initial: float, final: float
    ) -> None:
        """Move from ``current`` by the Metropolis test until the search is exhausted."""
        search, representation, operators, control = (
            run.search,
            run.representation,
            run.operators,
            run.control,
        )
        ratio = final / initial if initial > 0.0 else 0.0
        start = search.progress(deterministic=True)
        span = 1.0 - start
        temperature = initial
        decided = accepted = 0
        try:
            while True:
                search.next_iteration()
                moved = representation.move(current.genotype, operators)
                trial = run.evaluate(moved, parent=current)
                fraction = (search.progress(deterministic=True) - start) / span if span > 0 else 1.0
                temperature = initial * ratio**fraction if initial > 0.0 else 0.0
                # Drawn whatever the move, so that both representations stay in step.
                draw = control.random()
                delta = worsening(trial.score, current.score)
                if delta <= 0.0 or (temperature > 0.0 and draw < math.exp(-delta / temperature)):
                    current = trial
                    accepted += 1
                decided += 1
        finally:
            if decided:
                search.extras["last_temperature"] = temperature
                search.extras["acceptance_share"] = accepted / decided
