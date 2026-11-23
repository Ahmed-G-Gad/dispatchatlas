"""A particle swarm over random keys or job sequences, a job-shop search family.

:class:`ShopParticleSwarm` runs one swarm on either representation. What is written
once, and so is the same code on random keys and on job sequences, is the swarm and
its size, the topology and the way each particle finds its leader, the memory of
personal bests and of the best a leader holds, the order in which the particles
move, and the stop. What cannot be written once is the **update rule**, which turns
a particle's position, its personal best and its leader's best into the particle's
next position: velocity has no natural form for a permutation. Random keys therefore
move by the velocity update of a constriction swarm, and job sequences by the
discrete swarm of Pan, Tasgetiren and Liang (2007), two different algorithms for two
different spaces. Two runs of the family with one seed start from the same swarm and
then differ in their update rules, which draw different numbers, so this pair is a
weaker match than the other families of the package, whose arms differ in the
operators of the representation only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from dispatchatlas.core.model import Problem, Scalar
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
from dispatchatlas.solvers.shop.representation import (
    Keys,
    Representation,
    SequenceMove,
    reflect,
)
from dispatchatlas.solvers.shop.run import ShopRun

if TYPE_CHECKING:
    from dispatchatlas.solvers.shop.run import Evaluated

__all__ = ["TOPOLOGIES", "ShopParticleSwarm", "Topology", "constriction_coefficient"]

type Topology = Literal["global", "ring"]
"""How the particles of a swarm are connected, one of :data:`TOPOLOGIES`."""

TOPOLOGIES: tuple[Topology, ...] = ("global", "ring")
"""The topologies :class:`ShopParticleSwarm` takes: the whole swarm, or a ring of neighbours."""


def constriction_coefficient(phi: float) -> float:
    """Return the constriction coefficient for a total acceleration ``phi``.

    The velocity of a constriction swarm is multiplied by
    ``2 / (phi - 2 + sqrt(phi * phi - 4 * phi))``, which is the form
    ``2 / |2 - phi - sqrt(phi * phi - 4 * phi)|`` of Clerc and Kennedy (2002) for
    ``phi`` above 4, where ``phi`` is the sum of the swarm's two acceleration
    coefficients. It falls from 1 at ``phi = 4`` to 0.72984 at ``phi = 4.1``, and
    keeps falling as ``phi`` grows.

    Args:
        phi: The sum of the two acceleration coefficients, finite and above 4.

    Returns:
        The coefficient, in (0, 1).

    Raises:
        ValueError: If ``phi`` is not finite and above 4.
    """
    if not (math.isfinite(phi) and phi > 4.0):
        raise ValueError(
            f"the constriction coefficient needs a total acceleration above 4, got {phi!r}"
        )
    return 2.0 / (phi - 2.0 + math.sqrt(phi * phi - 4.0 * phi))


class _Update[G](Protocol):
    """The update rule of a swarm: the one part of it that depends on the representation."""

    def update(self, particle: int, position: G, own: G, leader: G, rng: np.random.Generator) -> G:
        """Return the next position of ``particle``, a genotype that its evaluation takes.

        Args:
            particle: The index of the particle.
            position: Its current position.
            own: Its personal best.
            leader: The personal best of its leader.
            rng: The operator stream.
        """
        ...

    def measures(self) -> dict[str, Scalar]:
        """Return what the update rule measured, to add to the extras of the result."""
        ...


class _ConstrictionUpdate:
    """The update rule of the keys: a constriction swarm in [0, 1], reflected at the bounds."""

    def __init__(self, size: int, keys: int, cognitive: float, social: float) -> None:
        self.cognitive = cognitive
        self.social = social
        self.constriction = constriction_coefficient(cognitive + social)
        self.velocity = np.zeros((size, keys))
        self.keys = keys
        self.updates = 0
        self.reflected = 0

    def update(
        self, particle: int, position: Keys, own: Keys, leader: Keys, rng: np.random.Generator
    ) -> Keys:
        """Return the position after one step of the velocity, folded into [0, 1]."""
        draws = rng.random((2, self.keys))
        velocity: Keys = self.constriction * (
            self.velocity[particle]
            + self.cognitive * draws[0] * (own - position)
            + self.social * draws[1] * (leader - position)
        )
        moved: Keys = position + velocity
        inside = (moved >= 0.0) & (moved <= 1.0)
        outside = self.keys - int(np.count_nonzero(inside))
        if outside:
            # A fold reverses the direction a key travels in, so an odd number of folds
            # negates its velocity: the slope of the folded line is (-1) ** floor(moved).
            turned = ~inside & (np.floor(moved) % 2.0 == 1.0)
            velocity = np.asarray(np.where(turned, -velocity, velocity), dtype=np.float64)
            moved = reflect(moved)
        self.velocity[particle] = velocity
        self.updates += 1
        self.reflected += outside
        return moved

    def measures(self) -> dict[str, Scalar]:
        """Return the share of the keys that the folds moved, once a particle has moved."""
        if not self.updates:
            return {}
        return {"reflected_share": self.reflected / (self.updates * self.keys)}


class _DiscreteUpdate:
    """The update rule of the job sequences: a move, then crossovers with two bests."""

    def __init__(
        self,
        representation: Representation[NDArray[np.intp]],
        move_probability: float,
        personal_probability: float,
        leader_probability: float,
        inheritance: float,
    ) -> None:
        self.representation = representation
        self.move_probability = move_probability
        self.personal_probability = personal_probability
        self.leader_probability = leader_probability
        self.inheritance = inheritance

    def update(
        self,
        particle: int,  # noqa: ARG002 - the discrete swarm keeps nothing for a particle
        position: NDArray[np.intp],
        own: NDArray[np.intp],
        leader: NDArray[np.intp],
        rng: np.random.Generator,
    ) -> NDArray[np.intp]:
        """Return the position after the move and the two crossovers that the numbers decide."""
        # The three numbers are drawn whatever they decide, before the operations draw theirs.
        coins = rng.random(3).tolist()
        representation, inheritance = self.representation, self.inheritance
        moved = position
        if coins[0] < self.move_probability:
            moved = representation.move(moved, rng)
        if coins[1] < self.personal_probability:
            moved = representation.recombine(own, moved, rng, inheritance)
        if coins[2] < self.leader_probability:
            moved = representation.recombine(leader, moved, rng, inheritance)
        return moved

    def measures(self) -> dict[str, Scalar]:
        """Return nothing: the update rule measures nothing of its own."""
        return {}


class _Memory[G]:
    """The positions and personal bests of a swarm, and the leader of each particle.

    A new position replaces its particle's personal best when its score is no
    worse, and the personal best is then stamped with the next number of a clock, so
    that among equal bests the most recent one is the one that leads. The leader of
    a particle is the best personal best of its neighbourhood, by the score and then
    by the stamp, whatever the topology.
    """

    def __init__(self, topology: Topology) -> None:
        self.topology = topology
        self.current: list[Evaluated[G]] = []
        self.best: list[Evaluated[G]] = []
        self.stamps: list[int] = []
        self._clock = 0
        self._leader = 0

    def begin(self, evaluated: Evaluated[G]) -> None:
        """Add the next particle, which holds ``evaluated`` as its position and its best."""
        self.current.append(evaluated)
        self.best.append(evaluated)
        self._clock += 1
        self.stamps.append(self._clock)
        if evaluated.score <= self.best[self._leader].score:
            self._leader = len(self.best) - 1

    def record(self, particle: int, evaluated: Evaluated[G]) -> None:
        """Move ``particle`` to ``evaluated``, and keep it as its best if it is no worse."""
        self.current[particle] = evaluated
        if evaluated.score <= self.best[particle].score:
            self.best[particle] = evaluated
            self._clock += 1
            self.stamps[particle] = self._clock
            if evaluated.score <= self.best[self._leader].score:
                self._leader = particle

    def leader_of(self, particle: int) -> int:
        """Return the particle whose personal best leads ``particle``.

        With the global topology it is the particle that holds the swarm's best,
        which the memory keeps up to date by the same rule: a new personal best
        that is no worse than the swarm's best leads. With the ring topology it is
        the best of ``particle`` and its two neighbours.
        """
        if self.topology == "global":
            return self._leader
        size = len(self.best)
        best, stamps = self.best, self.stamps
        leader = particle
        for other in ((particle - 1) % size, (particle + 1) % size):
            if (best[other].score, -stamps[other]) < (best[leader].score, -stamps[leader]):
                leader = other
        return leader


@dataclass(frozen=True)
class ShopParticleSwarm:
    """A particle swarm on random keys or job sequences, whose update rules differ by necessity.

    **What both representations share.** The swarm, its topology, its memory and its
    loop are the same code in both: only the update rule, which gives a particle its
    next position, depends on the representation, and it differs by necessity (see
    below), so the pair is a weaker match than the other families of the package.

    **The swarm.** ``swarm_size`` particles are drawn first: the run's operator
    stream gives ``swarm_size`` random solutions, one uniform number per slot each
    (:meth:`~dispatchatlas.solvers.shop.representation.Representation.initial`),
    before it gives anything else, so both representations draw the same numbers and
    start from the same job sequences. The run evaluates them in index order, so its
    first evaluation is the first particle's, with the same schedule on both
    representations. Every iteration then moves each particle once, in index order,
    and evaluates it at its new position, so an iteration is ``swarm_size``
    evaluations, and a run that ends at its evaluation limit has made exactly that
    many, wherever in an iteration it stops.

    **Memory, topology and leaders.** A particle remembers its personal best, the best
    position it has held. A new position replaces the personal best when its score is
    no worse, so on a tie the newer position becomes the personal best. A particle's
    leader is the particle whose personal best is the best of its neighbourhood: the
    whole swarm under the ``"global"`` topology, or the particle and its two
    neighbours in index order, the first and the last being neighbours, under the
    ``"ring"`` topology. Among equal bests the one set most recently leads. The memory
    is updated after each evaluation, and a particle reads it as the evaluations
    before its own, those of its iteration included, have left it. None of this draws a
    number. The update rule takes the particle's position, its personal best, and the
    personal best of its leader (the leader's best).

    **Keys: a constriction swarm**, after the standard swarm of Bratton and Kennedy
    (2007) and the constriction coefficient of Clerc and Kennedy (2002). A position is a
    vector of keys in [0, 1], and the velocity of a particle starts at zero. An update
    draws two arrays of uniform numbers from the operator stream, ``r1`` and then
    ``r2``, one number per key, and sets, key by key::

        v = χ (v + c1 r1 (p - x) + c2 r2 (l - x))
        x = x + v

    where ``x`` is the position, ``p`` the particle's personal best, ``l`` its
    leader's best, ``c1`` is ``cognitive``, ``c2`` is ``social``, and ``χ`` is
    :func:`constriction_coefficient` of ``c1 + c2``, 0.72984 for the defaults. The new
    position is folded into [0, 1] by reflection at the bounds
    (:func:`~dispatchatlas.solvers.shop.representation.reflect`), and the velocity of
    a key that the fold turned changes its sign once for each fold, so that a key
    that bounces off a bound leaves it as fast as it arrived: a key moved to 1.3 is
    placed at 0.7 with its velocity reversed, one moved to 2.25 is placed at 0.25
    after two folds with its velocity as it was, and a key inside [0, 1] and its
    velocity are left alone. The particle is evaluated at the folded position, so no
    fold skips an evaluation. A particle that is its own leader and its own personal
    best, with no velocity, does not move, and its evaluation repeats its last
    schedule. The extras hold ``reflected_share``, the share of the keys that the folds
    moved.

    **Job sequences: the discrete swarm** of Pan, Tasgetiren and Liang (2007), whose
    Eq. 10 for the permutation flow shop updates a position in three steps. Here an
    update applies up to three operations to the particle's position, in this order,
    each with a probability of its own (the chapter's ``w``, ``c1`` and ``c2``): one
    move of the kind ``sequence_move`` (an insert in the chapter, which calls this step
    the particle's velocity), with probability ``move_probability``; then the crossover
    of the representation, the precedence preservative crossover, with the particle's
    personal best, with probability ``personal_probability``; then the same crossover
    with its leader's best, with probability ``leader_probability``. In a crossover the
    best is the first parent and the particle's position the second, as the
    representation takes the fitter parent first, so each gene of the child comes from
    the best with probability ``inheritance``. The three numbers that decide whether the
    operations apply are drawn first, from the operator stream, at every update,
    whatever they decide; an operation that applies draws its own numbers after them.
    The chapter's seeding of the swarm by the NEH heuristic and its local search on the
    global best are not part of the family: every swarm starts from random solutions. A
    position that no operation changed is evaluated again.

    **Why the update rules differ.** Velocity has no natural form for a permutation:
    the positions of a particle are not real numbers, so no sum of a position and a
    velocity is a job sequence. The keys swarm is the standard swarm on the real
    vector that keys are, and the job-sequence swarm is a different algorithm that
    keeps the roles of inertia, personal best and leader. Their parameters differ
    (accelerations against probabilities), and so do the numbers they draw. Only the
    swarm, its size, its topology, its memory, its loop, and its stop are common to
    both.

    **Draws.** The family draws nothing from the control stream. The operator stream
    gives the swarm first, then the numbers of each update, which depend on the
    update rule and not on any score. The observer draws nothing. A job shop of one
    job has a single job sequence, which the first evaluation covers: the run stops
    after it, before any update.

    Besides the measures of the run and of its
    :class:`~dispatchatlas.solvers.shop.observers.TrajectoryObserver`, which sees the
    swarm and then every update as a move of the particle's last position, the extras
    of the result of the keys swarm hold ``reflected_share``.

    Attributes:
        representation: ``"keys"`` or ``"sequence"``
            (:data:`~dispatchatlas.solvers.shop.family.REPRESENTATIONS`).
        builder: The schedule builder, one of
            :data:`~dispatchatlas.solvers.shop.decoding.BUILDERS`.
        swarm_size: The number of particles, at least 1.
        topology: ``"global"`` or ``"ring"`` (:data:`TOPOLOGIES`).
        cognitive: The acceleration ``c1`` towards the particle's personal best.
            Only the keys swarm reads it. Non-negative and finite, with
            ``cognitive + social`` above 4.
        social: The acceleration ``c2`` towards the leader's best. Only the keys
            swarm reads it.
        move_probability: The probability that an update moves the particle
            first. Only the job-sequence swarm reads it. In [0, 1].
        personal_probability: The probability that an update then crosses the
            particle with its personal best. Only the job-sequence swarm reads it.
        leader_probability: The probability that an update then crosses the
            particle with its leader's best. Only the job-sequence swarm reads it.
        inheritance: The probability, in [0, 1], that a gene of a crossover's child
            comes from the best the particle is crossed with. Only the job-sequence
            swarm reads it.
        sequence_move: The move of job sequences, ``"insert"``, ``"swap"`` or
            ``"mixed"``. Only the job-sequence swarm reads it.
    """

    representation: RepresentationName = "sequence"
    builder: BuilderName = "active"
    swarm_size: int = 30
    topology: Topology = "global"
    cognitive: float = 2.05
    social: float = 2.05
    move_probability: float = 0.5
    personal_probability: float = 0.5
    leader_probability: float = 0.5
    inheritance: float = 0.5
    sequence_move: SequenceMove = "insert"

    info: ClassVar[SolverInfo] = SolverInfo(
        id="shop-particle-swarm",
        name="Job-shop particle swarm",
        family="swarm",
        summary="Moves a swarm of random keys by velocities that a constriction coefficient "
        "limits, or a swarm of job sequences by an insert move and crossovers with each "
        "particle's best and its leader's best, over a global or a ring topology of "
        "personal-best memories.",
        references=(
            "bratton2007defining",
            "pan2007discrete",
            "bean1994genetic",
            "bierwirth1995generalized",
            "bierwirth1996permutation",
        ),
        supports=JOB_SHOP_FEATURES,
        shape="job-shop",
    )

    def __post_init__(self) -> None:
        check_family(self.representation, self.builder)
        # The keys swarm has no move: its key arguments are the representation's defaults.
        check_moves("perturb", 0.1, self.sequence_move)
        size = self.swarm_size
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"swarm_size must be an integer of at least 1, got {size!r}")
        if self.topology not in TOPOLOGIES:
            raise ValueError(
                f"unknown topology {self.topology!r}; expected one of {list(TOPOLOGIES)}"
            )
        for name, acceleration in (("cognitive", self.cognitive), ("social", self.social)):
            if (
                isinstance(acceleration, bool)
                or not isinstance(acceleration, int | float)
                or not (math.isfinite(acceleration) and acceleration >= 0.0)
            ):
                raise ValueError(f"{name} must be non-negative and finite, got {acceleration!r}")
        phi = self.cognitive + self.social
        if not (math.isfinite(phi) and phi > 4.0):
            raise ValueError(
                f"cognitive + social must be finite and exceed 4 for the constriction "
                f"coefficient, got {phi!r}"
            )
        for name, probability in (
            ("move_probability", self.move_probability),
            ("personal_probability", self.personal_probability),
            ("leader_probability", self.leader_probability),
            ("inheritance", self.inheritance),
        ):
            if (
                isinstance(probability, bool)
                or not isinstance(probability, int | float)
                or not 0.0 <= probability <= 1.0
            ):
                raise ValueError(f"{name} must lie in [0, 1], got {probability!r}")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Draw the swarm, then move its particles until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        shop = JobShop.from_problem(problem)
        representation = make_representation(
            shop, self.representation, sequence_move=self.sequence_move
        )
        run = ShopRun(search, shop, self.builder, representation, TrajectoryObserver(shop))
        with search:
            self._swarm(run, self._update(representation, shop))
        run.finish()
        return search.result()

    def _update(self, representation: Representation[Any], shop: JobShop) -> _Update[Any]:
        """Return the update rule of the representation of the run."""
        if self.representation == "keys":
            return _ConstrictionUpdate(self.swarm_size, shop.size, self.cognitive, self.social)
        return _DiscreteUpdate(
            representation,
            self.move_probability,
            self.personal_probability,
            self.leader_probability,
            self.inheritance,
        )

    def _swarm[G](self, run: ShopRun[G], rule: _Update[G]) -> None:
        """Evaluate the swarm, then move its particles one by one until the search is exhausted."""
        search, representation, operators = run.search, run.representation, run.operators
        size = self.swarm_size
        # The whole swarm is drawn before any other draw from the operator stream.
        swarm = [representation.initial(operators) for _ in range(size)]
        memory = _Memory[G](self.topology)
        try:
            for genotype in swarm:
                memory.begin(run.evaluate(genotype))
                if run.shop.jobs < 2:
                    return
            while True:
                search.next_iteration()
                for particle in range(size):
                    current = memory.current[particle]
                    own = memory.best[particle]
                    leader = memory.best[memory.leader_of(particle)]
                    moved = rule.update(
                        particle, current.genotype, own.genotype, leader.genotype, operators
                    )
                    memory.record(particle, run.evaluate(moved, parent=current))
        finally:
            search.extras.update(rule.measures())
