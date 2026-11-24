"""A steady-state genetic algorithm over random keys or job sequences, a job-shop search family.

:class:`ShopSteadyStateGa` breeds one child per iteration and lets it replace one member of
its population. It is written once against
:class:`~dispatchatlas.solvers.shop.representation.Representation`: the population, the
tournaments, the choice of the parents, the replacement and the draws from the control stream
are the same code on random keys (Bean 1994, doi:10.1287/ijoc.6.2.154) and on job sequences
(Bierwirth 1995, doi:10.1007/BF01719250), and only the genotype and the representation's
recombination and move differ.
"""

from __future__ import annotations

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
from dispatchatlas.solvers.shop.graph import machine_orders
from dispatchatlas.solvers.shop.instance import JOB_SHOP_FEATURES, JobShop
from dispatchatlas.solvers.shop.observers import TrajectoryObserver
from dispatchatlas.solvers.shop.representation import (
    SEQUENCE_CROSSOVERS,
    KeyMove,
    SequenceCrossover,
    SequenceMove,
)
from dispatchatlas.solvers.shop.run import Evaluated, ShopRun
from dispatchatlas.solvers.shop.tabu import critical_block_descent

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["ShopSteadyStateGa"]


@dataclass(frozen=True)
class ShopSteadyStateGa:
    """A steady-state genetic algorithm on random keys or job sequences, one child per iteration.

    **The population.** The run draws ``population_size`` random solutions
    (:meth:`~dispatchatlas.solvers.shop.representation.Representation.initial`) from the
    operator stream, all of them before any other draw from it, and evaluates them in the order
    drawn. Both representations therefore start from the same population, and the first
    evaluation of the run is its first member. The population index of a member is the number of
    its slot, which a child keeps when it replaces the member. The members of the initial
    population are older the earlier they were drawn, and a child that enters is the youngest
    member. A job shop of one job has a single job sequence, which the run evaluates once
    before it stops.

    **An iteration** breeds one child, evaluates it, and places it, in this order:

    1. It draws ``2 * tournament_size + 2`` uniform numbers from the control stream, in one
       call and whatever the outcomes, so that the runs on the two representations read the
       same numbers at the same iteration. The first ``2 * tournament_size`` are the
       contestants of two tournaments, drawn with replacement: a number ``u`` names the member
       of population index ``floor(u * population_size)``. The next number decides the
       crossover, and the last the mutation.
    2. Each tournament selects its fittest contestant: the one of the lowest score and, among
       equal scores, of the lowest population index. The fitter of the two winners is the
       first parent and the other the second; when their scores are equal, the one of the
       lower population index is the first. A member that wins both tournaments is both
       parents.
    3. When the crossover number lies below ``crossover_rate``, the child is the
       representation's recombination of the first and the second parent, in this order.
       Otherwise it is the first parent.
    4. When the mutation number lies below ``mutation_rate``, one move of the representation
       changes the child.
    5. The child is evaluated, with the first parent as its parent for the observer, whatever
       the draws decided: a child that equals its first parent counts as an evaluation too.
    6. The child replaces the worst member of the population when its score is no worse than
       that member's. The worst member has the highest score and, among members of equal
       score, is the oldest. A child of a worse score than the worst member is dropped.
       Neither representation rejects duplicates: a child equal to a member enters the
       population like any other.

    **Recombination.** Each representation recombines in its own way, with the fitter parent
    first and the same ``inheritance``. Random keys cross by the parameterised uniform
    crossover
    (:func:`~dispatchatlas.solvers.shop.representation.parameterised_uniform_crossover`), in
    which each key of the child comes from the first parent with probability ``inheritance``,
    and job sequences by the precedence preservative crossover
    (:func:`~dispatchatlas.solvers.shop.representation.precedence_preservative_crossover`,
    Bierwirth, Mattfeld and Kopfer 1996, doi:10.1007/3-540-61723-X_995), in which each gene of
    the child does. ``sequence_crossover="jox"`` selects the job-based order crossover
    (:func:`~dispatchatlas.solvers.shop.representation.job_order_crossover`, Ono, Yamamura and
    Kobayashi 1996, doi:10.1109/ICEC.1996.542658) instead, in which a job keeps its positions
    in the first parent with that probability, so that its match to the keys' crossover is
    weaker.

    **Mutation.** A move of random keys redraws one key (``key_move="redraw"``) or perturbs
    two keys by normal steps of standard deviation ``key_step``, reflected into [0, 1]
    (``"perturb"``); a move of a job sequence is one insert or one swap (``sequence_move``).
    Each key move is matched to a sequence move, as in
    :class:`~dispatchatlas.solvers.shop.annealing.ShopAnnealing`: the redrawn key to the
    insert, and the two perturbed keys to the swap, for a ``key_step`` that
    :func:`~dispatchatlas.solvers.shop.distance.calibrate_step` sets so that their median
    distance is the swap's. The defaults are the redrawn key and the insert, a pair without a
    step to set. ``sequence_move="mixed"``, a swap or an insert with probability one half
    each, is accepted, but no key move is matched to it.

    **Budget.** The initial population counts toward the budget like every other evaluation,
    and every iteration evaluates one child, so a run under an evaluation limit makes exactly
    that many evaluations, whether the limit falls inside the initial population or after it,
    and an iteration limit counts children.

    **Local search.** With ``local_search``, the run is a Baldwinian memetic algorithm: every
    evaluated genotype, a member of the initial population or a child, is decoded by the
    builder as above, and the machine orders of its schedule then go to
    :func:`~dispatchatlas.solvers.shop.tabu.critical_block_descent`, the descent of
    :class:`~dispatchatlas.solvers.shop.brkga.ShopBrkga`. It is a best-improvement descent
    over the N5 swaps of the blocks of a critical path: each step applies the swap of the
    lowest exact makespan when that is below the current makespan, and the descent ends
    when no swap is, or when the budget ends. The descent reads the schedule, never the
    genotype, so it is the same in both representations: equal schedules are improved
    alike. The genotype is not changed, and nothing is written back into it: the individual
    keeps its genotype and its decoded job sequence, and takes the improved schedule, a
    semi-active one, and its score, which rank it in the tournaments and in the replacement.
    A descent steers by the makespan, so under another objective it can end in a schedule
    of a worse score; the individual then keeps the schedule it decoded to. A child's
    parent for the observer is its first parent, with that parent's improved schedule.

    **Counting with the local search.** Every neighbour that the descent evaluates exactly
    counts as one evaluation of the run's budget, through the run's search like a decode,
    and updates the incumbent and the trace; the observer sees it as a move of the schedule
    its step starts from. The swap a step applies is one of those neighbours and is not
    counted again. An evaluation limit therefore bounds the decodes and the evaluations of
    the local search together, and the incumbent is the best schedule that any decode or
    any neighbour built. The estimates that rank the swaps and end the scan of a step are
    not counted, and neither is a swap whose exact heads reveal a cycle, which builds no
    schedule. A run can end inside a descent; the individual being improved then enters no
    population. An iteration limit still counts children. The extras of the result add
    ``descent_evaluations``, the evaluations of the local search, and ``descent_moves``,
    the swaps it applied.

    Besides the measures of the run and of its
    :class:`~dispatchatlas.solvers.shop.observers.TrajectoryObserver`, which sees every member
    of the initial population and every child, as a move of its first parent, the extras of
    the result hold ``replacement_share`` once a child has been evaluated: the share of the
    children that entered the population.

    Attributes:
        representation: ``"keys"`` or ``"sequence"``
            (:data:`~dispatchatlas.solvers.shop.family.REPRESENTATIONS`).
        builder: The schedule builder, one of
            :data:`~dispatchatlas.solvers.shop.decoding.BUILDERS`.
        population_size: The number of members, an integer of at least 2.
        tournament_size: The number of contestants of each tournament, an integer of at
            least 1; with 1 a parent is a member drawn at random.
        crossover_rate: The probability, in [0, 1], that the child is a recombination of its
            parents and not a copy of the first.
        mutation_rate: The probability, in [0, 1], that one move changes the child.
        inheritance: The probability, in [0, 1], that a key or a gene of a recombined child
            comes from the first parent, which is the fitter one.
        key_move: The move of random keys, ``"redraw"`` or ``"perturb"``.
        key_step: The standard deviation of a key perturbation, positive and finite. Only
            ``"perturb"`` reads it.
        sequence_move: The move of job sequences, ``"insert"``, ``"swap"`` or ``"mixed"``.
        sequence_crossover: The crossover of job sequences, ``"ppx"`` or ``"jox"``.
        distance_every: The observer measures the phenotype distance of the first move and of
            every ``distance_every``-th one after it, an integer of at least 1. Its other
            measures count every evaluation, and no decision of the search depends on it.
        local_search: Improve the schedule of every evaluated genotype by the critical-block
            descent, without changing the genotype; off by default.
    """

    representation: RepresentationName = "sequence"
    builder: BuilderName = "active"
    population_size: int = 30
    tournament_size: int = 2
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    inheritance: float = 0.7
    key_move: KeyMove = "redraw"
    key_step: float = 0.1
    sequence_move: SequenceMove = "insert"
    sequence_crossover: SequenceCrossover = "ppx"
    distance_every: int = 1
    local_search: bool = False

    info: ClassVar[SolverInfo] = SolverInfo(
        id="shop-steady-state-ga",
        name="Job-shop steady-state genetic algorithm",
        family="population",
        summary="Evolves a population of random keys or job sequences one child at a time: "
        "two tournament winners are recombined, the child may be mutated by one move, and it "
        "replaces the worst member when it is no worse.",
        references=(
            "bean1994genetic",
            "bierwirth1995generalized",
            "bierwirth1996permutation",
            "ono1996genetic",
            "nowicki1996fast",
        ),
        supports=JOB_SHOP_FEATURES,
        shape="job-shop",
    )

    def __post_init__(self) -> None:
        check_family(self.representation, self.builder, self.distance_every)
        check_moves(self.key_move, self.key_step, self.sequence_move)
        if self.sequence_crossover not in SEQUENCE_CROSSOVERS:
            raise ValueError(
                f"unknown crossover {self.sequence_crossover!r}; "
                f"expected one of {list(SEQUENCE_CROSSOVERS)}"
            )
        _check_count("population_size", self.population_size, 2)
        _check_count("tournament_size", self.tournament_size, 1)
        _check_probability("crossover_rate", self.crossover_rate)
        _check_probability("mutation_rate", self.mutation_rate)
        _check_probability("inheritance", self.inheritance)
        if not isinstance(self.local_search, bool):
            raise ValueError(f"local_search must be true or false, got {self.local_search!r}")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Evolve the population, one child per iteration, until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        shop = JobShop.from_problem(problem)
        representation = make_representation(
            shop,
            self.representation,
            key_move=self.key_move,
            key_step=self.key_step,
            sequence_move=self.sequence_move,
            crossover=self.sequence_crossover,
        )
        observer = TrajectoryObserver(shop, self.distance_every)
        run = ShopRun(search, shop, self.builder, representation, observer)
        tally = _Tally()
        with search:
            # The whole population is drawn before any other draw from the operator stream.
            genotypes = [representation.initial(run.operators) for _ in range(self.population_size)]
            members = [self._individual(run, genotypes[0], None, tally)]
            if shop.jobs > 1:
                members.extend(
                    self._individual(run, genotype, None, tally) for genotype in genotypes[1:]
                )
                self._breed(run, _Population(members), tally)
        run.finish()
        if self.local_search:
            search.extras.update(descent_evaluations=tally.evaluations, descent_moves=tally.moves)
        return search.result()

    def _individual[G](
        self, run: ShopRun[G], genotype: G, parent: Evaluated[G] | None, tally: _Tally
    ) -> Evaluated[G]:
        """Evaluate a genotype and, with the local search, improve its schedule."""
        evaluated = run.evaluate(genotype, parent=parent)
        return _improve(run, evaluated, tally) if self.local_search else evaluated

    def _breed[G](self, run: ShopRun[G], population: _Population[G], tally: _Tally) -> None:
        """Breed one child per iteration into ``population`` until the search is exhausted."""
        search, representation, operators, control = (
            run.search,
            run.representation,
            run.operators,
            run.control,
        )
        size, contests = self.population_size, self.tournament_size
        width = 2 * contests + 2
        children = replaced = 0
        try:
            while True:
                search.next_iteration()
                # Drawn whatever the outcomes, so that both representations stay in step.
                *picks, crossing, mutating = control.random(width).tolist()
                slots = [int(draw * size) for draw in picks]
                first, second = population.parents(slots[:contests], slots[contests:])
                parent = population.members[first]
                genotype = parent.genotype
                if crossing < self.crossover_rate:
                    genotype = representation.recombine(
                        genotype, population.members[second].genotype, operators, self.inheritance
                    )
                if mutating < self.mutation_rate:
                    genotype = representation.move(genotype, operators)
                child = self._individual(run, genotype, parent, tally)
                children += 1
                replaced += population.admit(child)
        finally:
            if children:
                search.extras["replacement_share"] = replaced / children


def _improve[G](run: ShopRun[G], decoded: Evaluated[G], tally: _Tally) -> Evaluated[G]:
    """Return the individual with the schedule that the descent makes of its decoded one.

    The genotype and the job sequence stay. Every neighbour the descent evaluates counts
    through the run, and ``tally`` takes the evaluations and the swaps down, the evaluations
    even when the budget ends inside the descent. The individual keeps its decoded schedule
    when the improved one scores worse, which another objective than the makespan allows.
    """
    shop, search = run.shop, run.search
    counted = search.evaluations
    try:
        descent = critical_block_descent(
            shop, machine_orders(shop, decoded.decoded), run, parent=decoded, count_neighbours=True
        )
    finally:
        tally.evaluations += search.evaluations - counted
    tally.moves += descent.moves
    score = search.evaluator.score(descent.decoded)
    if score > decoded.score:
        return decoded
    return Evaluated(decoded.genotype, decoded.sequence, score, descent.decoded)


@dataclass(slots=True)
class _Tally:
    """What the local search of a run did, for the extras of the result."""

    evaluations: int = 0
    moves: int = 0


class _Population[G]:
    """The members of a steady-state population, by slot, and the keys that order them.

    A slot keeps its number when a child replaces its member, and the number is the member's
    population index. The age of a member is the order in which it entered the population: the
    members of the initial population in slot order, then each child that replaced a member.

    Args:
        members: The evaluated members of the initial population, in slot order.

    Attributes:
        members: The member of each slot.
    """

    __slots__ = ("_age", "_births", "_fit", "members")

    def __init__(self, members: Sequence[Evaluated[G]]) -> None:
        self.members = list(members)
        # One key per slot that sorts the slots as selection does: the lowest score first and,
        # among equal scores, the lowest slot.
        self._fit = [(m.score.violation, m.score.value, slot) for slot, m in enumerate(members)]
        # And one that sorts them as replacement does: the highest score first and, among equal
        # scores, the oldest. A lower birth number is older, so the key holds its negative.
        self._age = [(m.score.violation, m.score.value, -slot) for slot, m in enumerate(members)]
        self._births = len(self.members)

    def parents(self, left: Sequence[int], right: Sequence[int]) -> tuple[int, int]:
        """Return the slots of the first and of the second parent of two tournaments.

        Each tournament selects the fittest of the members at its slots, and the fitter of the
        two winners is the first parent.

        Args:
            left: The slots of the contestants of the first tournament.
            right: The slots of the contestants of the second tournament.
        """
        fit = self._fit
        one, other = min(left, key=fit.__getitem__), min(right, key=fit.__getitem__)
        return (one, other) if fit[one] <= fit[other] else (other, one)

    def worst(self) -> int:
        """Return the slot of the worst member, the oldest of the members of equal score."""
        age = self._age
        return age.index(max(age))

    def admit(self, child: Evaluated[G]) -> bool:
        """Replace the worst member by ``child`` unless the child's score is worse than its.

        Returns:
            Whether the child replaced the worst member.
        """
        slot = self.worst()
        score = child.score
        if score > self.members[slot].score:
            return False
        self.members[slot] = child
        self._fit[slot] = (score.violation, score.value, slot)
        self._age[slot] = (score.violation, score.value, -self._births)
        self._births += 1
        return True


def _check_count(name: str, value: object, least: int) -> None:
    """Raise :class:`ValueError` unless ``value`` is an integer of at least ``least``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise ValueError(f"{name} must be an integer of at least {least}, got {value!r}")


def _check_probability(name: str, value: object) -> None:
    """Raise :class:`ValueError` unless ``value`` is a number in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
