"""A biased random-key genetic algorithm for the job shop, with a local search and a write-back.

:class:`ShopBrkga` is the biased random-key genetic algorithm of Gonçalves and Resende
(2011, doi:10.1007/s10732-010-9143-1) on the job shop, after Gonçalves and Resende (2014,
doi:10.1111/itor.12044), who pair an active-schedule decoder with a local search and adjust
each chromosome to the schedule the search found. The local search is the library's own,
:func:`~dispatchatlas.solvers.shop.tabu.critical_block_descent`, not that paper's.

**Chromosome.** A chromosome is one random key in [0, 1) per slot of
:attr:`~dispatchatlas.solvers.shop.instance.JobShop.slot_job`
(:class:`~dispatchatlas.solvers.shop.representation.KeyRepresentation`). Sorting the slots by
key gives a job sequence, which the run's builder decodes into a schedule: the active one
(:func:`~dispatchatlas.solvers.shop.decoding.giffler_thompson`) by default.

**Generation.** The population is ranked by the scores of its chromosomes' schedules, the best
first and equal scores by position. The next population has the three parts of
:class:`~dispatchatlas.solvers.population.brkga.Brkga`, whose
:meth:`~dispatchatlas.solvers.population.brkga.Brkga.partition` sizes them:

1. the **elite**, the best chromosomes, copied with their schedules and never decoded again;
2. **mutants**, new chromosomes drawn like the initial ones;
3. **offspring** for the remaining places, each crossing an elite parent drawn uniformly from
   the elite with a non-elite parent drawn uniformly from the others: each key comes from the
   elite parent with probability ``inheritance``
   (:func:`~dispatchatlas.solvers.shop.representation.parameterised_uniform_crossover`).

**Local search.** With ``local_search``, every chromosome that is decoded, an initial one, a
mutant or an offspring, is improved. The machine orders of its schedule go to
:func:`~dispatchatlas.solvers.shop.tabu.critical_block_descent`, which swaps adjacent
operations of the blocks of a critical path while a swap lowers the makespan. The schedule of
the orders it ends in, a semi-active one, becomes the chromosome's schedule, and its score
ranks the chromosome. A descent steers by the makespan, so under another objective it can end
in a schedule of a worse score; the chromosome then keeps the schedule it decoded to.

**Write-back.** With ``write_back``, :func:`adjust_keys` rearranges the keys of an improved
chromosome so that they sort into the job sequence of the improved schedule, its operations by
start time (:func:`start_sequence`). Decoded by the semi-active builder, the written-back keys
give the improved schedule exactly. Decoded by the active builder they can give a longer one,
because an improved schedule need not be active and the builder may then order the operations
of a machine differently. The chromosome's value is therefore its improved schedule's, and
the elite are never decoded again.

**Counting.** A decoded chromosome counts one evaluation, and so does every swap the descent
applies, both through the run's search, which updates the incumbent and the trace with them.
The elite are not decoded again, the ranking and the write-back build no schedule, and the
estimates and the exact heads of swaps that the descent ranks and does not apply are not
counted, so an evaluation limit does not bound the work between two evaluations. A time limit
is checked before every decode and before every swap the descent applies, so a run ends within
the work between two of those checks of its limit. The result's extras add ``decodes``,
``descent_moves``, ``elite_members`` and ``mutant_members``.

**Streams.** The operator stream draws the whole initial population before any other draw,
then the mutants and the crossings of each generation; the control stream draws the parents of
each generation. The descent and the write-back draw nothing. A shop of one job has a single
job sequence, which the run evaluates once before it stops.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.population.brkga import Brkga
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.shop.decoding import BuilderName, JobSequence
from dispatchatlas.solvers.shop.family import check_family
from dispatchatlas.solvers.shop.graph import heads_tails, machine_orders
from dispatchatlas.solvers.shop.instance import JOB_SHOP_FEATURES, JobShop
from dispatchatlas.solvers.shop.observers import TrajectoryObserver
from dispatchatlas.solvers.shop.representation import KeyRepresentation, Keys, Representation
from dispatchatlas.solvers.shop.run import Evaluated, ShopRun
from dispatchatlas.solvers.shop.tabu import critical_block_descent

__all__ = ["ShopBrkga", "adjust_keys", "start_sequence"]

# The members of the elite and of the mutants when a count or a fraction is not given: the
# settings of Gonçalves and Resende (2014, Table 2) for the job shop.
_MEMBERS = 10


def start_sequence(shop: JobShop, orders: Sequence[Sequence[int]]) -> NDArray[np.intp]:
    """Return the job sequence of the semi-active schedule of machine orders.

    The schedule starts every operation at its head
    (:func:`~dispatchatlas.solvers.shop.graph.heads_tails`), and the sequence lists the
    operations by start time, those that start together by machine and then by job, as jobs:
    a job sequence names the job of each operation only, and the ``k``-th occurrence of a job
    stands for its ``k``-th operation. An operation never comes before an operation that
    precedes it in its job or on its machine. Only operations that take no time can make the
    sort by start time, machine and job alone put one there, and then the operation that
    precedes it comes first.

    The semi-active builder decodes the sequence into that schedule exactly: it processes every
    machine in the order of ``orders``, and starts every operation as early as the orders, the
    jobs and the release times allow.

    Args:
        shop: The job shop.
        orders: The machine orders, as
            :func:`~dispatchatlas.solvers.shop.graph.machine_orders` returns them, such as
            the final orders of :func:`~dispatchatlas.solvers.shop.tabu.critical_block_descent`.

    Returns:
        The job sequence, a new array.

    Raises:
        ModelError: If ``orders`` do not list every operation once on its machine, or form a
            cycle with the jobs.
    """
    result = heads_tails(shop, orders)
    if result is None:
        raise ModelError("the machine orders and the jobs form a cycle")
    starts, size = result[0], shop.size
    machine, job, successor = shop.machine, shop.job, shop.successor
    # Each operation waits for its job predecessor and for its machine predecessor.
    waiting = [int(before >= 0) for before in shop.predecessor]
    after = [-1] * size
    for order in orders:
        for previous, task in pairwise(order):
            after[previous] = task
            waiting[task] += 1
    ready = [
        (starts[task], machine[task], job[task], task) for task in range(size) if not waiting[task]
    ]
    heapq.heapify(ready)
    sequence: list[int] = []
    while ready:
        _, _, owner, task = heapq.heappop(ready)
        sequence.append(owner)
        for following in (successor[task], after[task]):
            if following >= 0:
                waiting[following] -= 1
                if not waiting[following]:
                    key = (starts[following], machine[following], job[following], following)
                    heapq.heappush(ready, key)
    return np.asarray(sequence, dtype=np.intp)


def adjust_keys(representation: Representation[Keys], keys: Keys, sequence: JobSequence) -> Keys:
    """Return the keys rearranged so that they sort into ``sequence``, with no two alike.

    The values of the keys stay, and move between slots as
    :meth:`~dispatchatlas.solvers.shop.representation.KeyRepresentation.follow` moves them,
    keeping the keys that already sort into ``sequence`` where they are. Slots sort by
    ``(key, slot)``, and a job sequence can put a job of a later slot before one of an earlier
    slot, which equal keys cannot express. A chromosome of a run can hold one value twice, since
    the write-back moves a value to another slot while the chromosomes it was bred from keep it
    where it was, and a child may take both. So each key that equals the one below it in
    ascending order first moves up by one unit in the last place, and then the keys sort into
    ``sequence`` exactly.

    Args:
        representation: The random-key representation of the shop.
        keys: The chromosome, one key per slot.
        sequence: The job sequence the keys must sort into.

    Returns:
        The adjusted keys, a new array.

    Raises:
        ValueError: If ``keys`` is not a row of float keys, one per slot.
        ModelError: If ``sequence`` is not a job sequence of the shop.
    """
    return representation.follow(_distinct(keys), sequence)


def _distinct(keys: Keys) -> Keys:
    """Return ``keys`` with each key that equals a smaller one moved up to a value of its own."""
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    if not (ordered[1:] == ordered[:-1]).any():
        return keys
    values = ordered.tolist()
    for place in range(1, len(values)):
        if values[place] <= values[place - 1]:
            values[place] = float(np.nextafter(values[place - 1], np.inf))
    separated = np.empty_like(keys)
    separated[order] = values
    return separated


@dataclass(frozen=True)
class ShopBrkga:
    """A biased random-key genetic algorithm on the job shop, with local search and write-back.

    The module describes the method. The run draws ``population_size`` chromosomes, decodes
    them in order, and then breeds generation after generation until a stop criterion fires; an
    iteration is a generation, and the initial population is not one. The result is the
    incumbent of the run's search, the best schedule any decode or any swap of the descent
    built, by the problem's objective.

    **The parts of a generation.** The elite holds ``elite_count`` chromosomes, or the
    ``elite_fraction`` of the population, and the mutants ``mutant_count`` or
    ``mutant_fraction`` likewise. Give at most one of a count and a fraction for each part; a
    part with neither holds 10 chromosomes. A count is used as given and refused when the
    population cannot hold it and leave a place for an offspring. A fraction is rounded by
    :meth:`~dispatchatlas.solvers.population.brkga.Brkga.partition`, which keeps at least one
    elite chromosome and one place for offspring.

    **Ranges.** The checks admit every setting of the two sources: an elite of 0.10 to 0.25 of
    the population, mutants of 0.10 to 0.30 and an inheritance above 0.5 and at most 0.8
    (Gonçalves and Resende 2011, Table 1), and 10 elite chromosomes and 10 mutants in a
    population of at least 150 with an inheritance of 0.85 (Gonçalves and Resende 2014,
    Table 2). The checks of the fractions and of the inheritance are those of
    :class:`~dispatchatlas.solvers.population.brkga.Brkga`, which refuses none of these values,
    so none of them is widened; counts are added. The defaults are the settings of the second
    source at its least population.

    Attributes:
        population_size: The number of chromosomes of a generation, an integer of at least 2.
        elite_fraction: The elite as a fraction of the population, in (0, 1), or ``None``.
        elite_count: The elite as a number of chromosomes, an integer of at least 1, or
            ``None``.
        mutant_fraction: The mutants as a fraction of the population, in [0, 1), or ``None``.
        mutant_count: The mutants as a number of chromosomes, an integer of at least 0, or
            ``None``.
        inheritance: The probability, in [0.5, 1), that a key of an offspring comes from its
            elite parent.
        local_search: Improve the schedule of every decoded chromosome by the critical-block
            descent.
        write_back: Rearrange the keys of an improved chromosome so that they sort into the job
            sequence of its improved schedule. It has no effect without ``local_search``.
        builder: The schedule builder, one of
            :data:`~dispatchatlas.solvers.shop.decoding.BUILDERS`. The semi-active one
            reproduces an improved schedule from its written-back keys exactly.
    """

    population_size: int = 150
    elite_fraction: float | None = None
    elite_count: int | None = None
    mutant_fraction: float | None = None
    mutant_count: int | None = None
    inheritance: float = 0.85
    local_search: bool = True
    write_back: bool = True
    builder: BuilderName = "active"

    info: ClassVar[SolverInfo] = SolverInfo(
        id="shop-brkga",
        name="Job-shop biased random-key genetic algorithm",
        family="population",
        summary="Keeps the elite random-key chromosomes, adds random mutants and crosses elite "
        "with non-elite parents in favour of the elite, decodes each chromosome into a job-shop "
        "schedule, improves it by a critical-block descent, and writes the improved schedule "
        "back into the keys.",
        references=(
            "goncalves2011biased",
            "goncalves2014extended",
            "bean1994genetic",
            "giffler1960algorithms",
            "nowicki1996fast",
        ),
        supports=JOB_SHOP_FEATURES,
        shape="job-shop",
    )

    def __post_init__(self) -> None:
        check_family("keys", self.builder)
        _check_count("population_size", self.population_size, 2)
        for part, count, fraction, least in (
            ("elite", self.elite_count, self.elite_fraction, 1),
            ("mutant", self.mutant_count, self.mutant_fraction, 0),
        ):
            if count is not None and fraction is not None:
                raise ValueError(f"give {part}_count or {part}_fraction, not both")
            if count is not None:
                _check_count(f"{part}_count", count, least)
            if fraction is not None:
                _check_fraction(f"{part}_fraction", fraction, positive=least >= 1)
        inheritance = self.inheritance
        if (
            isinstance(inheritance, bool)
            or not isinstance(inheritance, int | float)
            or not 0.5 <= inheritance < 1.0
        ):
            raise ValueError(f"inheritance must lie in [0.5, 1), got {inheritance!r}")
        for name in ("local_search", "write_back"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false, got {getattr(self, name)!r}")
        elite, mutants = self._shares()
        if elite + mutants >= 1.0:
            raise ValueError(
                f"the elite and the mutants leave no place for an offspring in a population of "
                f"{self.population_size}"
            )

    def partition(self) -> tuple[int, int]:
        """Return the numbers of elite and of mutant chromosomes of a generation.

        A count is returned as given. A fraction is rounded as
        :meth:`~dispatchatlas.solvers.population.brkga.Brkga.partition` rounds it.
        """
        elite, mutants = self._shares()
        generic = Brkga(
            population_size=None,
            elite_fraction=elite,
            mutant_fraction=mutants,
            inheritance=self.inheritance,
        )
        return generic.partition(self.population_size)

    def _shares(self) -> tuple[float, float]:
        """Return the elite and the mutants as fractions of the population."""
        size = self.population_size
        return (
            _share(self.elite_count, self.elite_fraction, size),
            _share(self.mutant_count, self.mutant_fraction, size),
        )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Evolve the chromosomes, generation after generation, until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        shop = JobShop.from_problem(problem)
        representation = KeyRepresentation(shop)
        run = ShopRun(search, shop, self.builder, representation, TrajectoryObserver(shop))
        tally = _Tally()
        with search:
            if shop.jobs < 2:
                # The shop has one job sequence: decode it, and stop.
                run.evaluate(representation.initial(run.operators))
            else:
                population = self._initial(run, tally)
                while True:
                    search.next_iteration()
                    population = self._generation(run, population, tally)
        run.finish()
        elite, mutants = self.partition()
        search.extras.update(
            decodes=search.evaluations - tally.moves,
            descent_moves=tally.moves,
            elite_members=elite,
            mutant_members=mutants,
        )
        return search.result()

    def _initial(self, run: ShopRun[Keys], tally: _Tally) -> list[Evaluated[Keys]]:
        """Draw the whole initial population, then decode and improve its chromosomes in order."""
        representation, operators = run.representation, run.operators
        chromosomes = [representation.initial(operators) for _ in range(self.population_size)]
        return [self._member(run, keys, None, tally) for keys in chromosomes]

    def _generation(
        self, run: ShopRun[Keys], population: Sequence[Evaluated[Keys]], tally: _Tally
    ) -> list[Evaluated[Keys]]:
        """Return the next population: the elite, then the mutants, then the offspring.

        Every draw is made before the first decode, and the number of draws depends on the
        sizes of the parts only. An offspring is decoded with its elite parent as its parent.
        """
        representation, operators, control = run.representation, run.operators, run.control
        elite_count, mutant_count = self.partition()
        ranked = sorted(
            range(len(population)),
            key=lambda member: (
                population[member].score.violation,
                population[member].score.value,
                member,
            ),
        )
        elite = [population[member] for member in ranked[:elite_count]]
        others = [population[member] for member in ranked[elite_count:]]
        offspring = len(population) - elite_count - mutant_count
        favoured = control.integers(len(elite), size=offspring).tolist()
        alternative = control.integers(len(others), size=offspring).tolist()
        mutants = [representation.initial(operators) for _ in range(mutant_count)]
        children = [
            (
                representation.recombine(
                    elite[first].genotype, others[second].genotype, operators, self.inheritance
                ),
                elite[first],
            )
            for first, second in zip(favoured, alternative, strict=True)
        ]
        members = list(elite)
        members.extend(self._member(run, keys, None, tally) for keys in mutants)
        members.extend(self._member(run, keys, parent, tally) for keys, parent in children)
        return members

    def _member(
        self,
        run: ShopRun[Keys],
        keys: Keys,
        parent: Evaluated[Keys] | None,
        tally: _Tally,
    ) -> Evaluated[Keys]:
        """Decode a chromosome and, with the local search, improve it; the result ranks it."""
        decoded = run.evaluate(keys, parent)
        return self._improve(run, decoded, tally) if self.local_search else decoded

    def _improve(
        self, run: ShopRun[Keys], decoded: Evaluated[Keys], tally: _Tally
    ) -> Evaluated[Keys]:
        """Return the chromosome that the local search makes of a decoded one.

        The descent counts its swaps through the run, and ``tally`` takes them down even when
        the budget ends inside it. The chromosome keeps its decoded schedule when the improved
        schedule scores worse, which another objective than the makespan allows.
        """
        shop, search = run.shop, run.search
        counted = search.evaluations
        try:
            descent = critical_block_descent(
                shop, machine_orders(shop, decoded.decoded), run, parent=decoded
            )
        finally:
            tally.moves += search.evaluations - counted
        improved = descent.decoded
        score = search.evaluator.score(improved)
        if score > decoded.score:
            return decoded
        keys, sequence = decoded.genotype, decoded.sequence
        if self.write_back:
            sequence = start_sequence(shop, descent.orders)
            keys = adjust_keys(run.representation, keys, sequence)
        return Evaluated(keys, sequence, score, improved)


@dataclass(slots=True)
class _Tally:
    """What a run did, for the extras of the result: the swaps its descents applied."""

    moves: int = 0


def _share(count: int | None, fraction: float | None, size: int) -> float:
    """Return the fraction of a population of ``size`` that a count or a fraction stands for."""
    if fraction is not None:
        return fraction
    return (_MEMBERS if count is None else count) / size


def _check_count(name: str, value: object, least: int) -> None:
    """Raise :class:`ValueError` unless ``value`` is an integer of at least ``least``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise ValueError(f"{name} must be an integer of at least {least}, got {value!r}")


def _check_fraction(name: str, value: object, *, positive: bool) -> None:
    """Raise :class:`ValueError` unless ``value`` lies in (0, 1), or in [0, 1) if not positive."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not (0.0 < value < 1.0 if positive else 0.0 <= value < 1.0)
    ):
        raise ValueError(f"{name} must lie in {'(0, 1)' if positive else '[0, 1)'}, got {value!r}")
