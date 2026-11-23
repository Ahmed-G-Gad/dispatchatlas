"""Random keys and job sequences: the two representations of a job-shop search family.

A search family of this package is written once, against :class:`Representation`,
which supplies its random solutions, its move, its recombination, and the job
sequence a solution stands for. Both representations reach the same builder of
:mod:`~dispatchatlas.solvers.shop.decoding` through that job sequence, so the
two runs of a family differ in the genotype and its operators only:

- :class:`KeyRepresentation` searches **random keys** (Bean 1994,
  doi:10.1287/ijoc.6.2.154): one key in [0, 1] per slot of
  :attr:`~dispatchatlas.solvers.shop.instance.JobShop.slot_job`, which
  :func:`~dispatchatlas.solvers.shop.decoding.sequence_from_keys` sorts into a
  job sequence by ``(key, slot)``;
- :class:`SequenceRepresentation` searches **job sequences** (Bierwirth 1995,
  doi:10.1007/BF01719250) themselves.

**Random solutions.** :meth:`Representation.initial` draws one uniform number per
slot, in both representations, and hands the draws to
:meth:`Representation.from_keys`: random keys keep them as the keys, and job
sequences sort the slots by them. Two generators in the same state therefore
give the same job sequence in either representation, and are left in the same
state.

**Moves.** Every move changes the genotype. A key move gives each of its keys a
new value, redrawn or perturbed and reflected into [0, 1] (:func:`reflect`). A
sequence move swaps two positions that hold different jobs, or moves the job at
one position to another position where the sequence changes, so neither draws
a null move. Whether the schedule changes is not imposed but measured, by the
observers of :mod:`~dispatchatlas.solvers.shop.observers`: a key that passes
only keys of its own job leaves the job sequence as it was.

**Recombination.** :func:`parameterised_uniform_crossover` takes each key, and
:func:`job_order_crossover` the positions of each job, from the first parent
with probability ``inheritance``. Callers pass the fitter parent first in both
representations.

Every operator returns a new array, leaves its arguments as they were, and
draws only from the generator it is given.
"""

from __future__ import annotations

import math
import operator
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dispatchatlas.core.errors import ModelError, SolverError
from dispatchatlas.solvers.shop.decoding import JobSequence, check_sequence, sequence_from_keys
from dispatchatlas.solvers.shop.instance import JobShop

__all__ = [
    "KEY_MOVES",
    "SEQUENCE_MOVES",
    "KeyMove",
    "KeyRepresentation",
    "Keys",
    "Representation",
    "SequenceMove",
    "SequenceRepresentation",
    "job_order_crossover",
    "parameterised_uniform_crossover",
    "reflect",
]

type Keys = NDArray[np.float64]
"""A vector of random keys, one per slot of a job shop."""

type KeyMove = Literal["perturb", "redraw"]
"""How a key move changes its keys, one of :data:`KEY_MOVES`."""

type SequenceMove = Literal["swap", "insert", "mixed"]
"""How a sequence move changes a job sequence, one of :data:`SEQUENCE_MOVES`."""

KEY_MOVES: tuple[KeyMove, ...] = ("perturb", "redraw")
"""The key moves :class:`KeyRepresentation` takes."""

SEQUENCE_MOVES: tuple[SequenceMove, ...] = ("swap", "insert", "mixed")
"""The sequence moves :class:`SequenceRepresentation` takes."""

# Proposals a key move draws for a key before it moves the key to an adjacent
# float, which only a step too small to change the key in floating point needs.
_PROPOSALS = 64


def reflect(values: ArrayLike) -> Keys:
    """Return finite ``values`` folded into [0, 1] by reflection at the bounds.

    A value beyond a bound is reflected at that bound, as often as needed:
    ``-0.2`` becomes ``0.2``, ``1.3`` becomes ``0.7``, and ``2.5`` becomes ``0.5``.
    Values in [0, 1] are kept exactly, and the fold of any other value is exact
    too. Clipping instead would pile keys up on the bounds, where they tie.

    Args:
        values: Finite numbers.

    Returns:
        A new array of the folded values.
    """
    folded = np.mod(np.abs(np.asarray(values, dtype=np.float64)), 2.0)
    return np.asarray(np.where(folded > 1.0, 2.0 - folded, folded), dtype=np.float64)


def _check_inheritance(inheritance: float) -> None:
    """Raise :class:`ValueError` unless ``inheritance`` lies in [0, 1]."""
    if not 0.0 <= inheritance <= 1.0:
        raise ValueError(f"inheritance must lie in [0, 1], got {inheritance!r}")


def _unit_keys(values: Keys, size: int) -> Keys:
    """Return ``values`` after checking they are ``size`` keys in [0, 1], in one row."""
    if values.shape != (size,):
        raise ValueError(f"expected {size} keys in one row, got shape {values.shape}")
    if not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("every key must lie in [0, 1]")
    return values


def parameterised_uniform_crossover(
    first: ArrayLike, second: ArrayLike, rng: np.random.Generator, inheritance: float
) -> Keys:
    """Return a child that takes each key from ``first`` with probability ``inheritance``.

    This is the parameterised uniform crossover (Spears and De Jong 1991): one
    uniform draw per key decides whether the child takes the key of the first
    parent, when the draw falls below ``inheritance``, or that of the second.
    The biased random-key genetic algorithm crosses its parents in the same way,
    with the elite parent first (:class:`~dispatchatlas.solvers.population.brkga.Brkga`).

    Args:
        first: The keys of the first parent, the fitter one by the caller's choice.
        second: The keys of the second parent, of the same shape.
        rng: The generator of the draws.
        inheritance: The probability, in [0, 1], that a key comes from ``first``.

    Returns:
        The keys of the child, a new array.

    Raises:
        ValueError: If the parents differ in shape or ``inheritance`` lies outside
            [0, 1].
    """
    _check_inheritance(inheritance)
    keys = np.asarray(first, dtype=np.float64)
    other = np.asarray(second, dtype=np.float64)
    if keys.ndim != 1 or keys.shape != other.shape:
        raise ValueError(f"the parents hold keys of shapes {keys.shape} and {other.shape}")
    return np.asarray(np.where(rng.random(keys.size) < inheritance, keys, other), dtype=np.float64)


def job_order_crossover(
    first: JobSequence, second: JobSequence, rng: np.random.Generator, inheritance: float
) -> NDArray[np.intp]:
    """Return a child with some jobs where ``first`` has them, the others in ``second``'s order.

    This is the job-based order crossover, JOX (Ono, Yamamura and Kobayashi 1996,
    doi:10.1109/ICEC.1996.542658), on job sequences. One uniform draw per job
    keeps the job when it falls below ``inheritance``. The child holds every
    occurrence of a kept job at the position it has in ``first``, and fills the
    other positions, from left to right, with the occurrences of the other jobs
    in the order they come in ``second``. Every job occurs in the child as often
    as in its parents, so the child is a job sequence of their job shop.

    Args:
        first: The first parent, the fitter one by the caller's choice.
        second: The second parent, a permutation of ``first``.
        rng: The generator of the draws.
        inheritance: The probability, in [0, 1], that a job keeps its positions
            in ``first``.

    Returns:
        The child, a new array.

    Raises:
        ValueError: If the parents are not one row of the same job indices, or
            ``inheritance`` lies outside [0, 1].
    """
    _check_inheritance(inheritance)
    jobs = np.asarray(first)
    other = np.asarray(second)
    if (
        jobs.ndim != 1
        or jobs.shape != other.shape
        or jobs.size == 0
        or jobs.dtype.kind not in "iu"
        or other.dtype.kind not in "iu"
    ):
        raise ValueError(
            f"the parents must be rows of job indices, got {jobs.shape} and {other.shape}"
        )
    ordered = np.sort(jobs)
    if ordered[0] < 0 or not np.array_equal(ordered, np.sort(other)):
        raise ValueError("the parents must hold the same non-negative job indices")
    kept = rng.random(int(ordered[-1]) + 1) < inheritance
    child = jobs.astype(np.intp)
    child[~kept[jobs]] = other[~kept[other]]
    return child


class Representation[G](ABC):
    """How a job-shop search family draws, varies and decodes its solutions.

    A family written against this interface runs the same control logic on
    either representation: it draws random solutions with :meth:`initial`,
    varies them with :meth:`move` and :meth:`recombine`, and hands them to
    :class:`~dispatchatlas.solvers.shop.run.ShopRun`, which decodes the job
    sequence :meth:`sequence` gives with the run's builder. ``G`` is the type of
    a solution, the genotype.

    Args:
        shop: The job shop.

    Attributes:
        shop: The job shop.
    """

    def __init__(self, shop: JobShop) -> None:
        self.shop = shop

    def initial(self, rng: np.random.Generator) -> G:
        """Return a random solution: one uniform draw per slot, handed to :meth:`from_keys`.

        Every representation draws the same numbers here, so two generators in
        the same state give the same job sequence in every representation, and
        are left in the same state. The job sequence is uniformly random.

        Args:
            rng: The generator of the draws.
        """
        return self.from_keys(rng.random(self.shop.size))

    @abstractmethod
    def from_keys(self, keys: ArrayLike) -> G:
        """Return the solution that random keys stand for: its job sequence is theirs.

        Args:
            keys: One key in [0, 1] per slot.

        Raises:
            ValueError: If the keys do not fit the slots or lie outside [0, 1].
        """

    @abstractmethod
    def sequence(self, genotype: G) -> NDArray[np.intp]:
        """Return the job sequence that a solution stands for, which the builders decode."""

    @abstractmethod
    def move(self, genotype: G, rng: np.random.Generator) -> G:
        """Return a neighbour of a solution, a new genotype that differs from it."""

    @abstractmethod
    def recombine(self, first: G, second: G, rng: np.random.Generator, inheritance: float) -> G:
        """Return a child of two solutions that favours ``first`` by ``inheritance``.

        Callers pass the fitter parent first.
        """

    @abstractmethod
    def follow(self, genotype: G, sequence: JobSequence) -> G:
        """Return a solution that stands for ``sequence``, keeping what it can of ``genotype``.

        A family writes the job sequence of an improved schedule, such as one
        that a local search found, back into its solution with it.

        Raises:
            ModelError: If ``sequence`` is not a job sequence of the shop.
        """

    def _job_sequence(self, sequence: JobSequence) -> NDArray[np.intp]:
        """Return ``sequence`` as an integer array after checking it is a job sequence."""
        array = np.asarray(sequence)
        if (
            array.ndim != 1
            or array.dtype.kind not in "iu"
            or not np.array_equal(np.sort(array), self.shop.slot_job)
        ):
            raise ModelError(
                f"the sequence is not a job sequence of problem {self.shop.problem.id!r}",
                check_sequence(self.shop, sequence),
            )
        return array.astype(np.intp)


class KeyRepresentation(Representation[Keys]):
    """Random keys: one key in [0, 1] per slot, sorted into a job sequence.

    Key ``i`` belongs to slot ``i`` of
    :attr:`~dispatchatlas.solvers.shop.instance.JobShop.slot_job`, and the job
    sequence lists the jobs of the slots by ascending key, ties in slot order
    (:func:`~dispatchatlas.solvers.shop.decoding.sequence_from_keys`).

    A move picks ``key_count`` distinct slots uniformly and gives each a new key.
    ``"perturb"`` adds to each a normal step of standard deviation ``key_step``
    and reflects the sum into [0, 1] (:func:`reflect`); ``"redraw"`` draws each
    anew, uniformly in [0, 1). A proposal equal to the key it replaces is drawn
    again, so every chosen key changes; a step too small to change a key in
    floating point, after 64 proposals, moves it to the adjacent float towards
    the middle of [0, 1]. Recombination is
    :func:`parameterised_uniform_crossover`.

    Args:
        shop: The job shop.
        key_move: ``"perturb"`` or ``"redraw"``.
        key_step: The standard deviation of a perturbation, positive and finite.
            Only ``"perturb"`` reads it; :func:`~dispatchatlas.solvers.shop.distance.calibrate_step`
            sets it so that a key move goes as far as a move of the job sequences.
        key_count: How many keys a move changes, from 1 to the number of slots.

    Attributes:
        key_move: How a move changes its keys.
        key_step: The standard deviation of a perturbation.
        key_count: How many keys a move changes.

    Raises:
        ValueError: If ``key_move`` is unknown, ``key_step`` is not positive and
            finite, or ``key_count`` lies outside 1 to the number of slots.
    """

    def __init__(
        self,
        shop: JobShop,
        key_move: KeyMove = "perturb",
        key_step: float = 0.1,
        key_count: int = 1,
    ) -> None:
        if key_move not in KEY_MOVES:
            raise ValueError(f"unknown key move {key_move!r}; expected one of {list(KEY_MOVES)}")
        if not (math.isfinite(key_step) and key_step > 0.0):
            raise ValueError(f"key_step must be positive and finite, got {key_step!r}")
        try:
            count = operator.index(key_count)
        except TypeError:
            raise ValueError(f"key_count must be an integer, got {key_count!r}") from None
        if isinstance(key_count, bool) or not 1 <= count <= shop.size:
            raise ValueError(f"key_count must lie in 1..{shop.size}, got {key_count!r}")
        super().__init__(shop)
        self.key_move: KeyMove = key_move
        self.key_step = float(key_step)
        self.key_count = count

    def __repr__(self) -> str:
        return (
            f"KeyRepresentation(shop={self.shop!r}, key_move={self.key_move!r}, "
            f"key_step={self.key_step!r}, key_count={self.key_count})"
        )

    def from_keys(self, keys: ArrayLike) -> Keys:
        """Return a copy of the keys, which are the solution itself.

        Raises:
            ValueError: If the keys do not fit the slots or lie outside [0, 1].
        """
        return _unit_keys(np.array(keys, dtype=np.float64), self.shop.size)

    def sequence(self, genotype: Keys) -> NDArray[np.intp]:
        """Return the job sequence of the keys: the jobs of the slots by ascending key.

        Raises:
            ValueError: If the genotype is not a row of finite keys, one per slot.
        """
        return sequence_from_keys(self._keys(genotype), self.shop.slot_job)

    def move(self, genotype: Keys, rng: np.random.Generator) -> Keys:
        """Return a copy of the keys in which ``key_count`` keys have new values.

        The slots are distinct and uniformly chosen, and each of their keys
        changes. The job sequence changes when a key passes a key of another job.

        Raises:
            ValueError: If the genotype is not a row of float keys, one per slot.
        """
        keys = self._keys(genotype)
        slots = rng.choice(keys.size, self.key_count, replace=False)
        old = keys[slots]
        values = self._propose(old, rng)
        for _ in range(_PROPOSALS):
            same = values == old
            if not same.any():
                break
            values[same] = self._propose(old[same], rng)
        else:
            same = values == old
            values[same] = np.nextafter(old[same], np.where(old[same] < 0.5, 1.0, 0.0))
        moved = keys.copy()
        moved[slots] = values
        return moved

    def recombine(
        self, first: Keys, second: Keys, rng: np.random.Generator, inheritance: float
    ) -> Keys:
        """Return :func:`parameterised_uniform_crossover` of the two parents' keys."""
        return parameterised_uniform_crossover(first, second, rng, inheritance)

    def follow(self, genotype: Keys, sequence: JobSequence) -> Keys:
        """Return the keys of ``genotype`` rearranged so that they sort into ``sequence``.

        The ``r``-th smallest key goes to the slot of the operation at position
        ``r`` of ``sequence``. The ``k``-th occurrence of a job there takes the
        slot of the job whose key comes ``k``-th among the job's keys, so each
        job's slots keep the order of their keys, and keys that already sort
        into ``sequence`` stay as they are. The keys keep their values, and when
        they are distinct, which they are with probability one, the new keys
        sort into ``sequence`` exactly.

        Raises:
            ValueError: If the genotype is not a row of float keys, one per slot.
            ModelError: If ``sequence`` is not a job sequence of the shop.
        """
        keys = self._keys(genotype)
        target = self._job_sequence(sequence)
        slot_job = self.shop.slot_job
        by_key = np.argsort(keys, kind="stable")
        # Each job's slots in the order of their keys, and each job's positions in
        # ``target`` in order, both job by job: the k-th position of a job takes
        # its k-th slot.
        slots = by_key[np.argsort(slot_job[by_key], kind="stable")]
        slot_at = np.empty(keys.size, dtype=np.intp)
        slot_at[np.argsort(target, kind="stable")] = slots
        followed = np.empty(keys.size, dtype=np.float64)
        followed[slot_at] = keys[by_key]
        return followed

    def _keys(self, genotype: Keys) -> Keys:
        """Return the genotype as a float array after checking it holds one key per slot."""
        keys = np.asarray(genotype)
        if keys.dtype.kind != "f" or keys.shape != (self.shop.size,):
            raise ValueError(
                f"expected {self.shop.size} keys in one row, got {keys.dtype} of shape {keys.shape}"
            )
        return keys.astype(np.float64, copy=False)

    def _propose(self, old: Keys, rng: np.random.Generator) -> Keys:
        """Return a proposed new value for each key of ``old``."""
        if self.key_move == "redraw":
            return rng.random(old.size)
        return reflect(old + self.key_step * rng.standard_normal(old.size))


class SequenceRepresentation(Representation[NDArray[np.intp]]):
    """Job sequences: the solution is the sequence the builders decode.

    A move draws two distinct positions ``i`` and ``j`` uniformly and either

    - **swaps** the jobs at ``i`` and ``j``, or
    - **inserts** the job at ``i`` at position ``j``, shifting the jobs between
      them by one place.

    A draw whose move would leave the sequence as it is is drawn again: a swap
    of two positions that hold the same job, or an insert whose job passes only
    occurrences of itself. Each move is therefore uniform over the swaps, or the
    inserts, that change the sequence. ``"mixed"`` first draws a swap or an
    insert with probability one half each. Recombination is
    :func:`job_order_crossover`.

    A job shop of one job has a single job sequence, which no move changes:
    :meth:`move` refuses it, and a family evaluates that sequence and stops.

    Args:
        shop: The job shop.
        move: ``"swap"``, ``"insert"``, or ``"mixed"``.

    Attributes:
        move_kind: The move, as ``move`` named it.

    Raises:
        ValueError: If ``move`` is unknown.
    """

    def __init__(self, shop: JobShop, move: SequenceMove = "swap") -> None:
        if move not in SEQUENCE_MOVES:
            raise ValueError(
                f"unknown sequence move {move!r}; expected one of {list(SEQUENCE_MOVES)}"
            )
        super().__init__(shop)
        self.move_kind: SequenceMove = move

    def __repr__(self) -> str:
        return f"SequenceRepresentation(shop={self.shop!r}, move={self.move_kind!r})"

    def from_keys(self, keys: ArrayLike) -> NDArray[np.intp]:
        """Return the job sequence the keys sort into.

        Raises:
            ValueError: If the keys do not fit the slots or lie outside [0, 1].
        """
        values = _unit_keys(np.asarray(keys, dtype=np.float64), self.shop.size)
        return sequence_from_keys(values, self.shop.slot_job)

    def sequence(self, genotype: NDArray[np.intp]) -> NDArray[np.intp]:
        """Return the genotype itself; the builder checks that it is a job sequence.

        Raises:
            ValueError: If the genotype is not a row of job indices, one per slot.
        """
        return self._jobs(genotype)

    def move(self, genotype: NDArray[np.intp], rng: np.random.Generator) -> NDArray[np.intp]:
        """Return a copy of the sequence changed by one swap or insert.

        Raises:
            ValueError: If the genotype is not a row of job indices, one per slot.
            SolverError: If the job shop has one job, so that no move exists.
        """
        sequence = self._jobs(genotype)
        if self.shop.jobs < 2:
            raise SolverError(
                f"problem {self.shop.problem.id!r} has one job, whose sequence no move changes"
            )
        kind = self.move_kind
        if kind == "mixed":
            kind = "swap" if rng.random() < 0.5 else "insert"
        size = sequence.size
        while True:
            first = int(rng.integers(size))
            second = int(rng.integers(size - 1))
            if second >= first:
                second += 1
            if kind == "swap":
                if sequence[first] != sequence[second]:
                    moved = sequence.astype(np.intp)
                    moved[first], moved[second] = sequence[second], sequence[first]
                    return moved
            elif (sequence[min(first, second) : max(first, second) + 1] != sequence[first]).any():
                return _insert(sequence, first, second)

    def recombine(
        self,
        first: NDArray[np.intp],
        second: NDArray[np.intp],
        rng: np.random.Generator,
        inheritance: float,
    ) -> NDArray[np.intp]:
        """Return :func:`job_order_crossover` of the two parents."""
        return job_order_crossover(first, second, rng, inheritance)

    def follow(
        self,
        genotype: NDArray[np.intp],  # noqa: ARG002 - a job sequence stands for itself
        sequence: JobSequence,
    ) -> NDArray[np.intp]:
        """Return a copy of ``sequence``, which is the solution that stands for it.

        Raises:
            ModelError: If ``sequence`` is not a job sequence of the shop.
        """
        return self._job_sequence(sequence)

    def _jobs(self, genotype: NDArray[np.intp]) -> NDArray[np.intp]:
        """Return the genotype as an integer array after checking its type and shape."""
        sequence = np.asarray(genotype)
        if sequence.dtype.kind not in "iu" or sequence.shape != (self.shop.size,):
            raise ValueError(
                f"expected {self.shop.size} job indices in one row, got {sequence.dtype} "
                f"of shape {sequence.shape}"
            )
        return sequence.astype(np.intp, copy=False)


def _insert(sequence: NDArray[np.integer[Any]], source: int, target: int) -> NDArray[np.intp]:
    """Return a copy of ``sequence`` whose entry at ``source`` has moved to ``target``."""
    moved = sequence.astype(np.intp)
    if source < target:
        moved[source:target] = sequence[source + 1 : target + 1]
    else:
        moved[target + 1 : source + 1] = sequence[target:source]
    moved[target] = sequence[source]
    return moved
