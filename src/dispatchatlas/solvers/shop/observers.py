"""Observers of the trajectory of a job-shop run.

An observer sees every evaluation that a :class:`~dispatchatlas.solvers.shop.run.ShopRun`
makes outside its calibration, with the evaluation it came from when the family
names one, and adds its measures to the extras of the result when the run
finishes. It reads schedules only: it draws from no random stream and
evaluates nothing, and the run times it apart from the decoding, so it changes
neither the search nor its evaluation count nor its decode time.

:class:`TrajectoryObserver` measures how a search moves through the schedules:

- ``revisit_share``: the share of evaluations that rebuild a schedule evaluated
  before in the run, told apart by the
  :func:`~dispatchatlas.solvers.shop.run.schedule_digest` of their start times;
- ``unchanged_move_share``: the share of moves, the evaluations with a parent,
  whose schedule is the parent's;
- ``phenotype_distance_mean`` and ``phenotype_distance_median``: the distance
  of a move between the schedules, the Kendall tau distance between their
  machine orders (:func:`~dispatchatlas.solvers.shop.graph.machine_orders`):
  the number of pairs of operations on one machine that the two schedules
  process in different orders, summed over the machines;
- ``observed_evaluations``, ``observed_moves`` and ``measured_moves``: the
  counts the shares and the distances are taken over.

A share or a distance over no evaluation is left out of the extras.
"""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from dispatchatlas.solvers.shop.graph import machine_orders

if TYPE_CHECKING:
    from dispatchatlas.core.model import Scalar
    from dispatchatlas.solvers.scheduling import Decoded
    from dispatchatlas.solvers.shop.instance import JobShop
    from dispatchatlas.solvers.shop.run import Evaluated

__all__ = ["TrajectoryObserver"]

# Evaluations whose machine-order ranks the observer keeps, so that a parent
# shared by many moves has its ranks computed once.
_KEPT = 2


class TrajectoryObserver:
    """Measures revisits, unchanged moves, and the phenotype distance of moves.

    Every evaluation it sees counts toward the share of revisits, every
    evaluation with a parent toward the share of unchanged moves, and the first
    move and every ``distance_every``-th one after it toward the phenotype
    distance, which is computed from the machine orders of both schedules. The
    distance compares every pair of operations on a machine with NumPy at once,
    so it takes time linear in the number of such pairs.

    Args:
        shop: The job shop of the run.
        distance_every: Measure the distance of one move in this many, at
            least 1.

    Attributes:
        shop: The job shop.
        distance_every: One move in this many is measured.
        evaluations: Evaluations seen.
        revisits: Evaluations whose schedule had been seen before.
        moves: Evaluations seen with a parent.
        unchanged: Moves whose schedule is the parent's.
        distances: The phenotype distance of each measured move, in order.
        pairs: The number of pairs of operations on one machine, the largest
            distance.

    Raises:
        ValueError: If ``distance_every`` is not an integer of at least 1.
    """

    def __init__(self, shop: JobShop, distance_every: int = 1) -> None:
        if (
            isinstance(distance_every, bool)
            or not isinstance(distance_every, int)
            or distance_every < 1
        ):
            raise ValueError(
                f"distance_every must be an integer of at least 1, got {distance_every!r}"
            )
        self.shop = shop
        self.distance_every = distance_every
        self.evaluations = 0
        self.revisits = 0
        self.moves = 0
        self.unchanged = 0
        self.distances: list[int] = []
        self._seen: set[bytes] = set()
        # Every pair of operations on one machine, as two arrays of task indices.
        machine = np.asarray(shop.machine, dtype=np.intp)
        firsts: list[NDArray[np.intp]] = [np.empty(0, dtype=np.intp)]
        seconds: list[NDArray[np.intp]] = [np.empty(0, dtype=np.intp)]
        for number in range(shop.machines):
            operations = np.flatnonzero(machine == number)
            earlier, later = np.triu_indices(operations.size, k=1)
            firsts.append(operations[earlier])
            seconds.append(operations[later])
        self._first = np.concatenate(firsts)
        self._second = np.concatenate(seconds)
        self.pairs = int(self._first.size)
        self._places = np.arange(shop.size, dtype=np.intp)
        self._kept: list[tuple[Evaluated[Any], NDArray[np.intp]]] = []

    def __repr__(self) -> str:
        return f"TrajectoryObserver(shop={self.shop!r}, distance_every={self.distance_every})"

    def observe(self, evaluated: Evaluated[Any], parent: Evaluated[Any] | None) -> None:
        """Count one evaluation, and measure it as a move of ``parent`` when given one."""
        self.evaluations += 1
        digest = evaluated.digest
        if digest in self._seen:
            self.revisits += 1
        else:
            self._seen.add(digest)
        if parent is None:
            return
        self.moves += 1
        if digest == parent.digest:
            self.unchanged += 1
        if (self.moves - 1) % self.distance_every == 0:
            self.distances.append(
                self._discordant(self._ranks_of(parent), self._ranks_of(evaluated))
            )

    def distance(self, first: Decoded, second: Decoded) -> int:
        """Return the Kendall tau distance between the machine orders of two schedules.

        It is the number of pairs of operations on one machine that the two
        schedules process in different orders, summed over the machines.

        Args:
            first: A schedule of the shop.
            second: Another schedule of the shop.

        Raises:
            ModelError: If a schedule leaves an operation unplaced.
        """
        return self._discordant(self._ranks(first), self._ranks(second))

    def measures(self) -> dict[str, Scalar]:
        """Return the counts, and the shares and distances over at least one evaluation."""
        measures: dict[str, Scalar] = {
            "observed_evaluations": self.evaluations,
            "observed_moves": self.moves,
            "measured_moves": len(self.distances),
        }
        if self.evaluations:
            measures["revisit_share"] = self.revisits / self.evaluations
        if self.moves:
            measures["unchanged_move_share"] = self.unchanged / self.moves
        if self.distances:
            values = np.asarray(self.distances, dtype=np.float64)
            measures["phenotype_distance_mean"] = float(values.mean())
            # The median of the sorted values: np.median imports numpy.ma on its first call in a
            # process, which would charge that import to the run that first finishes.
            ordered = np.sort(values)
            half = ordered.size // 2
            middle = (
                ordered[half] if ordered.size % 2 else (ordered[half - 1] + ordered[half]) / 2.0
            )
            measures["phenotype_distance_median"] = float(middle)
        return measures

    def _ranks(self, decoded: Decoded) -> NDArray[np.intp]:
        """Return the place of each operation in the machine orders, machine by machine.

        Two operations on one machine are processed in the order of their places.
        """
        orders = machine_orders(self.shop, decoded)
        listed = np.fromiter(chain.from_iterable(orders), dtype=np.intp, count=self.shop.size)
        ranks = np.empty(self.shop.size, dtype=np.intp)
        ranks[listed] = self._places
        return ranks

    def _ranks_of(self, evaluated: Evaluated[Any]) -> NDArray[np.intp]:
        """Return the ranks of an evaluation's schedule, kept for the latest evaluations."""
        kept = self._kept
        for index, (held, ranks) in enumerate(kept):
            if held is evaluated:
                kept.append(kept.pop(index))
                return ranks
        ranks = self._ranks(evaluated.decoded)
        kept.append((evaluated, ranks))
        if len(kept) > _KEPT:
            del kept[0]
        return ranks

    def _discordant(self, first: NDArray[np.intp], second: NDArray[np.intp]) -> int:
        """Return the pairs of operations on one machine that two rankings order differently."""
        one, other = self._first, self._second
        return int(np.count_nonzero((first[one] < first[other]) != (second[one] < second[other])))
