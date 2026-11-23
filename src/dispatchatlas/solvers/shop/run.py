"""The run of a job-shop search family: one representation, one builder, two random streams.

A search family of this package searches through a :class:`ShopRun`, which holds
the run's :class:`~dispatchatlas.solvers.search.Search`, the builder of the run
(:class:`~dispatchatlas.solvers.shop.decoding.JobSequenceDecoder`), the
:class:`~dispatchatlas.solvers.shop.representation.Representation` the family
searches, an optional :class:`Observer`, and two random generators derived from
the run's seed with :func:`~dispatchatlas.core.seeds.derive_seed`, each created
once and held for the run:

- :attr:`ShopRun.control`, the **control stream**, for the family's control
  logic: selection, replacement, acceptance, and the ties they break;
- :attr:`ShopRun.operators`, the **operator stream**, for the representation's
  operators: random solutions, moves, and recombination.

**Common random numbers.** The streams depend on the seed alone, not on the
representation or the builder. Two runs of a family with the same seed draw the
same control numbers, and when the first draw of the operator stream is a random
solution (:meth:`~dispatchatlas.solvers.shop.representation.Representation.initial`),
both start from the same job sequence whatever their representation. The digest
of the first schedule a run evaluates is recorded, so that a check can compare
the runs of a family.

**Evaluation.** :meth:`ShopRun.evaluate` hands the decoding of a genotype to
:meth:`~dispatchatlas.solvers.search.Search.assess_built`, which checks the
budget before the job sequence is computed and the schedule built, and counts
the schedule as one evaluation, in every representation. The builders of the
package never repair a sequence, and :meth:`ShopRun.finish` raises
:class:`~dispatchatlas.core.errors.SolverError` when the run's search decoded an
order through the precedence repair all the same.

**Measures.** :meth:`ShopRun.finish` writes into the extras of the result:

- ``first_schedule_digest``: the :func:`schedule_digest` of the first schedule
  evaluated, in hexadecimal;
- ``decode_seconds``: the time spent computing job sequences and building
  schedules, outside calibration;
- ``observer_seconds``: the time the observer took, which no evaluation counts;
- ``calibration_seconds`` and ``calibration_evaluations``: the time and the
  evaluations of the run's :meth:`ShopRun.calibration` blocks;
- ``decode_share``: ``decode_seconds`` over the run's elapsed time less
  ``observer_seconds`` and ``calibration_seconds``;
- ``repairs``: the decodes through the precedence repair, which is 0;
- the observer's own measures, such as those of
  :class:`~dispatchatlas.solvers.shop.observers.TrajectoryObserver`.
"""

from __future__ import annotations

import contextlib
import hashlib
import struct
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from dispatchatlas.core.errors import SolverError
from dispatchatlas.core.seeds import derive_seed
from dispatchatlas.core.solver import StopReason
from dispatchatlas.solvers.search import Exhausted, Search
from dispatchatlas.solvers.shop.decoding import BuilderName, JobSequenceDecoder

if TYPE_CHECKING:
    from dispatchatlas.core.model import Scalar
    from dispatchatlas.solvers.evaluation import Score
    from dispatchatlas.solvers.scheduling import Decoded
    from dispatchatlas.solvers.shop.instance import JobShop
    from dispatchatlas.solvers.shop.representation import Representation

__all__ = [
    "CONTROL_STREAM",
    "OPERATOR_STREAM",
    "Evaluated",
    "Observer",
    "ShopRun",
    "schedule_digest",
]

CONTROL_STREAM = "shop:control"
"""The namespace from which :class:`ShopRun` derives its control stream."""

OPERATOR_STREAM = "shop:operators"
"""The namespace from which :class:`ShopRun` derives its operator stream."""


def schedule_digest(decoded: Decoded) -> bytes:
    """Return the 16-byte BLAKE2b digest of a schedule's start times.

    The start times are hashed as little-endian 64-bit floats in task-index
    order, so the digest is the same on every platform. In one job shop the start
    times fix the schedule, so two schedules of the shop have equal digests
    exactly when they are equal, but for a collision of the 128-bit digest.

    Args:
        decoded: The schedule.

    Returns:
        The digest.
    """
    starts = decoded.starts
    return hashlib.blake2b(struct.pack(f"<{len(starts)}d", *starts), digest_size=16).digest()


class Evaluated[G]:
    """A genotype that a run evaluated, with its job sequence, schedule, and score.

    Attributes:
        genotype: The genotype, as the family handed it to :meth:`ShopRun.evaluate`.
        sequence: The job sequence the builder decoded. For job sequences it is
            the genotype itself.
        score: The score of the schedule.
        decoded: The schedule.
    """

    __slots__ = ("_digest", "decoded", "genotype", "score", "sequence")

    def __init__(
        self, genotype: G, sequence: NDArray[np.intp], score: Score, decoded: Decoded
    ) -> None:
        self.genotype = genotype
        self.sequence = sequence
        self.score = score
        self.decoded = decoded
        self._digest: bytes | None = None

    @property
    def digest(self) -> bytes:
        """The :func:`schedule_digest` of the schedule, computed when first read."""
        if self._digest is None:
            self._digest = schedule_digest(self.decoded)
        return self._digest


class Observer(Protocol):
    """What a :class:`ShopRun` reports its evaluations to.

    An observer reads what it is given and nothing else: it draws from no random
    stream and evaluates nothing, so a run with an observer makes the same
    decisions as one without. The run times its calls apart from the decoding.
    """

    def observe(self, evaluated: Evaluated[Any], parent: Evaluated[Any] | None) -> None:
        """See one evaluation, with the evaluation it came from when the family names one."""
        ...

    def measures(self) -> dict[str, Scalar]:
        """Return the measures to add to the extras of the result."""
        ...


class ShopRun[G]:
    """A run of a job-shop search family: it decodes, counts, observes, and measures.

    The family creates the run inside its solver's ``solve``, after the search,
    draws from :attr:`control` and :attr:`operators` only, evaluates every
    genotype with :meth:`evaluate`, and calls :meth:`finish` once the search is
    exhausted, before it returns the result::

        search = Search(problem, stop, seed, self.info)
        shop = JobShop.from_problem(problem)
        run = ShopRun(search, shop, "active", KeyRepresentation(shop), TrajectoryObserver(shop))
        with search:
            current = run.evaluate(run.representation.initial(run.operators))
            while True:
                search.next_iteration()
                trial = run.evaluate(
                    run.representation.move(current.genotype, run.operators), parent=current
                )
                ...  # accept or reject with run.control
        run.finish()
        return search.result()

    Args:
        search: The run's search, which counts every evaluation.
        shop: The job-shop view of the search's problem.
        builder: The schedule builder of the run.
        representation: The representation the family searches, of ``shop``.
        observer: Sees every evaluation outside calibration, or ``None``.

    Attributes:
        search: The run's search.
        shop: The job shop.
        decoder: The builder of the run.
        representation: The representation.
        observer: The observer, or ``None``.
        control: The control stream, a generator seeded with
            ``derive_seed(search.seed, CONTROL_STREAM)``.
        operators: The operator stream, a generator seeded with
            ``derive_seed(search.seed, OPERATOR_STREAM)``.
        decode_seconds: Seconds spent computing job sequences and building
            schedules, outside calibration.
        observer_seconds: Seconds the observer took.
        calibration_seconds: Seconds spent in :meth:`calibration` blocks.
        calibration_evaluations: Evaluations made in :meth:`calibration` blocks.
        first_digest: The :func:`schedule_digest` of the first schedule the run
            evaluated, ``None`` before the first evaluation.

    Raises:
        ValueError: If the representation belongs to another job shop, the job
            shop is not a view of the search's problem, the builder is unknown,
            or the search's seed is negative.
    """

    def __init__(
        self,
        search: Search,
        shop: JobShop,
        builder: BuilderName,
        representation: Representation[G],
        observer: Observer | None = None,
    ) -> None:
        if representation.shop is not shop:
            raise ValueError("the representation belongs to another job shop")
        if shop.problem is not search.problem and shop.problem != search.problem:
            raise ValueError(
                f"the job shop views problem {shop.problem.id!r}, not the search's problem "
                f"{search.problem.id!r}"
            )
        self.search = search
        self.shop = shop
        self.decoder = JobSequenceDecoder(shop, builder)
        self.representation = representation
        self.observer = observer
        self.control = np.random.default_rng(derive_seed(search.seed, CONTROL_STREAM))
        self.operators = np.random.default_rng(derive_seed(search.seed, OPERATOR_STREAM))
        self.decode_seconds = 0.0
        self.observer_seconds = 0.0
        self.calibration_seconds = 0.0
        self.calibration_evaluations = 0
        self.first_digest: bytes | None = None
        self._calibrating = False

    def evaluate(self, genotype: G, parent: Evaluated[G] | None = None) -> Evaluated[G]:
        """Decode a genotype with the run's builder and count it as one evaluation.

        The search checks the budget first, and only then is the job sequence
        computed and the schedule built, so a run at its limit builds nothing.
        The schedule updates the incumbent and the trace as every evaluation
        does, and the observer sees it, with ``parent``, unless the evaluation
        is part of a calibration.

        Args:
            genotype: A solution of the run's representation.
            parent: The evaluation that ``genotype`` was made from, such as the
                solution a move changed, which the observer compares it with, or
                ``None``.

        Returns:
            The evaluation.

        Raises:
            Exhausted: When the evaluation or time limit is reached before the
                genotype is decoded, or the target is reached by its schedule.
        """
        representation, decoder = self.representation, self.decoder
        built: list[NDArray[np.intp]] = []

        def build() -> Decoded:
            started = time.perf_counter()
            sequence = representation.sequence(genotype)
            decoded = decoder.decode(sequence)
            if not self._calibrating:
                self.decode_seconds += time.perf_counter() - started
            built.append(sequence)
            return decoded

        try:
            candidate = self.search.assess_built(build)
        except Exhausted as stop:
            # A schedule that reaches the target is counted before the search stops,
            # and it is the incumbent then.
            best = self.search.best
            if stop.reason is StopReason.TARGET_REACHED and built and best is not None:
                self._note(Evaluated(genotype, built[0], best.score, best.decoded), parent)
            raise
        evaluated = Evaluated(genotype, built[0], candidate.score, candidate.decoded)
        self._note(evaluated, parent)
        return evaluated

    @contextlib.contextmanager
    def calibration(self) -> Iterator[None]:
        """Time a block that calibrates the family, such as its initial temperature.

        The evaluations made in the block count toward the budget like any
        other, but the observer does not see them and their decoding is not
        counted in ``decode_seconds``: the block's time and evaluations are
        recorded as ``calibration_seconds`` and ``calibration_evaluations``.

        Raises:
            SolverError: If a calibration block is already open.
        """
        if self._calibrating:
            raise SolverError("calibration blocks of a job-shop run do not nest")
        self._calibrating = True
        evaluations = self.search.evaluations
        started = time.perf_counter()
        try:
            yield
        finally:
            self.calibration_seconds += time.perf_counter() - started
            self.calibration_evaluations += self.search.evaluations - evaluations
            self._calibrating = False

    def finish(self) -> None:
        """Write the run's measures and its observer's into the extras of the result.

        Call it once the search is exhausted, before ``search.result()``.

        Raises:
            SolverError: If the run's search decoded any order through the
                precedence repair, which no job-sequence builder calls.
        """
        repairs = self.search.evaluator.decoder.repairs
        if repairs:
            raise SolverError(
                f"solver {self.search.info.id!r} decoded {repairs} orders through the "
                "precedence repair, which the job-sequence builders never call"
            )
        extras = self.search.extras
        if self.observer is not None:
            started = time.perf_counter()
            measures = self.observer.measures()
            self.observer_seconds += time.perf_counter() - started
            extras.update(measures)
        searching = self.search.elapsed - self.observer_seconds - self.calibration_seconds
        if self.first_digest is not None:
            extras["first_schedule_digest"] = self.first_digest.hex()
        extras["decode_seconds"] = self.decode_seconds
        extras["decode_share"] = self.decode_seconds / searching if searching > 0.0 else 0.0
        extras["observer_seconds"] = self.observer_seconds
        extras["calibration_seconds"] = self.calibration_seconds
        extras["calibration_evaluations"] = self.calibration_evaluations
        extras["repairs"] = repairs

    def _note(self, evaluated: Evaluated[G], parent: Evaluated[G] | None) -> None:
        """Record the first digest, and show an evaluation outside calibration to the observer."""
        if self.first_digest is None:
            self.first_digest = evaluated.digest
        if self.observer is not None and not self._calibrating:
            started = time.perf_counter()
            self.observer.observe(evaluated, parent)
            self.observer_seconds += time.perf_counter() - started
