"""Random sampling of job sequences, the validation pair of the job-shop search families.

:class:`ShopRandomSampling` evaluates random solutions of its representation,
one per iteration, until a stop criterion fires, and returns the best. A random
solution is one uniform number per slot
(:meth:`~dispatchatlas.solvers.shop.representation.Representation.initial`):
random keys keep the numbers as the keys, and job sequences sort the slots by
them, so the two representations draw the same numbers and decode the same
uniformly random job sequence. Two runs with the same seed and builder, one on
each representation, therefore evaluate the same schedules in the same order,
and under an evaluation or iteration limit they make the same number of
evaluations: any difference between them comes from the code around the
representations, not from the representations, which makes the pair a check of
that code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.shop.decoding import BuilderName
from dispatchatlas.solvers.shop.family import RepresentationName, check_family, make_representation
from dispatchatlas.solvers.shop.instance import JOB_SHOP_FEATURES, JobShop
from dispatchatlas.solvers.shop.observers import TrajectoryObserver
from dispatchatlas.solvers.shop.run import ShopRun

__all__ = ["ShopRandomSampling"]


@dataclass(frozen=True)
class ShopRandomSampling:
    """Decode uniformly random job sequences of a job shop with one builder and keep the best.

    Every iteration draws a random solution of the representation from the
    run's operator stream and evaluates it through a
    :class:`~dispatchatlas.solvers.shop.run.ShopRun`, whose
    :class:`~dispatchatlas.solvers.shop.observers.TrajectoryObserver` sees
    every evaluation. The family has no control logic, so it draws nothing
    from the control stream. A job shop of one job has a single job sequence,
    which the run evaluates once before it stops.

    Attributes:
        representation: ``"keys"`` or ``"sequence"``
            (:data:`~dispatchatlas.solvers.shop.family.REPRESENTATIONS`).
        builder: The schedule builder, one of
            :data:`~dispatchatlas.solvers.shop.decoding.BUILDERS`.
    """

    representation: RepresentationName = "sequence"
    builder: BuilderName = "active"

    info: ClassVar[SolverInfo] = SolverInfo(
        id="shop-random-sampling",
        name="Job-shop random sampling",
        family="random-sampling",
        summary="Decodes uniformly random job sequences, drawn as random keys or as job "
        "sequences, with one job-shop schedule builder until the budget runs out and "
        "returns the best.",
        references=("bean1994genetic", "bierwirth1995generalized"),
        supports=JOB_SHOP_FEATURES,
        shape="job-shop",
    )

    def __post_init__(self) -> None:
        check_family(self.representation, self.builder)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Evaluate random solutions until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        shop = JobShop.from_problem(problem)
        representation = make_representation(shop, self.representation)
        run = ShopRun(search, shop, self.builder, representation, TrajectoryObserver(shop))
        with search:
            while True:
                search.next_iteration()
                run.evaluate(representation.initial(run.operators))
                if shop.jobs < 2:
                    break
        run.finish()
        return search.result()
