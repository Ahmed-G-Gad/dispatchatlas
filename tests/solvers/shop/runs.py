"""Small searches through a job-shop run, for the tests of the run and its observers."""

from __future__ import annotations

from typing import Any

from dispatchatlas.core import SolverInfo
from dispatchatlas.core.model import Scalar
from dispatchatlas.solvers.shop.instance import JOB_SHOP_FEATURES
from dispatchatlas.solvers.shop.run import Evaluated, ShopRun

INFO = SolverInfo(
    id="job-shop-run",
    name="Job-shop run",
    family="random-sampling",
    summary="Searches a job shop through a run of the tests.",
    supports=JOB_SHOP_FEATURES,
    shape="job-shop",
)


def climb(run: ShopRun[Any], *, worse: float = 0.2) -> list[float]:
    """Search by moves from the current solution, keeping a worse one with probability ``worse``.

    The first solution is random, every move is evaluated with the current
    solution as its parent, and the control stream decides whether a worse
    solution is kept. The search runs until its budget is spent.

    Returns:
        The value of every evaluation, in order.
    """
    representation = run.representation
    values: list[float] = []
    with run.search:
        current = run.evaluate(representation.initial(run.operators))
        values.append(current.score.value)
        while True:
            run.search.next_iteration()
            trial = run.evaluate(
                representation.move(current.genotype, run.operators), parent=current
            )
            values.append(trial.score.value)
            if trial.score <= current.score or run.control.random() < worse:
                current = trial
    return values


def sample(run: ShopRun[Any]) -> None:
    """Evaluate random solutions until the budget is spent."""
    with run.search:
        while True:
            run.search.next_iteration()
            run.evaluate(run.representation.initial(run.operators))


class Digests:
    """An observer that keeps the digest and the parent's digest of every evaluation it sees."""

    def __init__(self) -> None:
        self.digests: list[bytes] = []
        self.parents: list[bytes | None] = []

    def observe(self, evaluated: Evaluated[Any], parent: Evaluated[Any] | None) -> None:
        self.digests.append(evaluated.digest)
        self.parents.append(None if parent is None else parent.digest)

    def measures(self) -> dict[str, Scalar]:
        return {"digests": len(self.digests)}
