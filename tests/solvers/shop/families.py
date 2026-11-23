"""Helpers for the tests of the job-shop search families: a record of their runs, and small shops.

A family creates its :class:`ShopRun` inside ``solve``, so a test reaches the
run's evaluations, and the run itself, through the methods of the class only.
:class:`Recording` wraps :meth:`ShopRun.evaluate` and :meth:`ShopRun.finish` for
the length of a test and keeps what they see. The wrappers read the evaluation a
run returns and change nothing in it: the digest they read is computed once and
kept, the one that the run and its observer read too.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from dispatchatlas.core import Dependency, Problem, Resource, Task
from dispatchatlas.core.model import Scalar
from dispatchatlas.solvers.scheduling import Decoded
from dispatchatlas.solvers.shop.decoding import BuilderName
from dispatchatlas.solvers.shop.run import Evaluated, ShopRun

PAIRED_BUILDERS: tuple[BuilderName, ...] = ("semi-active", "active")
"""The builders under which the families' two representations are compared."""

TIMINGS = frozenset({"decode_seconds", "decode_share", "observer_seconds", "calibration_seconds"})
"""The extras that time a run, which differ between any two runs."""


@dataclass(frozen=True)
class Evaluation:
    """One evaluation of a run: its schedule, the schedule's digest and value, and its parent's."""

    decoded: Decoded
    digest: bytes
    value: float
    parent: bytes | None
    parent_value: float | None


class Recording:
    """Keeps every evaluation of every run, and every run that finishes, while a test lasts.

    Args:
        monkeypatch: The test's monkeypatch, which undoes the wrappers after it.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.evaluations: dict[ShopRun[Any], list[Evaluation]] = {}
        self.finished: list[ShopRun[Any]] = []
        evaluate, finish = ShopRun.evaluate, ShopRun.finish

        def recorded_evaluate(
            run: ShopRun[Any], genotype: Any, parent: Evaluated[Any] | None = None
        ) -> Evaluated[Any]:
            evaluated = evaluate(run, genotype, parent)
            self.evaluations.setdefault(run, []).append(
                Evaluation(
                    evaluated.decoded,
                    evaluated.digest,
                    evaluated.score.value,
                    None if parent is None else parent.digest,
                    None if parent is None else parent.score.value,
                )
            )
            return evaluated

        def recorded_finish(run: ShopRun[Any]) -> None:
            self.finished.append(run)
            finish(run)

        monkeypatch.setattr(ShopRun, "evaluate", recorded_evaluate)
        monkeypatch.setattr(ShopRun, "finish", recorded_finish)

    def runs(self) -> list[list[Evaluation]]:
        """Return the evaluations of each run, the runs in the order of their first evaluation."""
        return list(self.evaluations.values())


def untimed(extras: Mapping[str, Scalar]) -> dict[str, Scalar]:
    """Return the extras of a result without those that time the run."""
    return {key: value for key, value in extras.items() if key not in TIMINGS}


def one_job(operations: int = 2) -> Problem:
    """Return a job shop of one job, whose operations alternate between two machines."""
    tasks = tuple(
        Task.on(f"a{step}", f"m{step % 2}", float(step + 2)) for step in range(operations)
    )
    dependencies = tuple(Dependency(f"a{step - 1}", f"a{step}") for step in range(1, operations))
    return Problem(
        f"one-job-{operations}",
        tasks=tasks,
        resources=(Resource("m0"), Resource("m1")),
        dependencies=dependencies,
    )
