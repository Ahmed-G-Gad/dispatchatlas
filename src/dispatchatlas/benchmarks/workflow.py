"""Random task graphs on heterogeneous processors.

The ``workflow`` family generates the weighted task graphs of Topcuoglu, Hariri
and Wu (2002), the generator behind the HEFT and CPOP experiments. A graph has
``tasks`` tasks arranged in levels. Each task runs on any one of ``processors``
unrelated processors, with a processor-dependent duration, and each edge
carries a communication cost that is paid only when its two tasks run on
different processors.

The generator draws, in this order:

1. The height, ``ceil(U(0, 2 sqrt(v) / shape))`` levels, so its mean is
   ``sqrt(v) / shape``.
2. One width per level from ``U(0, 2 shape sqrt(v))``, mean
   ``shape * sqrt(v)``. The paper does not say how the widths are made to add
   up to ``v``; here they are rescaled, largest remainder first, so they sum to
   exactly ``v`` with at least one task per level.
3. For every task outside the last level, an out-degree from
   ``randint(1, 2 out_degree - 1)``, mean ``out_degree``, and that many
   distinct successors drawn uniformly from the next level, or the whole next
   level when it is narrower. The paper leaves the choice of successors open;
   joining consecutive levels only keeps the height equal to the length of
   the longest dependency chain. A task that still has no parent then
   receives one, drawn uniformly from the level above.
4. For every task, a mean duration ``w_i`` from ``U(0, 2 mean_duration)``, and
   for every processor a duration from
   ``U(w_i (1 - heterogeneity / 2), w_i (1 + heterogeneity / 2))``.
5. For every edge, a raw weight from ``U(0, 2)``. The weights are scaled so the
   mean communication cost is ``ccr`` times the mean duration over all tasks
   and processors, so every graph has the communication-to-computation ratio
   of the paper, not only the average graph.

Durations and communication costs are rounded to six decimals.

Communication uses one network endpoint per processor: every pair of
processors is joined by a link of bandwidth 1 and no latency, so the transfer
time of an edge equals its communication cost, and it is zero on one processor.

The paper draws the average duration ``w_DAG`` at random. Here it is the
``mean_duration`` parameter, since it only scales the instance.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import ClassVar

from dispatchatlas.benchmarks._common import (
    check_int,
    check_number,
    check_objectives,
    instance_id,
    instance_metadata,
)
from dispatchatlas.benchmarks.registry import FamilyInfo, FamilyRegistry
from dispatchatlas.core.model import Dependency, Link, Mode, Network, Problem, Resource, Task

__all__ = ["LayeredGraph", "RandomTaskGraph", "layered_graph", "register"]

_DECIMALS = 6
"""Generated durations and data volumes are rounded to this many decimals."""


@dataclass(frozen=True, slots=True)
class LayeredGraph:
    """A directed acyclic graph whose tasks are grouped into levels.

    Tasks are numbered level by level, so every edge goes from a lower to a
    higher task number and the numbering is a topological order.

    Attributes:
        levels: Level of each task, starting at 0.
        edges: Sorted ``(before, after)`` pairs with ``before < after``.
    """

    levels: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]

    @property
    def height(self) -> int:
        """Number of levels."""
        return max(self.levels, default=-1) + 1


def layered_graph(rng: random.Random, tasks: int, shape: float, out_degree: int) -> LayeredGraph:
    """Draw the levels and edges of a random task graph.

    Follows steps 1 to 3 of the module description: the height and level
    widths of Topcuoglu, Hariri and Wu (2002), and uniform out-degrees with
    mean ``out_degree`` towards the next level, completed so that every task
    below the first level has a parent in the level above. Edges join
    consecutive levels only, and there are at most ``2 * out_degree * tasks``
    of them, so the graph stays sparse as it grows.

    Args:
        rng: Source of every random draw.
        tasks: Number of tasks, at least 1.
        shape: Shape parameter. Values above 1 give short, wide graphs with
            much parallelism; values below 1 give long, narrow ones.
        out_degree: Mean number of successors of a task, at least 1.

    Returns:
        The levels and edges.

    Raises:
        ValueError: If a parameter is out of range.
    """
    check_int("tasks", tasks)
    check_number("shape", shape, positive=True)
    check_int("out_degree", out_degree)
    root = math.sqrt(tasks)
    height = min(tasks, max(1, math.ceil(rng.uniform(0.0, 2.0 * root / shape))))
    widths = _apportion(tasks, [rng.uniform(0.0, 2.0 * shape * root) for _ in range(height)])
    starts = [0]
    for width in widths:
        starts.append(starts[-1] + width)
    levels = [level for level, width in enumerate(widths) for _ in range(width)]

    edges: set[tuple[int, int]] = set()
    for task in range(starts[height - 1]):
        below = range(starts[levels[task] + 1], starts[levels[task] + 2])
        degree = rng.randint(1, 2 * out_degree - 1)
        for successor in rng.sample(below, min(degree, len(below))):
            edges.add((task, successor))
    for task in range(starts[1] if height > 1 else tasks, tasks):
        above = range(starts[levels[task] - 1], starts[levels[task]])
        if not any((parent, task) in edges for parent in above):
            edges.add((rng.choice(above), task))
    return LayeredGraph(tuple(levels), tuple(sorted(edges)))


def _apportion(total: int, weights: Sequence[float]) -> list[int]:
    """Split ``total`` into ``len(weights)`` parts of at least 1, proportional to ``weights``.

    The ``total - len(weights)`` units beyond the first of each part go out in
    proportion to the weights: first the integer parts of the shares, then one
    unit each to the largest fractional parts, lowest index first on ties.
    """
    count = len(weights)
    spare = total - count
    weight_sum = math.fsum(weights)
    if weight_sum > 0.0:
        shares = [spare * weight / weight_sum for weight in weights]
    else:
        shares = [spare / count] * count
    parts = [math.floor(share) for share in shares]
    remainder = spare - sum(parts)
    by_fraction = sorted(range(count), key=lambda i: (parts[i] - shares[i], i))
    for i in by_fraction[:remainder]:
        parts[i] += 1
    return [1 + part for part in parts]


@dataclass(frozen=True)
class RandomTaskGraph:
    """Random task graphs on heterogeneous processors (Topcuoglu, Hariri and Wu 2002).

    The draws follow the module description, in its order. The defaults of
    ``tasks``, ``shape``, ``out_degree``, ``ccr``, and ``heterogeneity`` come
    from the parameter sets of the paper; those of ``processors`` and
    ``mean_duration``, which the paper does not fix, are illustrative.

    Attributes:
        tasks: Number of tasks, ``v`` in the paper.
        shape: Shape parameter ``alpha``: above 1 gives wide graphs, below 1
            long ones.
        out_degree: Mean number of successors of a task.
        ccr: Communication-to-computation ratio, the mean communication cost
            over the mean duration.
        heterogeneity: Range percentage ``beta`` of a task's durations across
            processors, between 0 (identical processors) and 2.
        processors: Number of unrelated processors.
        mean_duration: Expected mean duration over tasks and processors,
            ``w_DAG`` in the paper.
        objectives: Objectives the problems declare, primary first. The
            graphs have no due dates, so due-date objectives are refused.
    """

    tasks: int = 20
    shape: float = 1.0
    out_degree: int = 2
    ccr: float = 1.0
    heterogeneity: float = 0.5
    processors: int = 4
    mean_duration: float = 10.0
    objectives: tuple[str, ...] = ("makespan",)

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="workflow",
        name="Random task graphs on heterogeneous processors",
        problem_class="workflow",
        summary="Layered random task graphs whose tasks each run on one of several "
        "unrelated processors, with communication costs paid between different "
        "processors, as generated for the HEFT experiments.",
        references=("topcuoglu2002performance",),
        objectives=("makespan",),
    )

    def __post_init__(self) -> None:
        check_int("tasks", self.tasks)
        check_int("out_degree", self.out_degree)
        check_int("processors", self.processors)
        check_number("shape", self.shape, positive=True)
        check_number("ccr", self.ccr)
        check_number("mean_duration", self.mean_duration, positive=True)
        check_number("heterogeneity", self.heterogeneity, maximum=2.0)
        object.__setattr__(self, "objectives", check_objectives(self.objectives, due_dates=False))

    def generate(self, seed: int = 0) -> Problem:
        """Return the task graph for ``seed``."""
        metadata = instance_metadata(self, seed)
        rng = random.Random(seed)
        graph = layered_graph(rng, self.tasks, self.shape, self.out_degree)
        low, high = 1.0 - self.heterogeneity / 2.0, 1.0 + self.heterogeneity / 2.0
        durations: list[list[float]] = []
        for _ in range(self.tasks):
            mean = rng.uniform(0.0, 2.0 * self.mean_duration)
            durations.append(
                [
                    round(rng.uniform(mean * low, mean * high), _DECIMALS)
                    for _ in range(self.processors)
                ]
            )
        weights = [rng.uniform(0.0, 2.0) for _ in graph.edges]
        mean_weight = fmean(weights) if weights else 0.0
        target = self.ccr * fmean(value for row in durations for value in row)
        scale = target / mean_weight if mean_weight > 0.0 else 0.0
        data = [round(weight * scale, _DECIMALS) for weight in weights]

        processors = tuple(Resource(f"p{j}", kind="processor") for j in range(self.processors))
        tasks = tuple(
            Task(
                f"t{i}",
                tuple(
                    Mode(duration, {processor.id: 1.0})
                    for duration, processor in zip(row, processors, strict=True)
                ),
            )
            for i, row in enumerate(durations)
        )
        dependencies = tuple(
            Dependency(f"t{before}", f"t{after}", data=volume)
            for (before, after), volume in zip(graph.edges, data, strict=True)
        )
        network = None
        if self.processors > 1 and any(volume > 0.0 for volume in data):
            network = Network(
                tuple(
                    Link(a.id, b.id, bandwidth=1.0)
                    for i, a in enumerate(processors)
                    for b in processors[i + 1 :]
                )
            )
        return Problem(
            id=instance_id(self, f"{self.tasks}x{self.processors}", seed, {}),
            tasks=tasks,
            resources=processors,
            dependencies=dependencies,
            objectives=self.objectives,
            network=network,
            family=self.info.id,
            metadata=metadata,
        )


def register(registry: FamilyRegistry) -> None:
    """Add the ``workflow`` family to ``registry``."""
    registry.register(RandomTaskGraph)
