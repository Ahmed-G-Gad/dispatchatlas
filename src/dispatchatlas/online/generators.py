"""Seeded generators of traces: arrivals and node availability.

Every generator draws from a stream derived from the caller's seed with
:func:`~dispatchatlas.core.seeds.derive_seed`, one stream per purpose:
``online-arrivals`` for arrival times, and ``online-availability:<node>`` for
each node's outages. The same seed therefore gives the same arrivals whatever
the availability model, and adding a node leaves the outages of the others
unchanged. Times are rounded to microseconds, so a trace written on one
platform hashes the same on every other.

Arrivals:

- :func:`poisson_arrivals`: exponential gaps at a rate, which
  :func:`rate_for_load` derives from an offered load.
- :func:`submitted_arrivals`: given submit times, such as those of the
  Alibaba 2018 batch jobs.

Availability, :func:`availability_events`: each node alternates between an
up time drawn from a time-to-failure distribution and a down time drawn from
a time-to-repair distribution, an alternating renewal process
(:class:`AlternatingRenewal`). :class:`Exponential`, :class:`Weibull` and
:class:`LogNormal` distributions are provided; any object with ``sample``
and ``describe`` methods can stand in for them, so a published availability
model plugs in the same way.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import Resource, Scalar
from dispatchatlas.core.seeds import derive_seed
from dispatchatlas.online.events import Arrival, Event, Failure, Repair, Trace, Workflow

__all__ = [
    "TIME_DECIMALS",
    "AlternatingRenewal",
    "Distribution",
    "Exponential",
    "LogNormal",
    "Weibull",
    "availability_events",
    "poisson_arrivals",
    "rate_for_load",
    "submitted_arrivals",
    "workflow_work",
]

TIME_DECIMALS = 6
"""Generated event times are rounded to this many decimals: microseconds."""


def _round(time: float) -> float:
    return round(time, TIME_DECIMALS)


def _finite(value: float, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{what} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ModelError(f"{what} must be finite, got {value!r}")
    return float(value)


def _positive(value: float, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{what} must be a number, got {value!r}")
    if not (math.isfinite(value) and value > 0.0):
        raise ModelError(f"{what} must be positive and finite, got {value!r}")
    return float(value)


class Distribution(Protocol):
    """A distribution of positive durations."""

    def sample(self, rng: random.Random) -> float:
        """Draw one duration from ``rng``."""
        ...

    def describe(self) -> dict[str, Scalar]:
        """Return the distribution's name and parameters, for records."""
        ...


@dataclass(frozen=True)
class Exponential:
    """Exponential durations with the given mean."""

    mean: float

    def __post_init__(self) -> None:
        _positive(self.mean, "exponential mean")

    def sample(self, rng: random.Random) -> float:
        """Draw one duration."""
        return rng.expovariate(1.0 / self.mean)

    def describe(self) -> dict[str, Scalar]:
        """Return ``{"distribution": "exponential", "mean": ...}``."""
        return {"distribution": "exponential", "mean": self.mean}


@dataclass(frozen=True)
class Weibull:
    """Weibull durations with a shape and a scale.

    The mean is ``scale * Gamma(1 + 1 / shape)``. A shape below 1 gives a
    falling hazard, as measured for the failures of many clusters; a shape of
    1 is the exponential distribution.
    """

    shape: float
    scale: float

    def __post_init__(self) -> None:
        _positive(self.shape, "Weibull shape")
        _positive(self.scale, "Weibull scale")

    @property
    def mean(self) -> float:
        """The mean duration."""
        return self.scale * math.gamma(1.0 + 1.0 / self.shape)

    def sample(self, rng: random.Random) -> float:
        """Draw one duration."""
        return rng.weibullvariate(self.scale, self.shape)

    def describe(self) -> dict[str, Scalar]:
        """Return ``{"distribution": "weibull", "shape": ..., "scale": ...}``."""
        return {"distribution": "weibull", "shape": self.shape, "scale": self.scale}


@dataclass(frozen=True)
class LogNormal:
    """Log-normal durations, ``exp(mu + sigma * Z)`` for a standard normal ``Z``.

    ``mu`` and ``sigma`` are the mean and standard deviation of the underlying
    normal distribution of the logarithm, the parameterisation of the fits of
    the Failure Trace Archive. ``mu`` may be negative; ``sigma`` must be
    positive. The mean is ``exp(mu + sigma**2 / 2)``. The distribution is
    heavy-tailed, as measured for the repair times of many clusters.

    Sampling uses ``rng.lognormvariate``, which draws through
    ``rng.normalvariate``, a rejection method, so the number of uniform
    numbers a duration consumes from ``rng`` varies.
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        _finite(self.mu, "log-normal mu")
        _positive(self.sigma, "log-normal sigma")

    @property
    def mean(self) -> float:
        """The mean duration."""
        return math.exp(self.mu + self.sigma**2 / 2.0)

    def sample(self, rng: random.Random) -> float:
        """Draw one duration."""
        return rng.lognormvariate(self.mu, self.sigma)

    def describe(self) -> dict[str, Scalar]:
        """Return ``{"distribution": "lognormal", "mu": ..., "sigma": ...}``."""
        return {"distribution": "lognormal", "mu": self.mu, "sigma": self.sigma}


@dataclass(frozen=True)
class AlternatingRenewal:
    """Up times and down times drawn in turn, each independently.

    Attributes:
        time_to_failure: Distribution of the time from a start or a repair to
            the next failure.
        time_to_repair: Distribution of the time from a failure to its repair.
    """

    time_to_failure: Distribution
    time_to_repair: Distribution

    def outages(
        self, rng: random.Random, horizon: float, start: float = 0.0
    ) -> list[tuple[float, float]]:
        """Return ``(failure, repair)`` times of the outages that begin before ``horizon``.

        The node is up at ``start``. The repair of the last outage is kept even
        when it falls after the horizon, so every failed node comes back.
        """
        outages: list[tuple[float, float]] = []
        time = start
        while True:
            failure = _round(time + self.time_to_failure.sample(rng))
            if failure >= horizon:
                return outages
            repair = _round(failure + self.time_to_repair.sample(rng))
            repair = max(repair, _round(failure + 10.0**-TIME_DECIMALS))
            if outages and failure <= outages[-1][1]:
                failure = _round(outages[-1][1] + 10.0**-TIME_DECIMALS)
                repair = max(repair, _round(failure + 10.0**-TIME_DECIMALS))
            outages.append((failure, repair))
            time = repair

    def describe(self) -> dict[str, Scalar]:
        """Return the model's parameters, flattened, for records."""
        described: dict[str, Scalar] = {"model": "alternating-renewal"}
        for prefix, distribution in (
            ("ttf", self.time_to_failure),
            ("ttr", self.time_to_repair),
        ):
            for key, value in distribution.describe().items():
                described[f"{prefix}_{key}"] = value
        return described


def poisson_arrivals(
    workflows: Sequence[Workflow], rate: float, seed: int, *, start: float = 0.0
) -> Trace:
    """Return the arrivals of ``workflows``, in order, as a Poisson process.

    The gaps between consecutive arrivals, and from ``start`` to the first
    one, are exponential with mean ``1 / rate``, drawn from the
    ``online-arrivals`` stream of ``seed``.

    Args:
        workflows: The workflows, in the order they arrive.
        rate: Arrivals per unit of time. :func:`rate_for_load` derives it from
            an offered load.
        seed: Root seed of the trace.
        start: Time the process starts.

    Returns:
        A trace of arrivals only.
    """
    _positive(rate, "arrival rate")
    rng = random.Random(derive_seed(seed, "online-arrivals"))
    time = float(start)
    events: list[Event] = []
    for workflow in workflows:
        time = _round(time + rng.expovariate(rate))
        events.append(Arrival(time, workflow))
    return Trace(tuple(events))


def submitted_arrivals(workflows: Sequence[Workflow], times: Sequence[float]) -> Trace:
    """Return the arrivals of ``workflows`` at the given submit times.

    Raises:
        ModelError: If the two sequences differ in length.
    """
    if len(workflows) != len(times):
        raise ModelError(f"{len(workflows)} workflows but {len(times)} submit times")
    return Trace(tuple(Arrival(float(t), w) for w, t in zip(workflows, times, strict=True)))


def workflow_work(workflow: Workflow) -> float:
    """Return the work of a workflow: over its tasks, the least units times duration.

    A task's work is the minimum over its modes of the mode's duration times
    the units it demands in total, the resource time the task needs on the
    mode that needs the least. It is a lower bound on what the workflow
    occupies.
    """
    return sum(
        min(mode.duration * sum(mode.demands.values()) for mode in task.modes)
        for task in workflow.tasks
    )


def rate_for_load(
    workflows: Sequence[Workflow], resources: Iterable[Resource], load: float
) -> float:
    """Return the arrival rate that offers ``load`` to the platform.

    The offered load is ``rate * W / C``: the rate times the mean
    :func:`workflow_work` ``W`` of the workflows, over the total capacity
    ``C`` of the resources that some mode of the workflows demands. Since the
    work counts each task on its cheapest mode, a load of 1 is an upper bound
    on the rate the platform can sustain, not the rate itself.

    Raises:
        ModelError: If ``load`` is not positive, no workflow is given, or no
            resource is demanded.
    """
    _positive(load, "load")
    if not workflows:
        raise ModelError("rate_for_load needs at least one workflow")
    demanded = {r for w in workflows for t in w.tasks for m in t.modes for r in m.demands}
    capacity = sum(resource.capacity for resource in resources if resource.id in demanded)
    if capacity <= 0.0:
        raise ModelError("no resource that the workflows demand is given")
    work = sum(workflow_work(w) for w in workflows) / len(workflows)
    if work <= 0.0:
        raise ModelError("the workflows hold no work")
    return load * capacity / work


def availability_events(
    nodes: Iterable[str],
    model: AlternatingRenewal,
    seed: int,
    horizon: float,
    *,
    start: float = 0.0,
) -> Trace:
    """Return the failures and repairs of ``nodes`` until ``horizon``.

    Each node draws its outages from its own stream,
    ``online-availability:<node>`` of ``seed``, so its outages do not depend
    on the other nodes or on the arrivals.

    Args:
        nodes: Ids of the nodes that can fail.
        model: The availability model.
        seed: Root seed of the trace.
        horizon: No failure begins at or after this time.
        start: Time at which every node is up.

    Returns:
        A trace of failures and repairs only.
    """
    _positive(horizon, "horizon")
    events: list[Event] = []
    for node in sorted(set(nodes)):
        rng = random.Random(derive_seed(seed, f"online-availability:{node}"))
        for failure, repair in model.outages(rng, horizon, start):
            events.append(Failure(failure, node))
            events.append(Repair(repair, node))
    return Trace(tuple(events))
