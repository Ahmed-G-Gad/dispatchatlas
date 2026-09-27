"""The continuum benchmark families: offloading, workflows, and pipelines.

Each family draws applications with its own structure and builds the problem
with :func:`~dispatchatlas.benchmarks.continuum.applications.build_problem`
on a :func:`~dispatchatlas.benchmarks.continuum.platform.three_tier_platform`
with the default hardware and links. All three share these parameters:

- ``devices``, ``edge_nodes``, ``fog_nodes``, ``cloud_nodes``: node counts.
  Applications are assigned to devices in turn.
- ``tiers``: comma-separated tiers a compute task may run on, ``"device"``
  meaning the issuing device itself.
- ``due_factor`` and ``deadline_factor``: due date and optional hard deadline
  of each application, as multiples of its contention-free response time
  counted from its release (see
  :func:`~dispatchatlas.benchmarks.continuum.applications.response_bound`).
- ``objectives``: objectives the problems declare, primary first.

Work and data volumes are drawn uniformly between half and one and a half
times their means. Work and times are rounded to microseconds and data to
kilobytes, which keeps every time value of the problems a decimal with at
most six digits.
Problems are named ``<family>-<size>-<seed>`` and record every parameter that
is not ``None``, plus the seed, in their metadata, as the other built-in
families do. Parameter defaults are illustrative sizes for quick experiments,
not calibrated to a measured workload.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.benchmarks._common import (
    check_int,
    check_number,
    check_objectives,
    instance_id,
    instance_metadata,
)
from dispatchatlas.benchmarks.continuum.applications import Application, build_problem
from dispatchatlas.benchmarks.continuum.platform import TIERS, Platform, three_tier_platform
from dispatchatlas.benchmarks.registry import FamilyInfo, FamilyRegistry
from dispatchatlas.benchmarks.workflow import LayeredGraph, layered_graph
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.model import Problem

__all__ = [
    "STRUCTURES",
    "ContinuumOffloading",
    "ContinuumPipeline",
    "ContinuumWorkflow",
    "fork_join_graph",
    "register",
]

STRUCTURES: tuple[str, ...] = ("layered", "fork-join")
"""Task-graph structures of :class:`ContinuumWorkflow`."""

_DECIMALS = 6
"""Work and times are rounded to microseconds."""

_DATA_DECIMALS = 3
"""Data volumes, in megabytes, are rounded to kilobytes."""


def _draw(rng: random.Random, mean: float, decimals: int = _DECIMALS) -> float:
    """Draw uniformly between half and one and a half times ``mean``, then round."""
    return round(mean * rng.uniform(0.5, 1.5), decimals)


def _parse_tiers(text: object) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise ValueError(f"tiers must be a comma-separated string, got {text!r}")
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = sorted(set(names) - set(TIERS))
    if not names or unknown:
        raise ValueError(f"tiers must name some of {', '.join(TIERS)}, got {text!r}")
    return tuple(tier for tier in TIERS if tier in names)


def _check_shared(
    tiers: object,
    devices: object,
    edge_nodes: object,
    fog_nodes: object,
    cloud_nodes: object,
    due_factor: object,
    deadline_factor: object,
) -> None:
    """Validate the parameters every continuum family shares."""
    check_int("devices", devices)
    counts = {"device": devices, "edge": edge_nodes, "fog": fog_nodes, "cloud": cloud_nodes}
    for tier in ("edge", "fog", "cloud"):
        check_int(f"{tier}_nodes", counts[tier], 0)
    if not any(counts[tier] for tier in _parse_tiers(tiers)):
        raise ValueError(f"no node of tiers {tiers!r} exists")
    check_number("due_factor", due_factor, minimum=1.0)
    if deadline_factor is not None:
        check_number("deadline_factor", deadline_factor, minimum=1.0)


def fork_join_graph(tasks: int) -> LayeredGraph:
    """Return a fork-join graph: one task forks ``tasks - 2`` parallel tasks that one task joins.

    With one or two tasks the graph is a single task or a chain of two.
    """
    check_int("tasks", tasks)
    if tasks <= 2:
        return LayeredGraph(tuple(range(tasks)), ((0, 1),) if tasks == 2 else ())
    middle = range(1, tasks - 1)
    edges = [(0, k) for k in middle] + [(k, tasks - 1) for k in middle]
    return LayeredGraph((0, *(1 for _ in middle), 2), tuple(sorted(edges)))


@dataclass(frozen=True)
class ContinuumOffloading:
    """Independent requests offloaded across an edge-fog-cloud continuum.

    Devices issue requests at the arrival times of a Poisson process of rate
    ``arrival_rate``. Each request is one compute task between a source and a
    sink on its device: the device uploads ``input_data``, the task processes
    ``work`` on the device itself or on an edge, fog, or cloud node, and
    ``output_data`` returns to the device. Placing the task trades processing
    speed against transfer time, transfer energy, and money.

    Per request the generator draws, in order: the time since the previous
    request, the work, the input volume, and the output volume.

    Attributes:
        applications: Number of requests.
        arrival_rate: Requests per second over all devices.
        work: Mean work of a request, in work units.
        input_data: Mean data a request uploads, in megabytes.
        output_data: Mean data a request returns, in megabytes.
        due_factor: Due date as a multiple of the best response time on an
            idle platform.
        deadline_factor: Hard deadline as a multiple of the same time, or
            ``None`` for none.
        tiers: Tiers a request may run on, comma-separated.
        devices: Number of devices.
        edge_nodes: Number of edge nodes.
        fog_nodes: Number of fog nodes.
        cloud_nodes: Number of cloud nodes.
        objectives: Objectives the problems declare, primary first.
    """

    applications: int = 20
    arrival_rate: float = 4.0
    work: float = 1.0
    input_data: float = 1.0
    output_data: float = 0.1
    due_factor: float = 2.0
    deadline_factor: float | None = None
    tiers: str = "device,edge,fog,cloud"
    devices: int = 4
    edge_nodes: int = 2
    fog_nodes: int = 1
    cloud_nodes: int = 1
    objectives: tuple[str, ...] = ("mean_response_time", "energy", "cost")

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="continuum-offloading",
        name="Edge-fog-cloud request offloading",
        problem_class="continuum",
        summary="Independent requests issued by devices at Poisson arrival times, each "
        "run on its device or offloaded to an edge, fog, or cloud node at the price of "
        "uploading its input and returning its result.",
        references=("mao2017survey", "gupta2017ifogsim", "fan2007power"),
        objectives=("mean_response_time", "energy", "cost"),
    )

    def __post_init__(self) -> None:
        check_int("applications", self.applications)
        check_number("arrival_rate", self.arrival_rate, positive=True)
        for name in ("work", "input_data", "output_data"):
            check_number(name, getattr(self, name))
        _check_shared(
            self.tiers,
            self.devices,
            self.edge_nodes,
            self.fog_nodes,
            self.cloud_nodes,
            self.due_factor,
            self.deadline_factor,
        )
        object.__setattr__(self, "objectives", check_objectives(self.objectives))

    def platform(self) -> Platform:
        """Return the platform the requests run on."""
        return three_tier_platform(
            devices=self.devices, edge=self.edge_nodes, fog=self.fog_nodes, cloud=self.cloud_nodes
        )

    def generate(self, seed: int = 0) -> Problem:
        """Return the problem for ``seed``."""
        metadata = instance_metadata(self, seed)
        rng = random.Random(seed)
        platform = self.platform()
        devices = platform.nodes("device")
        tiers = _parse_tiers(self.tiers)
        requests: list[Application] = []
        clock = 0.0
        for index in range(self.applications):
            clock += rng.expovariate(self.arrival_rate)
            work = _draw(rng, self.work)
            upload = _draw(rng, self.input_data, _DATA_DECIMALS)
            download = _draw(rng, self.output_data, _DATA_DECIMALS)
            requests.append(
                Application(
                    name=f"a{index}",
                    device=devices[index % len(devices)].id,
                    release=round(clock, _DECIMALS),
                    work=(work,),
                    upload=(upload,),
                    download=(download,),
                    tiers=(tiers,),
                )
            )
        return build_problem(
            instance_id(self, str(self.applications), seed, {}),
            platform,
            requests,
            objectives=self.objectives,
            due_factor=self.due_factor,
            deadline_factor=self.deadline_factor,
            family=self.info.id,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ContinuumWorkflow:
    """Workflows that devices submit to an edge-fog-cloud continuum.

    Each application is a task graph: a layered random graph drawn as in
    Topcuoglu, Hariri and Wu (2002), or a fork-join graph in which one task
    forks parallel tasks that one task joins. The device uploads data to the
    tasks without predecessors and receives the results of the tasks without
    successors. Every dependency, uploads and results included, carries a
    data volume whose mean takes ``ccr * work`` seconds to cross the access
    link between devices and the edge, so ``ccr`` is the ratio between moving
    one dependency's data off a device and processing one task on it.

    Per application the generator draws, in order: the time since the
    previous application (when ``arrival_rate`` is positive), the graph, the
    work of each task, the data of each edge, of each upload, and of each
    result, and the tasks pinned to the edge.

    Attributes:
        applications: Number of workflows.
        tasks: Compute tasks per workflow.
        structure: ``"layered"`` or ``"fork-join"``.
        shape: Shape parameter of layered graphs: above 1 gives wide graphs,
            below 1 long ones.
        out_degree: Mean number of successors of a task in layered graphs.
        work: Mean work of a task, in work units.
        ccr: Communication-to-computation ratio, as defined above.
        pinned_fraction: Share of each workflow's tasks, rounded to the
            nearest count, that run on edge nodes only, whatever ``tiers``
            allows.
        arrival_rate: Workflows per second, or 0 to release all at time 0.
        due_factor: Due date as a multiple of the contention-free response time.
        deadline_factor: Hard deadline as a multiple of the contention-free
            response time, or ``None`` for none.
        tiers: Tiers a task may run on, comma-separated.
        devices: Number of devices.
        edge_nodes: Number of edge nodes.
        fog_nodes: Number of fog nodes.
        cloud_nodes: Number of cloud nodes.
        objectives: Objectives the problems declare, primary first.
    """

    applications: int = 4
    tasks: int = 10
    structure: str = "layered"
    shape: float = 1.0
    out_degree: int = 2
    work: float = 1.0
    ccr: float = 1.0
    pinned_fraction: float = 0.0
    arrival_rate: float = 0.0
    due_factor: float = 2.0
    deadline_factor: float | None = None
    tiers: str = "device,edge,fog,cloud"
    devices: int = 4
    edge_nodes: int = 2
    fog_nodes: int = 1
    cloud_nodes: int = 1
    objectives: tuple[str, ...] = ("makespan", "energy", "cost")

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="continuum-workflow",
        name="Edge-fog-cloud workflows",
        problem_class="continuum",
        summary="Layered random or fork-join workflows submitted by devices, each task "
        "placed on its device or on an edge, fog, or cloud node, with data volumes set "
        "by a communication-to-computation ratio.",
        references=("topcuoglu2002performance", "gupta2017ifogsim", "fan2007power"),
        objectives=("makespan", "energy", "cost"),
    )

    def __post_init__(self) -> None:
        check_int("applications", self.applications)
        check_int("tasks", self.tasks)
        check_int("out_degree", self.out_degree)
        if self.structure not in STRUCTURES:
            raise ValueError(f"structure must be one of {', '.join(STRUCTURES)}")
        check_number("shape", self.shape, positive=True)
        for name in ("work", "ccr", "arrival_rate"):
            check_number(name, getattr(self, name))
        check_number("pinned_fraction", self.pinned_fraction, maximum=1.0)
        if self.pinned_fraction > 0.0 and self.edge_nodes == 0:
            raise ValueError("pinning tasks to the edge needs at least one edge node")
        _check_shared(
            self.tiers,
            self.devices,
            self.edge_nodes,
            self.fog_nodes,
            self.cloud_nodes,
            self.due_factor,
            self.deadline_factor,
        )
        object.__setattr__(self, "objectives", check_objectives(self.objectives))

    def platform(self) -> Platform:
        """Return the platform the workflows run on."""
        return three_tier_platform(
            devices=self.devices, edge=self.edge_nodes, fog=self.fog_nodes, cloud=self.cloud_nodes
        )

    def generate(self, seed: int = 0) -> Problem:
        """Return the problem for ``seed``."""
        metadata = instance_metadata(self, seed)
        rng = random.Random(seed)
        platform = self.platform()
        devices = platform.nodes("device")
        tiers = _parse_tiers(self.tiers)
        access = platform.network.link("device", "edge")
        if access is None or math.isinf(access.bandwidth):
            raise ConfigError("the ccr needs a finite bandwidth between devices and the edge")
        mean_data = self.ccr * self.work * access.bandwidth
        pinned_count = round(self.pinned_fraction * self.tasks)
        workflows: list[Application] = []
        clock = 0.0
        for index in range(self.applications):
            if self.arrival_rate > 0.0:
                clock += rng.expovariate(self.arrival_rate)
            if self.structure == "layered":
                graph = layered_graph(rng, self.tasks, self.shape, self.out_degree)
            else:
                graph = fork_join_graph(self.tasks)
            work = tuple(_draw(rng, self.work) for _ in range(self.tasks))
            edges = tuple((a, b, _draw(rng, mean_data, _DATA_DECIMALS)) for a, b in graph.edges)
            has_predecessor = {b for _, b in graph.edges}
            has_successor = {a for a, _ in graph.edges}
            upload = tuple(
                0.0 if k in has_predecessor else _draw(rng, mean_data, _DATA_DECIMALS)
                for k in range(self.tasks)
            )
            download = tuple(
                0.0 if k in has_successor else _draw(rng, mean_data, _DATA_DECIMALS)
                for k in range(self.tasks)
            )
            pinned = set(rng.sample(range(self.tasks), pinned_count))
            workflows.append(
                Application(
                    name=f"a{index}",
                    device=devices[index % len(devices)].id,
                    release=round(clock, _DECIMALS),
                    work=work,
                    edges=edges,
                    upload=upload,
                    download=download,
                    tiers=tuple(("edge",) if k in pinned else tiers for k in range(self.tasks)),
                )
            )
        return build_problem(
            instance_id(self, f"{self.applications}x{self.tasks}", seed, {}),
            platform,
            workflows,
            objectives=self.objectives,
            due_factor=self.due_factor,
            deadline_factor=self.deadline_factor,
            family=self.info.id,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ContinuumPipeline:
    """Periodic frames of device streams, each processed by a chain of stages.

    Each stream belongs to a device and emits a frame every ``period``
    seconds from a random phase. A frame is one application: the device
    uploads ``input_data``, and ``stages`` compute tasks process it in turn,
    each passing on ``reduction`` times the data it received, until the last
    returns its output to the device. All frames of a stream share the same
    stage work and data volumes. Because the data shrink along the chain,
    early stages favour nodes near the device and late stages favour fast
    remote nodes, the trade-off of splitting inference between device and
    cloud studied by Kang et al. (2017).

    Per stream the generator draws, in order: the phase, the work of each
    stage, and the frame size.

    Attributes:
        streams: Number of streams.
        frames: Frames per stream.
        stages: Compute stages per frame.
        period: Seconds between two frames of a stream.
        work: Mean work of a stage, in work units.
        input_data: Mean frame size, in megabytes.
        reduction: Ratio between the data a stage sends on and the data it
            received.
        due_factor: Due date as a multiple of the best response time on an
            idle platform.
        deadline_factor: Hard deadline as a multiple of the same time, or
            ``None`` for none.
        tiers: Tiers a stage may run on, comma-separated.
        devices: Number of devices.
        edge_nodes: Number of edge nodes.
        fog_nodes: Number of fog nodes.
        cloud_nodes: Number of cloud nodes.
        objectives: Objectives the problems declare, primary first.
    """

    streams: int = 4
    frames: int = 5
    stages: int = 3
    period: float = 1.0
    work: float = 0.5
    input_data: float = 2.0
    reduction: float = 0.5
    due_factor: float = 2.0
    deadline_factor: float | None = None
    tiers: str = "device,edge,fog,cloud"
    devices: int = 4
    edge_nodes: int = 2
    fog_nodes: int = 1
    cloud_nodes: int = 1
    objectives: tuple[str, ...] = ("mean_response_time", "energy", "cost")

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="continuum-pipeline",
        name="Edge-fog-cloud stream pipelines",
        problem_class="continuum",
        summary="Periodic frames from device streams, each passing through a chain of "
        "stages whose data shrink stage by stage, so each stage is placed on the device "
        "or on an edge, fog, or cloud node.",
        references=("kang2017neurosurgeon", "gupta2017ifogsim", "fan2007power"),
        objectives=("mean_response_time", "energy", "cost"),
    )

    def __post_init__(self) -> None:
        check_int("streams", self.streams)
        check_int("frames", self.frames)
        check_int("stages", self.stages)
        check_number("period", self.period, positive=True)
        check_number("reduction", self.reduction, positive=True)
        for name in ("work", "input_data"):
            check_number(name, getattr(self, name))
        _check_shared(
            self.tiers,
            self.devices,
            self.edge_nodes,
            self.fog_nodes,
            self.cloud_nodes,
            self.due_factor,
            self.deadline_factor,
        )
        object.__setattr__(self, "objectives", check_objectives(self.objectives))

    def platform(self) -> Platform:
        """Return the platform the streams run on."""
        return three_tier_platform(
            devices=self.devices, edge=self.edge_nodes, fog=self.fog_nodes, cloud=self.cloud_nodes
        )

    def generate(self, seed: int = 0) -> Problem:
        """Return the problem for ``seed``."""
        metadata = instance_metadata(self, seed)
        rng = random.Random(seed)
        platform = self.platform()
        devices = platform.nodes("device")
        tiers = _parse_tiers(self.tiers)
        frames: list[tuple[float, int, int, Application]] = []
        for stream in range(self.streams):
            phase = rng.uniform(0.0, self.period)
            work = tuple(_draw(rng, self.work) for _ in range(self.stages))
            sizes = [_draw(rng, self.input_data, _DATA_DECIMALS)]
            for _ in range(self.stages):
                sizes.append(round(sizes[-1] * self.reduction, _DATA_DECIMALS))
            edges = tuple((k, k + 1, sizes[k + 1]) for k in range(self.stages - 1))
            upload = (sizes[0],) + (0.0,) * (self.stages - 1)
            download = (0.0,) * (self.stages - 1) + (sizes[-1],)
            for frame in range(self.frames):
                release = round(phase + frame * self.period, _DECIMALS)
                application = Application(
                    name=f"s{stream}f{frame}",
                    device=devices[stream % len(devices)].id,
                    release=release,
                    work=work,
                    edges=edges,
                    upload=upload,
                    download=download,
                    tiers=(tiers,) * self.stages,
                )
                frames.append((release, stream, frame, application))
        frames.sort(key=lambda item: item[:3])
        return build_problem(
            instance_id(self, f"{self.streams}x{self.frames}x{self.stages}", seed, {}),
            platform,
            [application for *_, application in frames],
            objectives=self.objectives,
            due_factor=self.due_factor,
            deadline_factor=self.deadline_factor,
            family=self.info.id,
            metadata=metadata,
        )


def register(registry: FamilyRegistry) -> None:
    """Add the continuum families to ``registry``."""
    for factory in (ContinuumOffloading, ContinuumWorkflow, ContinuumPipeline):
        registry.register(factory)
