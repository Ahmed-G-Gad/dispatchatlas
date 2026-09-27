"""Compute nodes and network of an edge-fog-cloud continuum.

A :class:`Platform` is a set of compute nodes, each a
:class:`~dispatchatlas.core.model.Resource` whose capacity is its number of
cores and whose ``tier`` names its layer, plus the
:class:`~dispatchatlas.core.model.Network` whose links join the tiers.
:func:`three_tier_platform` builds the usual layout: devices that issue work,
then edge, fog, and cloud nodes of growing speed and distance.

Units: time in seconds; work in work units, the seconds one core of speed 1
needs; data in megabytes; bandwidth in megabytes per second; power in watts;
energy in joules; cost in an abstract currency. Durations are rounded to
microseconds, and the default link parameters keep transfer times of data
given in kilobytes to microseconds too, so models in integer time, such as
those of the exact solvers, represent them exactly.

The hardware in :data:`DEFAULT_NODES` and the links in :data:`DEFAULT_LINKS`
are illustrative orders of magnitude chosen so that no tier is best on every
objective. They are not measurements. Override them for any study that
depends on their values.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType

from dispatchatlas.benchmarks._common import check_int, check_number
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.model import Link, Mode, Network, Resource

__all__ = [
    "DEFAULT_LINKS",
    "DEFAULT_NODES",
    "TIERS",
    "NodeSpec",
    "Platform",
    "three_tier_platform",
]

TIERS: tuple[str, ...] = ("device", "edge", "fog", "cloud")
"""The tiers of :func:`three_tier_platform`, nearest to the data first."""

_DECIMALS = 6
"""Durations are rounded to this many decimals: microseconds."""


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Hardware of one compute node.

    A task holds one core while it runs, so a node runs up to ``cores`` tasks
    at once. Energy and cost follow the core model: a task on one of ``cores``
    cores adds ``(busy_power - idle_power) / cores`` watts and costs
    ``cost_rate / cores`` per second.

    Attributes:
        cores: Number of cores, the node's resource capacity.
        speed: Work units one core completes per second. A task of work ``w``
            runs for ``w / speed`` seconds.
        idle_power: Watts drawn while the node is on.
        busy_power: Watts drawn with every core busy.
        cost_rate: Currency per second with every core busy.
    """

    cores: int = 1
    speed: float = 1.0
    idle_power: float = 0.0
    busy_power: float = 0.0
    cost_rate: float = 0.0

    def __post_init__(self) -> None:
        check_int("cores", self.cores)
        check_number("speed", self.speed, positive=True)
        for name in ("idle_power", "busy_power", "cost_rate"):
            check_number(name, getattr(self, name))
        if self.busy_power < self.idle_power:
            raise ValueError("busy_power must be at least idle_power")


DEFAULT_NODES: Mapping[str, NodeSpec] = MappingProxyType(
    {
        "device": NodeSpec(cores=1, speed=1.0, idle_power=0.5, busy_power=2.5, cost_rate=0.0),
        "edge": NodeSpec(cores=2, speed=2.0, idle_power=4.0, busy_power=36.0, cost_rate=0.04),
        "fog": NodeSpec(cores=8, speed=2.0, idle_power=12.0, busy_power=108.0, cost_rate=0.096),
        "cloud": NodeSpec(cores=32, speed=3.0, idle_power=48.0, busy_power=432.0, cost_rate=0.288),
    }
)
"""Illustrative hardware per tier.

One work unit takes 1 s on a device, 0.5 s on an edge or fog node, and 1/3 s
in the cloud. It spends 2 J of busy energy on a device, 8 J on the edge, 6 J
in the fog, and 4 J in the cloud, and costs nothing on a device, 0.01 on the
edge, 0.006 in the fog, and 0.003 in the cloud. Idle power is a ninth of
full-load power on edge, fog, and cloud nodes and a fifth on a device. The
edge is thus the closest fast tier, the fog has the same speed with more
cores, and the cloud is the fastest and cheapest per unit of work but the
farthest away.
"""


def _join(a: str, b: str, *hops: Link) -> Link:
    """Return the link from ``a`` to ``b`` across consecutive ``hops``.

    The bandwidth is the narrowest hop's; latency, energy, and cost add up.
    """
    return Link(
        a,
        b,
        bandwidth=min(hop.bandwidth for hop in hops),
        latency=math.fsum(hop.latency for hop in hops),
        energy_per_unit=math.fsum(hop.energy_per_unit for hop in hops),
        cost_per_unit=math.fsum(hop.cost_per_unit for hop in hops),
    )


_ACCESS = Link("device", "edge", bandwidth=12.5, latency=0.005, energy_per_unit=0.5)
_METRO = Link("edge", "fog", bandwidth=25.0, latency=0.005, energy_per_unit=0.1)
_WIDE_AREA = Link(
    "fog", "cloud", bandwidth=5.0, latency=0.04, energy_per_unit=0.2, cost_per_unit=0.002
)

DEFAULT_LINKS: tuple[Link, ...] = (
    _join("device", "device", _ACCESS, _ACCESS),
    _ACCESS,
    _join("device", "fog", _ACCESS, _METRO),
    _join("device", "cloud", _ACCESS, _METRO, _WIDE_AREA),
    Link("edge", "edge", bandwidth=25.0, latency=0.002, energy_per_unit=0.1),
    _METRO,
    _join("edge", "cloud", _METRO, _WIDE_AREA),
    Link("fog", "fog", bandwidth=125.0, latency=0.001, energy_per_unit=0.02),
    _WIDE_AREA,
    Link("cloud", "cloud", bandwidth=1000.0, latency=0.0005, energy_per_unit=0.01),
)
"""Illustrative links between every pair of tiers, same-tier pairs included.

Three hops form a chain: wireless access from devices to the edge (12.5 MB/s,
5 ms, 0.5 J/MB), a metro network from the edge to the fog (25 MB/s, 5 ms,
0.1 J/MB), and a wide-area network from the fog to the cloud (5 MB/s, 40 ms,
0.2 J/MB, 0.002 per MB). A transfer between two tiers crosses every hop in
between: it gets the narrowest bandwidth and pays the sum of the latencies,
energies, and costs, so a device reaches the cloud at 5 MB/s after 50 ms.
Nodes of one tier exchange data over a local network.
"""


@dataclass(frozen=True)
class Platform:
    """Compute nodes grouped into tiers, and the links between the tiers.

    Each node is a resource with a ``tier``. Its speed is read from its
    ``"speed"`` metadata entry, 1 when absent.

    Attributes:
        resources: The nodes.
        network: Links between tiers. A transfer between two different nodes
            uses the link joining their tiers.
    """

    resources: tuple[Resource, ...]
    network: Network

    def __post_init__(self) -> None:
        resources = tuple(self.resources)
        object.__setattr__(self, "resources", resources)
        seen: set[str] = set()
        for resource in resources:
            if resource.id in seen:
                raise ConfigError(f"platform lists node {resource.id!r} twice")
            seen.add(resource.id)
            if resource.tier is None:
                raise ConfigError(f"platform node {resource.id!r} has no tier")
            speed = resource.metadata.get("speed", 1.0)
            if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not speed > 0.0:
                raise ConfigError(f"platform node {resource.id!r} has invalid speed {speed!r}")

    @cached_property
    def _by_id(self) -> Mapping[str, Resource]:
        return MappingProxyType({resource.id: resource for resource in self.resources})

    @cached_property
    def _by_tier(self) -> Mapping[str, tuple[Resource, ...]]:
        groups: dict[str, list[Resource]] = {}
        for resource in self.resources:
            groups.setdefault(resource.endpoint, []).append(resource)
        return MappingProxyType({tier: tuple(nodes) for tier, nodes in groups.items()})

    @property
    def tiers(self) -> tuple[str, ...]:
        """Tiers that have at least one node, in the order nodes are listed."""
        return tuple(self._by_tier)

    def nodes(self, tier: str) -> tuple[Resource, ...]:
        """Return the nodes of ``tier``, empty when it has none."""
        return self._by_tier.get(tier, ())

    def resource(self, node: str) -> Resource:
        """Return the node with id ``node``.

        Raises:
            ConfigError: If the platform has no such node.
        """
        try:
            return self._by_id[node]
        except KeyError:
            raise ConfigError(f"the platform has no node {node!r}") from None

    def speed(self, node: str) -> float:
        """Return the speed of node ``node``, in work units per second per core."""
        return float(self.resource(node).metadata.get("speed", 1.0))

    def transfer_time(self, data: float, source: str, target: str) -> float:
        """Return the time to move ``data`` from node ``source`` to node ``target``.

        Zero for no data or one node, otherwise the transfer time of the link
        joining the two nodes' tiers, as the core model computes it.

        Raises:
            ConfigError: If the network has no link between the two tiers.
        """
        if data <= 0.0 or source == target:
            return 0.0
        a, b = self.resource(source).endpoint, self.resource(target).endpoint
        link = self.network.link(a, b)
        if link is None:
            raise ConfigError(f"the platform network has no link between {a!r} and {b!r}")
        return link.transfer_time(data)

    def duration(self, work: float, node: str) -> float:
        """Return the seconds node ``node`` needs for ``work`` work units.

        That is ``work / speed``, rounded to microseconds.
        """
        return round(work / self.speed(node), _DECIMALS)

    def modes(self, work: float, nodes: Sequence[Resource]) -> tuple[Mode, ...]:
        """Return one mode per node for a task of ``work`` work units.

        Each mode holds one core of its node for :meth:`duration` seconds and
        is located on that node, so choosing a mode places the task.
        """
        return tuple(Mode(self.duration(work, node.id), {node.id: 1.0}) for node in nodes)


def three_tier_platform(
    *,
    devices: int = 4,
    edge: int = 2,
    fog: int = 1,
    cloud: int = 1,
    nodes: Mapping[str, NodeSpec] | None = None,
    links: Iterable[Link] = (),
) -> Platform:
    """Build a platform of devices and of edge, fog, and cloud nodes.

    Nodes are named ``<tier>-<index>``, for example ``edge-0``. Devices get
    the resource kind ``"device"``, the other nodes ``"node"``, and every node
    records its speed in its ``"speed"`` metadata entry.

    Args:
        devices: Number of devices.
        edge: Number of edge nodes.
        fog: Number of fog nodes.
        cloud: Number of cloud nodes.
        nodes: Hardware per tier replacing :data:`DEFAULT_NODES`, for example
            ``{"cloud": NodeSpec(cores=32, speed=6.0)}``.
        links: Links replacing the :data:`DEFAULT_LINKS` entry that joins the
            same pair of tiers.

    Returns:
        The platform.

    Raises:
        ConfigError: If a count is negative, or ``nodes`` or ``links`` name a
            tier outside :data:`TIERS`.
    """
    counts = {"device": devices, "edge": edge, "fog": fog, "cloud": cloud}
    for tier, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ConfigError(f"the number of {tier} nodes must be a non-negative integer")
    hardware = dict(DEFAULT_NODES)
    for tier, spec in (nodes or {}).items():
        if tier not in hardware:
            raise ConfigError(f"unknown tier {tier!r}; tiers: {', '.join(TIERS)}")
        if not isinstance(spec, NodeSpec):
            raise ConfigError(f"hardware of tier {tier!r} must be a NodeSpec")
        hardware[tier] = spec
    table = {_pair(link.a, link.b): link for link in DEFAULT_LINKS}
    for link in links:
        table[_pair(link.a, link.b)] = link

    resources: list[Resource] = []
    for tier in TIERS:
        spec = hardware[tier]
        for index in range(counts[tier]):
            resources.append(
                Resource(
                    f"{tier}-{index}",
                    capacity=float(spec.cores),
                    kind="device" if tier == "device" else "node",
                    tier=tier,
                    idle_power=spec.idle_power,
                    busy_power=spec.busy_power,
                    cost_rate=spec.cost_rate,
                    metadata={"speed": spec.speed},
                )
            )
    return Platform(tuple(resources), Network(tuple(table.values())))


def _pair(a: str, b: str) -> tuple[str, str]:
    """Return the tier pair of a link in :data:`TIERS` order."""
    if a not in TIERS or b not in TIERS:
        raise ConfigError(f"link {a!r} <-> {b!r} names a tier outside {', '.join(TIERS)}")
    return (a, b) if TIERS.index(a) <= TIERS.index(b) else (b, a)
