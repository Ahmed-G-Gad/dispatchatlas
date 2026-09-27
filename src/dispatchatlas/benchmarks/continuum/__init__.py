"""Scheduling across an edge-fog-cloud continuum.

Devices issue applications; compute nodes in the edge, fog, and cloud tiers,
and the devices themselves, run their tasks; a network carries data between
tiers. Placement is a decision: every compute task has one mode per node it
may run on, with duration ``work / speed`` on one core of that node. Each
application starts and ends on its device, through zero-duration source and
sink tasks, so offloading pays upload and download transfers, and the
response time of an application runs from its release to its sink. Because
nodes and links carry power, price, latency, and bandwidth, the core
objectives ``energy``, ``cost``, ``makespan``, and ``mean_response_time`` all
measure real consequences of placement.

Three families are registered:

- ``continuum-offloading``: independent requests at Poisson arrival times.
- ``continuum-workflow``: layered random or fork-join workflows.
- ``continuum-pipeline``: periodic frames through chains of stages.

The building blocks are public, so other workloads can use the same model::

    from dispatchatlas.benchmarks.continuum import (
        Application,
        NodeSpec,
        build_problem,
        three_tier_platform,
    )

    fast_cloud = NodeSpec(cores=16, speed=6.0, idle_power=60.0, busy_power=300.0)
    platform = three_tier_platform(devices=2, cloud=2, nodes={"cloud": fast_cloud})
    request = Application(
        "a0", "device-0", release=0.0, work=(2.0,), upload=(1.5,), download=(0.1,)
    )
    problem = build_problem("one-request", platform, [request], due_factor=2.0)
"""

from dispatchatlas.benchmarks.continuum.applications import (
    Application,
    build_problem,
    eligible_nodes,
    response_bound,
)
from dispatchatlas.benchmarks.continuum.families import (
    STRUCTURES,
    ContinuumOffloading,
    ContinuumPipeline,
    ContinuumWorkflow,
    fork_join_graph,
    register,
)
from dispatchatlas.benchmarks.continuum.platform import (
    DEFAULT_LINKS,
    DEFAULT_NODES,
    TIERS,
    NodeSpec,
    Platform,
    three_tier_platform,
)

__all__ = [
    "DEFAULT_LINKS",
    "DEFAULT_NODES",
    "STRUCTURES",
    "TIERS",
    "Application",
    "ContinuumOffloading",
    "ContinuumPipeline",
    "ContinuumWorkflow",
    "NodeSpec",
    "Platform",
    "build_problem",
    "eligible_nodes",
    "fork_join_graph",
    "register",
    "response_bound",
    "three_tier_platform",
]
