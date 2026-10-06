"""The helpers the built-in families share."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import ClassVar

import pytest

from dispatchatlas.benchmarks._common import (
    ceil_tolerant,
    check_bool,
    check_int,
    check_int_range,
    check_matrix,
    check_number,
    check_objectives,
    check_optional_positive,
    check_per_item,
    check_seed,
    eligible_nodes,
    floor_tolerant,
    format_tag,
    instance_id,
    instance_metadata,
    numbered_ids,
    round_half_up,
)
from dispatchatlas.benchmarks.continuum import three_tier_platform
from dispatchatlas.benchmarks.continuum.platform import Platform
from dispatchatlas.benchmarks.registry import FamilyInfo
from dispatchatlas.core import ModelError, Network, Resource


@dataclass(frozen=True)
class Toy:
    size: int = 3
    rate: float = 0.5
    flag: bool = False
    limit: float | None = None
    names: tuple[str, ...] = ("makespan",)
    scale: tuple[float, ...] = (1.0, 2.5)

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="toy", name="Toy", problem_class="toy", summary="A toy family.", references=()
    )


TAGS = {"rate": "r", "flag": "f", "limit": "cap", "names": ""}


def test_ids_tag_the_parameters_that_differ_from_their_defaults():
    assert instance_id(Toy(), "3", 7, TAGS) == "toy-3-7"
    toy = Toy(size=9, rate=1.25, flag=True, limit=2.0, names=("a", "b"))
    assert instance_id(toy, "9", 0, TAGS) == "toy-9-r1.25-f-cap2-a+b-0"
    assert instance_id(Toy(flag=True), "3", 1, {"flag": "f"}) == "toy-3-f-1"


def test_metadata_records_parameters_and_seed():
    metadata = instance_metadata(Toy(), 4)
    assert metadata == {
        "size": 3,
        "rate": 0.5,
        "flag": False,
        "names": "makespan",
        "scale": "1.0,2.5",
        "seed": 4,
    }


@pytest.mark.parametrize("seed", [-1, True, 1.0, "3", None])
def test_seeds_must_be_non_negative_integers(seed):
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        check_seed(seed)
    with pytest.raises(ValueError, match="seed"):
        instance_id(Toy(), "3", seed, TAGS)
    with pytest.raises(ValueError, match="seed"):
        instance_metadata(Toy(), seed)


def test_format_tag():
    assert format_tag("w", True) == "w"
    assert format_tag("w", False) == "now"
    assert format_tag("due", 1.5) == "due1.5"
    assert format_tag("due", 2.0) == "due2"
    assert format_tag("", ("makespan", "energy")) == "makespan+energy"
    assert format_tag("s", 49) == "s49"


def test_integer_checks():
    assert check_int("jobs", 3) == 3
    assert check_int("nodes", 0, 0) == 0
    for value, message in [(0, "at least 1"), (True, "integer"), (2.0, "integer")]:
        with pytest.raises(ValueError, match=message):
            check_int("jobs", value)
    assert check_int_range("low", 2, "high", 2) == (2, 2)
    with pytest.raises(ValueError, match=r"low \(3\) cannot exceed high \(2\)"):
        check_int_range("low", 3, "high", 2)


def test_bool_check_refuses_integers():
    assert check_bool("flag", False) is False
    with pytest.raises(ValueError, match="true or false"):
        check_bool("flag", 1)


@pytest.mark.parametrize(
    ("value", "options", "message"),
    [
        ("1", {}, "must be a number"),
        (True, {}, "must be a number"),
        (math.inf, {}, "must be a finite number"),
        (math.nan, {}, "must be a finite number"),
        (-0.5, {}, "must be at least 0"),
        (1.5, {"maximum": 1.0}, r"must be \[0, 1\]"),
        (0.5, {"minimum": 1.0}, "must be at least 1"),
        (0, {"positive": True}, "must be positive"),
    ],
)
def test_number_check_refuses(value, options, message):
    with pytest.raises(ValueError, match=message):
        check_number("x", value, **options)


def test_number_check_returns_floats():
    assert check_number("x", 2) == 2.0
    assert isinstance(check_number("x", 2), float)
    assert check_number("x", 1.0, maximum=1.0, positive=True) == 1.0
    assert check_optional_positive("x", None) is None
    assert check_optional_positive("x", 3) == 3.0
    with pytest.raises(ValueError, match="positive"):
        check_optional_positive("x", 0.0)


def test_objective_check():
    assert check_objectives("makespan") == ("makespan",)
    assert check_objectives(["makespan", "total_tardiness"]) == ("makespan", "total_tardiness")
    with pytest.raises(ValueError, match="this family generates none"):
        check_objectives(("max_lateness",), due_dates=False)
    with pytest.raises(ValueError, match="set due_factor to generate them"):
        check_objectives(("tardy_tasks",), due_dates=False, due_parameter="due_factor")
    for value, message in [((), "non-empty"), (("unknown",), "unknown"), ((1,), "unknown")]:
        with pytest.raises(ValueError, match=message):
            check_objectives(value)
    with pytest.raises(ValueError, match="repeat"):
        check_objectives(("makespan", "makespan"))


def test_eligible_nodes_follow_the_tiers_and_leave_out_the_devices_by_default():
    platform = three_tier_platform(devices=2, edge=2, fog=1, cloud=1)
    assert [node.id for node in eligible_nodes(platform, None)] == [
        "edge-0",
        "edge-1",
        "fog-0",
        "cloud-0",
    ]
    assert [node.id for node in eligible_nodes(platform, ["cloud", "edge"])] == [
        "cloud-0",
        "edge-0",
        "edge-1",
    ]
    assert [node.id for node in eligible_nodes(platform, ("device",))] == ["device-0", "device-1"]
    assert [node.id for node in eligible_nodes(platform, deque(["fog"]))] == ["fog-0"]
    assert eligible_nodes(platform, ("fog", "cloud")) == platform.nodes("fog") + platform.nodes(
        "cloud"
    )


@pytest.mark.parametrize(
    ("tiers", "message"),
    [
        ("cloud", "tiers must be a sequence of tier names, got 'cloud'; write one tier as a list"),
        (b"cloud", "tiers must be a sequence of tier names, got b'cloud'$"),
        (bytearray(b""), "tiers must be a sequence of tier names"),
        (3, "tiers must be a sequence of tier names, got 3$"),
        ({"cloud"}, "tiers must be a sequence of tier names"),
        ({"cloud": 1}, "tiers must be a sequence of tier names"),
        ({"cloud": 1}.keys(), "tiers must be a sequence of tier names"),
        (iter(["cloud"]), "tiers must be a sequence of tier names"),
        (("cloud", 3), "tiers must be a sequence of tier names"),
        (("cloud", None), "tiers must be a sequence of tier names"),
        (("cloud", ["fog"]), "tiers must be a sequence of tier names"),
        (("cloud", "cloud"), r"tiers repeat a name: \['cloud', 'cloud'\]"),
        (["fog", "cloud", "fog"], "tiers repeat a name"),
        ((), r"the platform has no node in tiers \[\]$"),
        (("orbit",), r"the platform has no node in tiers \['orbit'\]"),
        (("cloud", "orbit", "fog", "sea"), r"the platform has no node in tiers \['orbit', 'sea'\]"),
        (("Cloud",), r"the platform has no node in tiers \['Cloud'\]"),
    ],
)
def test_eligible_nodes_refuse_tiers_that_are_not_distinct_names_of_tiers_with_nodes(
    tiers, message
):
    platform = three_tier_platform(devices=1, edge=1, fog=1, cloud=1)
    with pytest.raises(ValueError, match=message):
        eligible_nodes(platform, tiers)


def test_eligible_nodes_refuse_platforms_that_cannot_run_a_task():
    with pytest.raises(ValueError, match=r"no node in tiers \['fog'\]"):
        eligible_nodes(three_tier_platform(fog=0), ("edge", "fog"))
    with pytest.raises(ValueError, match="the platform has no node in tiers other than 'device'"):
        eligible_nodes(three_tier_platform(edge=0, fog=0, cloud=0), None)
    assert len(eligible_nodes(three_tier_platform(edge=0, fog=0, cloud=0), ("device",))) == 4
    nodes = (Resource("a-0", tier="a"), Resource("b-0", capacity=0.5, tier="b"))
    platform = Platform(nodes, Network())
    with pytest.raises(ValueError, match=r"node 'b-0' has capacity 0\.5, less than the one core"):
        eligible_nodes(platform, None)
    assert eligible_nodes(platform, ("a",)) == nodes[:1]


def test_numbered_ids_share_one_width():
    assert numbered_ids("m", 3) == ["m0", "m1", "m2"]
    assert numbered_ids("j", 11)[:2] == ["j00", "j01"]
    assert numbered_ids("a", 9, start=1)[-1] == "a9"
    assert numbered_ids("a", 10, start=1)[:2] == ["a01", "a02"]
    assert numbered_ids("x", 0) == []


def test_builder_data_checks():
    assert check_matrix([(1.0, 2.0)], "times") == [[1.0, 2.0]]
    with pytest.raises(ModelError, match="times need at least one job and one machine"):
        check_matrix([], "times")
    with pytest.raises(ModelError, match="must list 2 values for every node"):
        check_matrix([[1.0, 2.0], [3.0]], "speeds", rows="node", columns="tier")
    assert check_per_item(None, 3, "weights") is None
    assert check_per_item([1, 2], 2, "weights") == [1.0, 2.0]
    with pytest.raises(ModelError, match="due_dates lists 1 values for 2 tasks"):
        check_per_item([1.0], 2, "due_dates", items="tasks")


def test_rounding_tolerates_floating_point_error():
    assert math.floor(2.3 * 100) == 229
    assert floor_tolerant(2.3 * 100) == 230
    assert math.ceil(0.07 * 100) == 8
    assert ceil_tolerant(0.07 * 100) == 7
    assert (floor_tolerant(2.5), ceil_tolerant(2.5)) == (2, 3)
    assert [round_half_up(value) for value in (2.5, 3.5, -2.5, 2.4999)] == [3, 4, -2, 2]
