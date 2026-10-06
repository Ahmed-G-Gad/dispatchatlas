"""The helpers the built-in families and readers share."""

from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import ClassVar

import pytest

from dispatchatlas.benchmarks._common import (
    READ_WITHOUT_DUE_DATES,
    ceil_tolerant,
    check_bool,
    check_int,
    check_int_range,
    check_matrix,
    check_number,
    check_objectives,
    check_optional_positive,
    check_per_item,
    check_problem_id,
    check_seed,
    check_source_path,
    decimal_number,
    default_problem_id,
    eligible_nodes,
    floor_tolerant,
    format_tag,
    id_from_name,
    instance_id,
    instance_metadata,
    numbered_ids,
    round_half_up,
)
from dispatchatlas.benchmarks.continuum import NodeSpec, three_tier_platform
from dispatchatlas.benchmarks.continuum.platform import Platform
from dispatchatlas.benchmarks.registry import FamilyInfo
from dispatchatlas.campaigns.config import check_name
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


@pytest.mark.parametrize(
    "value",
    [10**400, -(10**400), 10**5000],
    ids=["401 digits", "negative", "5001 digits"],
)
def test_number_check_refuses_an_integer_too_large_for_a_float(value):
    # float() raises OverflowError for these, and an integer of more than 4300 digits
    # cannot be written in a message either.
    message = "^x must be a finite number, got an integer too large for a float$"
    with pytest.raises(ValueError, match=message):
        check_number("x", value)
    with pytest.raises(ValueError, match=message):
        check_number("x", value, minimum=-math.inf, maximum=1.0, positive=True)
    with pytest.raises(ValueError, match=message):
        check_optional_positive("x", value)
    with pytest.raises(ValueError, match=r"^speed must be a finite number, got an integer too"):
        NodeSpec(speed=value)
    # The largest integers a float holds are still read.
    assert check_number("x", 2**1023) == 2.0**1023
    assert check_number("x", 10**308) == 1e308


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


def test_objective_check_words_the_missing_due_dates_for_its_caller():
    family = (
        r"^objectives \['max_lateness', 'tardy_tasks'\] need due dates; this family generates none$"
    )
    with pytest.raises(ValueError, match=family):
        check_objectives(("tardy_tasks", "makespan", "max_lateness"), due_dates=False)
    reader = (
        r"^objectives \['tardy_tasks'\] need due dates; the problems this reader builds have none$"
    )
    with pytest.raises(ValueError, match=reader):
        check_objectives("tardy_tasks", due_dates=False, no_due_dates=READ_WITHOUT_DUE_DATES)
    # A parameter that adds due dates is named whatever the other wording is.
    with pytest.raises(ValueError, match=r"need due dates; set due_factor to generate them$"):
        check_objectives(
            "tardy_tasks", due_dates=False, due_parameter="due_factor", no_due_dates="unused"
        )
    # The wording plays no part while the problems have due dates.
    assert check_objectives("tardy_tasks", no_due_dates="unused") == ("tardy_tasks",)


class TextPath:
    """A path-like object whose path is the text it is given."""

    def __init__(self, text: object) -> None:
        self.text = text

    def __fspath__(self) -> object:
        return self.text


def test_a_source_path_is_returned_as_a_path_without_opening_a_file(tmp_path):
    missing = tmp_path / "missing" / "run 01.dat"
    assert check_source_path(missing) == missing
    assert check_source_path(str(missing)) == missing
    assert check_source_path(TextPath(str(missing))) == missing
    assert check_source_path(PurePosixPath("runs/run.dat")) == Path("runs/run.dat")
    # A string is a path whatever it holds.
    assert check_source_path("") == Path()
    assert check_source_path('{"name": "run"}') == Path('{"name": "run"}')
    assert isinstance(check_source_path("run.dat"), Path)
    assert not missing.parent.exists()


@pytest.mark.parametrize(
    "value", [None, 5, 1.5, b"run.dat", bytearray(b"run.dat"), ["run.dat"], {}]
)
def test_a_source_path_of_another_type_is_refused(value):
    message = f"^source must be a path, got {type(value).__name__}$"
    with pytest.raises(ValueError, match=message):
        check_source_path(value)
    message = f"^source must be a table or a path, got {type(value).__name__}$"
    with pytest.raises(ValueError, match=message):
        check_source_path(value, "a table or a path")


@pytest.mark.parametrize(
    "value",
    [
        "run\0.dat",
        "\0",
        Path("runs/run\0.dat"),
        TextPath("run\0.dat"),
        TextPath(b"run.dat"),
        TextPath(7),
    ],
    ids=["null", "only a null", "null in a path", "null in a path-like", "bytes", "integer"],
)
def test_a_source_path_that_is_not_text_or_holds_a_null_character_is_refused(value):
    message = r"^a source path must be text without a null character$"
    with pytest.raises(ValueError, match=message):
        check_source_path(value)


def test_a_source_path_that_file_names_cannot_encode_is_refused():
    name = "lone\ud800.dat"
    try:
        os.fsencode(name)
    except UnicodeEncodeError:
        message = "^a source path must be text that the encoding of file names can express$"
        with pytest.raises(ValueError, match=message):
            check_source_path(name)
    else:
        # The encoding of file names on this platform expresses a lone surrogate.
        assert check_source_path(name) == Path(name)


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


@pytest.mark.parametrize("platform", [None, "cloud", 3, {}, three_tier_platform().resources])
def test_eligible_nodes_refuse_what_is_not_a_platform(platform):
    message = f"^platform must be a Platform, got {type(platform).__name__}$"
    for tiers in (None, ("cloud",), "cloud"):
        with pytest.raises(ValueError, match=message):
            eligible_nodes(platform, tiers)


def test_a_problem_id_is_kept_as_given_or_refused():
    assert check_problem_id(None) is None
    assert check_problem_id("run-1") == "run-1"
    # An id the caller gives is not rewritten.
    assert check_problem_id("_a b/c.") == "_a b/c."
    for value in ("", " ", " x", "x ", "x\n", "\t", 7, 0, False, b"x", ["x"], ("x",)):
        with pytest.raises(ValueError, match=r"^problem_id must be a non-empty string without"):
            check_problem_id(value)
    with pytest.raises(ValueError, match=r"surrounding spaces, got ''$"):
        check_problem_id("")


@pytest.mark.parametrize(
    ("name", "identifier"),
    [
        ("Montage_25", "Montage_25"),
        ("a.b-c_d", "a.b-c_d"),
        ("my flow (1)", "my-flow-1"),
        ("Map Reduce/1", "Map-Reduce-1"),
        ("a\nb\t\tc", "a-b-c"),
        ("_private run", "private-run"),
        ("..hidden", "hidden"),
        ("-._x_.-", "x_"),
        ("caf\N{LATIN SMALL LETTER E WITH ACUTE}", "caf"),
        ("na\N{LATIN SMALL LETTER I WITH DIAERESIS}ve run", "na-ve-run"),
        ("\N{ARABIC-INDIC DIGIT ONE}\N{ARABIC-INDIC DIGIT TWO}", ""),
        ("", ""),
        ("   ", ""),
        ("///", ""),
        ("_", ""),
        ("-.-", ""),
    ],
)
def test_an_id_made_from_a_name_can_name_a_file(name, identifier):
    assert id_from_name(name) == identifier
    if identifier:
        assert check_name(identifier, "problem id") == identifier
        assert id_from_name(identifier) == identifier
        assert check_problem_id(identifier) == identifier


def test_a_default_problem_id_comes_from_the_first_name_that_leaves_one():
    assert default_problem_id(("run 01 (b)", "Shared Name"), "fallback") == "run-01-b"
    assert default_problem_id(("---", "Shared Name"), "fallback") == "Shared-Name"
    assert default_problem_id(("", "_shared"), "fallback") == "shared"
    assert default_problem_id(("", "///"), "fallback") == "fallback"
    assert default_problem_id((), "fallback") == "fallback"
    assert default_problem_id(iter(["_", "b c"]), "fallback") == "b-c"


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("0", 0.0),
        ("12", 12.0),
        ("007", 7.0),
        ("-12", -12.0),
        ("0.5", 0.5),
        (".5", 0.5),
        ("5.", 5.0),
        ("-.5", -0.5),
        ("1e3", 1000.0),
        ("1E3", 1000.0),
        ("1e+3", 1000.0),
        ("1e-3", 0.001),
        ("2.05e1", 20.5),
        (".25e+6", 250000.0),
        ("1.7e308", 1.7e308),
        ("4.9e-324", 5e-324),
        ("1e-400", 0.0),
    ],
)
def test_decimal_numbers_are_read(text, value):
    number = decimal_number(text)
    assert isinstance(number, float)
    assert number == value
    assert number == float(text)


@pytest.mark.parametrize("text", ["-0", "-0.0", "-.0", "-0e5", "-1e-400"])
def test_a_negative_zero_is_read_as_zero(text):
    assert str(float(text)) == "-0.0"
    assert str(decimal_number(text)) == "0.0"


def test_a_decimal_number_too_large_for_a_float_is_an_infinity():
    assert decimal_number("1e999") == math.inf
    assert decimal_number("-1e999") == -math.inf
    assert decimal_number("9" * 400) == math.inf


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "+1",
        "+.5",
        "+1e3",
        " 1",
        "1 ",
        "1\n",
        "\t1",
        "1 000",
        "nan",
        "NaN",
        "-nan",
        "inf",
        "-inf",
        "Infinity",
        "1_000",
        "1__0",
        "_1",
        "1_",
        "1e1_0",
        "1,5",
        "1.2.3",
        "1e",
        "e5",
        "1e+",
        "1e5.0",
        "1e2e3",
        "--1",
        "-+1",
        "1-",
        "-",
        ".",
        "-.",
        "0x10",
        "1f",
        "1d0",
        "1/2",
        "one",
        "\N{ARABIC-INDIC DIGIT ONE}\N{ARABIC-INDIC DIGIT TWO}",
        "\N{FULLWIDTH DIGIT TWO}0",
        "1e\N{FULLWIDTH DIGIT FIVE}",
        "\N{DEVANAGARI DIGIT NINE}",
        "\N{SUPERSCRIPT TWO}",
        "1\N{NO-BREAK SPACE}",
        "\N{MINUS SIGN}1",
    ],
)
def test_other_texts_are_not_decimal_numbers(text):
    with pytest.raises(ValueError, match=r"^not a decimal number$"):
        decimal_number(text)


def test_long_texts_are_read_or_refused_in_time_proportional_to_their_length():
    digits = "9" * 200_000
    refused = [
        f"{digits}x",
        f"1.{digits}x",
        f".{digits}x",
        f"1e{digits}x",
        f"1e-{digits}x",
        f"{digits}e",
        "-" * 200_000,
        "1" + "e1" * 100_000,
        "1" + "." * 200_000,
        "+" + digits,
    ]
    started = time.perf_counter()
    for text in refused:
        # The message does not repeat the text.
        with pytest.raises(ValueError, match=r"^not a decimal number$"):
            decimal_number(text)
    assert decimal_number(digits) == math.inf
    assert decimal_number(f"1.{digits}") == 2.0
    assert decimal_number(f"1e-{digits}") == 0.0
    assert decimal_number("0." + "0" * 200_000 + "1") == 0.0
    # Each text takes milliseconds; a check that backtracks would take minutes.
    assert time.perf_counter() - started < 30.0


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
