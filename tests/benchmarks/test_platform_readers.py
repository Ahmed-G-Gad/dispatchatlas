"""What the readers onto a continuum platform have in common.

The readers of WfCommons instances, of Pegasus DAX files, and of the Alibaba 2018 batch
jobs choose the nodes of a platform, name their problems, and read numbers in one way.
"""

from __future__ import annotations

import json
import math
import re
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.benchmarks import (
    iter_alibaba_batch,
    read_pegasus_dax,
    read_wfcommons,
    three_tier_platform,
)
from dispatchatlas.benchmarks.continuum.platform import Platform
from dispatchatlas.campaigns.config import check_name
from dispatchatlas.core import ConfigError, Network, Problem, Resource, SerializationError

PLATFORM = three_tier_platform(devices=1, edge=1, fog=1, cloud=1)
NODES = ("edge-0", "fog-0", "cloud-0")
HALF = Platform((Resource("a-0", capacity=0.5, tier="a"),), Network())

Reader = Callable[..., Problem]


def wfcommons_text(number: str = "1.5", name: str = "one task") -> str:
    """A WfFormat instance of one task whose runtime is written as ``number``."""
    document = {
        "name": name,
        "schemaVersion": "1.5",
        "workflow": {
            "specification": {"tasks": [{"id": "a", "name": "a"}]},
            "execution": {"tasks": [{"id": "a", "runtimeInSeconds": 0}]},
        },
    }
    text = json.dumps(document, ensure_ascii=False)
    marker = '"runtimeInSeconds": 0'
    assert text.count(marker) == 1
    return text.replace(marker, f'"runtimeInSeconds": {number}')


def pegasus_text(number: str = "1.5", name: str = "one task") -> str:
    """A DAX document of one job whose runtime is written as ``number``."""
    return f'<adag name="{name}"><job id="a" runtime="{number}"/></adag>\n'


def alibaba_lines(number: str = "1.5") -> list[str]:
    """The table of one job of one task whose ``plan_cpu`` is written as ``number``."""
    return [f"M1,1,j_1,1,Terminated,10,20,{number},0.5\n"]


FILE_READERS: dict[str, tuple[Reader, Callable[..., str], str, str]] = {
    "read_wfcommons": (read_wfcommons, wfcommons_text, ".json", "wfcommons"),
    "read_pegasus_dax": (read_pegasus_dax, pegasus_text, ".xml", "pegasus-dax"),
}
"""The readers of one problem per file: the function, a source, an extension, a fallback id."""


def written(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def readers(tmp_path: Path) -> dict[str, Reader]:
    """Each reader on a source of one task, by its name, as a function of a platform."""
    document = written(tmp_path, "one.json", wfcommons_text())
    dax = written(tmp_path, "one.xml", pegasus_text())

    def alibaba(platform: Any, **options: Any) -> Problem:
        (problem,) = iter_alibaba_batch(alibaba_lines(), platform, **options)
        return problem

    return {
        "read_wfcommons": lambda platform, **options: read_wfcommons(document, platform, **options),
        "read_pegasus_dax": lambda platform, **options: read_pegasus_dax(dax, platform, **options),
        "iter_alibaba_batch": alibaba,
    }


# ------------------------------------------------------------------ nodes


@pytest.mark.parametrize("tiers", [None, ["device", "cloud"], ("cloud", "edge"), deque(["fog"])])
def test_the_readers_choose_the_same_nodes(readers, tiers):
    expected = list(NODES) if tiers is None else [f"{tier}-0" for tier in tiers]
    for read in readers.values():
        (task,) = read(PLATFORM, tiers=tiers).tasks
        assert [mode.location for mode in task.modes] == expected
        assert [dict(mode.demands) for mode in task.modes] == [{node: 1.0} for node in expected]


REFUSED: list[tuple[Any, dict[str, Any], str]] = [
    (None, {}, "^platform must be a Platform, got NoneType$"),
    ("cloud", {}, "^platform must be a Platform, got str$"),
    (PLATFORM.resources, {"tiers": ("cloud",)}, "^platform must be a Platform, got tuple$"),
    (PLATFORM, {"tiers": "cloud"}, "^tiers must be a sequence of tier names, got 'cloud'; write"),
    (PLATFORM, {"tiers": b"cloud"}, "^tiers must be a sequence of tier names, got b'cloud'$"),
    (PLATFORM, {"tiers": {"cloud"}}, "^tiers must be a sequence of tier names"),
    (PLATFORM, {"tiers": {"cloud": 1}.keys()}, "^tiers must be a sequence of tier names"),
    (PLATFORM, {"tiers": iter(["cloud"])}, "^tiers must be a sequence of tier names"),
    (PLATFORM, {"tiers": 3}, "^tiers must be a sequence of tier names, got 3$"),
    (PLATFORM, {"tiers": ("cloud", 3)}, "^tiers must be a sequence of tier names"),
    (PLATFORM, {"tiers": ("cloud", "cloud")}, r"^tiers repeat a name: \['cloud', 'cloud'\]$"),
    (PLATFORM, {"tiers": ["fog", "cloud", "fog"]}, "^tiers repeat a name"),
    (PLATFORM, {"tiers": ()}, r"^the platform has no node in tiers \[\]$"),
    (PLATFORM, {"tiers": ("orbit", "cloud")}, r"^the platform has no node in tiers \['orbit'\]$"),
    (
        three_tier_platform(fog=0),
        {"tiers": ("fog", "cloud")},
        r"^the platform has no node in tiers \['fog'\]$",
    ),
    (
        three_tier_platform(edge=0, fog=0, cloud=0),
        {},
        "^the platform has no node in tiers other than 'device'$",
    ),
    (HALF, {}, r"^node 'a-0' has capacity 0\.5, less than the one core a task holds$"),
    (PLATFORM, {"objectives": "speed"}, "^unknown objective 'speed'$"),
    (PLATFORM, {"objectives": ()}, "^objectives must be a non-empty list of objective names$"),
    (PLATFORM, {"objectives": ("makespan", "makespan")}, "^objectives repeat a name"),
    (PLATFORM, {"objectives": "total_tardiness"}, r"^objectives \['total_tardiness'\] need due"),
    (PLATFORM, {"objectives": ("makespan", "tardy_tasks")}, r"\['tardy_tasks'\] need due dates"),
    (PLATFORM, {"objectives": ["max_lateness"]}, r"\['max_lateness'\] need due dates"),
]


@pytest.mark.parametrize(("platform", "options", "message"), REFUSED)
def test_the_readers_refuse_the_same_platforms_tiers_and_objectives(
    readers, platform, options, message
):
    found = set()
    for name, read in readers.items():
        with pytest.raises(ConfigError) as caught:
            read(platform, **options)
        prefix = f"invalid arguments for {name}: "
        assert str(caught.value).startswith(prefix)
        found.add(str(caught.value).removeprefix(prefix))
    # The three readers refuse the value in the same words.
    (refusal,) = found
    assert re.search(message, refusal)


def test_the_readers_report_a_number_too_large_for_a_float_as_a_configuration_error(readers):
    for name, read in readers.items():
        option = "edge_data" if name == "iter_alibaba_batch" else "bytes_per_unit"
        message = (
            f"^invalid arguments for {name}: {option} must be a finite number, "
            "got an integer too large for a float$"
        )
        for value in (10**400, 10**5000):
            with pytest.raises(ConfigError, match=message):
                read(PLATFORM, **{option: value})


def test_no_reader_gives_a_problem_without_due_dates_a_due_date_objective(readers):
    for read in readers.values():
        problem = read(PLATFORM, objectives=("makespan", "energy", "cost"))
        assert problem.objectives == ("makespan", "energy", "cost")
        assert all(task.due is None for task in problem.tasks)


# ------------------------------------------------------------------ problem ids


@pytest.mark.parametrize(
    ("stem", "identifier"),
    [
        ("Montage_25", "Montage_25"),
        ("my flow (1)", "my-flow-1"),
        ("run.2024.tar", "run.2024.tar"),
        ("_draft.run", "draft.run"),
        ("..hidden", "hidden"),
        # A file name that leaves nothing gives way to the name written in the file.
        ("---", "one-task"),
        ("_", "one-task"),
    ],
)
def test_a_file_names_its_problem_in_every_reader_of_one_problem_per_file(
    tmp_path, stem, identifier
):
    for read, text, extension, _ in FILE_READERS.values():
        problem = read(written(tmp_path, stem + extension, text()), PLATFORM)
        assert problem.id == identifier
        assert check_name(problem.id, "problem id") == identifier
        assert {task.group for task in problem.tasks} == {identifier}
        assert problem.metadata["name"] == "one task"


@pytest.mark.parametrize("name", ["///", "_", " "])
def test_a_file_without_a_usable_name_is_called_after_its_format(tmp_path, name):
    for read, text, extension, fallback in FILE_READERS.values():
        problem = read(written(tmp_path, "---" + extension, text(name=name)), PLATFORM)
        assert problem.id == fallback
        assert problem.metadata["name"] == name


def test_files_that_share_a_written_name_get_ids_of_their_own(tmp_path):
    for read, text, extension, _ in FILE_READERS.values():
        paths = [written(tmp_path, f"run-00{number}{extension}", text()) for number in (1, 2)]
        problems = [read(path, PLATFORM) for path in paths]
        assert [problem.id for problem in problems] == ["run-001", "run-002"]
        assert [problem.metadata["name"] for problem in problems] == ["one task", "one task"]


@pytest.mark.parametrize("problem_id", ["", " ", " x", "x ", "x\n", 7, ["x"]])
def test_an_unusable_problem_id_is_refused_and_never_replaced(tmp_path, problem_id):
    for name, (read, text, extension, _) in FILE_READERS.items():
        path = written(tmp_path, "one" + extension, text())
        with pytest.raises(ConfigError) as caught:
            read(path, PLATFORM, problem_id=problem_id)
        assert str(caught.value) == (
            f"invalid arguments for {name}: problem_id must be a non-empty string "
            f"without surrounding spaces, got {problem_id!r}"
        )


def test_a_given_problem_id_is_kept_as_written(tmp_path):
    for read, text, extension, _ in FILE_READERS.values():
        problem = read(written(tmp_path, "one" + extension, text()), PLATFORM, problem_id="_a b/c.")
        assert problem.id == "_a b/c."
        assert {task.group for task in problem.tasks} == {"_a b/c."}


# ------------------------------------------------------------------ numbers


@pytest.mark.parametrize("number", ["1.5", "1.50", "0.15e1", "15e-1", "1.5E0", "1.5e+0"])
def test_the_readers_read_the_same_numbers(tmp_path, number):
    durations = [PLATFORM.duration(1.5, node) for node in NODES]
    for read, text, extension, _ in FILE_READERS.values():
        problem = read(written(tmp_path, "one" + extension, text(number)), PLATFORM)
        assert [mode.duration for mode in problem.tasks[0].modes] == durations
    (job,) = iter_alibaba_batch(alibaba_lines(number), PLATFORM)
    assert job.tasks[0].metadata["plan_cpu"] == 1.5


@pytest.mark.parametrize(
    "number",
    [
        "nan",
        "NaN",
        "inf",
        "Infinity",
        "-Infinity",
        "1_500",
        "+1.5",
        "1e",
        "0x10",
        "1e999",
        pytest.param("9" * 400, id="400 digits"),
        "\N{ARABIC-INDIC DIGIT ONE}\N{ARABIC-INDIC DIGIT FIVE}",
        "\N{FULLWIDTH DIGIT ONE}.\N{FULLWIDTH DIGIT FIVE}",
    ],
)
def test_the_readers_refuse_the_same_texts_and_numbers_too_large_for_a_float(tmp_path, number):
    for read, text, extension, _ in FILE_READERS.values():
        with pytest.raises(SerializationError):
            read(written(tmp_path, "one" + extension, text(number)), PLATFORM)
    with pytest.raises(SerializationError, match="row 1: plan_cpu must be a finite decimal number"):
        list(iter_alibaba_batch(alibaba_lines(number), PLATFORM))


def test_the_readers_read_a_negative_zero_as_zero(tmp_path):
    for read, text, extension, _ in FILE_READERS.values():
        problem = read(written(tmp_path, "one" + extension, text("-0.0")), PLATFORM)
        for mode in problem.tasks[0].modes:
            assert mode.duration == 0.0
            assert math.copysign(1.0, mode.duration) == 1.0
    (job,) = iter_alibaba_batch(alibaba_lines("-0.0"), PLATFORM)
    assert str(job.tasks[0].metadata["plan_cpu"]) == "0.0"
