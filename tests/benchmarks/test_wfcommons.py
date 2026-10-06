"""Tests of the WfCommons WfFormat reader."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from dispatchatlas.benchmarks.continuum import NodeSpec, three_tier_platform
from dispatchatlas.benchmarks.wfcommons import SCHEMA_VERSIONS, read_wfcommons
from dispatchatlas.campaigns.config import check_name
from dispatchatlas.core import ConfigError, Link, Problem, SerializationError, check_schedule
from dispatchatlas.solvers.scheduling import Decoder

PLATFORM = three_tier_platform(devices=1, edge=1, fog=1, cloud=1)

FILES = {
    "raw": 5_000_000,
    "chunk_a": 2_000_000,
    "chunk_b": 3_000_000,
    "part_a": 1_000_000,
    "part_b": 500_000,
    "log_b": 7,
    "merged": 250_000,
    "summary": 100,
}
TASKS = [
    # id, runtime, parents, children, inputs, outputs
    ("split", 10, [], ["map_a"], ["raw"], ["chunk_a", "chunk_b"]),
    ("map_a", 20.5, ["split"], ["reduce"], ["chunk_a"], ["part_a"]),
    ("map_b", 30, ["split"], ["reduce"], ["chunk_b"], ["part_b", "log_b"]),
    ("reduce", 5, ["map_a", "map_b"], ["report"], ["part_a", "part_b"], ["merged"]),
    ("report", 0, ["reduce"], [], ["merged"], ["summary"]),
]
EDGES = [
    ("split", "map_a", 2.0),
    ("split", "map_b", 3.0),
    ("map_a", "reduce", 1.0),
    ("map_b", "reduce", 0.5),
    ("reduce", "report", 0.25),
]


Document = dict[str, Any]
Change = Callable[[Document], None]


def split_document(version: object = "1.5") -> Document:
    """The workflow in the layout of schema 1.5 and later."""
    return {
        "name": "Map Reduce/1",
        "schemaVersion": version,
        "workflow": {
            "specification": {
                "tasks": [
                    {
                        "name": key.split("_")[0],
                        "id": key,
                        "parents": parents,
                        "children": children,
                        "inputFiles": inputs,
                        "outputFiles": outputs,
                    }
                    for key, _, parents, children, inputs, outputs in TASKS
                ],
                "files": [{"id": name, "sizeInBytes": size} for name, size in FILES.items()],
            },
            "execution": {
                "makespanInSeconds": 70.0,
                "executedAt": "2024-01-01T00:00:00Z",
                "tasks": [{"id": key, "runtimeInSeconds": runtime} for key, runtime, *_ in TASKS],
            },
        },
    }


def flat_document() -> Document:
    """The same workflow in the layout of schema 1.4, which lists parents only."""
    return {
        "name": "Map Reduce/1",
        "schemaVersion": "1.4",
        "workflow": {
            "makespanInSeconds": 70.0,
            "executedAt": "2024-01-01T00:00:00Z",
            "tasks": [
                {
                    "name": key,
                    "type": "compute",
                    "parents": parents,
                    "runtimeInSeconds": runtime,
                    "files": [
                        {"name": name, "sizeInBytes": FILES[name], "link": "input"}
                        for name in inputs
                    ]
                    + [
                        {"name": name, "sizeInBytes": FILES[name], "link": "output"}
                        for name in outputs
                    ],
                }
                for key, runtime, parents, _, inputs, outputs in TASKS
            ],
        },
    }


def edges_of(problem: Problem) -> list[tuple[str, str, float]]:
    return [(d.before, d.after, d.data) for d in problem.dependencies]


def test_reads_tasks_placements_and_data():
    problem = read_wfcommons(split_document(), PLATFORM)
    assert problem.id == "Map-Reduce-1"
    assert [task.id for task in problem.tasks] == [key for key, *_ in TASKS]
    for task, (_, runtime, *_) in zip(problem.tasks, TASKS, strict=True):
        assert [mode.location for mode in task.modes] == ["edge-0", "fog-0", "cloud-0"]
        for mode in task.modes:
            assert mode.location is not None
            assert mode.duration == PLATFORM.duration(runtime, mode.location)
        assert task.group == problem.id
        assert task.release == 0.0
    assert [(d.before, d.after, d.data) for d in problem.dependencies] == EDGES
    assert problem.network == PLATFORM.network
    assert problem.objectives == ("makespan",)
    assert dict(problem.metadata) == {
        "name": "Map Reduce/1",
        "schema_version": "1.5",
        "source": "wfcommons",
    }


def test_the_flat_layout_of_schema_1_4_reads_the_same_workflow():
    split = read_wfcommons(split_document(), PLATFORM)
    flat = read_wfcommons(flat_document(), PLATFORM)
    assert flat.tasks == split.tasks
    assert flat.dependencies == split.dependencies
    assert flat.metadata["schema_version"] == "1.4"


@pytest.mark.parametrize("version", ["1.5", "1.6"])
def test_split_layout_versions_are_read(version):
    assert version in SCHEMA_VERSIONS
    problem = read_wfcommons(split_document(version), PLATFORM)
    assert len(problem.dependencies) == len(EDGES)


@pytest.mark.parametrize("version", [None, "1.3", "2.0", 1.5])
def test_other_schema_versions_are_refused(version):
    with pytest.raises(SerializationError, match="schemaVersion"):
        read_wfcommons(split_document(version), PLATFORM)


def test_reads_files_and_names_problems_after_them(tmp_path):
    path = tmp_path / "montage-small.json"
    path.write_text(json.dumps(split_document()), encoding="utf-8")
    problem = read_wfcommons(path, PLATFORM)
    # The file names the problem; the name of the instance stays in the metadata.
    assert problem.id == "montage-small"
    assert problem.metadata["name"] == "Map Reduce/1"
    assert {task.group for task in problem.tasks} == {"montage-small"}
    assert problem == read_wfcommons(split_document(), PLATFORM, problem_id="montage-small")
    assert read_wfcommons(str(path), PLATFORM) == problem
    assert read_wfcommons(path, PLATFORM, problem_id="chosen").id == "chosen"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SerializationError, match="invalid JSON"):
        read_wfcommons(path, PLATFORM)


def test_default_ids_of_documents_and_of_files(tmp_path):
    document = split_document()
    assert read_wfcommons(document, PLATFORM).id == "Map-Reduce-1"
    document["name"] = "_private run"
    assert check_name(read_wfcommons(document, PLATFORM).id, "problem id") == "private-run"
    for name in ("", "///", "  ", "_", 5, None):
        document["name"] = name
        assert read_wfcommons(document, PLATFORM).id == "wfcommons"
    del document["name"]
    nameless = read_wfcommons(document, PLATFORM)
    assert nameless.id == "wfcommons"
    assert "name" not in nameless.metadata
    # A file is named after itself, whatever its instance is called.
    spaced = tmp_path / "run 01 (b).json"
    spaced.write_text(json.dumps(split_document()), encoding="utf-8")
    assert check_name(read_wfcommons(spaced, PLATFORM).id, "problem id") == "run-01-b"
    # A file name that leaves nothing gives way to the instance name, then to the fallback.
    bare = tmp_path / "---.json"
    bare.write_text(json.dumps(split_document()), encoding="utf-8")
    assert read_wfcommons(bare, PLATFORM).id == "Map-Reduce-1"
    bare.write_text(json.dumps(document), encoding="utf-8")
    assert read_wfcommons(bare, PLATFORM).id == "wfcommons"


@pytest.mark.parametrize("problem_id", ["", " x", "x ", "\n", 5, [], ("a",)])
def test_an_invalid_problem_id_is_refused(problem_id):
    with pytest.raises(ConfigError, match="problem_id must be a non-empty string"):
        read_wfcommons(split_document(), PLATFORM, problem_id=problem_id)


def test_tiers_units_and_objectives_are_configurable():
    problem = read_wfcommons(
        split_document(),
        PLATFORM,
        tiers=["device", "cloud"],
        bytes_per_unit=1.0,
        objectives=("makespan", "energy"),
    )
    assert [mode.location for mode in problem.tasks[0].modes] == ["device-0", "cloud-0"]
    assert problem.dependencies[0].data == 2_000_000.0
    assert problem.objectives == ("makespan", "energy")
    with pytest.raises(ConfigError):
        read_wfcommons(split_document(), three_tier_platform(edge=0, fog=0, cloud=0))
    with pytest.raises(ConfigError):
        read_wfcommons(split_document(), PLATFORM, bytes_per_unit=0.0)


@pytest.mark.parametrize(
    ("tiers", "message"),
    [
        ("cloud", "tiers must be a sequence of tier names"),
        (3, "tiers must be a sequence of tier names"),
        (["cloud", 3], "tiers must be a sequence of tier names"),
        (["cloud", "cloud"], "tiers repeat a name"),
        (("fog", "cloud", "fog"), "tiers repeat a name"),
        ([], "no node in tiers"),
        (["nowhere"], "no node in tiers"),
    ],
)
def test_invalid_tiers_are_refused(tiers, message):
    with pytest.raises(ConfigError, match=message):
        read_wfcommons(split_document(), PLATFORM, tiers=tiers)


@pytest.mark.parametrize("source", [None, 3, [], b"instance.json"])
def test_a_source_that_is_neither_a_document_nor_a_path_is_refused(source):
    with pytest.raises(ConfigError, match="source must be a mapping or a path"):
        read_wfcommons(source, PLATFORM)


@pytest.mark.parametrize("platform", [None, "cloud"])
def test_a_platform_of_another_type_is_refused(platform):
    with pytest.raises(ConfigError, match="platform must be a Platform"):
        read_wfcommons(split_document(), platform)


def test_a_string_is_a_path_and_never_json_text():
    with pytest.raises(OSError, match="Errno"):
        read_wfcommons(json.dumps(split_document()), PLATFORM)


def test_transfers_appear_in_decoded_schedules():
    problem = read_wfcommons(split_document(), PLATFORM)
    decoder = Decoder(problem)
    # split on the edge, map_a in the cloud: 2 MB cross the metro and wide-area links.
    modes = [0, 2, 0, 0, 0]
    decoded = decoder.decode(list(range(problem.size)), modes)
    link = PLATFORM.network.link("edge", "cloud")
    assert isinstance(link, Link)
    transfer = link.latency + 2.0 / link.bandwidth
    assert decoded.starts[1] == pytest.approx(decoded.ends[0] + transfer)
    local = decoder.decode(list(range(problem.size)), [0, 0, 0, 0, 0])
    assert local.starts[1] == pytest.approx(local.ends[0])
    assert check_schedule(problem, decoded.schedule(problem)) == ()


def _without_runtime(document: Document) -> None:
    document["workflow"]["execution"]["tasks"].pop()


def _set_runtime(value: object) -> Change:
    def change(document: Document) -> None:
        document["workflow"]["execution"]["tasks"][0]["runtimeInSeconds"] = value

    return change


def _set_task(field: str, value: object, index: int = 0) -> Change:
    def change(document: Document) -> None:
        document["workflow"]["specification"]["tasks"][index][field] = value

    return change


def _set_size(value: object, index: int = 1) -> Change:
    def change(document: Document) -> None:
        document["workflow"]["specification"]["files"][index]["sizeInBytes"] = value

    return change


def _duplicate_file(document: Document) -> None:
    document["workflow"]["specification"]["files"].append({"id": "chunk_a", "sizeInBytes": 1})


def _drop_edge_file(document: Document) -> None:
    files = document["workflow"]["specification"]["files"]
    files[:] = [item for item in files if item["id"] != "part_a"]


def _drop_specification(document: Document) -> None:
    del document["workflow"]["specification"]


def _drop_execution(document: Document) -> None:
    del document["workflow"]["execution"]


def _repeat_execution(runtime: float) -> Change:
    def change(document: Document) -> None:
        entry = {"id": "split", "runtimeInSeconds": runtime}
        document["workflow"]["execution"]["tasks"].append(entry)

    return change


def _unlisted_execution(document: Document) -> None:
    entry = {"id": "ghost", "runtimeInSeconds": 1.0}
    document["workflow"]["execution"]["tasks"].append(entry)


def _rename(old: str, new: str) -> Change:
    """Rename a task everywhere the document mentions it."""

    def change(document: Document) -> None:
        workflow = document["workflow"]
        for entry in workflow["specification"]["tasks"] + workflow["execution"]["tasks"]:
            if entry["id"] == old:
                entry["id"] = new
            for field in ("parents", "children"):
                if field in entry:
                    entry[field] = [new if key == old else key for key in entry[field]]

    return change


def _two_huge_files(document: Document) -> None:
    # map_b passes two files to reduce; each size is a float, their sum is not.
    specification = document["workflow"]["specification"]
    for item in specification["files"]:
        if item["id"] in ("part_b", "log_b"):
            item["sizeInBytes"] = 1e308
    specification["tasks"][3]["inputFiles"] = ["part_a", "part_b", "log_b"]


SURROUNDED = "id must be a non-empty string without surrounding spaces"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_without_runtime, "no execution runtime"),
        (_set_runtime(-1.0), "non-negative"),
        (_set_runtime(True), "must be a number"),
        (_set_runtime("10"), "must be a number"),
        (_set_runtime(10**400), "finite and non-negative"),
        (_repeat_execution(600), "execution task 'split' is listed twice"),
        (_repeat_execution(10), "execution task 'split' is listed twice"),
        (_unlisted_execution, "execution task 'ghost' is not a task of the specification"),
        (_set_task("children", ["ghost"]), "unknown child"),
        (_set_task("parents", ["ghost"], index=2), "unknown parent"),
        (_set_task("id", "map_a"), "task 'map_a' is listed twice"),
        (_set_task("id", ""), "non-empty string"),
        (_set_task("id", "split "), f"specification task 0 {SURROUNDED}"),
        (_rename("split", " split"), f"execution task 0 {SURROUNDED}"),
        (_rename("map_a", "map_a\n"), f"execution task 1 {SURROUNDED}"),
        (_set_task("children", "map_a"), "JSON array"),
        (_set_task("parents", ["map_a"], index=1), "depends on itself"),
        (_set_task("parents", ["reduce"]), "cycle"),
        (_set_size(1.5), "non-negative integer"),
        (_set_size(False), "non-negative integer"),
        (_set_size(-3), "non-negative integer"),
        (_set_size(-3.0), "non-negative integer"),
        (_set_size(float("inf")), "non-negative integer"),
        (_set_size(float("nan")), "non-negative integer"),
        (_set_size(10**400), "too large to express"),
        (_two_huge_files, "from 'map_b' to 'reduce' are too large to express"),
        (_duplicate_file, "two different sizes"),
        (_drop_edge_file, "has no size"),
        (_drop_specification, "workflow.specification"),
        (_drop_execution, "workflow.execution"),
    ],
)
def test_inconsistent_documents_are_refused(change, message):
    document = split_document()
    change(document)
    with pytest.raises(SerializationError, match=message):
        read_wfcommons(document, PLATFORM)


def test_flat_layout_refusals():
    document = flat_document()
    document["workflow"]["tasks"][0]["files"][0]["link"] = "sideways"
    with pytest.raises(SerializationError, match="link"):
        read_wfcommons(document, PLATFORM)
    document = flat_document()
    del document["workflow"]["tasks"][1]["runtimeInSeconds"]
    with pytest.raises(SerializationError, match="runtimeInSeconds"):
        read_wfcommons(document, PLATFORM)
    document = flat_document()
    document["workflow"]["tasks"][1]["name"] = "split"
    with pytest.raises(SerializationError, match="task 'split' is listed twice"):
        read_wfcommons(document, PLATFORM)
    document = flat_document()
    document["workflow"]["tasks"][0]["name"] = "split "
    with pytest.raises(SerializationError, match="task 0 name must be a non-empty string without"):
        read_wfcommons(document, PLATFORM)


def test_values_that_a_float_cannot_hold_are_refused():
    # 2,000,000 bytes in units of the smallest positive float.
    with pytest.raises(SerializationError, match="from 'split' to 'map_a' are too large"):
        read_wfcommons(split_document(), PLATFORM, bytes_per_unit=5e-324)
    slow = three_tier_platform(
        devices=1, edge=1, fog=1, cloud=1, nodes={"edge": NodeSpec(cores=1, speed=0.001)}
    )
    document = split_document()
    _set_runtime(1e308)(document)
    assert read_wfcommons(document, PLATFORM).task("split").min_duration > 1e307
    with pytest.raises(SerializationError, match="task 'split' cannot be represented"):
        read_wfcommons(document, slow)


def test_a_negative_zero_runtime_is_a_zero_duration():
    document = split_document()
    _set_runtime(-0.0)(document)
    for mode in read_wfcommons(document, PLATFORM).task("split").modes:
        assert mode.duration == 0.0
        assert math.copysign(1.0, mode.duration) == 1.0


def test_zero_runtimes_and_unused_file_sizes_are_accepted():
    document = split_document()
    document["workflow"]["specification"]["files"] = [
        item for item in document["workflow"]["specification"]["files"] if item["id"] != "raw"
    ]
    problem = read_wfcommons(document, PLATFORM)
    assert problem.task("report").min_duration == 0.0


def test_the_document_is_not_modified():
    document = split_document()
    before = copy.deepcopy(document)
    read_wfcommons(document, PLATFORM)
    assert document == before


def test_whole_sizes_written_as_reals_are_read(tmp_path):
    text = json.dumps(split_document())
    reals = text.replace('"sizeInBytes": 2000000}', '"sizeInBytes": 2e6}')
    reals = reals.replace('"sizeInBytes": 3000000}', '"sizeInBytes": 3000000.0}')
    assert reals.count("2e6") == 1
    assert reals.count("3000000.0") == 1
    path = tmp_path / "reals.json"
    path.write_text(reals, encoding="utf-8")
    assert read_wfcommons(path, PLATFORM) == read_wfcommons(
        split_document(), PLATFORM, problem_id="reals"
    )
    flat = flat_document()
    for task in flat["workflow"]["tasks"]:
        for item in task["files"]:
            item["sizeInBytes"] = float(item["sizeInBytes"])
    assert read_wfcommons(flat, PLATFORM) == read_wfcommons(flat_document(), PLATFORM)


def test_numbers_of_a_document_may_be_numpy_scalars():
    document = split_document()
    files = document["workflow"]["specification"]["files"]
    runs = document["workflow"]["execution"]["tasks"]
    files[1]["sizeInBytes"] = np.int64(2_000_000)
    files[2]["sizeInBytes"] = np.float64(3_000_000.0)
    runs[0]["runtimeInSeconds"] = np.int64(10)
    runs[1]["runtimeInSeconds"] = np.float32(20.5)
    assert read_wfcommons(document, PLATFORM) == read_wfcommons(split_document(), PLATFORM)
    for size in (np.float64(0.5), np.int64(-1), np.True_):
        files[1]["sizeInBytes"] = size
        with pytest.raises(SerializationError, match="non-negative integer"):
            read_wfcommons(document, PLATFORM)
    # An integer keeps every digit: files of 2**53 + 1 bytes and 1 byte add up to 2**53 + 2.
    document = split_document()
    specification = document["workflow"]["specification"]
    assert [specification["files"][index]["id"] for index in (4, 5)] == ["part_b", "log_b"]
    specification["files"][4]["sizeInBytes"] = np.int64(2**53 + 1)
    specification["files"][5]["sizeInBytes"] = np.int64(1)
    specification["tasks"][3]["inputFiles"] = ["part_a", "part_b", "log_b"]
    problem = read_wfcommons(document, PLATFORM, bytes_per_unit=1.0)
    data = {(d.before, d.after): d.data for d in problem.dependencies}
    assert data["map_b", "reduce"] == float(2**53 + 2)


def test_a_file_with_a_byte_order_mark_is_read(tmp_path):
    path = tmp_path / "marked.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(split_document()).encode("utf-8"))
    assert read_wfcommons(path, PLATFORM) == read_wfcommons(
        split_document(), PLATFORM, problem_id="marked"
    )


def test_files_that_are_not_utf_8_json_are_refused(tmp_path):
    text = json.dumps(split_document())
    path = tmp_path / "instance.json"
    path.write_bytes(text.encode("utf-16"))
    with pytest.raises(SerializationError, match=r"invalid JSON in .* not UTF-8 text"):
        read_wfcommons(path, PLATFORM)
    renamed = text.replace("Map Reduce", "Café")
    assert renamed != text
    path.write_bytes(renamed.encode("latin-1"))
    with pytest.raises(SerializationError, match=r"invalid JSON in .* not UTF-8 text"):
        read_wfcommons(path, PLATFORM)
    # Arrays nested beyond what the parser follows, and an integer beyond what it converts.
    path.write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
    with pytest.raises(SerializationError):
        read_wfcommons(path, PLATFORM)
    digits = text.replace('"runtimeInSeconds": 10}', '"runtimeInSeconds": ' + "9" * 5000 + "}")
    assert digits != text
    path.write_text(digits, encoding="utf-8")
    with pytest.raises(SerializationError):
        read_wfcommons(path, PLATFORM)


def test_a_file_whose_objects_repeat_a_key_is_refused(tmp_path):
    text = json.dumps(split_document())
    path = tmp_path / "instance.json"
    twice = text.replace(
        '"runtimeInSeconds": 10}', '"runtimeInSeconds": 10, "runtimeInSeconds": 600}'
    )
    assert twice != text
    path.write_text(twice, encoding="utf-8")
    with pytest.raises(SerializationError, match="an object repeats the key 'runtimeInSeconds'"):
        read_wfcommons(path, PLATFORM)
    assert text.endswith("}")
    path.write_text(text[:-1] + ', "workflow": {}}', encoding="utf-8")
    with pytest.raises(SerializationError, match="an object repeats the key 'workflow'"):
        read_wfcommons(path, PLATFORM)


def test_a_file_that_a_task_lists_twice_counts_once():
    document = split_document()
    tasks = document["workflow"]["specification"]["tasks"]
    tasks[0]["outputFiles"] = ["chunk_a", "chunk_b", "chunk_a"]
    tasks[1]["inputFiles"] = ["chunk_a", "chunk_a"]
    assert read_wfcommons(document, PLATFORM) == read_wfcommons(split_document(), PLATFORM)


def test_a_file_read_by_several_children_is_carried_to_each():
    document = split_document()
    # map_b reads chunk_a as well, the file that map_a already reads.
    document["workflow"]["specification"]["tasks"][2]["inputFiles"] = ["chunk_a", "chunk_b"]
    problem = read_wfcommons(document, PLATFORM)
    data = {(d.before, d.after): d.data for d in problem.dependencies}
    assert data["split", "map_a"] == 2.0
    assert data["split", "map_b"] == 5.0


def test_pairs_that_share_no_bytes_are_dependencies_without_data():
    document = split_document()
    specification = document["workflow"]["specification"]
    specification["tasks"][1]["inputFiles"] = []  # map_a reads no file of split
    assert specification["files"][4]["id"] == "part_b"
    specification["files"][4]["sizeInBytes"] = 0  # the only file map_b passes on is empty
    problem = read_wfcommons(document, PLATFORM)
    assert [(d.before, d.after, d.data) for d in problem.dependencies] == [
        ("split", "map_a", 0.0),
        ("split", "map_b", 3.0),
        ("map_a", "reduce", 1.0),
        ("map_b", "reduce", 0.0),
        ("reduce", "report", 0.25),
    ]
    # split on the edge, map_a in the cloud: the order holds and nothing is transferred.
    decoded = Decoder(problem).decode(list(range(problem.size)), [0, 2, 0, 0, 0])
    assert decoded.starts[1] == pytest.approx(decoded.ends[0])
    assert check_schedule(problem, decoded.schedule(problem)) == ()


def test_a_pair_that_only_one_of_its_tasks_declares_is_a_dependency():
    # The fixture lists every pair under ``parents`` and split -> map_b nowhere else.
    document = split_document()
    tasks = document["workflow"]["specification"]["tasks"]
    assert (tasks[0]["id"], tasks[0]["children"]) == ("split", ["map_a"])
    assert (tasks[2]["id"], tasks[2]["parents"]) == ("map_b", ["split"])
    # Under ``children`` only: split names map_b, which names no parent.
    tasks[0]["children"] = ["map_a", "map_b"]
    tasks[2]["parents"] = []
    assert edges_of(read_wfcommons(document, PLATFORM)) == EDGES
    # Every pair under ``children`` only.
    for task in tasks:
        task["parents"] = []
    assert edges_of(read_wfcommons(document, PLATFORM)) == EDGES
    # Every pair under ``parents`` only.
    document = split_document()
    for task in document["workflow"]["specification"]["tasks"]:
        del task["children"]
    assert edges_of(read_wfcommons(document, PLATFORM)) == EDGES


def test_the_flat_layout_reads_pairs_declared_by_children():
    document = flat_document()
    tasks = document["workflow"]["tasks"]
    assert all("children" not in task for task in tasks)
    assert (tasks[2]["name"], tasks[2]["parents"]) == ("map_b", ["split"])
    tasks[2]["parents"] = []
    tasks[0]["children"] = ["map_b"]
    assert edges_of(read_wfcommons(document, PLATFORM)) == EDGES
    del tasks[0]["children"]
    assert edges_of(read_wfcommons(document, PLATFORM)) == [EDGES[0], *EDGES[2:]]
    for task in tasks:
        del task["parents"]
    assert edges_of(read_wfcommons(document, PLATFORM)) == []


def test_a_file_between_tasks_that_are_not_parent_and_child_is_not_represented():
    document = split_document()
    specification = document["workflow"]["specification"]
    tasks = specification["tasks"]
    # split writes 900 MB that report reads, and report is not a declared child of split.
    assert (tasks[0]["id"], tasks[4]["id"], tasks[4]["parents"]) == ("split", "report", ["reduce"])
    specification["files"].append({"id": "side", "sizeInBytes": 900_000_000})
    tasks[0]["outputFiles"] = ["chunk_a", "chunk_b", "side"]
    tasks[4]["inputFiles"] = ["merged", "side"]
    problem = read_wfcommons(document, PLATFORM)
    assert edges_of(problem) == EDGES
    assert math.fsum(d.data for d in problem.dependencies) == 6.75
    # The file lies on no dependency, so it needs no size either.
    del specification["files"][-1]
    assert read_wfcommons(document, PLATFORM) == problem


ALIGN = "PIPELINE:ALIGN:STEP__sample.1__ID0000001"
SORT = "main.cwl#sort/run_#0b1f-22"


def recorded_document() -> Document:
    """Two tasks with the identifiers and the fields that recorded executions carry."""
    return {
        "name": "recorded",
        "description": "A hand-written instance.",
        "createdAt": "2024-01-01T00:00:00+00:00",
        "schemaVersion": "1.6",
        "author": {"name": "someone", "email": "someone@example.org"},
        "runtimeSystem": {"name": "engine", "version": "1.0", "url": "https://example.org"},
        "workflow": {
            "specification": {
                "tasks": [
                    {
                        "name": "align",
                        "id": ALIGN,
                        "parents": [],
                        "children": [SORT],
                        "inputFiles": ["/data/in/sample.1.fq"],
                        "outputFiles": ["#task/9f/sample.1.bam", "#task/9f/align.log"],
                    },
                    {
                        "name": "sort",
                        "id": SORT,
                        "parents": [ALIGN],
                        "children": [],
                        "inputFiles": ["#task/9f/sample.1.bam"],
                        "outputFiles": ["/data/out/sample.1.sorted.bam"],
                    },
                ],
                "files": [
                    {"id": "/data/in/sample.1.fq", "sizeInBytes": 9_000_000},
                    {"id": "#task/9f/sample.1.bam", "sizeInBytes": 1_500_000},
                    {"id": "#task/9f/align.log", "sizeInBytes": 0},
                    {"id": "/data/out/sample.1.sorted.bam", "sizeInBytes": 1_400_000},
                ],
            },
            "execution": {
                "makespanInSeconds": 12.5,
                "executedAt": "2024-01-01T00:00:00+00:00",
                "tasks": [
                    {
                        "id": ALIGN,
                        "runtimeInSeconds": 8.0,
                        "executedAt": "2024-01-01T00:00:00+00:00",
                        "command": {"program": "align", "arguments": ["--threads", "8"]},
                        "coreCount": 8,
                        "avgCPU": 640.5,
                        "readBytes": 9_000_000,
                        "writtenBytes": 1_500_000,
                        "memoryInBytes": 4_000_000_000,
                        "priority": 20,
                        "machines": ["node-a"],
                    },
                    {
                        "id": SORT,
                        "runtimeInSeconds": 4.5,
                        "executedAt": "2024-01-01T00:00:08+00:00",
                        "coreCount": 1.0,
                        "machines": ["node-a"],
                    },
                ],
                "machines": [
                    {
                        "nodeName": "node-a",
                        "system": "linux",
                        "architecture": "x86_64",
                        "release": "5.15",
                        "memoryInBytes": 128_000_000_000,
                        "cpu": {"coreCount": 48, "speedInMHz": 2100, "vendor": "vendor"},
                    }
                ],
            },
        },
    }


def test_recorded_identifiers_are_kept():
    problem = read_wfcommons(recorded_document(), PLATFORM)
    assert [task.id for task in problem.tasks] == [ALIGN, SORT]
    assert [(d.before, d.after, d.data) for d in problem.dependencies] == [(ALIGN, SORT, 1.5)]


def test_recorded_cores_machines_memory_and_start_times_are_not_read():
    document = recorded_document()
    problem = read_wfcommons(document, PLATFORM)
    # Eight recorded cores and a recorded machine: still the runtime on one core of each node.
    for task, runtime in zip(problem.tasks, (8.0, 4.5), strict=True):
        assert [mode.location for mode in task.modes] == ["edge-0", "fog-0", "cloud-0"]
        for mode in task.modes:
            assert mode.location is not None
            assert mode.duration == PLATFORM.duration(runtime, mode.location)
            assert dict(mode.demands) == {mode.location: 1.0}
        assert task.release == 0.0
    for key in ("description", "createdAt", "author", "runtimeSystem"):
        del document[key]
    execution = document["workflow"]["execution"]
    execution["tasks"] = [
        {"id": entry["id"], "runtimeInSeconds": entry["runtimeInSeconds"]}
        for entry in execution["tasks"]
    ]
    for key in ("makespanInSeconds", "executedAt", "machines"):
        del execution[key]
    assert read_wfcommons(document, PLATFORM) == problem


def test_workflow_inputs_and_results_are_on_no_dependency():
    document = recorded_document()
    files = document["workflow"]["specification"]["files"]
    assert [item["id"] for item in files][::3] == [
        "/data/in/sample.1.fq",
        "/data/out/sample.1.sorted.bam",
    ]
    # Neither the file that no task writes nor the file that no task reads needs a size.
    del files[3], files[0]
    problem = read_wfcommons(document, PLATFORM)
    assert math.fsum(d.data for d in problem.dependencies) == 1.5


def test_files_that_share_an_instance_name_get_their_own_ids(tmp_path):
    first = tmp_path / "run-001.json"
    second = tmp_path / "run-002.json"
    document = recorded_document()
    first.write_text(json.dumps(document), encoding="utf-8")
    document["workflow"]["execution"]["tasks"][0]["runtimeInSeconds"] = 16.0
    second.write_text(json.dumps(document), encoding="utf-8")
    problems = [read_wfcommons(path, PLATFORM) for path in (first, second)]
    assert [problem.id for problem in problems] == ["run-001", "run-002"]
    assert [problem.metadata["name"] for problem in problems] == ["recorded", "recorded"]
