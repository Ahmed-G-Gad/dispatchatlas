"""Tests of the WfCommons WfFormat reader."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from dispatchatlas.benchmarks.continuum import three_tier_platform
from dispatchatlas.benchmarks.wfcommons import SCHEMA_VERSIONS, read_wfcommons
from dispatchatlas.core import ConfigError, Link, SerializationError, check_schedule
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
    assert read_wfcommons(path, PLATFORM) == read_wfcommons(split_document(), PLATFORM)
    nameless = split_document()
    del nameless["name"]
    path.write_text(json.dumps(nameless), encoding="utf-8")
    assert read_wfcommons(str(path), PLATFORM).id == "montage-small"
    assert read_wfcommons(path, PLATFORM, problem_id="chosen").id == "chosen"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SerializationError, match="invalid JSON"):
        read_wfcommons(path, PLATFORM)


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


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_without_runtime, "no execution runtime"),
        (_set_runtime(-1.0), "non-negative"),
        (_set_runtime(True), "must be a number"),
        (_set_runtime("10"), "must be a number"),
        (_set_task("children", ["ghost"]), "unknown child"),
        (_set_task("parents", ["ghost"], index=2), "unknown parent"),
        (_set_task("id", "map_a"), "unique"),
        (_set_task("id", ""), "non-empty string"),
        (_set_task("children", "map_a"), "JSON array"),
        (_set_task("parents", ["map_a"], index=1), "depends on itself"),
        (_set_task("parents", ["reduce"]), "cycle"),
        (_set_size(1.5), "non-negative integer"),
        (_set_size(False), "non-negative integer"),
        (_set_size(-3), "non-negative integer"),
        (_duplicate_file, "two different sizes"),
        (_drop_edge_file, "has no size"),
        (_drop_specification, "workflow.specification"),
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
