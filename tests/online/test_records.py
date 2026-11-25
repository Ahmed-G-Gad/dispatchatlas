"""Record files: round trip, atomic writing, and refusal of torn lines."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import pytest

from dispatchatlas.core.errors import SerializationError
from dispatchatlas.online import (
    RECORDS_FORMAT,
    OnlineEft,
    OnlineResult,
    PlanningClock,
    RecordWriter,
    read_records,
    run_online,
    write_records,
)
from tests.online.cases import TwoNodes, fork_trace


def _runs() -> list[OnlineResult]:
    return [
        run_online(TwoNodes(), fork_trace(), planner, clock=PlanningClock(rate=rate))
        for planner in (None, OnlineEft())
        for rate in (1.0, math.inf)
    ]


def test_records_round_trip(tmp_path: Path):
    runs = _runs()
    path = write_records(tmp_path / "job.jsonl.gz", runs, {"job": "fork", "trace": "x"})
    header, documents = read_records(path)
    assert header == {"format": RECORDS_FORMAT, "version": 1, "job": "fork", "trace": "x"}
    assert [OnlineResult.from_dict(d) for d in documents] == runs
    assert documents[0]["config"]["clock"] == {"mode": "counted", "rate": 1.0}
    assert documents[1]["config"]["clock"]["rate"] == "inf"


def test_equal_runs_give_equal_bytes(tmp_path: Path):
    runs = [run_online(TwoNodes(), fork_trace(), None, clock=PlanningClock(rate=1.0))]
    first = write_records(tmp_path / "a.jsonl.gz", runs).read_bytes()
    second = write_records(tmp_path / "b.jsonl.gz", runs).read_bytes()
    assert first == second


def test_the_target_appears_only_when_the_writer_closes(tmp_path: Path):
    target = tmp_path / "job.jsonl.gz"
    writer = RecordWriter(target, {"job": "fork"})
    writer.write(_runs()[0])
    assert not target.exists()
    assert writer.temporary.exists()
    writer.close()
    assert target.exists()
    assert not writer.temporary.exists()
    assert len(read_records(target)[1]) == 1


def test_a_job_that_fails_leaves_its_finished_runs_beside_the_target(tmp_path: Path):
    target = tmp_path / "job.jsonl.gz"
    run = _runs()[0]
    temporary: list[Path] = []

    def die() -> None:
        with RecordWriter(target) as writer:
            writer.write(run)
            temporary.append(writer.temporary)
            raise RuntimeError("the job died")

    with pytest.raises(RuntimeError):
        die()
    assert not target.exists()
    assert temporary[0].exists()
    header, documents = read_records(temporary[0])
    assert header["format"] == RECORDS_FORMAT
    assert len(documents) == 1


def test_a_torn_last_line_is_refused(tmp_path: Path):
    path = write_records(tmp_path / "job.jsonl.gz", _runs())
    text = gzip.decompress(path.read_bytes())
    torn = tmp_path / "torn.jsonl.gz"
    torn.write_bytes(gzip.compress(text[: len(text) - 40]))
    with pytest.raises(SerializationError, match="torn"):
        read_records(torn)
    cut = tmp_path / "cut.jsonl.gz"
    cut.write_bytes(path.read_bytes()[:-30])
    with pytest.raises(SerializationError):
        read_records(cut)


def test_a_line_that_is_not_a_json_object_is_refused(tmp_path: Path):
    header = json.dumps({"format": RECORDS_FORMAT, "version": 1}).encode()
    for body in (b"[1, 2]\n", b'{"a": 1\n'):
        path = tmp_path / "bad.jsonl.gz"
        path.write_bytes(gzip.compress(header + b"\n" + body))
        with pytest.raises(SerializationError):
            read_records(path)
    other = tmp_path / "other.jsonl.gz"
    other.write_bytes(gzip.compress(b'{"format": "something"}\n'))
    with pytest.raises(SerializationError, match="not a version 1"):
        read_records(other)
    empty = tmp_path / "empty.jsonl.gz"
    empty.write_bytes(gzip.compress(b""))
    with pytest.raises(SerializationError, match="empty"):
        read_records(empty)


def test_a_run_document_of_another_format_is_refused():
    document = _runs()[0].to_dict()
    document["format"] = "x"
    with pytest.raises(SerializationError):
        OnlineResult.from_dict(document)
