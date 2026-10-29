from __future__ import annotations

import os
from pathlib import Path

import pytest

from dispatchatlas.processes import rows as rows_module
from dispatchatlas.processes.rows import (
    RowWriter,
    TornRowsError,
    canonical_key,
    count_lines,
    done_keys,
    read_rows,
    repair_rows,
)


def test_a_row_is_one_utf8_line_with_one_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[int] = []
    real = os.fsync

    def counted(fd: int) -> None:
        synced.append(fd)
        real(fd)

    monkeypatch.setattr(os, "fsync", counted)
    path = tmp_path / "rows.jsonl"
    with RowWriter(path) as writer:
        assert writer.write({"a": 1}) == 1
        assert writer.write({"name": "Zoë", "values": [1.5, None]}) == 2
    data = path.read_bytes()
    assert data == b'{"a":1}\n' + '{"name":"Zoë","values":[1.5,null]}\n'.encode()
    assert not data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data
    assert len(synced) == 2


@pytest.mark.parametrize("row", [{"x": float("nan")}, {"x": object()}, [1, 2]])
def test_a_row_that_is_not_a_json_object_is_refused(tmp_path: Path, row: object) -> None:
    with (
        RowWriter(tmp_path / "rows.jsonl", fsync=False) as writer,
        pytest.raises((TypeError, ValueError)),
    ):
        writer.write(row)  # type: ignore[arg-type]
    assert (tmp_path / "rows.jsonl").read_bytes() == b""


def test_rows_are_read_back_in_order(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    with RowWriter(path, fsync=False) as writer:
        for index in range(3):
            writer.write({"i": index})
    assert [row["i"] for row in read_rows(path)] == [0, 1, 2]


@pytest.mark.parametrize(
    ("content", "line", "offset", "reason"),
    [
        (b'{"a":1}\n{"a":2', 2, 8, "no line feed"),
        (b'{"a":1}\n\n{"a":2}\n', 2, 8, "blank"),
        (b'{"a":1}\n[1]\n', 2, 8, "not a JSON object"),
        (b'{"a":1}\n{"a":\n{"a":3}\n', 2, 8, "not valid JSON"),
    ],
)
def test_a_torn_or_foreign_line_is_refused_with_its_place(
    tmp_path: Path, content: bytes, line: int, offset: int, reason: str
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_bytes(content)
    with pytest.raises(TornRowsError, match=reason) as caught:
        list(read_rows(path))
    assert (caught.value.line_number, caught.value.byte_offset) == (line, offset)
    assert "rows-repair" in str(caught.value)


def test_done_keys_are_canonical(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    assert done_keys(path, lambda row: row["k"]) == set()
    with RowWriter(path, fsync=False) as writer:
        writer.write({"k": ["a", 1, ["b"]]})
        writer.write({"k": "plain"})
    found = done_keys(path, lambda row: row["k"])
    assert found == {canonical_key(("a", 1, ("b",))), canonical_key("plain")}
    assert canonical_key(("a", 1)) == canonical_key(["a", 1]) == '["a",1]'
    path.write_bytes(path.read_bytes() + b'{"k":')
    with pytest.raises(TornRowsError):
        done_keys(path, lambda row: row["k"])


def test_a_torn_tail_is_moved_aside(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    kept = b'{"a":1}\n{"a":2}\n'
    path.write_bytes(kept + b'{"a":3,"b"')
    report = repair_rows(path)
    assert report.aside is not None
    assert report.aside.read_bytes() == b'{"a":3,"b"'
    assert (report.removed_bytes, report.kept_lines) == (10, 2)
    assert path.read_bytes() == kept
    assert len(list(read_rows(path))) == 2
    again = repair_rows(path)
    assert again.aside is None
    assert again.removed_bytes == 0
    assert path.read_bytes() == kept


def test_lines_are_counted_incrementally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rows_module, "_CHUNK", 7)
    path = tmp_path / "rows.jsonl"
    assert count_lines(path) == (0, 0)
    path.write_bytes(b"a\nbb\nccc\n")
    first, size = count_lines(path)
    with path.open("ab") as stream:
        stream.write(b"dddd\neeeee\nf")
    more, new_size = count_lines(path, offset=size)
    assert (first, first + more) == (3, 5)
    assert count_lines(path) == (5, new_size)


def test_a_row_is_in_the_file_before_its_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rows.jsonl"
    seen: list[bytes] = []
    real = os.fsync

    def look(fd: int) -> None:
        seen.append(path.read_bytes())  # through another handle: flushed bytes only
        real(fd)

    monkeypatch.setattr(os, "fsync", look)
    with RowWriter(path) as writer:
        writer.write({"a": 1})
        writer.write({"b": 2})
    assert seen == [b'{"a":1}\n', b'{"a":1}\n{"b":2}\n']


def test_a_row_that_is_not_json_names_its_field(tmp_path: Path) -> None:
    with RowWriter(tmp_path / "rows.jsonl", fsync=False) as writer:
        with pytest.raises(ValueError, match=r"field 'x\.y' is nan"):
            writer.write({"x": {"y": float("nan")}})
        with pytest.raises(ValueError, match=r"field 'z\[1\]' is a set"):
            writer.write({"z": [1, {2}]})
    assert rows_module.unwritable_field({"a": [1, 2.5, None, {"b": "c"}]}) is None
    assert (
        rows_module.unwritable_field({(1, 2): 3}) == "the row has a key of type tuple, not a string"
    )


def test_repeated_keys_are_listed_in_order_of_first_sight(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    assert rows_module.duplicate_keys(path, lambda row: row["k"]) == []
    with RowWriter(path, fsync=False) as writer:
        for key in ("b", "a", "b", "c", "a", "b"):
            writer.write({"k": key})
    assert rows_module.duplicate_keys(path, lambda row: row["k"]) == ['"b"', '"a"']
