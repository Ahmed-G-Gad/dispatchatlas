"""A rows file: one JSON object per line, appended by one writer, read back strictly.

:class:`RowWriter` appends each row as one line, UTF-8 without a byte-order
mark, ending in a line feed, in one write followed by a flush and an
``fsync``, so a crash loses at most the row being written and leaves at most
a torn last line. :func:`read_rows` refuses such a line, with its number and
byte offset, rather than skip it; :func:`repair_rows` moves the torn bytes
aside so the file can be resumed. :func:`done_keys` reads back the keys the
rows hold, through a caller's key function, for a resume.

Keys are compared through :func:`canonical_key`, so a tuple key and the list
that JSON gives back for it are one key.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Self

__all__ = [
    "Key",
    "RepairReport",
    "RowWriter",
    "TornRowsError",
    "canonical_key",
    "count_lines",
    "done_keys",
    "read_rows",
    "repair_rows",
]

type Key = str | int | tuple[Key, ...] | list[Key]
"""What identifies a call: a string, an integer, or a sequence of keys; JSON-able."""

_CHUNK = 1 << 20


def canonical_key(key: Any) -> str:
    """Return the canonical text of a key: compact JSON with sorted object keys.

    Raises:
        ValueError: If the key cannot be written as JSON.
    """
    try:
        return json.dumps(key, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"key {key!r} cannot be written as JSON: {error}") from error


class TornRowsError(ValueError):
    """A rows file holds a line that is not one complete JSON object.

    Attributes:
        path: The rows file.
        line_number: The line, counted from 1.
        byte_offset: Offset of the line's first byte in the file.
        reason: What is wrong with the line.
    """

    def __init__(self, path: Path, line_number: int, byte_offset: int, reason: str) -> None:
        self.path = path
        self.line_number = line_number
        self.byte_offset = byte_offset
        self.reason = reason
        super().__init__(
            f"{path}: line {line_number} at byte {byte_offset} {reason}. A torn last line is "
            "what a writer killed mid-row leaves; one that is not last means another writer "
            f"touched the file. To move a torn tail aside: dispatchatlas processes rows-repair "
            f"{path}"
        )


class RowWriter:
    """Appends rows to a file, one JSON object per line, with an ``fsync`` per row.

    Args:
        path: The rows file; created when absent, appended to otherwise.
        fsync: Call ``os.fsync`` after each row (default). ``False`` only
            flushes, for tests and scratch files.

    A lock makes :meth:`write` safe from several threads, but a file must
    have one writing object: two writers in two processes could interleave.
    """

    def __init__(self, path: str | Path, *, fsync: bool = True) -> None:
        self.path = Path(path)
        self._fsync = fsync
        self._lock = threading.Lock()
        self._count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("ab")

    @property
    def count(self) -> int:
        """Rows written by this writer."""
        return self._count

    def write(self, row: Mapping[str, Any]) -> int:
        """Append one row and return the number of rows written so far.

        Raises:
            TypeError: If the row is not a mapping.
            ValueError: If the row cannot be written as JSON (a value that
                is not JSON-able, a NaN or an infinity).
        """
        if not isinstance(row, Mapping):
            raise TypeError(f"a row is a mapping, not {type(row).__name__}")
        try:
            text = json.dumps(dict(row), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"the row cannot be written as JSON: {error}") from error
        data = text.encode("utf-8") + b"\n"
        with self._lock:
            self._stream.write(data)
            self._stream.flush()
            if self._fsync:
                os.fsync(self._stream.fileno())
            self._count += 1
            return self._count

    def close(self) -> None:
        """Close the file."""
        with self._lock:
            self._stream.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()


def read_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield the rows of a rows file, in order.

    Raises:
        TornRowsError: At the first line that is blank, is not a JSON
            object, or is the last line and lacks its line feed. Rows
            before it have been yielded.
        OSError: If the file cannot be read.
    """
    target = Path(path)
    offset = 0
    for number, line in enumerate(_lines(target), start=1):
        if not line.endswith(b"\n"):
            raise TornRowsError(target, number, offset, "has no line feed at the end of the file")
        body = line[:-1]
        if not body.strip():
            raise TornRowsError(target, number, offset, "is blank")
        try:
            row = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise TornRowsError(target, number, offset, "is not valid JSON") from None
        if not isinstance(row, dict):
            raise TornRowsError(target, number, offset, "is not a JSON object")
        yield row
        offset += len(line)


def done_keys(path: str | Path, key_of: Callable[[Mapping[str, Any]], Any]) -> set[str]:
    """Return the canonical keys of the rows of a file; empty when it does not exist.

    Raises:
        TornRowsError: If a line is torn (see :func:`read_rows`); no key is
            returned then.
    """
    target = Path(path)
    if not target.exists():
        return set()
    return {canonical_key(key_of(row)) for row in read_rows(target)}


@dataclass(frozen=True)
class RepairReport:
    """What :func:`repair_rows` did.

    Attributes:
        path: The rows file.
        aside: The file holding the bytes removed, ``None`` when nothing was.
        removed_bytes: Bytes removed from the end of the rows file.
        kept_lines: Complete lines the rows file holds now.
    """

    path: Path
    aside: Path | None
    removed_bytes: int
    kept_lines: int


def repair_rows(path: str | Path) -> RepairReport:
    """Move the bytes after the last line feed of a rows file aside.

    The bytes go to ``<path>.torn-<UTC time>.bin`` and the file is truncated
    after its last line feed. Kept lines are neither parsed nor rewritten,
    so a malformed line that is not last stays (and :func:`read_rows` still
    refuses it): that is another writer's damage, not a torn tail.

    Raises:
        OSError: If the file cannot be read or written.
    """
    target = Path(path)
    size = target.stat().st_size
    lines, _ = count_lines(target)
    with target.open("rb") as stream:
        keep = _last_line_feed(stream, size) + 1
    if keep == size:
        return RepairReport(path=target, aside=None, removed_bytes=0, kept_lines=lines)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    aside = target.with_name(f"{target.name}.torn-{stamp}.bin")
    with target.open("rb") as source, aside.open("xb") as copy:
        source.seek(keep)
        shutil.copyfileobj(source, copy)
        copy.flush()
        os.fsync(copy.fileno())
    with target.open("r+b") as stream:
        stream.truncate(keep)
        stream.flush()
        os.fsync(stream.fileno())
    return RepairReport(path=target, aside=aside, removed_bytes=size - keep, kept_lines=lines)


def count_lines(path: str | Path, *, offset: int = 0) -> tuple[int, int]:
    """Return the line feeds of a file from byte ``offset``, and the file's size.

    A watcher counts a growing file incrementally by passing the size it
    got last time; a size below ``offset`` means the file was truncated. A
    missing file counts as empty.
    """
    target = Path(path)
    try:
        stream = target.open("rb")
    except FileNotFoundError:
        return 0, 0
    with stream:
        size = os.fstat(stream.fileno()).st_size
        if offset > size:  # the file shrank: the caller counts again from 0
            return 0, size
        stream.seek(offset)
        count = 0
        position = offset
        while chunk := stream.read(_CHUNK):
            count += chunk.count(b"\n")
            position += len(chunk)
    return count, position


def _lines(path: Path) -> Iterator[bytes]:
    """Yield the lines of a file with their line feeds, the last one possibly without."""
    with path.open("rb") as stream:
        yield from stream


def _last_line_feed(stream: BinaryIO, size: int) -> int:
    """Return the offset of the last line feed of a binary stream of ``size`` bytes, or -1."""
    position = size
    while position > 0:
        start = max(0, position - _CHUNK)
        stream.seek(start)
        chunk = stream.read(position - start)
        found = chunk.rfind(b"\n")
        if found >= 0:
            return start + found
        position = start
    return -1
