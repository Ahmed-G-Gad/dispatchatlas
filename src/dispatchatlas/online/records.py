"""Record files of online runs: one compressed JSON-lines file per job.

A record file is gzip-compressed UTF-8 text. Its first line is a header, a
JSON object with ``format`` ``dispatchatlas.online-records``, ``version``, and
whatever the job adds (its configuration, the digests of its traces); every
further line is one run, an :meth:`~dispatchatlas.online.runner.OnlineResult.to_dict`
document. Every line ends with a line feed.

:class:`RecordWriter` writes the file to a temporary name beside the target
and renames it into place when the job closes it, so a reader never sees a
partial file at the target. Each run is flushed as it is written; a job that
fails with an exception closes the temporary file, which keeps the runs it
had finished, and leaves it in place.
:func:`read_records` refuses a file whose stream or last line is cut, or
whose line is not a JSON object, a torn line, rather than return part of it.
The gzip header carries no time or name, so equal runs give equal bytes,
except for each pass's ``wall``, the measured seconds of its planner call,
which even a counted run records.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import tempfile
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Self

from dispatchatlas.core.errors import SerializationError
from dispatchatlas.online.runner import OnlineResult

__all__ = ["RECORDS_FORMAT", "RecordWriter", "read_records", "write_records"]

RECORDS_FORMAT = "dispatchatlas.online-records"
"""Value of the ``format`` field of a record file's header."""

_VERSION = 1


def _line(document: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SerializationError(f"a record cannot be written as JSON: {error}") from error
    return text.encode("utf-8") + b"\n"


class RecordWriter:
    """Writes one record file atomically; use it as a context manager.

    Args:
        path: The record file to create. It replaces an existing file only
            when the writer closes without error.
        header: Fields of the header line, besides ``format`` and ``version``.

    Raises:
        SerializationError: If the header cannot be written as JSON. The
            temporary file is closed and removed first.

    Example::

        with RecordWriter(path, {"job": "loop-pilot"}) as writer:
            for run in runs:
                writer.write(run)
    """

    def __init__(self, path: str | Path, header: Mapping[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        self.temporary = Path(temporary)
        self._raw: IO[bytes] = os.fdopen(handle, "wb")
        self._stream = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
        self.count = 0
        fields = dict(header or {})
        fields.update({"format": RECORDS_FORMAT, "version": _VERSION})
        try:
            self._write(_line(fields))
        except BaseException:
            # A writer that never exists leaves no file behind.
            self._stream.close()
            self._raw.close()
            self.temporary.unlink(missing_ok=True)
            raise

    def _write(self, data: bytes) -> None:
        self._stream.write(data)
        self._stream.flush(zlib.Z_SYNC_FLUSH)
        self._raw.flush()

    def write(self, run: OnlineResult | Mapping[str, Any]) -> None:
        """Append one run."""
        document = run.to_dict() if isinstance(run, OnlineResult) else run
        self._write(_line(document))
        self.count += 1

    def close(self) -> Path:
        """Finish the file and move it to :attr:`path`. Returns the path."""
        self._stream.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        self.temporary.replace(self.path)
        return self.path

    def abandon(self) -> Path:
        """Close the file without moving it: :attr:`temporary` keeps the runs written so far.

        The closed temporary file is a complete record file that
        :func:`read_records` reads. A job killed before it closes leaves a cut
        stream instead, which the reader refuses.
        """
        self._stream.close()
        self._raw.close()
        return self.temporary

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if kind is None:
            self.close()
        else:
            self.abandon()


def write_records(
    path: str | Path,
    runs: Iterable[OnlineResult | Mapping[str, Any]],
    header: Mapping[str, Any] | None = None,
) -> Path:
    """Write a record file of ``runs`` atomically. Returns the path."""
    with RecordWriter(path, header) as writer:
        for run in runs:
            writer.write(run)
    return Path(path)


def read_records(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a record file: its header and its run documents.

    Raises:
        SerializationError: If the stream is cut or corrupt, a line is torn
            (cut, or not a JSON object), or the header is not a record header.
    """
    path = Path(path)
    try:
        with gzip.open(path, "rb") as stream:
            data = stream.read()
    except (EOFError, gzip.BadGzipFile, zlib.error) as error:
        raise SerializationError(
            f"{path}: the compressed stream is cut or corrupt: {error}"
        ) from error
    if not data:
        raise SerializationError(f"{path}: empty record file")
    if not data.endswith(b"\n"):
        raise SerializationError(f"{path}: the last line is torn")
    documents: list[dict[str, Any]] = []
    for number, line in enumerate(io.BytesIO(data), start=1):
        try:
            document = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SerializationError(f"{path}: line {number} is torn: {error}") from error
        if not isinstance(document, dict):
            raise SerializationError(f"{path}: line {number} is not a JSON object")
        documents.append(document)
    header = documents[0]
    if header.get("format") != RECORDS_FORMAT or header.get("version") != _VERSION:
        raise SerializationError(f"{path}: not a version {_VERSION} {RECORDS_FORMAT} file")
    return header, documents[1:]
