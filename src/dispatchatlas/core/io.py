"""File helpers: atomic writes and JSON documents."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dispatchatlas.core.errors import SerializationError

__all__ = ["atomic_write_bytes", "atomic_write_json", "atomic_write_text", "read_json"]


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` so readers never see a partial file.

    The data goes to a temporary file in the same directory, is flushed to disk,
    and then replaces ``path`` in one step.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path: Path, text: str) -> Path:
    """Write UTF-8 text with LF line endings atomically. Returns the path."""
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any, *, indent: int | None = 2) -> Path:
    """Write a JSON document atomically with sorted keys. Returns the path.

    Raises:
        SerializationError: If ``value`` cannot be represented in JSON.
    """
    try:
        text = json.dumps(value, indent=indent, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SerializationError(f"cannot write {path} as JSON: {error}") from error
    return atomic_write_text(path, text + "\n")


def read_json(path: Path) -> Any:
    """Read a JSON document.

    Raises:
        SerializationError: If the file is not UTF-8 text or not valid JSON.
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise SerializationError(f"{path} is not UTF-8 text: {error}") from error
    except json.JSONDecodeError as error:
        raise SerializationError(f"invalid JSON in {path}: {error}") from error
