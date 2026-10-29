"""The child side of a call: one function called once in a process of its own.

``python -m dispatchatlas.processes.child`` reads one JSON document from its
standard input::

    {"schema": "dispatchatlas.call/1", "key": ..., "target": "package.module:function",
     "arguments": {...}, "cpu": 7, "group": "<name>", "attempt": 1,
     "clean_tokens": [...]}

and then, in this order: joins its process group by name
(:func:`~dispatchatlas.processes.groups.join_group`), pins itself to ``cpu``
and reads its mask back (:func:`~dispatchatlas.processes.affinity.pin_self`),
on POSIX systems starts a thread that ends it when its parent dies, points
``sys.stdout`` at the standard error stream so the function's prints cannot
corrupt the reply, imports the target, and calls it with the arguments. The
**last line** of its standard output is the reply, one JSON object:
``{"row": {...}, "facts": {...}}`` with exit code 0, or ``{"error": {"type":
..., "message": ...}, "facts": {...}}`` with exit code 1. A traceback goes
to the standard error stream only. Each failing step has its own error type:
``BadDocument``, ``GroupRefused``, ``PinningRefused``, ``TargetNotFound``,
``RowNotMapping``, or the class of what the function raised.

``facts`` holds ``pid``, ``cpu``, ``affinity_inherited`` (the mask the
process started with), ``affinity_mask`` (the mask read back after pinning),
``group`` (how the group was joined), ``threads`` (the thread-limit variables
of numerical libraries as the process saw them), ``peak_rss_bytes``,
``started_utc``, ``ended_utc``, and ``wall_seconds``.

The module imports the standard library and the leaf modules of this
package only: a process started for every call should not pay for loading
the library's solvers. For the same reason the error-naming rule of
:func:`dispatchatlas.campaigns.execution.exception_error` is repeated here
rather than imported.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from dispatchatlas.processes.affinity import current_mask, pin_self
from dispatchatlas.processes.groups import end_with_parent_process, join_group
from dispatchatlas.processes.memory import peak_memory_bytes

__all__ = [
    "SCHEMA",
    "THREAD_VARIABLES",
    "child_main",
    "clean_message",
    "describe_error",
    "locate_callable",
    "run_child",
]

SCHEMA = "dispatchatlas.call/1"
"""The schema of the document a child reads."""

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
"""Environment variables that limit the thread pools of numerical libraries."""

_MESSAGE_LIMIT = 2000
_TRACEBACK_LINE = re.compile(r'File "[^\n]*')
_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:[\\/][^\s\"'<>|]*")
_UNC_PATH = re.compile(r"\\\\[^\s\"'<>|\\]+\\[^\s\"'<>|]*")
_POSIX_PATH = re.compile(r"(?<![\w.:/\\~-])/(?:[^\s/\"'<>|]+/)+[^\s/\"'<>|]*")
_SPACE = re.compile(r"\s+")


class _RefusedError(Exception):
    """A step before the call failed; ``kind`` names it."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def clean_message(text: str, *, tokens: Iterable[str] = (), limit: int = _MESSAGE_LIMIT) -> str:
    """Return an error message with file paths and named tokens removed.

    Removed: everything from ``File "`` to the end of its line (a traceback
    frame), Windows drive paths, UNC paths, and absolute POSIX paths of two
    or more segments, each replaced by ``<path>``; then every token,
    ignoring case, replaced by ``<token>``. Whitespace is collapsed and the
    result cut to ``limit`` characters.
    """
    cleaned = _TRACEBACK_LINE.sub("<frame>", text)
    cleaned = _UNC_PATH.sub("<path>", cleaned)
    cleaned = _WINDOWS_PATH.sub("<path>", cleaned)
    cleaned = _POSIX_PATH.sub("<path>", cleaned)
    for token in sorted({item for item in tokens if item}, key=len, reverse=True):
        cleaned = re.sub(re.escape(token), "<token>", cleaned, flags=re.IGNORECASE)
    cleaned = _SPACE.sub(" ", cleaned).strip()
    if len(cleaned) > limit:
        cleaned = cleaned[: max(0, limit - 3)] + "..."
    return cleaned


def describe_error(error: BaseException, *, tokens: Iterable[str] = ()) -> dict[str, str]:
    """Return ``{"type", "message"}`` for an exception, the message cleaned.

    Classes of the ``builtins`` module keep their plain name; others are
    qualified by their module. An empty message becomes the type's name.
    """
    cls = type(error)
    module = cls.__module__
    name = cls.__qualname__ if module == "builtins" else f"{module}.{cls.__qualname__}"
    return {"type": name, "message": clean_message(str(error) or name, tokens=tokens)}


def locate_callable(target: str) -> Callable[..., Any]:
    """Import ``"package.module:function"`` (or ``"package.module:Class.method"``).

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If an attribute on the path is missing.
        TypeError: If the target is malformed or not callable.
    """
    module_name, colon, path = target.partition(":")
    if not colon or not module_name or not path:
        raise TypeError(f"a target is 'package.module:function', not {target!r}")
    found: Any = importlib.import_module(module_name)
    for part in path.split("."):
        found = getattr(found, part)
    if not callable(found):
        raise TypeError(f"{target} is not callable")
    return found  # type: ignore[no-any-return]


def run_child(
    document: Mapping[str, Any],
    *,
    resolve: Callable[[str], Callable[..., Any]] | None = None,
) -> tuple[dict[str, Any], int]:
    """Run one call described by ``document`` in the calling process.

    Args:
        document: The call document (see the module docstring).
        resolve: Turns a target into a function; :func:`locate_callable`
            by default. A caller whose functions live in its own entry
            script passes its own.

    Returns:
        The reply object and the exit code: 0 for a row, 1 for an error.
    """
    started = time.perf_counter()
    started_utc = _utc_now()
    tokens = tuple(str(token) for token in document.get("clean_tokens") or ())
    cpu = document.get("cpu")
    facts: dict[str, Any] = {
        "pid": os.getpid(),
        "cpu": cpu,
        "affinity_inherited": current_mask(),
        "affinity_mask": None,
        "group": None,
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "peak_rss_bytes": None,
        "started_utc": started_utc,
        "ended_utc": None,
        "wall_seconds": None,
    }
    reply: dict[str, Any]
    try:
        function, arguments = _prepare(document, facts, resolve)
        row = function(**arguments)
        if not isinstance(row, Mapping):
            raise _RefusedError(
                "RowNotMapping", f"the function returned {type(row).__name__}, not a mapping"
            )
        reply, code = {"row": dict(row)}, 0
    except _RefusedError as refusal:
        message = clean_message(str(refusal), tokens=tokens)
        reply, code = {"error": {"type": refusal.kind, "message": message}}, 1
    except Exception as error:  # every failure of the call becomes an error reply
        traceback.print_exc(file=sys.stderr)
        reply, code = {"error": describe_error(error, tokens=tokens)}, 1
    facts["peak_rss_bytes"] = peak_memory_bytes()
    facts["ended_utc"] = _utc_now()
    facts["wall_seconds"] = time.perf_counter() - started
    reply["facts"] = facts
    return reply, code


def _prepare(
    document: Mapping[str, Any],
    facts: dict[str, Any],
    resolve: Callable[[str], Callable[..., Any]] | None,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Join the group, pin, and locate the function; raise :class:`_RefusedError` on failure."""
    if document.get("schema") != SCHEMA:
        raise _RefusedError("BadDocument", f"the document's schema is not {SCHEMA}")
    group = document.get("group")
    if group is not None:
        try:
            facts["group"] = join_group(str(group))
        except RuntimeError as error:
            raise _RefusedError("GroupRefused", str(error)) from None
        end_with_parent_process()
    cpu = document.get("cpu")
    if cpu is not None:
        try:
            report = pin_self(int(cpu))
        except (RuntimeError, ValueError) as error:
            raise _RefusedError("PinningRefused", str(error)) from None
        facts["affinity_inherited"] = report.inherited
        facts["affinity_mask"] = report.read
    else:
        facts["affinity_mask"] = current_mask()
    target = document.get("target")
    if not isinstance(target, str):
        raise _RefusedError("BadDocument", "the document names no target")
    arguments = document.get("arguments") or {}
    if not isinstance(arguments, Mapping):
        raise _RefusedError("BadDocument", "the document's arguments are not an object")
    try:
        function = (resolve or locate_callable)(target)
    except (ImportError, AttributeError, TypeError) as error:
        traceback.print_exc(file=sys.stderr)
        raise _RefusedError("TargetNotFound", f"{target}: {error}") from None
    return function, dict(arguments)


def child_main(
    argv: Sequence[str] | None = None,
    *,
    resolve: Callable[[str], Callable[..., Any]] | None = None,
) -> int:
    """Read the document from standard input, run the call, write the reply line.

    Args:
        argv: Ignored apart from ``--help``; the document comes on stdin.
        resolve: See :func:`run_child`.

    Returns:
        The exit code: 0 for a row, 1 for an error reply.
    """
    arguments = sys.argv[1:] if argv is None else list(argv)
    if "--help" in arguments or "-h" in arguments:
        sys.stdout.write(__doc__ or "")
        return 0
    reply_stream = sys.stdout
    sys.stdout = sys.stderr  # a print from the function must not reach the reply
    try:
        document: Any = None
        problem: str | None = None
        try:
            document = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            problem = clean_message(str(error))
        if problem is None and not isinstance(document, Mapping):
            problem = "the document is not an object"
        if problem is None:
            reply, code = run_child(document, resolve=resolve)
        else:
            reply = {
                "error": {"type": "BadDocument", "message": problem},
                "facts": {"pid": os.getpid()},
            }
            code = 1
        try:
            line = json.dumps(reply, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            line = json.dumps(
                {
                    "error": {
                        "type": "RowNotSerializable",
                        "message": clean_message(f"the row cannot be written as JSON: {error}"),
                    },
                    "facts": reply.get("facts"),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            code = 1
        # A leading line feed ends whatever the function wrote to the descriptor directly.
        data = "\n" + line + "\n"
        binary = getattr(reply_stream, "buffer", None)
        if binary is not None:
            reply_stream.flush()
            binary.write(data.encode("utf-8"))
            binary.flush()
        else:
            reply_stream.write(data)
            reply_stream.flush()
    finally:
        sys.stdout = reply_stream
    return code


def _utc_now() -> str:
    """Return the current UTC time in ISO 8601 with microseconds and a ``Z`` suffix."""
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(child_main())
