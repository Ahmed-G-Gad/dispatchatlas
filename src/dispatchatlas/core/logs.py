"""Logging helpers.

The library logs through the standard :mod:`logging` module under the
``dispatchatlas`` logger and sends nothing to a stream of its own accord, so
applications decide what is shown. :func:`configure_logging` is a convenience
for scripts and the command line. :class:`SessionLog` writes what is logged
during one piece of work, such as a campaign session, to a file with UTC time
stamps, and :class:`Lowered` lets chosen loggers record from a lower level
meanwhile. Neither changes what the other handlers show.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import IO

__all__ = [
    "LOG_FILE_LIMIT",
    "LOG_FORMAT",
    "Lowered",
    "SessionLog",
    "configure_logging",
    "get_logger",
]

ROOT_LOGGER = "dispatchatlas"

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
"""Format of a line of a :class:`SessionLog` file; the time is in UTC."""

LOG_FILE_LIMIT = 1 << 26
"""Most characters a :class:`SessionLog` writes to its file by default."""

_KEPT = 10_000
"""Most records a :class:`SessionLog` keeps while it has no file."""

_FLUSH_SECONDS = 1.0
"""Longest a record below ``WARNING`` waits in the buffer of a log file for the next write."""

logging.getLogger(ROOT_LOGGER).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return the library logger for a module.

    Args:
        name: Module name, usually ``__name__``. Names outside the
            ``dispatchatlas`` package are nested under it.

    Returns:
        A logger in the ``dispatchatlas`` hierarchy.
    """
    if name == ROOT_LOGGER or name.startswith(ROOT_LOGGER + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")


def configure_logging(level: int | str = logging.INFO) -> None:
    """Send library log records to standard error with a compact format.

    Calling it more than once replaces the handler it installed before.

    Args:
        level: Minimum level to show, as a number or a name such as ``"DEBUG"``.
    """
    logger = logging.getLogger(ROOT_LOGGER)
    for handler in list(logger.handlers):
        if getattr(handler, "_dispatchatlas", False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler._dispatchatlas = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(level)


class Lowered:
    """Lets loggers record from a lower level while a ``with`` block runs, and shows no more.

    Each logger of ``names`` that lets fewer records through than ``level``
    asks for gets that level for the block; a logger that lets them through
    already is left alone. Every handler the records of such a logger can
    reach, on the logger, on the loggers above it as far as its records
    propagate, and on the loggers below it, gets a filter for the block. The
    filter holds back what the lowering added and nothing else: a record of
    a lowered logger, or of a logger below one, passes when its logger let
    it through before the block, and a record of any other logger passes
    always. Those handlers therefore show inside the block what they show
    outside it, and the block leaves the levels and the filters as it found
    them. A handler that is added inside the block is not filtered.

    Args:
        names: Names of the loggers to lower.
        level: The level they are lowered to.
    """

    def __init__(self, names: Sequence[str], level: int) -> None:
        self.names = tuple(names)
        self.level = level
        self._earlier: dict[logging.Logger, int] = {}
        self._gate = _Before(self.names, self._earlier)
        self._gated: list[logging.Handler] = []

    def __enter__(self) -> Lowered:
        self.apply()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.undo()

    def apply(self) -> None:
        """Lower the loggers and filter the handlers their records reach."""
        loggers = [logging.getLogger(name) for name in self.names]
        if self._earlier or all(item.getEffectiveLevel() <= self.level for item in loggers):
            return
        self._gated = self._handlers()
        for handler in self._gated:
            handler.addFilter(self._gate)
        for logger in loggers:
            # An outer logger lowered before this one may have lowered it as well.
            if logger.getEffectiveLevel() > self.level:
                self._earlier[logger] = logger.level
                logger.setLevel(self.level)

    def undo(self) -> None:
        """Give the loggers the levels they had and take the filter off the handlers."""
        for logger, level in self._earlier.items():
            logger.setLevel(level)
        self._earlier.clear()
        for handler in self._gated:
            handler.removeFilter(self._gate)
        self._gated = []

    def _handlers(self) -> list[logging.Handler]:
        """Return the handlers that the records of the named loggers can reach."""
        found: dict[int, logging.Handler] = {}
        known = list(logging.Logger.manager.loggerDict.items())
        for name in self.names:
            reached: logging.Logger | None = logging.getLogger(name)
            while reached is not None:
                found.update((id(handler), handler) for handler in reached.handlers)
                reached = reached.parent if reached.propagate else None
            for other, below in known:
                if other.startswith(name + ".") and isinstance(below, logging.Logger):
                    found.update((id(handler), handler) for handler in below.handlers)
        return list(found.values())


class _Before(logging.Filter):
    """Passes the records their loggers let through before some loggers were lowered.

    ``earlier`` maps each lowered logger to the level it had. The level a
    logger answered to before is the first one set on the way from it to
    the root, with the earlier level in the place of a lowered one.
    """

    def __init__(self, names: tuple[str, ...], earlier: Mapping[logging.Logger, int]) -> None:
        super().__init__()
        self._names = frozenset(names)
        self._prefixes = tuple(name + "." for name in names)
        self._earlier = earlier

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        if name not in self._names and not name.startswith(self._prefixes):
            return True
        logger: logging.Logger | None = logging.getLogger(name)
        while logger is not None:
            level = self._earlier.get(logger, logger.level)
            if level != logging.NOTSET:
                return record.levelno >= level
            logger = logger.parent
        return True


class SessionLog:
    """A file that receives what the library logs while a ``with`` block runs.

    Inside the block every record of ``level`` or higher that reaches the
    ``dispatchatlas`` logger is kept, and :meth:`open` names the file: the
    records kept so far are written to it, and every later one as it
    arrives. Each line states the UTC time of its record to the millisecond,
    the level, the logger, and the message (:data:`LOG_FORMAT`). The file is
    appended to, never replaced. A block that ends before :meth:`open`
    writes nothing.

    The loggers of ``lowered`` record from ``level`` inside the block, as
    :class:`Lowered` arranges it, so what the other handlers show stays as
    it was. Every other logger of the library keeps its level, and the file
    receives from it what it lets through anyway: a record that is not made
    costs the code that would log it next to nothing, and one that is made
    costs its making and one buffered write.

    The file is written through a buffer. A record of the level ``WARNING``
    or higher is on disk when it is written, and so is a record that
    arrives a second or more after the last one that was; the records of a
    burst below ``WARNING`` reach the disk with the next such record or at
    the end of the block. At most ``limit`` characters are written: later
    records are counted, and the last line of the file states how many were
    left out.

    Args:
        level: Lowest level of the records the file receives.
        lowered: Names of the loggers that record from ``level`` inside the
            block.
        limit: Most characters written to the file.
    """

    def __init__(
        self,
        level: int = logging.INFO,
        *,
        lowered: Sequence[str] = (),
        limit: int = LOG_FILE_LIMIT,
    ) -> None:
        self.level = level
        self.path: Path | None = None
        self._handler = _FileHandler(level, limit)
        self._lowered = Lowered(lowered, level)

    def __enter__(self) -> SessionLog:
        self._lowered.apply()
        logging.getLogger(ROOT_LOGGER).addHandler(self._handler)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        logging.getLogger(ROOT_LOGGER).removeHandler(self._handler)
        self._handler.close()
        self._lowered.undo()

    def open(self, path: str | Path) -> Path:
        """Write the records kept so far, and every later one, to ``path``.

        Args:
            path: The log file. Its directory is created when missing.

        Returns:
            The path.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handler.open(self.path)
        return self.path

    def write(self, created: float, level: int, name: str, message: str) -> None:
        """Add a record that did not reach the loggers of this process.

        The record goes to the file only, with its own time, so a worker
        process can hand over what it logged.

        Args:
            created: When the record was made, in seconds since the epoch.
            level: Level of the record.
            name: Name of its logger.
            message: Its message.
        """
        if level < self.level:
            return
        record = logging.makeLogRecord(
            {
                "name": name,
                "levelno": level,
                "levelname": logging.getLevelName(level),
                "msg": message,
                "created": created,
                "msecs": (created - int(created)) * 1000.0,
            }
        )
        self._handler.handle(record)


class _UtcFormatter(logging.Formatter):
    """Formats the time of a record in UTC, as ISO 8601 to the millisecond."""

    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03dZ"

    @staticmethod
    def converter(seconds: float | None = None) -> time.struct_time:
        """Return a time in seconds since the epoch as a UTC time."""
        return time.gmtime(seconds)


class _FileHandler(logging.Handler):
    """Keeps records until it has a file, then writes each one through a buffer.

    Args:
        level: Lowest level of the records it takes.
        limit: Most characters it writes; later records are counted.
    """

    def __init__(self, level: int, limit: int) -> None:
        super().__init__(level)
        self.setFormatter(_UtcFormatter(LOG_FORMAT))
        self._kept: list[logging.LogRecord] = []
        self._stream: IO[str] | None = None
        self._limit = limit
        self._written = 0
        self._left_out = 0
        self._flushed = 0.0

    def open(self, path: Path) -> None:
        """Write the records kept so far to ``path``, and every later one."""
        self.acquire()
        try:
            if self._stream is not None:
                self._stream.close()
            self._stream = path.open("a", encoding="utf-8", newline="\n")
            kept, self._kept = self._kept, []
            for record in kept:
                self.emit(record)
        finally:
            self.release()

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the record, or write it when the handler has a file."""
        try:
            if self._stream is None:
                if len(self._kept) < _KEPT:
                    self._kept.append(record)
                return
            if self._written >= self._limit:
                self._left_out += 1
                return
            line = self.format(record) + "\n"
            self._stream.write(line)
            self._written += len(line)
            now = time.monotonic()
            if record.levelno >= logging.WARNING or now - self._flushed >= _FLUSH_SECONDS:
                self._stream.flush()
                self._flushed = now
        except Exception:  # a record that cannot be written is reported, not raised
            self.handleError(record)

    def close(self) -> None:
        """State how many records were left out, if any, and close the file."""
        self.acquire()
        try:
            if self._stream is not None:
                if self._left_out:
                    note = logging.makeLogRecord(
                        {
                            "name": __name__,
                            "levelno": logging.WARNING,
                            "levelname": logging.getLevelName(logging.WARNING),
                            "msg": f"{self._left_out} later record(s) were left out: the file "
                            f"reached its limit of {self._limit} characters",
                        }
                    )
                    self._stream.write(self.format(note) + "\n")
                self._stream.close()
                self._stream = None
        finally:
            self.release()
        super().close()
