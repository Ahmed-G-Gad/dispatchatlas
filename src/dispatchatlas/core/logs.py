"""Logging helpers.

The library logs through the standard :mod:`logging` module under the
``dispatchatlas`` logger and sends nothing to a stream of its own accord, so
applications decide what is shown. :func:`configure_logging` is a convenience
for scripts and the command line. :class:`SessionLog` writes what is logged
during one piece of work, such as a campaign session, to a file with UTC time
stamps, and leaves what the other handlers show as it is.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from types import TracebackType
from typing import IO

__all__ = ["LOG_FORMAT", "SessionLog", "configure_logging", "get_logger"]

ROOT_LOGGER = "dispatchatlas"

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
"""Format of a line of a :class:`SessionLog` file; the time is in UTC."""

_KEPT = 10_000
"""Most records a :class:`SessionLog` keeps while it has no file."""

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


class SessionLog:
    """A file that receives what the library logs while a ``with`` block runs.

    Inside the block every record of ``level`` or higher that reaches the
    ``dispatchatlas`` logger is kept, and :meth:`open` names the file: the
    records kept so far are written to it, and every later one as it
    arrives. Each line states the UTC time of its record to the millisecond,
    the level, the logger, and the message (:data:`LOG_FORMAT`). The file is
    appended to, never replaced. A block that ends before :meth:`open`
    writes nothing.

    When the ``dispatchatlas`` logger lets fewer records through than
    ``level`` asks for, the block lowers the level of the logger and gives
    every handler the records reach, on that logger and on the loggers above
    it, a filter at the earlier level. Those handlers therefore show what
    they showed before, and the level and the filters are as they were when
    the block ends.

    Args:
        level: Lowest level of the records the file receives.
    """

    def __init__(self, level: int = logging.INFO) -> None:
        self.level = level
        self.path: Path | None = None
        self._handler = _FileHandler(level)
        self._gate: _AtLeast | None = None
        self._gated: list[logging.Handler] = []
        self._earlier = logging.NOTSET

    def __enter__(self) -> SessionLog:
        logger = logging.getLogger(ROOT_LOGGER)
        effective = logger.getEffectiveLevel()
        if effective > self.level:
            self._gate = _AtLeast(effective)
            reached: logging.Logger | None = logger
            while reached is not None:
                for handler in reached.handlers:
                    handler.addFilter(self._gate)
                    self._gated.append(handler)
                reached = reached.parent if reached.propagate else None
            self._earlier = logger.level
            logger.setLevel(self.level)
        logger.addHandler(self._handler)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        logger = logging.getLogger(ROOT_LOGGER)
        logger.removeHandler(self._handler)
        self._handler.close()
        if self._gate is not None:
            logger.setLevel(self._earlier)
            for handler in self._gated:
                handler.removeFilter(self._gate)
            self._gate = None
            self._gated.clear()

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


class _AtLeast(logging.Filter):
    """Lets the records of a level or higher through."""

    def __init__(self, level: int) -> None:
        super().__init__()
        self._level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self._level


class _UtcFormatter(logging.Formatter):
    """Formats the time of a record in UTC, as ISO 8601 to the millisecond."""

    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03dZ"

    @staticmethod
    def converter(seconds: float | None = None) -> time.struct_time:
        """Return a time in seconds since the epoch as a UTC time."""
        return time.gmtime(seconds)


class _FileHandler(logging.Handler):
    """Keeps records until it has a file, then writes each one as it arrives."""

    def __init__(self, level: int) -> None:
        super().__init__(level)
        self.setFormatter(_UtcFormatter(LOG_FORMAT))
        self._kept: list[logging.LogRecord] = []
        self._stream: IO[str] | None = None

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
            self._stream.write(self.format(record) + "\n")
            self._stream.flush()
        except Exception:  # a record that cannot be written is reported, not raised
            self.handleError(record)

    def close(self) -> None:
        """Close the file."""
        self.acquire()
        try:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        finally:
            self.release()
        super().close()
