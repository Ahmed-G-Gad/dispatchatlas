"""Logging helpers.

The library logs through the standard :mod:`logging` module under the
``dispatchatlas`` logger and attaches no handlers of its own, so applications
decide what is shown. :func:`configure_logging` is a convenience for scripts
and the command line.
"""

from __future__ import annotations

import logging

__all__ = ["configure_logging", "get_logger"]

ROOT_LOGGER = "dispatchatlas"

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
