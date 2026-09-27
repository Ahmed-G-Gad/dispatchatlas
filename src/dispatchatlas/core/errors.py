"""Exception hierarchy shared by every DispatchAtlas package.

Every error raised on purpose by the library derives from
:class:`DispatchAtlasError`, so callers can catch the whole family with one
clause. The concrete classes also derive from the matching built-in exception
(``ValueError``, ``RuntimeError``, ``ImportError``), so generic handlers keep
working.
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "DispatchAtlasError",
    "MissingDependencyError",
    "ModelError",
    "SerializationError",
    "SolverError",
    "UnsupportedProblemError",
]


class DispatchAtlasError(Exception):
    """Base class of every error raised on purpose by DispatchAtlas."""


class ModelError(DispatchAtlasError, ValueError):
    """A problem, task, resource, or schedule violates the data model.

    Attributes:
        issues: One human-readable line per detected defect. A problem that
            fails several checks reports all of them at once.
    """

    def __init__(self, message: str, issues: tuple[str, ...] = ()) -> None:
        self.issues = issues
        if issues:
            message = message + "\n" + "\n".join(f"  - {issue}" for issue in issues)
        super().__init__(message)


class SerializationError(DispatchAtlasError, ValueError):
    """A document cannot be read or written in a DispatchAtlas format."""


class ConfigError(DispatchAtlasError, ValueError):
    """A configuration file or parameter set is invalid."""


class SolverError(DispatchAtlasError, RuntimeError):
    """A solver cannot run as requested, for example with unbounded stop criteria."""


class UnsupportedProblemError(SolverError):
    """A solver was asked to solve a problem with features it does not model."""


class MissingDependencyError(DispatchAtlasError, ImportError):
    """An optional third-party package required by a feature is not installed.

    The message names the package and the extra that provides it, for example
    ``pip install "dispatchatlas[exact]"``.
    """

    def __init__(self, package: str, extra: str, feature: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            f"{feature} requires the optional package '{package}'. "
            f'Install it with: pip install "dispatchatlas[{extra}]"'
        )
