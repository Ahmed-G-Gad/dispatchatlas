"""The solver registry and plugin discovery.

Solvers register a class (or any factory) together with its
:class:`~dispatchatlas.core.solver.SolverInfo`. The default registry holds every
built-in solver and, unless disabled, the solvers that installed packages
publish under the ``dispatchatlas.solvers`` entry-point group.

A plugin package declares, in its ``pyproject.toml``::

    [project.entry-points."dispatchatlas.solvers"]
    my_solvers = "my_package.solvers:register"

where ``register(registry)`` calls :meth:`SolverRegistry.register` for each of
its solvers.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from functools import cache
from importlib.metadata import entry_points
from typing import Any

from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.logs import get_logger
from dispatchatlas.core.solver import Solver, SolverInfo

__all__ = [
    "BUILT_IN_MODULES",
    "PLUGIN_GROUP",
    "SolverRegistry",
    "default_registry",
    "load_plugins",
]

PLUGIN_GROUP = "dispatchatlas.solvers"
"""Entry-point group scanned for solver plugins."""

BUILT_IN_MODULES: tuple[str, ...] = (
    "dispatchatlas.solvers.rules",
    "dispatchatlas.solvers.constructive",
    "dispatchatlas.solvers.trajectory",
    "dispatchatlas.solvers.population",
    "dispatchatlas.solvers.continuous",
    "dispatchatlas.solvers.swarm",
    "dispatchatlas.solvers.multiobjective",
    "dispatchatlas.solvers.learning",
    "dispatchatlas.solvers.exact",
    "dispatchatlas.solvers.hybrid",
    "dispatchatlas.solvers.shop",
)
"""Modules whose ``register(registry)`` function adds the built-in solvers."""

_LOG = get_logger(__name__)

type SolverFactory = Callable[..., Solver]


class SolverRegistry:
    """A collection of solvers addressable by id."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[SolverInfo, SolverFactory]] = {}

    def register(
        self,
        factory: SolverFactory,
        info: SolverInfo | None = None,
        *,
        replace_existing: bool = False,
    ) -> None:
        """Add a solver.

        Args:
            factory: Callable that builds the solver from keyword parameters.
                Usually the solver's dataclass.
            info: The solver's metadata. Defaults to ``factory.info``. When the
                factory is a dataclass and ``info.parameters`` is empty, the
                parameters and their defaults are read from its fields.
            replace_existing: Allow replacing a solver with the same id.

        Raises:
            ConfigError: If the id is taken and ``replace_existing`` is false, or
                no metadata is available.
        """
        resolved: object = info if info is not None else getattr(factory, "info", None)
        if not isinstance(resolved, SolverInfo):
            raise ConfigError(f"no SolverInfo given for factory {factory!r}")
        info = resolved
        candidate: object = factory
        is_dataclass = isinstance(candidate, type) and dataclasses.is_dataclass(candidate)
        if not info.parameters and is_dataclass:
            info = replace(info, parameters=_dataclass_defaults(candidate))
        if info.id in self._entries and not replace_existing:
            raise ConfigError(f"solver {info.id!r} is already registered")
        self._entries[info.id] = (info, factory)

    def create(self, solver_id: str, **parameters: Any) -> Solver:
        """Build a solver by id with the given parameter values.

        Raises:
            ConfigError: If the id is unknown or a parameter is not accepted.
        """
        info, factory = self._entry(solver_id)
        unknown = set(parameters) - set(info.parameters)
        if unknown:
            raise ConfigError(
                f"solver {solver_id!r} has no parameters {sorted(unknown)}; "
                f"known: {sorted(info.parameters)}"
            )
        try:
            return factory(**parameters)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"invalid parameters for solver {solver_id!r}: {error}") from error

    def info(self, solver_id: str) -> SolverInfo:
        """Return the metadata of a registered solver."""
        return self._entry(solver_id)[0]

    def infos(self) -> tuple[SolverInfo, ...]:
        """Return the metadata of every solver, sorted by id."""
        return tuple(self._entries[key][0] for key in sorted(self._entries))

    def ids(self) -> tuple[str, ...]:
        """Return every registered id, sorted."""
        return tuple(sorted(self._entries))

    def copy(self) -> SolverRegistry:
        """Return an independent copy of this registry."""
        clone = SolverRegistry()
        clone._entries = dict(self._entries)
        return clone

    def __contains__(self, solver_id: object) -> bool:
        return solver_id in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids())

    def __len__(self) -> int:
        return len(self._entries)

    def _entry(self, solver_id: str) -> tuple[SolverInfo, SolverFactory]:
        try:
            return self._entries[solver_id]
        except KeyError:
            raise ConfigError(
                f"unknown solver {solver_id!r}; run 'dispatchatlas solvers list' to see ids"
            ) from None


def _dataclass_defaults(factory: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for item in dataclasses.fields(factory):
        if not item.init:
            continue
        if item.default is not dataclasses.MISSING:
            defaults[item.name] = item.default
        elif item.default_factory is not dataclasses.MISSING:
            defaults[item.name] = item.default_factory()
    return defaults


@cache
def _built_in() -> SolverRegistry:
    registry = SolverRegistry()
    for module_name in BUILT_IN_MODULES:
        importlib.import_module(module_name).register(registry)
    return registry


def load_plugins(registry: SolverRegistry, *, group: str = PLUGIN_GROUP) -> list[str]:
    """Register the solvers published by installed plugins.

    Each entry point must load a callable that takes the registry. A plugin
    that fails to load is skipped with a warning and registers none of its
    solvers, so one broken package cannot disable or half-install the others.

    Returns:
        Names of the entry points that loaded.
    """
    loaded: list[str] = []
    for entry in _entry_points(group):
        staged = registry.copy()
        try:
            hook = entry.load()
            hook(staged)
        except Exception as error:  # a broken plugin must not disable the others
            _LOG.warning("solver plugin %r failed to load: %s", entry.name, error)
            continue
        registry._entries = staged._entries
        loaded.append(entry.name)
    return loaded


def _entry_points(group: str) -> Iterable[Any]:
    return sorted(entry_points(group=group), key=lambda entry: entry.name)


def default_registry(*, plugins: bool = True) -> SolverRegistry:
    """Return a registry of the built-in solvers, plus plugins by default.

    Each call returns a fresh copy, so callers may register more solvers
    without affecting others.

    Args:
        plugins: Also load solvers from the ``dispatchatlas.solvers``
            entry-point group.
    """
    registry = _built_in().copy()
    if plugins:
        load_plugins(registry)
    return registry
