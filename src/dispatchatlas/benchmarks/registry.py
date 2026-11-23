"""The benchmark-family registry and plugin discovery.

A benchmark family is a parameterised generator of problems. Each family is a
dataclass whose fields are its parameters and whose ``generate(seed)`` method
returns one :class:`~dispatchatlas.core.model.Problem`. Equal parameters and
seed always give the same problem.

Families register like solvers. Installed packages can add families through
the ``dispatchatlas.benchmarks`` entry-point group::

    [project.entry-points."dispatchatlas.benchmarks"]
    my_families = "my_package.families:register"
"""

from __future__ import annotations

import dataclasses
import importlib
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import cache
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

from dispatchatlas.core._frozen import FrozenMap
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.logs import get_logger
from dispatchatlas.core.model import Problem

__all__ = [
    "BUILT_IN_MODULES",
    "PLUGIN_GROUP",
    "FamilyInfo",
    "FamilyRegistry",
    "Generator",
    "default_registry",
    "generate",
    "load_plugins",
]

PLUGIN_GROUP = "dispatchatlas.benchmarks"
"""Entry-point group scanned for benchmark-family plugins."""

BUILT_IN_MODULES: tuple[str, ...] = (
    "dispatchatlas.benchmarks.shop",
    "dispatchatlas.benchmarks.taillard",
    "dispatchatlas.benchmarks.machines",
    "dispatchatlas.benchmarks.project",
    "dispatchatlas.benchmarks.workflow",
    "dispatchatlas.benchmarks.continuum",
)
"""Modules whose ``register(registry)`` function adds the built-in families."""

_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_LOG = get_logger(__name__)


@dataclass(frozen=True)
class FamilyInfo:
    """Metadata describing a benchmark family.

    Attributes:
        id: Registry key in kebab case, for example ``"flow-shop"``.
        name: Human-readable name.
        problem_class: The scheduling problem class, for example
            ``"job-shop"`` or ``"continuum"``.
        summary: One sentence describing what the generated problems model.
        references: Bibliography keys, DOIs, or URLs of the sources the
            generator follows.
        objectives: Objectives the generated problems declare by default.
        parameters: Default parameter values. Filled from the dataclass
            fields when registered.
    """

    id: str
    name: str
    problem_class: str
    summary: str
    references: tuple[str, ...] = ()
    objectives: tuple[str, ...] = ("makespan",)
    parameters: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not _SLUG.match(self.id):
            raise ConfigError(f"family id {self.id!r} must be lowercase kebab case")
        if not self.name or not self.summary:
            raise ConfigError(f"family {self.id!r} needs a name and a summary")
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "objectives", tuple(self.objectives))
        object.__setattr__(self, "parameters", FrozenMap(self.parameters))


@runtime_checkable
class Generator(Protocol):
    """The interface of a benchmark family."""

    @property
    def info(self) -> FamilyInfo:
        """Metadata of the family."""
        ...

    def generate(self, seed: int = 0) -> Problem:
        """Return the problem for ``seed``."""
        ...


class FamilyRegistry:
    """A collection of benchmark families addressable by id."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[FamilyInfo, Any]] = {}

    def register(
        self, factory: Any, info: FamilyInfo | None = None, *, replace_existing: bool = False
    ) -> None:
        """Add a family.

        Args:
            factory: Callable building the generator from keyword parameters,
                usually the family's dataclass.
            info: The family's metadata. Defaults to ``factory.info``.
            replace_existing: Allow replacing a family with the same id.

        Raises:
            ConfigError: If the id is taken, or no metadata is available.
        """
        resolved: object = info if info is not None else getattr(factory, "info", None)
        if not isinstance(resolved, FamilyInfo):
            raise ConfigError(f"no FamilyInfo given for factory {factory!r}")
        info = resolved
        candidate: object = factory
        is_dataclass = isinstance(candidate, type) and dataclasses.is_dataclass(candidate)
        if not info.parameters and is_dataclass:
            info = replace(info, parameters=_dataclass_defaults(candidate))
        if info.id in self._entries and not replace_existing:
            raise ConfigError(f"benchmark family {info.id!r} is already registered")
        self._entries[info.id] = (info, factory)

    def create(self, family_id: str, **parameters: Any) -> Generator:
        """Build a family's generator with the given parameter values.

        Raises:
            ConfigError: If the id is unknown or a parameter is invalid.
        """
        info, factory = self._entry(family_id)
        unknown = set(parameters) - set(info.parameters)
        if unknown:
            raise ConfigError(
                f"family {family_id!r} has no parameters {sorted(unknown)}; "
                f"known: {sorted(info.parameters)}"
            )
        try:
            generator: Generator = factory(**parameters)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"invalid parameters for family {family_id!r}: {error}") from error
        return generator

    def generate(self, family_id: str, seed: int = 0, **parameters: Any) -> Problem:
        """Generate one problem of a family."""
        return self.create(family_id, **parameters).generate(seed)

    def info(self, family_id: str) -> FamilyInfo:
        """Return the metadata of a registered family."""
        return self._entry(family_id)[0]

    def infos(self) -> tuple[FamilyInfo, ...]:
        """Return the metadata of every family, sorted by id."""
        return tuple(self._entries[key][0] for key in sorted(self._entries))

    def ids(self) -> tuple[str, ...]:
        """Return every registered id, sorted."""
        return tuple(sorted(self._entries))

    def copy(self) -> FamilyRegistry:
        """Return an independent copy of this registry."""
        clone = FamilyRegistry()
        clone._entries = dict(self._entries)
        return clone

    def __contains__(self, family_id: object) -> bool:
        return family_id in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids())

    def __len__(self) -> int:
        return len(self._entries)

    def _entry(self, family_id: str) -> tuple[FamilyInfo, Any]:
        try:
            return self._entries[family_id]
        except KeyError:
            raise ConfigError(
                f"unknown benchmark family {family_id!r}; "
                "run 'dispatchatlas benchmarks list' to see ids"
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
def _built_in() -> FamilyRegistry:
    registry = FamilyRegistry()
    for module_name in BUILT_IN_MODULES:
        importlib.import_module(module_name).register(registry)
    return registry


def load_plugins(registry: FamilyRegistry, *, group: str = PLUGIN_GROUP) -> list[str]:
    """Register the families published by installed plugins.

    Returns:
        Names of the entry points that loaded. A plugin that fails is skipped
        with a warning and registers none of its families.
    """
    loaded: list[str] = []
    for entry in sorted(entry_points(group=group), key=lambda item: item.name):
        staged = registry.copy()
        try:
            entry.load()(staged)
        except Exception as error:  # a broken plugin must not disable the others
            _LOG.warning("benchmark plugin %r failed to load: %s", entry.name, error)
            continue
        registry._entries = staged._entries
        loaded.append(entry.name)
    return loaded


def default_registry(*, plugins: bool = True) -> FamilyRegistry:
    """Return a registry of the built-in families, plus plugins by default."""
    registry = _built_in().copy()
    if plugins:
        load_plugins(registry)
    return registry


def generate(family_id: str, seed: int = 0, **parameters: Any) -> Problem:
    """Generate one problem of a registered family.

    Example:
        >>> from dispatchatlas.benchmarks import generate
        >>> problem = generate("job-shop", seed=1, jobs=6, machines=4)
    """
    return default_registry().generate(family_id, seed, **parameters)
