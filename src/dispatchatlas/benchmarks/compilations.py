"""Dated compilations of best known bounds for Taillard's and Demirkol's job shops.

Two web compilations list a lower and an upper bound on the optimal makespan of every instance
of the suites ``"taillard-job-shop"`` (ta01 to ta80) and ``"demirkol-job-shop"`` (dmu01 to
dmu80), and revise them as new bounds are found. The library records each compilation as it
stood on its date, in its own file under ``dispatchatlas/data/bounds/compilations``:

- ``scheduleopt-2026-11-21``: the file ``bks.json`` of the ScheduleOpt benchmark collection,
  at one commit.
- ``optimizizer-2026-11-08``: the Taillard and Demirkol tables of Optimizizer, as of the
  update date its pages print.

The two compilations stay separate. :func:`compilation_bounds` returns what one of them
lists, with its source and date; :func:`best_known` returns, for each instance, the tighter
upper bound and the tighter lower bound of the two, each with the compilation or compilations
that list it, so no value is attributed to a source that does not print it. These bounds are
distinct from the values that :mod:`dispatchatlas.benchmarks.bounds` reads from two published
papers.

The trivial lower bound of a job shop is the larger of the greatest machine load, the sum of
the processing times of the operations on one machine, and the greatest job length.
:func:`trivial_lower_bound` computes it from an instance, :func:`best_known` carries it for
every instance, and :func:`upper_equals_trivial_lower_bound` says whether the best known upper
bound reaches it, in which case the instance is solved by that bound alone.

Example::

    from dispatchatlas.benchmarks.compilations import best_known

    known = best_known("demirkol-job-shop")["dmu43"]
    print(known.lower, known.lower_sources, known.upper, known.upper_sources)
"""

from __future__ import annotations

import datetime
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from dispatchatlas.benchmarks.suites import suite
from dispatchatlas.core.errors import ConfigError, ModelError
from dispatchatlas.core.model import Problem

__all__ = [
    "BestKnown",
    "Compilation",
    "CompiledBound",
    "best_known",
    "compilation_bounds",
    "compilations",
    "trivial_lower_bound",
    "upper_equals_trivial_lower_bound",
]

SUITES = ("taillard-job-shop", "demirkol-job-shop")
"""The suites that the compilations cover."""

_COMPILATIONS = ("scheduleopt-2026-11-21", "optimizizer-2026-11-08")
_META_KEYS = frozenset({"id", "name", "url", "file", "version", "date", "retrieved", "licence"})


@dataclass(frozen=True, slots=True)
class Compilation:
    """One dated compilation of bounds, as the data file records it.

    Attributes:
        id: Identifier, also the name of the data file, for example
            ``"scheduleopt-2026-11-21"``.
        name: Name of the compilation.
        url: Page or repository that holds it.
        file: The file or pages of ``url`` that the values come from.
        version: The commit of the repository, or the update date that the pages print.
        date: The date of the compilation: the date of the commit, or the update date.
        retrieved: The date the values were read.
        licence: The licence the source states, or ``None`` if it states none.
    """

    id: str
    name: str
    url: str
    file: str
    version: str
    date: datetime.date
    retrieved: datetime.date
    licence: str | None


@dataclass(frozen=True, slots=True)
class CompiledBound:
    """The bounds one compilation lists for one instance.

    Attributes:
        compilation: Id of the compilation that lists them.
        date: Date of that compilation.
        lower: Lower bound on the optimal makespan.
        upper: Upper bound on the optimal makespan.
    """

    compilation: str
    date: datetime.date
    lower: int
    upper: int

    @property
    def optimal(self) -> bool:
        """Whether the two bounds are equal, so that the compilation lists the optimum."""
        return self.lower == self.upper


@dataclass(frozen=True, slots=True)
class BestKnown:
    """The tighter bounds of the compilations for one instance, with their sources.

    Attributes:
        instance: Instance name.
        lower: The largest lower bound that any compilation lists.
        lower_sources: Ids of the compilations that list ``lower``, in the order of
            :func:`compilations`.
        upper: The smallest upper bound that any compilation lists.
        upper_sources: Ids of the compilations that list ``upper``, in the same order.
        trivial_lower: The trivial lower bound of the instance, computed from it.
        listed: What each compilation lists, by id.
    """

    instance: str
    lower: int
    lower_sources: tuple[str, ...]
    upper: int
    upper_sources: tuple[str, ...]
    trivial_lower: int
    listed: Mapping[str, CompiledBound]

    @property
    def optimal(self) -> bool:
        """Whether the tighter bounds are equal, so that the optimum is known."""
        return self.lower == self.upper

    @property
    def upper_equals_trivial_lower(self) -> bool:
        """Whether the best known upper bound equals the trivial lower bound."""
        return self.upper == self.trivial_lower


def compilations() -> Mapping[str, Compilation]:
    """Return the recorded compilations by id, with their source, version, and date.

    Returns:
        A read-only mapping in a fixed order: the ScheduleOpt compilation, then Optimizizer's.
    """
    return MappingProxyType({key: value[0] for key, value in _read_all().items()})


def compilation_bounds(compilation: str) -> Mapping[str, CompiledBound]:
    """Return what one compilation lists, by instance name.

    Args:
        compilation: Id of a compilation, a key of :func:`compilations`.

    Returns:
        A read-only mapping from each of ta01 to ta80 and dmu01 to dmu80 to its bounds.

    Raises:
        ConfigError: If the id is unknown, or its data file is refused: it must have the table
            ``compilation`` with the keys of :class:`Compilation` (``licence`` optional) and
            the table ``bounds`` with an entry ``lower`` and ``upper`` for each instance of
            the covered suites and no other, positive whole numbers with ``lower`` not above
            ``upper``.
    """
    return _read_all_checked(compilation)[1]


def best_known(suite_id: str) -> Mapping[str, BestKnown]:
    """Return the tighter of the compilations' bounds for each instance of a suite.

    The upper bound is the smallest that any compilation lists, the lower bound the largest,
    each with the compilations that list it, so a value that one compilation alone lists names
    that compilation, and equal values name both.

    Args:
        suite_id: ``"taillard-job-shop"`` or ``"demirkol-job-shop"``.

    Returns:
        A read-only mapping from instance name to its best known bounds, in the order of the
        suite. The trivial lower bound of each instance is computed from the instance, so the
        first call builds every instance of the suite.

    Raises:
        ConfigError: If the suite is not one of those covered.
    """
    if suite_id not in SUITES:
        raise ConfigError(f"no compilations cover suite {suite_id!r}; they cover {list(SUITES)}")
    return _best_known(suite_id)


def trivial_lower_bound(problem: Problem) -> int:
    """Return the trivial lower bound on the makespan of a job shop.

    It is the larger of the greatest machine load, the total processing time of the operations
    that demand one machine, and the greatest job length, the total processing time of one
    job's operations. Both are computed from the instance alone.

    Args:
        problem: A job shop whose every operation belongs to a job group and demands one
            machine in its only mode, as the problems of :mod:`dispatchatlas.benchmarks.taillard`
            and :mod:`dispatchatlas.benchmarks.demirkol` do.

    Returns:
        The bound, a whole number when the processing times are whole.

    Raises:
        ModelError: If the problem is not a job shop, or an operation belongs to no group.
    """
    if "job-shop" not in problem.shapes:
        raise ModelError(f"problem {problem.id!r} is not a job shop")
    if sum(len(indices) for indices in problem.groups.values()) != len(problem.tasks):
        raise ModelError(f"problem {problem.id!r} has operations that belong to no job group")
    loads: dict[str, float] = {}
    longest = 0.0
    for indices in problem.groups.values():
        length = 0.0
        for index in indices:
            mode = problem.tasks[index].modes[0]
            machine = next(iter(mode.demands))
            loads[machine] = loads.get(machine, 0.0) + mode.duration
            length += mode.duration
        longest = max(longest, length)
    return round(max(*loads.values(), longest))


def upper_equals_trivial_lower_bound(suite_id: str, instance: str) -> bool:
    """Return whether the best known upper bound of an instance equals its trivial lower bound.

    Args:
        suite_id: ``"taillard-job-shop"`` or ``"demirkol-job-shop"``.
        instance: Instance name, for example ``"ta73"``.

    Returns:
        ``True`` when the best known upper bound (:func:`best_known`) is the trivial lower
        bound (:func:`trivial_lower_bound`), so the optimum is known from the instance alone.

    Raises:
        ConfigError: If the suite is not covered or the instance is not one of it.
    """
    known = best_known(suite_id)
    if instance not in known:
        raise ConfigError(f"suite {suite_id!r} has no instance {instance!r} with recorded bounds")
    return known[instance].upper_equals_trivial_lower


# ------------------------------------------------------------------ reading


@cache
def _best_known(suite_id: str) -> Mapping[str, BestKnown]:
    names = suite(suite_id).instances
    tables = {key: compilation_bounds(key) for key in _COMPILATIONS}
    result: dict[str, BestKnown] = {}
    for name in names:
        listed = {key: table[name] for key, table in tables.items()}
        upper = min(item.upper for item in listed.values())
        lower = max(item.lower for item in listed.values())
        result[name] = BestKnown(
            instance=name,
            lower=lower,
            lower_sources=tuple(key for key, item in listed.items() if item.lower == lower),
            upper=upper,
            upper_sources=tuple(key for key, item in listed.items() if item.upper == upper),
            trivial_lower=trivial_lower_bound(suite(suite_id).load(name)),
            listed=MappingProxyType(listed),
        )
    return MappingProxyType(result)


def _read_all_checked(compilation: str) -> tuple[Compilation, Mapping[str, CompiledBound]]:
    table = _read_all()
    if compilation not in table:
        raise ConfigError(f"unknown compilation {compilation!r}; known: {list(table)}")
    return table[compilation]


@cache
def _read_all() -> Mapping[str, tuple[Compilation, Mapping[str, CompiledBound]]]:
    return MappingProxyType({key: _read(key) for key in _COMPILATIONS})


def _read(key: str) -> tuple[Compilation, Mapping[str, CompiledBound]]:
    path = files("dispatchatlas.data").joinpath("bounds", "compilations", f"{key}.toml")
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"the compilation {key!r} is not valid TOML: {error}") from error
    return _parse(key, document)


def _parse(
    key: str, document: Mapping[str, Any]
) -> tuple[Compilation, Mapping[str, CompiledBound]]:
    """Check a document of one compilation and build its records."""
    where = f"compilation {key!r}"
    if set(document) != {"compilation", "bounds"}:
        raise ConfigError(f"{where}: the tables must be 'compilation' and 'bounds'")
    head, rows = document["compilation"], document["bounds"]
    if not isinstance(head, Mapping) or not isinstance(rows, Mapping):
        raise ConfigError(f"{where}: 'compilation' and 'bounds' must be tables")
    unknown = sorted(set(head) - _META_KEYS)
    absent = sorted(_META_KEYS - {"licence"} - set(head))
    if unknown or absent:
        raise ConfigError(f"{where}: unknown keys {unknown}, missing keys {absent}")
    if head["id"] != key:
        raise ConfigError(f"{where}: the id is {head['id']!r}")
    strings = {name: head[name] for name in ("name", "url", "file", "version")}
    if not all(isinstance(value, str) and value for value in strings.values()):
        raise ConfigError(f"{where}: name, url, file, and version must be text")
    dates = {name: _date(where, name, head[name]) for name in ("date", "retrieved")}
    head_record = Compilation(
        id=key,
        licence=_optional_text(where, head.get("licence")),
        date=dates["date"],
        retrieved=dates["retrieved"],
        **strings,
    )
    expected = [name for suite_id in SUITES for name in suite(suite_id).instances]
    if sorted(rows) != sorted(expected):
        raise ConfigError(
            f"{where}: bounds must cover exactly the {len(expected)} instances of {list(SUITES)}"
        )
    bounds: dict[str, CompiledBound] = {}
    for name in expected:
        row = rows[name]
        if not isinstance(row, Mapping) or set(row) != {"lower", "upper"}:
            raise ConfigError(f"{where}, instance {name!r}: needs exactly lower and upper")
        lower, upper = (_whole(where, name, row[field]) for field in ("lower", "upper"))
        if lower > upper:
            raise ConfigError(
                f"{where}, instance {name!r}: the lower bound {lower} is above the upper bound "
                f"{upper}"
            )
        bounds[name] = CompiledBound(key, head_record.date, lower, upper)
    return head_record, MappingProxyType(bounds)


def _date(where: str, name: str, value: object) -> datetime.date:
    if not isinstance(value, str):
        raise ConfigError(f"{where}: {name} must be an ISO date in quotes, got {value!r}")
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        raise ConfigError(f"{where}: {name} must be an ISO date, got {value!r}") from None


def _optional_text(where: str, value: object) -> str | None:
    if value is None or (isinstance(value, str) and value):
        return value
    raise ConfigError(f"{where}: licence must be text when given, got {value!r}")


def _whole(where: str, name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{where}, instance {name!r}: a bound must be a positive whole number")
    return value
