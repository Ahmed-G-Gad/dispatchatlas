"""Reference values of a suite: the best known bounds, with every source's printed values.

The best known bounds on the optimal makespan of a benchmark instance move as work is
published, and the sources that list them disagree. This module reads the bounds recorded for
a suite in ``dispatchatlas/data/bounds`` as :class:`~dispatchatlas.benchmarks.suites.Bound`
objects that keep the values of every source beside the reference value:

- The reference upper bound is the smallest upper bound that any source prints, whether it
  lists it as best known or reports it as newly found.
- The reference lower bound is the largest lower bound that any source prints.
- An instance is optimal when the two are equal.

Every instance names at least two sources, each a key of the bibliography
(:mod:`dispatchatlas.references`) with the table and page that print its values, so that no
reference value rests on one table. A file that breaks a rule is refused as a whole.

The file of a suite is ``<suite id>.toml``. Each instance has a table with ``upper``,
``lower`` and ``optimal``, and one inline table per source, named by its bibliography key,
with the values that source prints: its ``upper`` and ``lower`` bound, its ``found_upper``
bound if it reports one as newly found, and ``where``, the table and page::

    [ta43]
    upper = 1846
    lower = 1809
    optimal = false
    peng2015tabu = { upper = 1848, lower = 1809, found_upper = 1846, where = "Table 8, p. 160" }
    goncalves2014extended = { upper = 1848, lower = 1809, where = "Table A7, p. 244" }

The bounds recorded are those of Taillard's job shops ta01 to ta50 (suite
``"taillard-job-shop"``), from Peng, Lü and Cheng (2015) and Gonçalves and Resende (2014).
The instances ta51 to ta80 are not included. The recorded bounds are separate from
:attr:`~dispatchatlas.benchmarks.suites.Suite.bounds`, which holds the bounds listed with a
suite's own instance files.

Example::

    from dispatchatlas.analysis import AnalysisConfig
    from dispatchatlas.benchmarks.bounds import reference_values

    config = AnalysisConfig(references=reference_values("taillard-job-shop", "upper"))
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Collection, Mapping
from functools import cache
from importlib.resources import files
from types import MappingProxyType
from typing import Any, Literal

from dispatchatlas.benchmarks.suites import Bound, BoundSource, suite
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.references import bibliography

__all__ = ["BoundKind", "load_bounds", "reference_values"]

type BoundKind = Literal["upper", "lower"]
"""Which reference bound of an instance: the best known upper or the best known lower bound."""

_INSTANCE_KEYS = ("upper", "lower", "optimal")
_SOURCE_KEYS = frozenset({"upper", "lower", "found_upper", "where"})


def load_bounds(suite_id: str) -> Mapping[str, Bound]:
    """Return the reference bounds recorded for a suite, by instance name.

    Each bound holds the reference upper and lower bound, whether they meet, and, in
    :attr:`~dispatchatlas.benchmarks.suites.Bound.upper_sources` and
    :attr:`~dispatchatlas.benchmarks.suites.Bound.lower_sources`, every value that a source
    prints, with the table and page. The file is checked as a whole, and is refused when:

    - a table has a key that the format does not know, or names an instance that the suite
      does not have;
    - a source is not a key of the bibliography, or an instance has fewer than two sources;
    - a lower bound is above its upper bound;
    - ``optimal`` is not true exactly when the two bounds are equal;
    - the upper bound is not the smallest value that its sources print, or the lower bound is
      not the largest.

    Args:
        suite_id: Id of a suite, for example ``"taillard-job-shop"``.

    Returns:
        A read-only mapping from instance name to bound, in the order of the suite's
        instances. It holds the instances that have recorded bounds, which for
        ``"taillard-job-shop"`` are ta01 to ta50.

    Raises:
        ConfigError: If the suite is unknown, no bounds are recorded for it, or its file is
            refused.
    """
    return _read(suite_id)


def reference_values(suite_id: str, kind: BoundKind) -> Mapping[str, float]:
    """Return the reference upper or lower bound of each instance that has recorded bounds.

    The mapping serves as :attr:`~dispatchatlas.analysis.config.AnalysisConfig.references`,
    for the relative percentage deviation against the best known upper bound or the best known
    lower bound.

    Args:
        suite_id: Id of a suite, for example ``"taillard-job-shop"``.
        kind: ``"upper"`` for the best known upper bound, ``"lower"`` for the best known lower
            bound.

    Returns:
        A read-only mapping from instance name to value.

    Raises:
        ConfigError: If ``kind`` is neither, or :func:`load_bounds` raises.
    """
    chosen = _kind(kind)
    values = {
        name: bound.upper if chosen == "upper" else bound.lower
        for name, bound in load_bounds(suite_id).items()
    }
    return MappingProxyType({name: value for name, value in values.items() if value is not None})


def _kind(kind: object) -> BoundKind:
    if kind == "upper":
        return "upper"
    if kind == "lower":
        return "lower"
    raise ConfigError(f"kind must be 'upper' or 'lower', got {kind!r}")


@cache
def _read(suite_id: str) -> Mapping[str, Bound]:
    instances = suite(suite_id).instances
    folder = files("dispatchatlas.data").joinpath("bounds")
    path = folder.joinpath(f"{suite_id}.toml")
    if not path.is_file():
        recorded = sorted(
            item.name.removesuffix(".toml")
            for item in folder.iterdir()
            if item.name.endswith(".toml")
        )
        raise ConfigError(
            f"no reference bounds are recorded for suite {suite_id!r}; recorded: {recorded}"
        )
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(
            f"the bounds of suite {suite_id!r} are not valid TOML: {error}"
        ) from error
    parsed = _parse(suite_id, document, instances)
    return MappingProxyType({name: parsed[name] for name in instances if name in parsed})


def _parse(
    suite_id: str, document: Mapping[str, Any], instances: Collection[str]
) -> dict[str, Bound]:
    """Check a document of recorded bounds against the rules of the module and build its bounds."""
    known = bibliography()
    parsed: dict[str, Bound] = {}
    for name, table in document.items():
        if name not in instances or not isinstance(table, Mapping):
            raise ConfigError(
                f"bounds of suite {suite_id!r}: unknown key {name!r}; the keys of the file are "
                "the instances of the suite"
            )
        parsed[name] = _bound(f"bounds of suite {suite_id!r}, instance {name!r}", table, known)
    return parsed


def _bound(where: str, table: Mapping[str, Any], known: Mapping[str, object]) -> Bound:
    """Check the table of one instance and build its bound."""
    absent = [key for key in _INSTANCE_KEYS if key not in table]
    if absent:
        raise ConfigError(f"{where}: missing {absent}")
    upper = _value(where, "upper", table["upper"])
    lower = _value(where, "lower", table["lower"])
    optimal = table["optimal"]
    if not isinstance(optimal, bool):
        raise ConfigError(f"{where}: optimal must be true or false, got {optimal!r}")
    uppers: list[BoundSource] = []
    lowers: list[BoundSource] = []
    sources = [key for key in table if key not in _INSTANCE_KEYS]
    for source in sources:
        fields = table[source]
        if not isinstance(fields, Mapping):
            raise ConfigError(f"{where}: unknown key {source!r}")
        if source not in known:
            raise ConfigError(
                f"{where}: unknown source {source!r}; a source is a key of the bibliography"
            )
        listed_uppers, listed_lower = _printed(f"{where}, source {source!r}", source, fields)
        uppers.extend(listed_uppers)
        lowers.append(listed_lower)
    if len(sources) < 2:
        raise ConfigError(f"{where}: needs at least two sources, has {len(sources)}")
    if lower > upper:
        raise ConfigError(f"{where}: the lower bound {lower:g} is above the upper bound {upper:g}")
    if optimal != (upper == lower):
        raise ConfigError(
            f"{where}: optimal is {'true' if optimal else 'false'}, but the upper bound is "
            f"{upper:g} and the lower bound {lower:g}; an instance is optimal when they are equal"
        )
    smallest = min(item.value for item in uppers)
    if upper != smallest:
        raise ConfigError(
            f"{where}: the upper bound {upper:g} is not the smallest value its sources print, "
            f"{smallest:g}"
        )
    largest = max(item.value for item in lowers)
    if lower != largest:
        raise ConfigError(
            f"{where}: the lower bound {lower:g} is not the largest value its sources print, "
            f"{largest:g}"
        )
    return Bound(
        lower=lower,
        upper=upper,
        optimal=optimal,
        upper_sources=tuple(uppers),
        lower_sources=tuple(lowers),
    )


def _printed(
    where: str, source: str, fields: Mapping[str, Any]
) -> tuple[list[BoundSource], BoundSource]:
    """Check what one source prints for an instance.

    Returns:
        The upper bounds it prints, the listed one and then the newly found one if any, and the
        lower bound it prints.
    """
    unknown = sorted(set(fields) - _SOURCE_KEYS)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}")
    absent = [key for key in ("upper", "lower", "where") if key not in fields]
    if absent:
        raise ConfigError(f"{where}: missing {absent}")
    place = fields["where"]
    if not isinstance(place, str) or not place or place != place.strip():
        raise ConfigError(f"{where}: where must name the table and page, got {place!r}")
    listed = _value(where, "upper", fields["upper"])
    uppers = [BoundSource(source, listed, place)]
    if "found_upper" in fields:
        found = _value(where, "found_upper", fields["found_upper"])
        if found >= listed:
            raise ConfigError(
                f"{where}: found_upper {found:g} is not below the upper bound {listed:g} that "
                "the source lists"
            )
        uppers.append(BoundSource(source, found, place, found=True))
    return uppers, BoundSource(source, _value(where, "lower", fields["lower"]), place)


def _value(where: str, name: str, value: object) -> float:
    """Return a bound as a float, refusing anything but a positive finite number."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ConfigError(f"{where}: {name} must be a positive number, got {value!r}")
    return float(value)
