"""Helpers shared by the built-in benchmark families.

The families follow common conventions, which these helpers implement:

- **Parameters** are checked in ``__post_init__`` and refused with
  :class:`ValueError`, which the family registry reports as a
  :class:`~dispatchatlas.core.errors.ConfigError`: :func:`check_int`,
  :func:`check_int_range`, :func:`check_bool`, :func:`check_number`,
  :func:`check_optional_positive`, and :func:`check_objectives`.
- **Instances** are named ``<family>-<label>[-<tag>...]-<seed>`` and record
  their parameters and seed as metadata: :func:`instance_id`,
  :func:`instance_metadata`, and :func:`check_seed`.
- **Numbered entities** such as machines and jobs get ids zero-padded to one
  width, so that ids sort in index order: :func:`numbered_ids`.
- **Builders** check the data they are given and refuse it with
  :class:`~dispatchatlas.core.errors.ModelError`: :func:`check_matrix` and
  :func:`check_per_item`.
- **Integers computed from floats** tolerate rounding error:
  :func:`floor_tolerant`, :func:`ceil_tolerant`, and :func:`round_half_up`.
- **Readers onto a continuum platform** choose the nodes that may run tasks
  from the same ``tiers`` argument and refuse the same values:
  :func:`eligible_nodes`.

The module is private to :mod:`dispatchatlas.benchmarks`. Its names have no
leading underscore because the modules of the package import them.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import Resource, Scalar
from dispatchatlas.core.objectives import is_registered

if TYPE_CHECKING:
    from dispatchatlas.benchmarks.continuum.platform import Platform

__all__ = [
    "DUE_DATE_OBJECTIVES",
    "ROUNDING_SLACK",
    "ceil_tolerant",
    "check_bool",
    "check_int",
    "check_int_range",
    "check_matrix",
    "check_number",
    "check_objectives",
    "check_optional_positive",
    "check_per_item",
    "check_seed",
    "eligible_nodes",
    "floor_tolerant",
    "format_tag",
    "instance_id",
    "instance_metadata",
    "numbered_ids",
    "round_half_up",
]

DUE_DATE_OBJECTIVES = frozenset({"total_tardiness", "max_lateness", "tardy_tasks"})
"""Built-in objectives that are constant when no task has a due date."""

ROUNDING_SLACK = 1e-9
"""Slack that absorbs floating-point error before rounding: ``2.3 * 100`` is just below 230."""


# ------------------------------------------------------------------ instances


def check_seed(seed: object) -> int:
    """Return ``seed`` if it is a non-negative integer.

    Raises:
        ValueError: If ``seed`` is not a non-negative integer. A ``bool`` is
            refused too.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    return seed


def instance_id(generator: Any, label: str, seed: object, tags: Mapping[str, str]) -> str:
    """Return the problem id ``<family>-<label>[-<tag>...]-<seed>`` of one instance.

    Args:
        generator: A family dataclass instance whose ``info.id`` is the family id.
        label: Short description of the size, such as ``"20x5"``.
        seed: The seed of the instance.
        tags: Id prefix by parameter name. Every listed parameter that is not
            ``None`` and differs from its field default adds a tag, formatted
            by :func:`format_tag`, in field order. An empty prefix makes the
            tag the bare value, as suits the objectives.

    Raises:
        ValueError: If ``seed`` is not a non-negative integer.
    """
    number = check_seed(seed)
    parts = [generator.info.id, label]
    for item in dataclasses.fields(generator):
        prefix = tags.get(item.name)
        value = getattr(generator, item.name)
        if prefix is not None and value is not None and value != item.default:
            parts.append(format_tag(prefix, value))
    parts.append(str(number))
    return "-".join(parts)


def instance_metadata(generator: Any, seed: object) -> dict[str, Scalar]:
    """Return the metadata of one instance: its parameters and its seed.

    Every dataclass field that is not ``None`` is recorded under its name.
    Tuples and lists become comma-separated strings; other values must be
    scalars. The seed is recorded under ``"seed"``.

    Raises:
        ValueError: If ``seed`` is not a non-negative integer.
    """
    number = check_seed(seed)
    metadata: dict[str, Scalar] = {}
    for item in dataclasses.fields(generator):
        value = getattr(generator, item.name)
        if value is None:
            continue
        if isinstance(value, tuple | list):
            metadata[item.name] = ",".join(str(element) for element in value)
        else:
            metadata[item.name] = value
    metadata["seed"] = number
    return metadata


def format_tag(prefix: str, value: object) -> str:
    """Format one parameter value for a problem id.

    A ``bool`` gives the prefix alone when true and ``no<prefix>`` when false,
    a tuple joins its items with ``+``, and a float uses the shortest general
    format, so ``format_tag("due", 1.5)`` is ``"due1.5"``.
    """
    if isinstance(value, bool):
        return prefix if value else f"no{prefix}"
    if isinstance(value, tuple):
        return prefix + "+".join(str(item) for item in value)
    if isinstance(value, float):
        return f"{prefix}{value:g}"
    return f"{prefix}{value}"


# ------------------------------------------------------------------ parameters


def check_int(name: str, value: object, minimum: int = 1) -> int:
    """Return ``value`` if it is an integer of at least ``minimum``.

    Raises:
        ValueError: If ``value`` is not an integer, is a ``bool``, or is below
            ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def check_int_range(
    low_name: str, low: object, high_name: str, high: object, minimum: int = 1
) -> tuple[int, int]:
    """Return two integer parameters that form a range ``minimum <= low <= high``.

    Raises:
        ValueError: If either value fails :func:`check_int` or ``low`` exceeds
            ``high``.
    """
    first = check_int(low_name, low, minimum)
    last = check_int(high_name, high, minimum)
    if first > last:
        raise ValueError(f"{low_name} ({first}) cannot exceed {high_name} ({last})")
    return first, last


def check_bool(name: str, value: object) -> bool:
    """Return ``value`` if it is a ``bool``.

    Raises:
        ValueError: If ``value`` is not a ``bool``; ``0`` and ``1`` are refused.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false, got {value!r}")
    return value


def check_number(
    name: str,
    value: object,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
    positive: bool = False,
) -> float:
    """Return ``value`` as a float if it is a finite number within bounds.

    Args:
        name: Parameter name for the error message.
        value: The value to check. Integers are accepted and converted.
        minimum: Smallest accepted value.
        maximum: Largest accepted value, or ``None`` for no upper bound.
        positive: Refuse zero as well.

    Raises:
        ValueError: If ``value`` is not a number, is a ``bool``, is not
            finite, or lies outside the bounds.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if number < minimum or (maximum is not None and number > maximum):
        bounds = f"[{minimum:g}, {maximum:g}]" if maximum is not None else f"at least {minimum:g}"
        raise ValueError(f"{name} must be {bounds}, got {value!r}")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def check_optional_positive(name: str, value: object) -> float | None:
    """Return ``value`` as a positive float, or ``None`` when it is ``None``.

    Raises:
        ValueError: If ``value`` is neither ``None`` nor a positive finite
            number.
    """
    if value is None:
        return None
    return check_number(name, value, positive=True)


def check_objectives(
    value: object, *, due_dates: bool = True, due_parameter: str | None = None
) -> tuple[str, ...]:
    """Return objective names as a tuple after checking them.

    Args:
        value: One objective name or a sequence of them, primary first.
        due_dates: Whether the generated problems have due dates. When
            ``False``, the objectives of :data:`DUE_DATE_OBJECTIVES` are
            refused, since they would be constant.
        due_parameter: The parameter that adds due dates, named in the error
            message, or ``None`` when the family never generates them.

    Raises:
        ValueError: If the names are empty, repeated, or not registered, or
            name a due-date objective while the problems have no due dates.
    """
    names = (value,) if isinstance(value, str) else value
    if not isinstance(names, tuple | list) or not names:
        raise ValueError("objectives must be a non-empty list of objective names")
    result = tuple(names)
    for name in result:
        if not isinstance(name, str) or not is_registered(name):
            raise ValueError(f"unknown objective {name!r}")
    if len(set(result)) != len(result):
        raise ValueError(f"objectives repeat a name: {list(result)}")
    needing = sorted(DUE_DATE_OBJECTIVES.intersection(result))
    if needing and not due_dates:
        remedy = (
            "this family generates none"
            if due_parameter is None
            else f"set {due_parameter} to generate them"
        )
        raise ValueError(f"objectives {needing} need due dates; {remedy}")
    return result


def eligible_nodes(platform: Platform, tiers: object) -> tuple[Resource, ...]:
    """Return the nodes of a continuum platform that may run the tasks of a reader.

    A reader gives every task one mode per returned node, and each mode holds
    one core of its node.

    Args:
        platform: The platform.
        tiers: The tiers whose nodes may run tasks: a sequence of tier names
            other than a string, each tier once and each with a node. ``None``
            stands for every tier of the platform except ``"device"``.

    Returns:
        The nodes, tier by tier in the order of ``tiers``, or in the order of
        the platform for ``None``.

    Raises:
        ValueError: If ``tiers`` is a string, is not a sequence of strings,
            repeats a name, or names a tier without a node; if no node is
            left; or if a node has less than one core.
    """
    if tiers is None:
        chosen = [tier for tier in platform.tiers if tier != "device"]
        if not chosen:
            raise ValueError("the platform has no node in tiers other than 'device'")
    else:
        if (
            isinstance(tiers, str | bytes | bytearray)
            or not isinstance(tiers, Sequence)
            or not all(isinstance(tier, str) for tier in tiers)
        ):
            hint = "; write one tier as a list of one name" if isinstance(tiers, str) else ""
            raise ValueError(f"tiers must be a sequence of tier names, got {tiers!r}{hint}")
        chosen = list(tiers)
        if len(set(chosen)) != len(chosen):
            raise ValueError(f"tiers repeat a name: {chosen}")
        empty = [tier for tier in chosen if not platform.nodes(tier)]
        if empty or not chosen:
            raise ValueError(f"the platform has no node in tiers {empty}")
    nodes = tuple(node for tier in chosen for node in platform.nodes(tier))
    for node in nodes:
        if node.capacity < 1.0:
            raise ValueError(
                f"node {node.id!r} has capacity {node.capacity:g}, less than the one core "
                "a task holds"
            )
    return nodes


# ------------------------------------------------------------------ builder data


def numbered_ids(prefix: str, count: int, *, start: int = 0) -> list[str]:
    """Return ``count`` ids ``<prefix><index>``, indices zero-padded to one width.

    For example ``numbered_ids("m", 12)`` runs from ``"m00"`` to ``"m11"``, and
    ``numbered_ids("a", 3, start=1)`` gives ``["a1", "a2", "a3"]``.
    """
    width = len(str(start + max(count, 1) - 1))
    return [f"{prefix}{index:0{width}d}" for index in range(start, start + count)]


def check_matrix(
    values: Sequence[Sequence[float]],
    what: str,
    *,
    rows: str = "job",
    columns: str = "machine",
) -> list[list[float]]:
    """Return a non-empty rectangular matrix as a list of row lists.

    Args:
        values: The matrix, row by row.
        what: Description of the matrix for error messages.
        rows: What a row stands for, for error messages.
        columns: What a column stands for, for error messages.

    Raises:
        ModelError: If the matrix has no row or no column, or its rows differ
            in length.
    """
    matrix = [list(row) for row in values]
    if not matrix or not matrix[0]:
        raise ModelError(f"{what} need at least one {rows} and one {columns}")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ModelError(f"{what} must list {width} values for every {rows}")
    return matrix


def check_per_item(
    values: Sequence[float] | None, count: int, what: str, *, items: str = "jobs"
) -> list[float] | None:
    """Return one value per item as floats, or ``None`` when ``values`` is ``None``.

    Args:
        values: The values, or ``None`` when the data are absent.
        count: Number of items, and so of values.
        what: Name of the values for error messages, such as ``"due_dates"``.
        items: What the items are, for error messages.

    Raises:
        ModelError: If the number of values differs from ``count``.
    """
    if values is None:
        return None
    result = [float(value) for value in values]
    if len(result) != count:
        raise ModelError(f"{what} lists {len(result)} values for {count} {items}")
    return result


# ------------------------------------------------------------------ rounding


def floor_tolerant(value: float) -> int:
    """Return ``floor(value)``, reading a value just below an integer as that integer.

    The slack is :data:`ROUNDING_SLACK`, so ``floor_tolerant(2.3 * 100)`` is 230.
    """
    return math.floor(value + ROUNDING_SLACK)


def ceil_tolerant(value: float) -> int:
    """Return ``ceil(value)``, reading a value just above an integer as that integer.

    The slack is :data:`ROUNDING_SLACK`, so ``ceil_tolerant(0.07 * 100)`` is 7.
    """
    return math.ceil(value - ROUNDING_SLACK)


def round_half_up(value: float) -> int:
    """Round to the nearest integer, halves up: 2.5 gives 3 and -2.5 gives -2."""
    return math.floor(value + 0.5)
