"""Multiple-comparison corrections of a family of p-values.

:func:`adjust_p_values` returns adjusted p-values: a hypothesis is rejected
at family-wise level ``alpha`` when its adjusted p-value is at most ``alpha``.
With ``m`` hypotheses and the p-values sorted ascending as
``p_(1) <= ... <= p_(m)``:

- **Bonferroni** (Dunn 1961): ``min(1, m p)``.
- **Holm** (1979), step-down: the running maximum of
  ``min(1, (m - i + 1) p_(i))``.
- **Hochberg** (1988), step-up: from the largest p-value down, the running
  minimum of ``min(1, (m - i + 1) p_(i))``.
- **Shaffer** (1986), static step-down for all pairwise comparisons of ``k``
  groups: Holm's multiplier is replaced by ``t_i``, the largest number of
  hypotheses that can be true at once and is at most ``m - i + 1``. The
  possible numbers of true hypotheses are ``S(k)``, with ``S(0) = S(1) = {0}``
  and ``S(k)`` the union over ``j = 1..k`` of ``C(j, 2) + S(k - j)``.
- **Finner** (1993), step-down: the running maximum of
  ``min(1, 1 - (1 - p_(i))^(m / i))``.

Equal p-values are ordered by their keys, when given, and then by position,
so the result never depends on the incidental order of the input.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from dispatchatlas.core.errors import ConfigError

__all__ = ["CORRECTIONS", "Correction", "adjust_p_values", "shaffer_hypothesis_counts"]

type Correction = Literal["none", "bonferroni", "holm", "hochberg", "shaffer", "finner"]
"""A multiple-comparison correction."""

CORRECTIONS: tuple[Correction, ...] = (
    "none",
    "bonferroni",
    "holm",
    "hochberg",
    "shaffer",
    "finner",
)


def adjust_p_values(
    p_values: Sequence[float],
    method: Correction = "holm",
    *,
    keys: Sequence[tuple[str, str]] | None = None,
    groups: int | None = None,
) -> tuple[float, ...]:
    """Return the adjusted p-values of a family, in the order of the input.

    Args:
        p_values: One p-value per hypothesis, each in ``[0, 1]``.
        method: The correction.
        keys: For pairwise comparisons, the two group names of each
            hypothesis. They break ties between equal p-values and, for
            Shaffer's correction, give the number of groups.
        groups: For Shaffer's correction, the number of groups ``k``; the
            family must hold all ``k (k - 1) / 2`` pairs. Defaults to the
            number of distinct names in ``keys``.

    Raises:
        ConfigError: For an unknown method, a p-value outside ``[0, 1]``,
            keys that do not match the p-values, or a Shaffer family that is
            not complete.
    """
    values = [_probability(value) for value in p_values]
    if keys is not None and len(keys) != len(values):
        raise ConfigError("give one key per p-value")
    if method not in CORRECTIONS:
        raise ConfigError(f"unknown correction {method!r}; choose one of {list(CORRECTIONS)}")
    total = len(values)
    if method == "none" or total == 0:
        return tuple(values)
    if method == "bonferroni":
        return tuple(min(1.0, total * value) for value in values)
    order = sorted(range(total), key=lambda i: (values[i], keys[i] if keys else ("", ""), i))
    adjusted = [0.0] * total
    if method == "hochberg":
        running = 1.0
        for position, index in enumerate(reversed(order)):
            running = min(running, 1.0, (position + 1) * values[index])
            adjusted[index] = running
        return tuple(adjusted)
    if method == "shaffer":
        counts = shaffer_hypothesis_counts(_shaffer_groups(total, keys, groups))
        multipliers = [max(c for c in counts if c <= total - step) for step in range(total)]
    running = 0.0
    for step, index in enumerate(order):
        value = values[index]
        if method == "holm":
            candidate = min(1.0, (total - step) * value)
        elif method == "shaffer":
            candidate = min(1.0, multipliers[step] * value)
        else:
            candidate = min(1.0, 1.0 - (1.0 - value) ** (total / (step + 1)))
        running = max(running, candidate)
        adjusted[index] = running
    return tuple(adjusted)


def shaffer_hypothesis_counts(groups: int) -> tuple[int, ...]:
    """Return ``S(k)``: the numbers of pairwise hypotheses that can be true at once.

    Splitting ``k`` groups into blocks of equal groups makes the pairs within
    each block true, so ``S(k)`` is the union over the size ``j`` of one block
    of ``C(j, 2) + S(k - j)`` (Shaffer 1986).

    Raises:
        ConfigError: If ``groups`` is negative or not an integer.
    """
    if isinstance(groups, bool) or not isinstance(groups, int) or groups < 0:
        raise ConfigError(f"the number of groups must be a non-negative integer, got {groups!r}")
    feasible: list[set[int]] = [{0}, {0}]
    for k in range(2, groups + 1):
        found: set[int] = set()
        for j in range(1, k + 1):
            found.update(math.comb(j, 2) + rest for rest in feasible[k - j])
        feasible.append(found)
    return tuple(sorted(feasible[groups]))


def _shaffer_groups(total: int, keys: Sequence[tuple[str, str]] | None, groups: int | None) -> int:
    if groups is None:
        if keys is None:
            raise ConfigError("Shaffer's correction needs the number of groups or the pair keys")
        groups = len({name for pair in keys for name in pair})
    if total != math.comb(groups, 2):
        raise ConfigError(
            f"Shaffer's correction needs all {math.comb(groups, 2)} pairs of {groups} groups, "
            f"got {total}; use Holm's or Finner's correction for a partial family"
        )
    return groups


def _probability(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"p-values must be numbers, got {value!r}")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ConfigError(f"p-values must lie in [0, 1], got {value!r}")
    return number
