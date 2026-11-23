"""Parameter spaces: the parameters of a solver and the configurations drawn from them.

A :class:`ParameterSpace` lists parameters of the four kinds of the irace package
(López-Ibáñez et al. 2016, p. 56): a :class:`Real` parameter takes a number in a
closed range, an :class:`Integer` parameter a whole number in a closed range, an
:class:`Ordinal` parameter one value of an ordered list, and a :class:`Categorical`
parameter one value of an unordered list. A configuration gives every parameter a
value, as a read-only mapping from the parameter's name to it.

The sampling distributions are those of iterated racing (López-Ibáñez et al. 2016,
p. 46 and footnote 2 on p. 48):

- the first configurations are uniform over the space (:meth:`ParameterSpace.sample`);
- later configurations are drawn around a parent configuration
  (:meth:`ParameterSpace.sample_around`). A numerical parameter is drawn from a
  normal distribution truncated to its range, with the parent's value as its mean
  and the standard deviation that a :class:`SamplingModel` holds. An ordinal
  parameter is handled as the integer index of its value. A categorical parameter
  is drawn from the discrete distribution that the model holds;
- an integer is drawn from the normal distribution truncated to
  ``[low, high + 1]`` and centred half a unit above the parent, and the draw is
  rounded down, so that the extreme integers keep the probability of the interior
  ones instead of half of it.

Every parameter draws exactly one uniform number from the generator it is given,
whatever its value, so the stream a configuration consumes does not depend on the
outcome of its draws.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from dispatchatlas.analysis.stats.distributions import normal_cdf, normal_ppf
from dispatchatlas.core._frozen import FrozenMap
from dispatchatlas.core.errors import ConfigError

__all__ = [
    "DIGITS",
    "Categorical",
    "Configuration",
    "Integer",
    "Ordinal",
    "Parameter",
    "ParameterSpace",
    "Real",
    "SamplingModel",
    "Value",
]

type Value = int | float | str
"""The value of one parameter: a number, or one of the values of a list."""

type Configuration = Mapping[str, Value]
"""A value for every parameter of a space, by the parameter's name.

The configurations the package returns are read-only, hashable, and picklable.
"""

DIGITS = 4
"""Decimal places of a real parameter's values, and the precision of the distance between
two values: the default of irace (López-Ibáñez et al. 2016, p. 56)."""

_BELOW_ONE = math.nextafter(1.0, 0.0)
_TINY = 5e-324


@dataclass(frozen=True, slots=True)
class Real:
    """A real parameter: a number in the closed range ``[low, high]``.

    Its values are rounded to ``digits`` decimal places and kept inside the range
    (López-Ibáñez et al. 2016, p. 56).

    Attributes:
        name: The name of the parameter, the key of its value in a configuration.
        low: The smallest value.
        high: The largest value, above ``low``.
        digits: Decimal places of a value, from 1 to 15. The default is that of
            irace, :data:`DIGITS`.
    """

    name: str
    low: float
    high: float
    digits: int = DIGITS

    def __post_init__(self) -> None:
        _check_name(self.name)
        low = _finite(self.low, f"the low end of {self.name!r}")
        high = _finite(self.high, f"the high end of {self.name!r}")
        if not low < high:
            raise ConfigError(f"parameter {self.name!r} needs low < high, got {low} and {high}")
        if isinstance(self.digits, bool) or not isinstance(self.digits, int):
            raise ConfigError(f"digits of {self.name!r} must be an integer, got {self.digits!r}")
        if not 1 <= self.digits <= 15:
            raise ConfigError(f"digits of {self.name!r} must lie in 1..15, got {self.digits}")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def spread(self) -> float:
        """The initial standard deviation of a draw around a parent, ``(high - low) / 2``.

        This is the value of López-Ibáñez et al. (2016, p. 47).
        """
        return (self.high - self.low) / 2.0

    def sample(self, rng: np.random.Generator) -> float:
        """Return a value uniform over ``[low, high]``, rounded to ``digits`` places."""
        return self._round(self.low + rng.random() * (self.high - self.low))

    def sample_around(self, rng: np.random.Generator, centre: float, spread: float) -> float:
        """Return a value from the normal distribution truncated to the range.

        Args:
            rng: The generator, from which one uniform number is drawn.
            centre: The mean of the normal distribution, the parent's value.
            spread: Its standard deviation, positive.

        Returns:
            The value, rounded to ``digits`` places.

        Raises:
            ConfigError: If ``spread`` is not positive and finite.
        """
        _check_spread(spread)
        return self._round(_truncated_normal(rng.random(), centre, spread, self.low, self.high))

    def distance(self, first: float, second: float) -> float:
        """Return the absolute difference of two values over the range, or 0 below ``10**-digits``.

        This is the distance of the soft restart (López-Ibáñez et al. 2016, p. 48).
        """
        gap = abs(first - second) / (self.high - self.low)
        return gap if gap > 10.0**-self.digits else 0.0

    def _round(self, value: float) -> float:
        return min(max(round(value, self.digits), self.low), self.high)


@dataclass(frozen=True, slots=True)
class Integer:
    """An integer parameter: a whole number in the closed range ``[low, high]``.

    Attributes:
        name: The name of the parameter, the key of its value in a configuration.
        low: The smallest value.
        high: The largest value, above ``low``.
    """

    name: str
    low: int
    high: int

    def __post_init__(self) -> None:
        _check_name(self.name)
        for label, value in (("low", self.low), ("high", self.high)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"the {label} end of {self.name!r} must be an integer: {value!r}")
        if not self.low < self.high:
            raise ConfigError(
                f"parameter {self.name!r} needs low < high, got {self.low} and {self.high}"
            )

    @property
    def spread(self) -> float:
        """The initial standard deviation of a draw around a parent, ``(high - low) / 2``.

        This is the value of López-Ibáñez et al. (2016, p. 47).
        """
        return (self.high - self.low) / 2.0

    def sample(self, rng: np.random.Generator) -> int:
        """Return a whole number uniform over ``[low, high]``."""
        count = self.high - self.low + 1
        return self.low + min(int(rng.random() * count), count - 1)

    def sample_around(self, rng: np.random.Generator, centre: float, spread: float) -> int:
        """Return a whole number from the truncated normal distribution, rounded down.

        The draw comes from the normal distribution truncated to ``[low, high + 1]``
        and centred at ``centre + 0.5``, and its integer part is the value. Rounding
        a draw from the range ``[low, high]`` to the nearest integer would give each
        extreme integer half the interval of an interior one
        (López-Ibáñez et al. 2016, footnote 2, p. 48).

        Args:
            rng: The generator, from which one uniform number is drawn.
            centre: The parent's value.
            spread: The standard deviation of the normal distribution, positive.

        Raises:
            ConfigError: If ``spread`` is not positive and finite.
        """
        _check_spread(spread)
        draw = _truncated_normal(rng.random(), centre + 0.5, spread, self.low, self.high + 1.0)
        return min(math.floor(draw), self.high)

    def distance(self, first: float, second: float) -> float:
        """Return the absolute difference of two values over the range, or 0 below ``1e-4``.

        This is the distance of the soft restart (López-Ibáñez et al. 2016, p. 48),
        with the threshold of the default number of digits.
        """
        gap = abs(first - second) / (self.high - self.low)
        return gap if gap > 10.0**-DIGITS else 0.0


@dataclass(frozen=True, slots=True)
class Ordinal:
    """An ordinal parameter: one value of an ordered list.

    The sampling treats the parameter as the integer index of its value in the
    list (López-Ibáñez et al. 2016, p. 46), so a draw around a parent tends to land
    on a neighbour of the parent's value.

    Attributes:
        name: The name of the parameter, the key of its value in a configuration.
        levels: The values from the smallest to the largest: at least two, all different.
    """

    name: str
    levels: Sequence[Value]

    def __post_init__(self) -> None:
        _check_name(self.name)
        object.__setattr__(self, "levels", _levels(self.name, self.levels))

    @property
    def spread(self) -> float:
        """The initial standard deviation of a draw around a parent, ``(len(levels) - 1) / 2``.

        This is the value of López-Ibáñez et al. (2016, p. 47) with the range of the
        indices, ``|X| - 1`` (p. 48).
        """
        return (len(self.levels) - 1) / 2.0

    def sample(self, rng: np.random.Generator) -> Value:
        """Return one value uniformly."""
        count = len(self.levels)
        return self.levels[min(int(rng.random() * count), count - 1)]

    def sample_around(self, rng: np.random.Generator, centre: Value, spread: float) -> Value:
        """Return the value whose index is drawn as an :class:`Integer` is.

        Args:
            rng: The generator, from which one uniform number is drawn.
            centre: The parent's value, one of the levels.
            spread: The standard deviation, in indices, positive.

        Raises:
            ConfigError: If ``centre`` is not a level or ``spread`` is not positive and finite.
        """
        _check_spread(spread)
        count = len(self.levels)
        draw = _truncated_normal(rng.random(), self.index(centre) + 0.5, spread, 0.0, float(count))
        return self.levels[min(math.floor(draw), count - 1)]

    def index(self, value: Value) -> int:
        """Return the position of ``value`` in the list, from 0.

        Raises:
            ConfigError: If ``value`` is not a level.
        """
        return _index(self.name, self.levels, value)

    def distance(self, first: Value, second: Value) -> float:
        """Return 0 for equal values and 1 otherwise (López-Ibáñez et al. 2016, p. 48)."""
        return 0.0 if first == second else 1.0


@dataclass(frozen=True, slots=True)
class Categorical:
    """A categorical parameter: one value of an unordered list.

    Attributes:
        name: The name of the parameter, the key of its value in a configuration.
        levels: The values: at least two, all different.
    """

    name: str
    levels: Sequence[Value]

    def __post_init__(self) -> None:
        _check_name(self.name)
        object.__setattr__(self, "levels", _levels(self.name, self.levels))

    def sample(self, rng: np.random.Generator) -> Value:
        """Return one value uniformly."""
        count = len(self.levels)
        return self.levels[min(int(rng.random() * count), count - 1)]

    def sample_weighted(self, rng: np.random.Generator, weights: Sequence[float]) -> Value:
        """Return one value with probability proportional to its weight.

        Args:
            rng: The generator, from which one uniform number is drawn.
            weights: One weight per level, in the order of ``levels``: not negative,
                and not all zero.

        Raises:
            ConfigError: If the weights do not fit the levels.
        """
        if len(weights) != len(self.levels):
            raise ConfigError(
                f"{self.name!r} has {len(self.levels)} levels but {len(weights)} weights"
            )
        if any(not (math.isfinite(w) and w >= 0.0) for w in weights) or not any(weights):
            raise ConfigError(
                f"the weights of {self.name!r} must be finite, not negative, not all 0"
            )
        threshold = rng.random() * math.fsum(weights)
        total = 0.0
        for level, weight in zip(self.levels, weights, strict=True):
            total += weight
            if threshold < total:
                return level
        return self.levels[-1]

    def index(self, value: Value) -> int:
        """Return the position of ``value`` in the list, from 0.

        Raises:
            ConfigError: If ``value`` is not a level.
        """
        return _index(self.name, self.levels, value)

    def distance(self, first: Value, second: Value) -> float:
        """Return 0 for equal values and 1 otherwise (López-Ibáñez et al. 2016, p. 48)."""
        return 0.0 if first == second else 1.0


type Parameter = Real | Integer | Ordinal | Categorical
"""One parameter of a space."""


@dataclass(frozen=True, slots=True)
class SamplingModel:
    """How the values around a parent are drawn: the distributions iterated racing updates.

    Attributes:
        spreads: The standard deviation of every numerical parameter, by name: a
            :class:`Real` and an :class:`Integer` in their own units, an
            :class:`Ordinal` in indices.
        weights: The probability of every level of every categorical parameter, by
            name, in the order of its levels.
    """

    spreads: Mapping[str, float]
    weights: Mapping[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "spreads", FrozenMap(self.spreads))
        object.__setattr__(self, "weights", FrozenMap(self.weights))


@dataclass(frozen=True, slots=True)
class ParameterSpace:
    """The parameters of a solver, each with its kind and domain.

    Attributes:
        parameters: At least one parameter, with different names, stored as a tuple.
            The order is the order in which the values of a configuration are drawn.
    """

    parameters: Sequence[Parameter]

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        if not parameters:
            raise ConfigError("a parameter space needs at least one parameter")
        for parameter in parameters:
            if not isinstance(parameter, (Real, Integer, Ordinal, Categorical)):
                raise ConfigError(f"not a parameter: {parameter!r}")
        names = [parameter.name for parameter in parameters]
        if len(set(names)) != len(names):
            raise ConfigError(f"parameter names must differ, got {names}")
        object.__setattr__(self, "parameters", parameters)

    def __len__(self) -> int:
        return len(self.parameters)

    @property
    def names(self) -> tuple[str, ...]:
        """The names of the parameters, in order."""
        return tuple(parameter.name for parameter in self.parameters)

    def sample(self, rng: np.random.Generator, count: int = 1) -> tuple[Configuration, ...]:
        """Return ``count`` configurations drawn uniformly over the space.

        Each parameter takes one draw, in the order of ``parameters``, configuration
        after configuration.

        Args:
            rng: The generator.
            count: How many configurations, not negative.

        Raises:
            ConfigError: If ``count`` is negative or not an integer.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ConfigError(f"count must be a non-negative integer, got {count!r}")
        return tuple(
            FrozenMap({parameter.name: parameter.sample(rng) for parameter in self.parameters})
            for _ in range(count)
        )

    def initial_model(self) -> SamplingModel:
        """Return the model of the first iteration: the initial spreads and uniform weights."""
        spreads: dict[str, float] = {}
        weights: dict[str, tuple[float, ...]] = {}
        for parameter in self.parameters:
            if isinstance(parameter, Categorical):
                count = len(parameter.levels)
                weights[parameter.name] = (1.0 / count,) * count
            else:
                spreads[parameter.name] = parameter.spread
        return SamplingModel(spreads, weights)

    def sample_around(
        self, rng: np.random.Generator, parent: Configuration, model: SamplingModel
    ) -> Configuration:
        """Return a configuration drawn around ``parent`` with the distributions of ``model``.

        A numerical parameter takes the parent's value as its centre and the model's
        spread, and a categorical parameter the model's probabilities. Each parameter
        takes one draw, in the order of ``parameters``.

        Raises:
            ConfigError: If the parent lacks a parameter or holds a value that is not
                of its kind, or the model lacks the distribution of a parameter.
        """
        values: dict[str, Value] = {}
        for parameter in self.parameters:
            name = parameter.name
            if name not in parent:
                raise ConfigError(f"the parent configuration lacks the parameter {name!r}")
            if isinstance(parameter, Categorical):
                values[name] = parameter.sample_weighted(rng, _entry(model.weights, name))
            elif isinstance(parameter, Ordinal):
                values[name] = parameter.sample_around(
                    rng, parent[name], _entry(model.spreads, name)
                )
            else:
                values[name] = parameter.sample_around(
                    rng, _number(name, parent[name]), _entry(model.spreads, name)
                )
        return FrozenMap(values)

    def distance(self, first: Configuration, second: Configuration) -> float:
        """Return the largest distance between the values of two configurations.

        A parameter's distance is that of its kind: the normalised absolute
        difference for a numerical parameter, and 0 or 1 otherwise
        (López-Ibáñez et al. 2016, p. 48). The distance is 0 for configurations
        that the soft restart treats as the same.

        Raises:
            ConfigError: If a configuration lacks a parameter.
        """
        largest = 0.0
        for parameter in self.parameters:
            name = parameter.name
            if name not in first or name not in second:
                raise ConfigError(f"a configuration lacks the parameter {name!r}")
            if isinstance(parameter, (Real, Integer)):
                gap = parameter.distance(_number(name, first[name]), _number(name, second[name]))
            else:
                gap = parameter.distance(first[name], second[name])
            largest = max(largest, gap)
        return largest


def _truncated_normal(u: float, mean: float, spread: float, low: float, high: float) -> float:
    """Return the quantile ``u`` of the normal distribution truncated to ``[low, high]``.

    The quantile is that of the normal distribution at the probability ``u`` of the
    way from its distribution function at ``low`` to its distribution function at
    ``high``, the inverse-transform draw of a truncated distribution. The mean lies
    inside the range, so the two distribution functions differ, and the probability
    is held inside the open unit interval against rounding at its ends.
    """
    lower = normal_cdf((low - mean) / spread)
    upper = normal_cdf((high - mean) / spread)
    probability = min(max(lower + u * (upper - lower), _TINY), _BELOW_ONE)
    return min(max(mean + spread * normal_ppf(probability), low), high)


def _check_name(name: object) -> None:
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise ConfigError(f"a parameter name must be a non-empty string without margins: {name!r}")


def _finite(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"{what} must be a finite number, got {value!r}")
    return float(value)


def _check_spread(spread: float) -> None:
    if isinstance(spread, bool) or not (math.isfinite(spread) and spread > 0.0):
        raise ConfigError(f"the spread must be positive and finite, got {spread!r}")


def _levels(name: str, levels: object) -> tuple[Value, ...]:
    if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
        raise ConfigError(f"the levels of {name!r} must be a sequence, got {levels!r}")
    values = tuple(levels)
    if len(values) < 2:
        raise ConfigError(f"{name!r} needs at least two levels, got {len(values)}")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ConfigError(f"a level of {name!r} must be a number or a string, got {value!r}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigError(f"a level of {name!r} must be finite, got {value!r}")
        if isinstance(value, str) and not value:
            raise ConfigError(f"a level of {name!r} must not be an empty string")
    for position, value in enumerate(values):
        if value in values[:position]:
            raise ConfigError(f"the levels of {name!r} must differ, got {value!r} twice")
    return values


def _index(name: str, levels: Sequence[Value], value: Value) -> int:
    for position, level in enumerate(levels):
        if level == value:
            return position
    raise ConfigError(f"{value!r} is not a level of {name!r}")


def _number(name: str, value: Value) -> float:
    if isinstance(value, (bool, str)):
        raise ConfigError(f"parameter {name!r} needs a number, got {value!r}")
    return float(value)


def _entry[T](table: Mapping[str, T], name: str) -> T:
    if name not in table:
        raise ConfigError(f"the sampling model lacks the parameter {name!r}")
    return table[name]
