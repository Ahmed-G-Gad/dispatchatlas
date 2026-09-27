"""Numeric axes: ranges, tick marks, tick labels, and the mapping to pixels.

A linear axis extends its data range to the nearest multiples of a nice
step, a number ``1``, ``2``, or ``5`` times a power of ten, chosen so that
about six ticks cover the range. A logarithmic axis (base 10) extends its
range to the nearest numbers of the ``1, 2, 5, 10, 20, ...`` sequence and
places ticks at the numbers of that sequence in its range, at powers of ten
alone when the sequence would give more than nine ticks, and at nice linear
steps when it gives fewer than three. The labels of one axis share one style: the fewest
decimals that tell the ticks apart, or exponent form when some tick is very
large or very small. A label depends on the tick values only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from dispatchatlas.core.errors import ConfigError

__all__ = ["Axis", "format_tick", "log_axis", "numeric_axis", "tick_labels"]

_SEQUENCE = (1.0, 2.0, 5.0)
_MAX_LOG_TICKS = 8


@dataclass(frozen=True)
class Axis:
    """A numeric axis.

    Attributes:
        label: Axis title.
        low: Value at the start of the axis.
        high: Value at the end of the axis, above ``low``.
        log: Whether the scale is logarithmic; then ``low`` is positive.
        ticks: Tick positions within ``[low, high]``, ascending.
        labels: The label of each tick.
    """

    label: str
    low: float
    high: float
    log: bool = False
    ticks: tuple[float, ...] = ()
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (math.isfinite(self.low) and math.isfinite(self.high) and self.low < self.high):
            raise ConfigError(
                f"an axis needs finite ends with low < high, got {self.low, self.high}"
            )
        if self.log and self.low <= 0.0:
            raise ConfigError(f"a logarithmic axis needs a positive range, got {self.low}")
        if len(self.labels) != len(self.ticks):
            raise ConfigError("an axis needs one label per tick")

    def fraction(self, value: float) -> float:
        """Return where ``value`` lies along the axis, from 0 to 1, clamping outside values."""
        if math.isnan(value):
            raise ValueError("cannot place NaN on an axis")
        if self.log:
            if value <= self.low:
                return 0.0
            span = math.log10(self.high) - math.log10(self.low)
            share = (math.log10(value) - math.log10(self.low)) / span
        else:
            share = (value - self.low) / (self.high - self.low)
        return min(1.0, max(0.0, share))

    def position(self, value: float, start: float, end: float) -> float:
        """Return the pixel of ``value`` on an axis drawn from ``start`` to ``end``."""
        return start + self.fraction(value) * (end - start)


def numeric_axis(
    label: str,
    values: Iterable[float],
    *,
    low: float | None = None,
    high: float | None = None,
    log: bool = False,
    ticks: int = 6,
) -> Axis:
    """Return an axis covering the finite ``values``.

    Args:
        label: Axis title.
        values: Data values; infinite and NaN values are ignored.
        low: Fixed lower end. Without one, the range extends to a nice value.
        high: Fixed upper end.
        log: Use a logarithmic scale; non-positive values are then ignored.
        ticks: About how many ticks a linear axis gets.

    Returns:
        The axis. Without data, a linear axis spans 0 to 1 and a logarithmic
        one 1 to 10.
    """
    if log:
        return log_axis(label, values, low=low, high=high)
    finite = [value for value in values if math.isfinite(value)]
    data_low = min(finite, default=0.0 if low is None else low)
    data_high = max(finite, default=1.0 if high is None else high)
    if low is not None:
        data_low = low
    if high is not None:
        data_high = high
    if data_high <= data_low:
        margin = abs(data_low) * 0.1 or 1.0
        if low is None:
            data_low -= margin
        if high is None:
            data_high = data_low + (2.0 * margin if low is None else margin)
    step = _nice((_nice(data_high - data_low, rounded=False)) / max(1, ticks - 1), rounded=True)
    start = data_low if low is not None else math.floor(data_low / step + 1e-9) * step
    stop = data_high if high is not None else math.ceil(data_high / step - 1e-9) * step
    first = math.ceil(start / step - 1e-9)
    last = math.floor(stop / step + 1e-9)
    positions = tuple(_clean(index * step, step) for index in range(first, last + 1))
    return Axis(label, start, stop, False, positions, tick_labels(positions, step))


def log_axis(
    label: str,
    values: Iterable[float],
    *,
    low: float | None = None,
    high: float | None = None,
) -> Axis:
    """Return a logarithmic axis covering the positive finite ``values``.

    Without fixed ends, the range extends to the nearest numbers of the
    ``1, 2, 5`` sequence around the data.
    """
    positive = [value for value in values if math.isfinite(value) and value > 0.0]
    data_low = low if low is not None else _sequence_below(min(positive, default=1.0))
    data_high = high if high is not None else _sequence_above(max(positive, default=10.0))
    if data_low <= 0.0:
        raise ConfigError(f"a logarithmic axis needs a positive lower end, got {data_low}")
    if data_high <= data_low:
        data_high = _sequence_above(data_low * 1.000001)
    sequence = [
        value
        for value in _sequence_between(data_low, data_high)
        if data_low * (1 - 1e-9) <= value <= data_high * (1 + 1e-9)
    ]
    if len(sequence) > _MAX_LOG_TICKS + 1:
        powers = [value for value in sequence if _is_power_of_ten(value)]
        every = max(1, math.ceil(len(powers) / _MAX_LOG_TICKS))
        sequence = powers[::every]
    if len(sequence) < 3:
        inner = numeric_axis(label, (), low=data_low, high=data_high)
        return Axis(label, data_low, data_high, True, inner.ticks, inner.labels)
    ticks = tuple(sequence)
    return Axis(label, data_low, data_high, True, ticks, tick_labels(ticks))


def _is_power_of_ten(value: float) -> bool:
    exponent = math.log10(value)
    return abs(exponent - round(exponent)) < 1e-9


def tick_labels(ticks: Sequence[float], step: float | None = None) -> tuple[str, ...]:
    """Return the labels of an axis's ticks, all in one style.

    Ticks are written in exponent form, such as ``2e-5`` or ``1.5e6``, when
    any of them is at least ``1e5`` or, other than zero, below ``1e-4``;
    otherwise as plain decimals.

    Args:
        ticks: The tick values.
        step: Distance between neighbouring ticks of a linear axis, which
            fixes the number of decimals; without it each value is written
            with the digits it needs.
    """
    magnitudes = [abs(tick) for tick in ticks if tick != 0.0]
    exponent = bool(magnitudes) and (max(magnitudes) >= 1e5 or min(magnitudes) < 1e-4)
    return tuple(format_tick(tick, step, exponent=exponent) for tick in ticks)


def format_tick(value: float, step: float | None = None, *, exponent: bool = False) -> str:
    """Return the label of one tick.

    Args:
        value: The tick value.
        step: Distance between neighbouring ticks of a linear axis, which
            fixes the number of decimals; without it the value is written
            with the digits it needs, up to six decimals.
        exponent: Write the value in exponent form, such as ``1.5e6``.
    """
    if exponent:
        if value == 0.0:
            return "0"
        digits = 6 if step is None else max(0, _exponent(abs(value)) - _exponent(step))
        mantissa, power = f"{value:.{min(digits, 6)}e}".split("e")
        if "." in mantissa:
            mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa}e{int(power)}"
    if step is None:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return "0" if text in ("", "-0") else text
    decimals = max(0, -_exponent(step))
    return f"{value + 0.0:.{decimals}f}"


def _exponent(value: float) -> int:
    return math.floor(math.log10(value) + 1e-9)


def _nice(value: float, *, rounded: bool) -> float:
    """Return the nice number ``1``, ``2``, or ``5`` times a power of ten nearest ``value``."""
    if value <= 0.0 or not math.isfinite(value):
        return 1.0
    exponent = _exponent(value)
    fraction = value / 10.0**exponent
    if rounded:
        nice = 1.0 if fraction < 1.5 else 2.0 if fraction < 3.0 else 5.0 if fraction < 7.0 else 10.0
    else:
        nice = (
            1.0 if fraction <= 1.0 else 2.0 if fraction <= 2.0 else 5.0 if fraction <= 5.0 else 10.0
        )
    return nice * 10.0**exponent


def _clean(value: float, step: float) -> float:
    """Remove the floating-point noise that multiplying the step leaves in a tick."""
    return round(value, max(0, -_exponent(step)) + 3) + 0.0


def _sequence_between(low: float, high: float) -> list[float]:
    values: list[float] = []
    for exponent in range(_exponent(low) - 1, _exponent(high) + 2):
        values.extend(multiple * 10.0**exponent for multiple in _SEQUENCE)
    return sorted({float(f"{value:.12g}") for value in values})


def _sequence_below(value: float) -> float:
    return max(v for v in _sequence_between(value, value) if v <= value * (1 + 1e-9))


def _sequence_above(value: float) -> float:
    return min(v for v in _sequence_between(value, value) if v >= value * (1 - 1e-9))
