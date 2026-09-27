"""Chart layouts shared by the figure families.

Two layouts cover every family except the critical-difference diagram:

- :func:`xy_chart` draws series against two numeric axes, as lines, step
  functions, scatter points, and translucent bands;
- :func:`row_chart` draws one row per category, such as a solver or a
  machine, against a numeric horizontal axis, with box plots or bars.

Both put the title on top, the plot with its ticks, grid, and axis titles in
the middle, and the legend and notes below the plot. The legend wraps onto
as many rows as it needs and the canvas grows to hold it, so no element is
ever placed outside the canvas, however many series a chart has. Series
lines are thinned in pixel space before drawing: points that fall within
half a pixel of each other are merged, which leaves the drawing unchanged
and bounds the size of the output.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from dispatchatlas.analysis.figures.axes import Axis
from dispatchatlas.analysis.figures.drawing import (
    Figure,
    Line,
    Marker,
    Point,
    Polygon,
    Polyline,
    Rect,
    Shape,
    Stroke,
    Text,
    text_width,
    truncate,
    wrap,
)
from dispatchatlas.analysis.figures.theme import SeriesStyle, Theme

__all__ = [
    "Bar",
    "BoxStats",
    "Guide",
    "LegendEntry",
    "Row",
    "Series",
    "box_stats",
    "draw_notes",
    "draw_title",
    "label_for",
    "placeholder",
    "row_chart",
    "xy_chart",
]

PAD = 12.0
"""Blank margin around the canvas, in pixels."""

TICK = 4.0
"""Length of a tick mark."""

LEGEND_ROW = 18.0
"""Height of one legend row."""

MAX_MARKERS = 10
"""Markers drawn along one series line."""

_SWATCH = 22.0
_BAND_OPACITY = 0.12
_BOX_OPACITY = 0.35


@dataclass(frozen=True)
class Series:
    """One series of an x-y chart, in data coordinates.

    Attributes:
        label: Legend text.
        points: The points, ascending in ``x`` for lines and steps.
        style: Colour, marker, and dash.
        kind: ``"line"`` joins the points, ``"step"`` draws a
            right-continuous step function that holds each value until the
            next point, and ``"points"`` draws markers only.
        markers: Draw :data:`MAX_MARKERS` markers along a line or step
            function, at evenly spaced positions.
        band: ``(x, low, high)`` triples drawn as a translucent band,
            stepped like the line when ``kind`` is ``"step"``.
        until: For a step function, the ``x`` up to which the last value holds.
        legend: Give the series a legend entry.
    """

    label: str
    points: tuple[Point, ...]
    style: SeriesStyle
    kind: Literal["line", "step", "points"] = "line"
    markers: bool = True
    band: tuple[tuple[float, float, float], ...] = ()
    until: float | None = None
    legend: bool = True


@dataclass(frozen=True, slots=True)
class LegendEntry:
    """One legend item.

    Attributes:
        label: Text.
        style: Colour, marker, and dash of the swatch.
        kind: ``"line"`` for a line with its marker, ``"box"`` for a translucent
            square like a box plot's, ``"bar"`` for an opaque square like a bar's.
    """

    label: str
    style: SeriesStyle
    kind: Literal["line", "box", "bar"] = "line"


@dataclass(frozen=True, slots=True)
class Guide:
    """A labelled vertical line across the plot, such as a makespan.

    Attributes:
        x: Position in data coordinates.
        label: Text written above the plot at the line.
    """

    x: float
    label: str


@dataclass(frozen=True, slots=True)
class BoxStats:
    """The five numbers a box plot draws.

    Attributes:
        minimum: Smallest value, the end of the lower whisker.
        lower: First quartile, the lower edge of the box.
        median: Median, the line inside the box.
        upper: Third quartile, the upper edge of the box.
        maximum: Largest value, the end of the upper whisker.
        count: Number of values.
    """

    minimum: float
    lower: float
    median: float
    upper: float
    maximum: float
    count: int


def box_stats(values: Sequence[float]) -> BoxStats:
    """Return the five-number summary of at least one value.

    Quartiles follow the inclusive method of :func:`statistics.quantiles`,
    type 7 in the taxonomy of Hyndman and Fan (1996); a single value is its
    own quartiles.

    Raises:
        ValueError: If ``values`` is empty.
    """
    ordered = sorted(values)
    if not ordered:
        raise ValueError("a box plot needs at least one value")
    if len(ordered) == 1:
        lower = middle = upper = ordered[0]
    else:
        lower, middle, upper = statistics.quantiles(ordered, n=4, method="inclusive")
    return BoxStats(ordered[0], lower, middle, upper, ordered[-1], len(ordered))


@dataclass(frozen=True)
class Bar:
    """A horizontal bar in a row chart.

    Attributes:
        row: Index of the row.
        start: Left end in data coordinates.
        end: Right end.
        color: ``#RRGGBB`` fill.
        text: Text written inside the bar when it fits.
        tooltip: Text a viewer shows on hovering, SVG only.
    """

    row: int
    start: float
    end: float
    color: str
    text: str = ""
    tooltip: str = ""


@dataclass(frozen=True)
class Row:
    """One category of a row chart.

    Attributes:
        label: Text left of the row.
        box: Box-plot statistics drawn in the row, if any.
        style: Colour of the box.
        note: Text right of the box, such as a count of failed runs.
    """

    label: str
    box: BoxStats | None = None
    style: SeriesStyle | None = None
    note: str = ""


@dataclass(frozen=True)
class _Frame:
    left: float
    top: float
    right: float
    bottom: float

    def x(self, axis: Axis, value: float) -> float:
        return axis.position(value, self.left, self.right)

    def y(self, axis: Axis, value: float) -> float:
        return axis.position(value, self.bottom, self.top)


def xy_chart(
    name: str,
    title: str,
    description: str,
    *,
    x: Axis,
    y: Axis,
    series: Sequence[Series],
    notes: Sequence[str] = (),
    theme: Theme,
) -> Figure:
    """Draw series against two numeric axes.

    Args:
        name: File stem of the figure.
        title: Title drawn at the top.
        description: Text alternative of the figure.
        x: Horizontal axis.
        y: Vertical axis.
        series: The series, drawn in order, each with a legend entry unless
            it asks for none.
        notes: Lines of text under the legend, such as caveats.
        theme: The appearance.

    Returns:
        The figure.
    """
    shapes: list[Shape] = []
    top = draw_title(shapes, title, theme)
    label_width = max((text_width(label, theme.tick_size) for label in y.labels), default=0.0)
    last_label = x.labels[-1] if x.labels else ""
    left = PAD + theme.label_size + 8.0 + label_width + TICK + 4.0
    right = theme.width - PAD - max(8.0, text_width(last_label, theme.tick_size) / 2.0)
    plot_top = top + theme.tick_size * 0.7
    frame = _Frame(left, plot_top, right, plot_top + theme.plot_height)
    _y_axis(shapes, frame, y, theme)
    bottom = _x_axis(shapes, frame, x, theme)
    for item in series:
        if item.band:
            shapes.append(_band(frame, x, y, item))
    for slot, item in enumerate(series):
        shapes.extend(_series(frame, x, y, item, theme, (slot, len(series))))
    shapes.append(_plot_border(frame, theme))
    legend = [LegendEntry(item.label, item.style) for item in series if item.legend]
    bottom = _legend(shapes, legend, bottom + 10.0, frame.left, theme)
    bottom = draw_notes(shapes, notes, bottom, theme)
    return Figure(name, title, description, theme.width, bottom + PAD, tuple(shapes), theme)


def row_chart(
    name: str,
    title: str,
    description: str,
    *,
    x: Axis,
    rows: Sequence[Row],
    bars: Sequence[Bar] = (),
    row_height: float = 26.0,
    legend: Sequence[LegendEntry] = (),
    guides: Sequence[Guide] = (),
    notes: Sequence[str] = (),
    theme: Theme,
) -> Figure:
    """Draw one row per category against a numeric horizontal axis.

    Args:
        name: File stem of the figure.
        title: Title drawn at the top.
        description: Text alternative of the figure.
        x: Horizontal axis.
        rows: The categories, drawn top to bottom, with optional box plots.
        bars: Horizontal bars placed in the rows.
        row_height: Height of a row, in pixels.
        legend: Legend entries, drawn under the axis.
        guides: Labelled vertical lines.
        notes: Lines of text under the legend.
        theme: The appearance.

    Returns:
        The figure.
    """
    shapes: list[Shape] = []
    top = draw_title(shapes, title, theme) + (theme.tick_size + 6.0 if guides else 0.0)
    size = min(theme.tick_size, max(6.0, row_height * 0.75))
    widest = max((text_width(row.label, size) for row in rows), default=0.0)
    label_width = min(widest, theme.width * 0.3)
    notes_width = max((text_width(row.note, theme.tick_size) for row in rows), default=0.0)
    left = PAD + label_width + 8.0
    right = theme.width - PAD - max(8.0, notes_width + (8.0 if notes_width else 0.0))
    right = max(right, left + theme.width * 0.3)
    frame = _Frame(left, top + 4.0, right, top + 4.0 + row_height * max(1, len(rows)))
    every = max(1, math.ceil((size + 1.0) / row_height))
    for index, row in enumerate(rows):
        if index % every:
            continue
        centre = frame.top + (index + 0.5) * row_height
        shapes.append(
            Text(
                left - 8.0,
                centre + size * 0.35,
                truncate(row.label, size, label_width),
                size,
                theme.foreground,
                anchor="end",
            )
        )
    bottom = _x_axis(shapes, frame, x, theme)
    for index, row in enumerate(rows):
        if row.box is not None and row.style is not None:
            shapes.extend(_box(frame, x, index, row_height, row.box, row.style))
        if row.note:
            end = frame.x(x, row.box.maximum) if row.box is not None else frame.left
            start = min(end + 6.0, frame.right + 4.0)
            shapes.append(
                Text(
                    start,
                    frame.top + (index + 0.5) * row_height + theme.tick_size * 0.35,
                    truncate(row.note, theme.tick_size, theme.width - PAD - start),
                    theme.tick_size,
                    theme.muted,
                )
            )
    for bar in bars:
        shapes.extend(_bar(frame, x, bar, row_height, theme))
    for guide in guides:
        position = frame.x(x, guide.x)
        dashed = Stroke(theme.foreground, 1.0, (4.0, 3.0))
        shapes.append(Line((position, frame.top), (position, frame.bottom), dashed))
        half = text_width(guide.label, theme.tick_size) / 2.0
        centre = min(max(position, PAD + half), theme.width - PAD - half)
        shapes.append(
            Text(centre, frame.top - 6.0, guide.label, theme.tick_size, theme.foreground, "middle")
        )
    shapes.append(_plot_border(frame, theme))
    bottom = _legend(shapes, legend, bottom + 10.0, frame.left, theme)
    bottom = draw_notes(shapes, notes, bottom, theme)
    return Figure(name, title, description, theme.width, bottom + PAD, tuple(shapes), theme)


def placeholder(name: str, title: str, reason: str, theme: Theme) -> Figure:
    """Return a figure that states why there is nothing to draw."""
    shapes: list[Shape] = []
    top = draw_title(shapes, title, theme)
    bottom = draw_notes(shapes, [reason], top + 8.0, theme)
    return Figure(name, title, reason, theme.width, bottom + PAD, tuple(shapes), theme)


def label_for(identifier: str, labels: Mapping[str, str] | None) -> str:
    """Return the display label of an id: its entry in ``labels``, else the id itself."""
    if labels is None:
        return identifier
    return labels.get(identifier, identifier)


# ------------------------------------------------------------------- layout


def draw_title(shapes: list[Shape], title: str, theme: Theme) -> float:
    """Add the title, wrapped to the canvas, and return the height it takes."""
    baseline = PAD + theme.title_size
    for line in wrap(title, theme.title_size, theme.width - 2.0 * PAD):
        shapes.append(Text(PAD, baseline, line, theme.title_size, theme.foreground, bold=True))
        baseline += theme.title_size * 1.3
    return baseline - theme.title_size * 1.3 + 12.0


def _y_axis(shapes: list[Shape], frame: _Frame, axis: Axis, theme: Theme) -> None:
    grid = Stroke(theme.grid, 1.0)
    line = Stroke(theme.axis, 1.0)
    for tick, label in zip(axis.ticks, axis.labels, strict=True):
        y = frame.y(axis, tick)
        shapes.append(Line((frame.left, y), (frame.right, y), grid, "grid"))
        shapes.append(Line((frame.left - TICK, y), (frame.left, y), line, "axis"))
        shapes.append(
            Text(
                frame.left - TICK - 4.0,
                y + theme.tick_size * 0.35,
                label,
                theme.tick_size,
                theme.foreground,
                "end",
            )
        )
    shapes.append(
        Text(
            PAD + theme.label_size,
            (frame.top + frame.bottom) / 2.0,
            truncate(axis.label, theme.label_size, frame.bottom - frame.top),
            theme.label_size,
            theme.foreground,
            "middle",
            vertical=True,
        )
    )


def _x_axis(shapes: list[Shape], frame: _Frame, axis: Axis, theme: Theme) -> float:
    """Add the horizontal axis and return the baseline of its title."""
    grid = Stroke(theme.grid, 1.0)
    line = Stroke(theme.axis, 1.0)
    baseline = frame.bottom + TICK + 4.0 + theme.tick_size
    for tick, label in zip(axis.ticks, axis.labels, strict=True):
        x = frame.x(axis, tick)
        shapes.append(Line((x, frame.top), (x, frame.bottom), grid, "grid"))
        shapes.append(Line((x, frame.bottom), (x, frame.bottom + TICK), line, "axis"))
        shapes.append(Text(x, baseline, label, theme.tick_size, theme.foreground, "middle"))
    title_baseline = baseline + 8.0 + theme.label_size
    centre = (frame.left + frame.right) / 2.0
    room = 2.0 * min(centre - PAD, theme.width - PAD - centre)
    shapes.append(
        Text(
            centre,
            title_baseline,
            truncate(axis.label, theme.label_size, room),
            theme.label_size,
            theme.foreground,
            "middle",
        )
    )
    return title_baseline


def _plot_border(frame: _Frame, theme: Theme) -> Polyline:
    return Polyline(
        (
            (frame.left, frame.top),
            (frame.left, frame.bottom),
            (frame.right, frame.bottom),
        ),
        Stroke(theme.axis, 1.0),
    )


def _legend(
    shapes: list[Shape], entries: Sequence[LegendEntry], top: float, left: float, theme: Theme
) -> float:
    """Add the legend, wrapping entries onto rows, and return the bottom of the last row."""
    if not entries:
        return top
    size = theme.legend_size
    available = theme.width - PAD - left
    x, row = left, 0
    for entry in entries:
        text = truncate(entry.label, size, available - _SWATCH - 6.0)
        width = _SWATCH + 6.0 + text_width(text, size) + 14.0
        if x > left and x + width > theme.width - PAD:
            x, row = left, row + 1
        middle = top + row * LEGEND_ROW + LEGEND_ROW / 2.0
        shapes.extend(_swatch(entry, x, middle, theme))
        shapes.append(Text(x + _SWATCH + 6.0, middle + size * 0.35, text, size, theme.foreground))
        x += width
    return top + (row + 1) * LEGEND_ROW


def _swatch(entry: LegendEntry, x: float, middle: float, theme: Theme) -> list[Shape]:
    style = entry.style
    if entry.kind == "bar":
        return [Rect(x + 4.0, middle - 6.0, _SWATCH - 8.0, 12.0, style.color)]
    if entry.kind == "box":
        return [
            Rect(x + 4.0, middle - 6.0, _SWATCH - 8.0, 12.0, style.color, _BOX_OPACITY),
            Rect(x + 4.0, middle - 6.0, _SWATCH - 8.0, 12.0, None, stroke=Stroke(style.color)),
        ]
    stroke = Stroke(style.color, theme.line_width, style.dash)
    return [
        Line((x, middle), (x + _SWATCH, middle), stroke),
        Marker(x + _SWATCH / 2.0, middle, style.marker, 3.5, style.color),
    ]


def draw_notes(shapes: list[Shape], notes: Sequence[str], top: float, theme: Theme) -> float:
    """Add notes, wrapped to the canvas, and return the bottom of the last line."""
    size = theme.note_size
    baseline = top + size + 2.0
    for note in notes:
        for line in wrap(note, size, theme.width - 2.0 * PAD):
            shapes.append(Text(PAD, baseline, line, size, theme.muted, italic=True))
            baseline += size * 1.35
    return baseline - size * 1.35 + size * 0.35 if notes else top


# ------------------------------------------------------------------- series


def _series(
    frame: _Frame, x: Axis, y: Axis, item: Series, theme: Theme, slot: tuple[int, int]
) -> list[Shape]:
    """Draw a series: its line, then its markers.

    Markers of lines and steps sit at :data:`MAX_MARKERS` evenly spaced
    positions, shifted by a fraction of the spacing that depends on the
    series' place among all series, so series that coincide still show
    every marker.
    """
    if not item.points:
        return []
    pixels = [(frame.x(x, px), frame.y(y, py)) for px, py in item.points]
    stroke = Stroke(item.style.color, theme.line_width, item.style.dash)
    if item.kind == "points":
        return [Marker(px, py, item.style.marker, 3.5, item.style.color) for px, py in pixels]
    if item.kind == "step":
        until = frame.x(x, item.until) if item.until is not None else pixels[-1][0]
        outline = _steps(pixels, until)
    else:
        outline = tuple(_thin(pixels))
    drawn: list[Shape] = []
    if len(outline) > 1:
        drawn.append(Polyline(outline, stroke))
    if item.markers:
        drawn.extend(
            Marker(px, py, item.style.marker, 3.5, item.style.color)
            for px, py in _along(outline, frame, slot)
        )
    return drawn


def _along(outline: Sequence[Point], frame: _Frame, slot: tuple[int, int]) -> list[Point]:
    """Return points of a left-to-right outline at evenly spaced, series-shifted positions."""
    if len(outline) == 1:
        return list(outline)
    index, count = slot
    spacing = (frame.right - frame.left) / MAX_MARKERS
    phase = (index + 0.5) / max(1, count)
    found: list[Point] = []
    segment = 0
    for step in range(MAX_MARKERS):
        target = frame.left + (step + phase) * spacing
        if target < outline[0][0] or target > outline[-1][0]:
            continue
        while segment + 1 < len(outline) - 1 and outline[segment + 1][0] <= target:
            segment += 1
        (x0, y0), (x1, y1) = outline[segment], outline[segment + 1]
        share = 0.0 if x1 <= x0 else (target - x0) / (x1 - x0)
        found.append((target, y0 + share * (y1 - y0)))
    return found


def _band(frame: _Frame, x: Axis, y: Axis, item: Series) -> Polygon:
    upper = [(frame.x(x, px), frame.y(y, high)) for px, _, high in item.band]
    lower = [(frame.x(x, px), frame.y(y, low)) for px, low, _ in item.band]
    if item.kind == "step":
        until = frame.x(x, item.until) if item.until is not None else upper[-1][0]
        upper, lower = list(_steps(upper, until)), list(_steps(lower, until))
    else:
        upper, lower = _thin(upper), _thin(lower)
    return Polygon(tuple(upper + lower[::-1]), item.style.color, _BAND_OPACITY)


def _merged(pixels: Sequence[Point]) -> list[Point]:
    """Merge the knots of a step function that lie within half a pixel of each other.

    Of knots closer than half a pixel, the first keeps its position and takes
    the level of the last, which is how the function looks at that scale.
    """
    merged: list[Point] = []
    for x, y in pixels:
        if merged and x - merged[-1][0] < 0.5:
            merged[-1] = (merged[-1][0], y)
        else:
            merged.append((x, y))
    return merged


def _steps(pixels: Sequence[Point], until: float) -> tuple[Point, ...]:
    """Return the outline of a right-continuous step function through the knots."""
    knots = _merged(pixels)
    outline: list[Point] = [knots[0]]
    for (_, previous), (x, y) in pairwise(knots):
        outline.append((x, previous))
        outline.append((x, y))
    if until > outline[-1][0]:
        outline.append((until, outline[-1][1]))
    return tuple(_thin(outline))


def _thin(pixels: Sequence[Point]) -> list[Point]:
    """Drop points within half a pixel of the last kept one, keeping the final point."""
    kept: list[Point] = []
    for index, (x, y) in enumerate(pixels):
        if kept and abs(x - kept[-1][0]) < 0.5 and abs(y - kept[-1][1]) < 0.5:
            if index == len(pixels) - 1:
                kept[-1] = (x, y)
            continue
        kept.append((x, y))
    return kept


# --------------------------------------------------------------- rows


def _box(
    frame: _Frame, x: Axis, index: int, row_height: float, box: BoxStats, style: SeriesStyle
) -> list[Shape]:
    centre = frame.top + (index + 0.5) * row_height
    half = min(9.0, row_height * 0.35)
    low, q1, mid, q3, high = (
        frame.x(x, value) for value in (box.minimum, box.lower, box.median, box.upper, box.maximum)
    )
    whisker = Stroke(style.color, 1.2)
    return [
        Line((low, centre), (q1, centre), whisker),
        Line((q3, centre), (high, centre), whisker),
        Line((low, centre - half / 2.0), (low, centre + half / 2.0), whisker),
        Line((high, centre - half / 2.0), (high, centre + half / 2.0), whisker),
        Rect(q1, centre - half, max(q3 - q1, 0.0), 2.0 * half, style.color, _BOX_OPACITY),
        Rect(q1, centre - half, max(q3 - q1, 0.0), 2.0 * half, None, stroke=Stroke(style.color)),
        Line((mid, centre - half), (mid, centre + half), Stroke(style.color, 2.4)),
    ]


def _bar(frame: _Frame, x: Axis, bar: Bar, row_height: float, theme: Theme) -> list[Shape]:
    inset = min(3.0, row_height * 0.12)
    top = frame.top + bar.row * row_height + inset
    start, end = frame.x(x, bar.start), frame.x(x, bar.end)
    width = max(end - start, 1.0)
    height = max(row_height - 2.0 * inset, 1.0)
    shapes: list[Shape] = [Rect(start, top, width, height, bar.color, tooltip=bar.tooltip)]
    size = min(theme.tick_size, height * 0.75)
    if bar.text and size >= 7.0 and text_width(bar.text, size) + 6.0 <= width:
        shapes.append(
            Text(
                start + width / 2.0,
                top + height / 2.0 + size * 0.35,
                bar.text,
                size,
                _contrast(bar.color),
                "middle",
            )
        )
    return shapes


def _contrast(color: str) -> str:
    """Return black or white, whichever reads better on ``color``."""
    red, green, blue = (int(color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    luminance = sum(
        weight * (channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4)
        for weight, channel in zip((0.2126, 0.7152, 0.0722), (red, green, blue), strict=True)
    )
    return "#000000" if luminance > math.sqrt(1.05 * 0.05) - 0.05 else "#FFFFFF"
