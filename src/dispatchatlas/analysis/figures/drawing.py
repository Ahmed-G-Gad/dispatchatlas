"""The drawing model: a figure is a canvas, a title, a description, and shapes.

Chart builders compute a layout in pixels, with the origin at the top left
and ``y`` growing downwards, and emit a flat list of shapes: lines,
polylines, polygons, rectangles, markers, and text. :func:`to_svg
<dispatchatlas.analysis.figures.svg.to_svg>` and :func:`to_tex
<dispatchatlas.analysis.figures.tex.to_tex>` serialise the same list, so the
SVG and TikZ versions of a figure share every coordinate.

Numbers are written with at most two decimals and no trailing zeros
(:func:`number`), so a figure's bytes depend on its data only. Text widths
are estimated from the character count (:func:`text_width`); the estimate is
generous, so labels fit in either format.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from dispatchatlas.analysis.figures.theme import DEFAULT_THEME, Theme

__all__ = [
    "Anchor",
    "Figure",
    "Line",
    "Marker",
    "Point",
    "Polygon",
    "Polyline",
    "Rect",
    "Shape",
    "Stroke",
    "Text",
    "bounds",
    "number",
    "text_width",
    "truncate",
    "wrap",
]

type Point = tuple[float, float]
"""A position in pixels: ``(x, y)`` with ``y`` growing downwards."""

type Anchor = Literal["start", "middle", "end"]
"""Which point of a text its position names: its start, middle, or end."""

CHARACTER_WIDTH = 0.6
"""Estimated width of one character, as a fraction of the type size."""

DESCENT = 0.25
"""Depth of descenders below the baseline, as a fraction of the type size."""


@dataclass(frozen=True, slots=True)
class Stroke:
    """How a line is drawn.

    Attributes:
        color: ``#RRGGBB`` colour.
        width: Line width in pixels.
        dash: Dash pattern in pixels, empty for a solid line.
        opacity: Opacity from 0 to 1.
    """

    color: str
    width: float = 1.0
    dash: tuple[float, ...] = ()
    opacity: float = 1.0


@dataclass(frozen=True, slots=True)
class Line:
    """A straight segment.

    Attributes:
        start: First end.
        end: Second end.
        stroke: How it is drawn.
        role: Style class for restyling the SVG, such as ``"axis"``.
    """

    start: Point
    end: Point
    stroke: Stroke
    role: str = ""


@dataclass(frozen=True, slots=True)
class Polyline:
    """An open path through points.

    Attributes:
        points: The points, in drawing order.
        stroke: How it is drawn.
    """

    points: tuple[Point, ...]
    stroke: Stroke


@dataclass(frozen=True, slots=True)
class Polygon:
    """A closed, filled path.

    Attributes:
        points: The corners, in drawing order.
        fill: ``#RRGGBB`` fill colour.
        opacity: Fill opacity from 0 to 1.
    """

    points: tuple[Point, ...]
    fill: str
    opacity: float = 1.0


@dataclass(frozen=True, slots=True)
class Rect:
    """An axis-aligned rectangle.

    Attributes:
        x: Left edge.
        y: Top edge.
        width: Width, at least 0.
        height: Height, at least 0.
        fill: ``#RRGGBB`` fill colour, or ``None`` for no fill.
        opacity: Fill opacity from 0 to 1.
        stroke: Outline, or ``None`` for none.
        tooltip: Text a viewer shows on hovering, SVG only.
    """

    x: float
    y: float
    width: float
    height: float
    fill: str | None
    opacity: float = 1.0
    stroke: Stroke | None = None
    tooltip: str = ""


@dataclass(frozen=True, slots=True)
class Marker:
    """A data marker centred on a point.

    Attributes:
        x: Centre.
        y: Centre.
        shape: One of :data:`~dispatchatlas.analysis.figures.theme.MARKERS`.
        size: Radius of the circle that holds the shape, in pixels.
        color: ``#RRGGBB`` colour.
        filled: Filled, or outlined only.
    """

    x: float
    y: float
    shape: str
    size: float
    color: str
    filled: bool = True


@dataclass(frozen=True, slots=True)
class Text:
    """A single line of text.

    Attributes:
        x: Horizontal position of the anchor.
        y: Baseline, or, for vertical text, the horizontal position of the
            baseline.
        text: The text, without line breaks.
        size: Type size in pixels.
        color: ``#RRGGBB`` colour.
        anchor: Which point of the text ``x`` names.
        bold: Set in bold.
        italic: Set in italics.
        vertical: Rotated a quarter turn counter-clockwise, reading upwards.
    """

    x: float
    y: float
    text: str
    size: float
    color: str
    anchor: Anchor = "start"
    bold: bool = False
    italic: bool = False
    vertical: bool = False


type Shape = Line | Polyline | Polygon | Rect | Marker | Text
"""Anything a figure is drawn from."""


@dataclass(frozen=True)
class Figure:
    """A figure, ready to be written as SVG or TikZ.

    Attributes:
        name: File stem, for example ``"performance-profile"``.
        title: Title, drawn at the top and stored as the SVG title.
        description: Text alternative: what the figure shows, stored as the
            SVG description and as comments in the TikZ document.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        shapes: What is drawn, in painting order.
        theme: The theme the layout was computed with.
    """

    name: str
    title: str
    description: str
    width: float
    height: float
    shapes: tuple[Shape, ...]
    theme: Theme = DEFAULT_THEME

    def svg(self) -> str:
        """Return the figure as a standalone SVG document."""
        from dispatchatlas.analysis.figures.svg import to_svg  # noqa: PLC0415 - avoids a cycle

        return to_svg(self)

    def tex(self) -> str:
        """Return the figure as a standalone LaTeX document holding one TikZ picture."""
        from dispatchatlas.analysis.figures.tex import to_tex  # noqa: PLC0415 - avoids a cycle

        return to_tex(self)

    def render(self, fmt: str) -> str:
        """Return the figure in format ``"svg"`` or ``"tex"``.

        Raises:
            ValueError: For another format.
        """
        if fmt == "svg":
            return self.svg()
        if fmt == "tex":
            return self.tex()
        raise ValueError(f"unknown figure format {fmt!r}; use 'svg' or 'tex'")


def number(value: float) -> str:
    """Format a coordinate: at most two decimals, no trailing zeros, no negative zero.

    Raises:
        ValueError: If ``value`` is not finite.
    """
    if not math.isfinite(value):
        raise ValueError(f"cannot draw at a non-finite coordinate {value!r}")
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def text_width(text: str, size: float) -> float:
    """Return a generous estimate of the width of ``text`` set at ``size`` pixels."""
    return CHARACTER_WIDTH * size * len(text)


def truncate(text: str, size: float, width: float) -> str:
    """Return ``text`` shortened so that it fits in ``width`` pixels.

    The middle gives way to an ellipsis, so that names that differ only at
    their end, such as numbered variants, stay apart.
    """
    if text_width(text, size) <= width:
        return text
    keep = max(1, math.floor(width / (CHARACTER_WIDTH * size)) - 1)
    head = (keep + 1) // 2
    tail = keep - head
    return text[:head] + "\u2026" + (text[len(text) - tail :] if tail else "")


def wrap(text: str, size: float, width: float) -> list[str]:
    """Break ``text`` into lines that fit in ``width`` pixels, at spaces where possible."""
    limit = max(8, math.floor(width / (CHARACTER_WIDTH * size)))
    lines: list[str] = []
    current = ""
    for word in text.split():
        rest = word
        while len(rest) > limit:
            if current:
                lines.append(current)
                current = ""
            lines.append(rest[:limit])
            rest = rest[limit:]
        candidate = f"{current} {rest}" if current else rest
        if len(candidate) <= limit:
            current = candidate
        else:
            lines.append(current)
            current = rest
    if current:
        lines.append(current)
    return lines


def bounds(shape: Shape) -> tuple[float, float, float, float]:
    """Return a shape's extent as ``(left, top, right, bottom)``, estimating text size."""
    if isinstance(shape, Line):
        xs: Sequence[float] = (shape.start[0], shape.end[0])
        ys: Sequence[float] = (shape.start[1], shape.end[1])
        return min(xs), min(ys), max(xs), max(ys)
    if isinstance(shape, (Polyline, Polygon)):
        xs = [point[0] for point in shape.points]
        ys = [point[1] for point in shape.points]
        return min(xs), min(ys), max(xs), max(ys)
    if isinstance(shape, Rect):
        return shape.x, shape.y, shape.x + shape.width, shape.y + shape.height
    if isinstance(shape, Marker):
        return (
            shape.x - shape.size,
            shape.y - shape.size,
            shape.x + shape.size,
            shape.y + shape.size,
        )
    extent = text_width(shape.text, shape.size)
    offset = {"start": 0.0, "middle": extent / 2.0, "end": extent}[shape.anchor]
    descent = DESCENT * shape.size
    if shape.vertical:
        return (
            shape.x - shape.size,
            shape.y - extent + offset,
            shape.x + descent,
            shape.y + offset,
        )
    return shape.x - offset, shape.y - shape.size, shape.x - offset + extent, shape.y + descent
