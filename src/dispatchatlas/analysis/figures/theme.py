"""Colours, type sizes, and canvas dimensions shared by every figure.

A :class:`Theme` fixes everything about a figure's appearance that is not
data: the canvas width, the height of a plot area, the series palette, the
colours of text and axes, and the type sizes. Figures are laid out in pixels;
the TikZ output maps one pixel to :attr:`Theme.tex_scale` big points, so the
two formats share one layout.

The series palette is the colour-universal-design palette of Okabe and Ito,
as published by Wong (2011): eight colours that stay distinguishable under
the common forms of colour-vision deficiency. Series beyond the eighth reuse
a colour, so every series also gets a marker shape and, from the ninth on, a
dash pattern; the triple (colour, marker, dash) is unique for the first 168
series.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dispatchatlas.core.errors import ConfigError

__all__ = [
    "DASHES",
    "DEFAULT_THEME",
    "MARKERS",
    "PALETTE",
    "PALETTE_REFERENCE",
    "SeriesStyle",
    "Theme",
]

PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow
)
"""The Okabe-Ito palette, ordered so that yellow, the colour with the least
contrast on white, comes last."""

PALETTE_REFERENCE = "wong2011points"
"""Bibliography key of the source of :data:`PALETTE`."""

MARKERS: tuple[str, ...] = (
    "circle",
    "square",
    "triangle",
    "diamond",
    "triangle-down",
    "cross",
    "plus",
)
"""Marker shapes, assigned to series in turn."""

DASHES: tuple[tuple[float, ...], ...] = ((), (6.0, 3.0), (1.5, 2.5))
"""Dash patterns in pixels: solid for the first eight series, then dashed, then dotted."""

_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


@dataclass(frozen=True, slots=True)
class SeriesStyle:
    """How one series is drawn.

    Attributes:
        color: Hexadecimal ``#RRGGBB`` colour.
        marker: One of :data:`MARKERS`.
        dash: Dash pattern in pixels, empty for a solid line.
    """

    color: str
    marker: str
    dash: tuple[float, ...]


@dataclass(frozen=True)
class Theme:
    """The appearance of a figure.

    Attributes:
        width: Canvas width in pixels.
        plot_height: Height of the plot area of a chart with a numeric
            vertical axis, in pixels. Charts with one row per category grow
            with the number of rows instead.
        palette: Series colours, as ``#RRGGBB``.
        foreground: Colour of text.
        muted: Colour of secondary text, such as notes.
        axis: Colour of axis lines and ticks.
        grid: Colour of grid lines.
        background: Canvas colour.
        font_family: CSS font family of the SVG output. The TikZ output uses
            the document's font.
        title_size: Title type size, in pixels.
        label_size: Axis-title type size.
        tick_size: Tick-label type size.
        legend_size: Legend type size.
        note_size: Note and caption type size.
        line_width: Width of series lines, in pixels.
        tex_scale: Big points per pixel in the TikZ output. The default maps
            the 720-pixel canvas to 432 bp, the width of a text block.
    """

    width: float = 720.0
    plot_height: float = 300.0
    palette: tuple[str, ...] = PALETTE
    foreground: str = "#1F2328"
    muted: str = "#57606A"
    axis: str = "#57606A"
    grid: str = "#D8DEE4"
    background: str = "#FFFFFF"
    font_family: str = "Helvetica, Arial, sans-serif"
    title_size: float = 15.0
    label_size: float = 13.0
    tick_size: float = 11.0
    legend_size: float = 12.0
    note_size: float = 11.0
    line_width: float = 1.8
    tex_scale: float = 0.6

    def __post_init__(self) -> None:
        if not self.palette:
            raise ConfigError("a theme needs at least one palette colour")
        colors = (*self.palette, self.foreground, self.muted, self.axis, self.grid, self.background)
        for color in colors:
            if not isinstance(color, str) or not _HEX.match(color):
                raise ConfigError(f"theme colours must be written #RRGGBB, got {color!r}")
        object.__setattr__(self, "palette", tuple(color.upper() for color in self.palette))
        for name in ("foreground", "muted", "axis", "grid", "background"):
            object.__setattr__(self, name, getattr(self, name).upper())
        minimums = {"width": 360.0, "plot_height": 120.0}
        sizes = ("title_size", "label_size", "tick_size", "legend_size", "note_size")
        for name in ("width", "plot_height", *sizes, "line_width", "tex_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise ConfigError(f"theme {name} must be a positive number, got {value!r}")
            if value < minimums.get(name, 0.0):
                raise ConfigError(f"theme {name} must be at least {minimums[name]:g}, got {value}")
            object.__setattr__(self, name, float(value))
        family = self.font_family
        if (
            not isinstance(family, str)
            or not family.strip()
            or any(c in family for c in "<>&\"';{}")
        ):
            raise ConfigError(f"invalid font family {family!r}")

    def series(self, index: int) -> SeriesStyle:
        """Return the style of the series at position ``index``, from 0."""
        colors = len(self.palette)
        return SeriesStyle(
            color=self.palette[index % colors],
            marker=MARKERS[index % len(MARKERS)],
            dash=DASHES[(index // colors) % len(DASHES)],
        )


DEFAULT_THEME = Theme()
"""The theme figures use unless given another."""
