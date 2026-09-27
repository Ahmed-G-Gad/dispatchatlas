"""Writing figures as SVG documents.

The document is ASCII: characters outside it are written as numeric
character references. Every colour is written as a presentation attribute,
so any SVG renderer shows the figure as designed, and the elements that a
page may want to restyle carry classes (``da-axis``, ``da-grid``, ``da-text``).
The theme's colours and font are also published as CSS custom properties
(``--da-fg``, ``--da-axis``, ``--da-grid``, ``--da-font``) in the ``style``
attribute of the ``svg`` element itself, so they are scoped to the figure:
an SVG inlined in a web page neither reads nor overrides the page's own
variables. The embedded style sheet only refers to those variables, and
only elements drawn in the theme's own text, note, axis, and grid colours
carry the classes it styles, so series colours are never overridden.
"""

from __future__ import annotations

import math

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
    number,
)

__all__ = ["escape", "marker_outline", "to_svg"]

_STYLE = (
    "<style>"
    ".da-figure text{font-family:var(--da-font)}"
    ".da-figure .da-text{fill:var(--da-fg)}"
    ".da-figure .da-muted{fill:var(--da-muted)}"
    ".da-figure .da-axis{stroke:var(--da-axis)}"
    ".da-figure .da-grid{stroke:var(--da-grid)}"
    "</style>"
)


def escape(text: str) -> str:
    """Escape text for SVG content or attributes, as ASCII.

    The five XML special characters become entities, other non-ASCII
    characters become numeric references, and control characters, which
    XML forbids, are dropped.
    """
    pieces: list[str] = []
    for char in text:
        code = ord(char)
        if char == "&":
            pieces.append("&amp;")
        elif char == "<":
            pieces.append("&lt;")
        elif char == ">":
            pieces.append("&gt;")
        elif char == '"':
            pieces.append("&quot;")
        elif char == "'":
            pieces.append("&#39;")
        elif code < 32 or 127 <= code < 160:
            continue
        elif code > 126:
            pieces.append(f"&#{code};")
        else:
            pieces.append(char)
    return "".join(pieces)


def to_svg(figure: Figure) -> str:
    """Return ``figure`` as a standalone SVG document, ending with a newline."""
    theme = figure.theme
    width, height = number(figure.width), number(figure.height)
    variables = (
        f"--da-fg:{theme.foreground};--da-muted:{theme.muted};--da-axis:{theme.axis};"
        f"--da-grid:{theme.grid};--da-font:{theme.font_family}"
    )
    text_classes = {theme.foreground: "da-text", theme.muted: "da-muted"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" class="da-figure" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'font-family="{escape(theme.font_family)}" style="{escape(variables)}">',
        f"<title>{escape(figure.title)}</title>",
        f"<desc>{escape(figure.description)}</desc>",
        _STYLE,
        f'<rect width="{width}" height="{height}" fill="{theme.background}"/>',
    ]
    lines.extend(_shape(shape, text_classes) for shape in figure.shapes)
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _shape(shape: Shape, text_classes: dict[str, str]) -> str:
    if isinstance(shape, Line):
        role = f' class="da-{shape.role}"' if shape.role else ""
        return (
            f'<line x1="{number(shape.start[0])}" y1="{number(shape.start[1])}" '
            f'x2="{number(shape.end[0])}" y2="{number(shape.end[1])}"{role}'
            f"{_stroke(shape.stroke)}/>"
        )
    if isinstance(shape, Polyline):
        return (
            f'<polyline points="{_points(shape.points)}" fill="none" '
            f'stroke-linejoin="round"{_stroke(shape.stroke)}/>'
        )
    if isinstance(shape, Polygon):
        return (
            f'<polygon points="{_points(shape.points)}" fill="{shape.fill}"'
            f'{_opacity("fill-opacity", shape.opacity)} stroke="none"/>'
        )
    if isinstance(shape, Rect):
        fill = shape.fill if shape.fill is not None else "none"
        stroke = _stroke(shape.stroke) if shape.stroke is not None else ""
        body = (
            f'<rect x="{number(shape.x)}" y="{number(shape.y)}" width="{number(shape.width)}" '
            f'height="{number(shape.height)}" fill="{fill}"'
            f"{_opacity('fill-opacity', shape.opacity)}{stroke}"
        )
        if shape.tooltip:
            return f"{body}><title>{escape(shape.tooltip)}</title></rect>"
        return body + "/>"
    if isinstance(shape, Marker):
        return _marker(shape)
    return _text(shape, text_classes.get(shape.color, ""))


def _stroke(stroke: Stroke) -> str:
    parts = f' stroke="{stroke.color}" stroke-width="{number(stroke.width)}"'
    if stroke.dash:
        parts += f' stroke-dasharray="{" ".join(number(value) for value in stroke.dash)}"'
    return parts + _opacity("stroke-opacity", stroke.opacity)


def _opacity(attribute: str, value: float) -> str:
    return "" if value >= 1.0 else f' {attribute}="{number(value)}"'


def _points(points: tuple[Point, ...]) -> str:
    return " ".join(f"{number(x)},{number(y)}" for x, y in points)


def _marker(marker: Marker) -> str:
    x, y, r = marker.x, marker.y, marker.size
    paint = (
        f' fill="{marker.color}" stroke="none"'
        if marker.filled
        else f' fill="none" stroke="{marker.color}" stroke-width="1.2"'
    )
    if marker.shape == "circle":
        return f'<circle cx="{number(x)}" cy="{number(y)}" r="{number(r)}"{paint}/>'
    if marker.shape in ("cross", "plus"):
        stroke = f' stroke="{marker.color}" stroke-width="1.6"'
        if marker.shape == "plus":
            path = (
                f"M{number(x - r)},{number(y)}H{number(x + r)}"
                f"M{number(x)},{number(y - r)}V{number(y + r)}"
            )
        else:
            d = r * math.sqrt(0.5)
            path = (
                f"M{number(x - d)},{number(y - d)}L{number(x + d)},{number(y + d)}"
                f"M{number(x - d)},{number(y + d)}L{number(x + d)},{number(y - d)}"
            )
        return f'<path d="{path}" fill="none"{stroke}/>'
    return f'<polygon points="{_points(marker_outline(marker))}"{paint}/>'


def marker_outline(marker: Marker) -> tuple[Point, ...]:
    """Return the corners of a polygonal marker: square, diamond, or triangle."""
    x, y, r = marker.x, marker.y, marker.size
    if marker.shape == "square":
        d = r * 0.85
        return ((x - d, y - d), (x + d, y - d), (x + d, y + d), (x - d, y + d))
    if marker.shape == "diamond":
        return ((x, y - r), (x + r, y), (x, y + r), (x - r, y))
    if marker.shape == "triangle-down":
        return ((x - r, y - r * 0.6), (x + r, y - r * 0.6), (x, y + r))
    return ((x, y - r), (x + r, y + r * 0.6), (x - r, y + r * 0.6))


def _text(text: Text, css: str) -> str:
    attributes = [f'class="{css}"'] if css else []
    attributes += [f'x="{number(text.x)}"', f'y="{number(text.y)}"']
    if text.anchor != "start":
        attributes.append(f'text-anchor="{text.anchor}"')
    attributes.append(f'font-size="{number(text.size)}"')
    if text.bold:
        attributes.append('font-weight="bold"')
    if text.italic:
        attributes.append('font-style="italic"')
    attributes.append(f'fill="{text.color}"')
    if text.vertical:
        attributes.append(f'transform="rotate(-90 {number(text.x)} {number(text.y)})"')
    return f"<text {' '.join(attributes)}>{escape(text.text)}</text>"
