r"""Writing figures as standalone LaTeX documents holding one TikZ picture.

The document needs only the ``tikz`` package. Its single ``tikzpicture``
uses the pixel coordinates of the SVG version, scaled by
:attr:`~dispatchatlas.analysis.figures.theme.Theme.tex_scale` big points per
pixel and with the ``y`` axis flipped, so both formats share one layout. The
colour definitions sit inside the picture, which can therefore be copied
into another document as it is. Plain TikZ rather than PGFPlots keeps the
output independent of PGFPlots versions and lets the SVG and TikZ versions
agree to the coordinate.

Text is escaped for LaTeX, and the few non-ASCII characters figures use,
such as Greek letters in axis titles, are written as LaTeX commands, so the
file is ASCII and compiles with the default font encoding. Type sizes map to
the nearest standard size command. The title and description open the file
as comments.
"""

from __future__ import annotations

import math
import textwrap
import unicodedata

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
from dispatchatlas.analysis.figures.svg import marker_outline

__all__ = ["ascii_text", "tex_escape", "to_tex"]

_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
    '"': "''",
}

_SYMBOLS = {
    "\u03b1": (r"$\alpha$", "alpha"),
    "\u03ba": (r"$\kappa$", "kappa"),
    "\u03c1": (r"$\rho$", "rho"),
    "\u03c4": (r"$\tau$", "tau"),
    "\u2264": (r"$\leq$", "<="),
    "\u2265": (r"$\geq$", ">="),
    "\u00d7": (r"$\times$", "x"),
    "\u221e": (r"$\infty$", "inf"),
    "\u2212": ("-", "-"),
    "\u2013": ("--", "-"),
    "\u2014": ("---", "-"),
    "\u2026": (r"\ldots{}", "..."),
    "\u00b7": (r"$\cdot$", "."),
}

_ACCENTS = {
    "\u0301": "'",
    "\u0300": "`",
    "\u0308": '"',
    "\u0302": "^",
    "\u030c": "v",
    "\u0303": "~",
}
"""LaTeX accent commands of the combining marks of accented Latin letters."""

_SIZES = ((7.3, r"\scriptsize"), (8.3, r"\footnotesize"), (9.4, r"\small"))

_ANCHORS = {"start": "base west", "middle": "base", "end": "base east"}

_PATH_POINTS_PER_LINE = 6


def tex_escape(text: str) -> str:
    """Return ``text`` as LaTeX source that typesets it literally, in ASCII.

    Special characters become commands, the symbols figures use (Greek
    letters, comparison signs, dashes) become their LaTeX forms, accented
    Latin letters become accent commands, and any other non-ASCII character
    becomes a question mark.
    """
    pieces: list[str] = []
    for char in text:
        if char in _SPECIAL:
            pieces.append(_SPECIAL[char])
        elif char in _SYMBOLS:
            pieces.append(_SYMBOLS[char][0])
        elif ord(char) < 32:
            continue
        elif ord(char) < 127:
            pieces.append(char)
        else:
            pieces.append(_accented(char))
    return "".join(pieces)


def ascii_text(text: str) -> str:
    """Return ``text`` folded to plain ASCII, for comments: accents dropped, symbols spelled."""
    pieces: list[str] = []
    for char in text:
        if char in _SYMBOLS:
            pieces.append(_SYMBOLS[char][1])
            continue
        if char.isspace():
            pieces.append(" ")
            continue
        base = unicodedata.normalize("NFKD", char)[:1]
        pieces.append(base if base.isascii() and base.isprintable() else "?")
    return "".join(pieces)


def to_tex(figure: Figure) -> str:
    """Return ``figure`` as a standalone LaTeX document, ending with a newline."""
    scale = figure.theme.tex_scale
    colors = sorted(
        {figure.theme.background} | {color for shape in figure.shapes for color in _colors(shape)}
    )
    width, height = number(figure.width), number(figure.height)
    lines = [f"% {ascii_text(figure.title)}"]
    lines.extend(f"% {line}" for line in textwrap.wrap(ascii_text(figure.description), 96))
    lines.extend(
        [
            r"\documentclass[border=0pt]{standalone}",
            r"\usepackage{tikz}",
            r"\begin{document}",
            rf"\begin{{tikzpicture}}[x={number(scale)}bp,y=-{number(scale)}bp]",
        ]
    )
    lines.extend(rf"\definecolor{{{_name(color)}}}{{HTML}}{{{color[1:]}}}" for color in colors)
    lines.append(rf"\useasboundingbox (0,0) rectangle ({width},{height});")
    lines.append(rf"\fill[{_name(figure.theme.background)}] (0,0) rectangle ({width},{height});")
    for shape in figure.shapes:
        lines.extend(_shape(shape, scale))
    lines.extend([r"\end{tikzpicture}", r"\end{document}"])
    return "\n".join(lines) + "\n"


def _accented(char: str) -> str:
    decomposed = unicodedata.normalize("NFD", char)
    base, marks = decomposed[:1], decomposed[1:]
    if base.isascii() and base.isalpha() and len(marks) == 1 and marks in _ACCENTS:
        return f"\\{_ACCENTS[marks]}{{{base}}}"
    return "?"


def _name(color: str) -> str:
    return "c" + color[1:].upper()


def _colors(shape: Shape) -> list[str]:
    if isinstance(shape, (Line, Polyline)):
        return [shape.stroke.color]
    if isinstance(shape, Polygon):
        return [shape.fill]
    if isinstance(shape, Rect):
        found = [] if shape.fill is None else [shape.fill]
        return found + ([] if shape.stroke is None else [shape.stroke.color])
    return [shape.color]


def _stroke(stroke: Stroke, scale: float) -> str:
    options = [f"draw={_name(stroke.color)}", f"line width={number(stroke.width * scale)}bp"]
    if stroke.dash:
        pattern = " ".join(
            f"{'on' if index % 2 == 0 else 'off'} {number(value * scale)}bp"
            for index, value in enumerate(stroke.dash)
        )
        options.append(f"dash pattern={pattern}")
    if stroke.opacity < 1.0:
        options.append(f"draw opacity={number(stroke.opacity)}")
    return ",".join(options)


def _fill(color: str, opacity: float) -> str:
    options = f"fill={_name(color)}"
    return options if opacity >= 1.0 else f"{options},fill opacity={number(opacity)}"


def _path(points: tuple[Point, ...], closed: bool = False) -> str:
    coordinates = [f"({number(x)},{number(y)})" for x, y in points]
    rows = [
        " -- ".join(coordinates[start : start + _PATH_POINTS_PER_LINE])
        for start in range(0, len(coordinates), _PATH_POINTS_PER_LINE)
    ]
    body = "\n  -- ".join(rows)
    return body + (" -- cycle" if closed else "")


def _shape(shape: Shape, scale: float) -> list[str]:
    if isinstance(shape, Line):
        start, end = shape.start, shape.end
        return [
            rf"\draw[{_stroke(shape.stroke, scale)}] ({number(start[0])},{number(start[1])})"
            rf" -- ({number(end[0])},{number(end[1])});"
        ]
    if isinstance(shape, Polyline):
        return [rf"\draw[{_stroke(shape.stroke, scale)},line join=round] {_path(shape.points)};"]
    if isinstance(shape, Polygon):
        return [rf"\fill[{_fill(shape.fill, shape.opacity)}] {_path(shape.points, closed=True)};"]
    if isinstance(shape, Rect):
        corners = (
            f"({number(shape.x)},{number(shape.y)}) rectangle "
            f"({number(shape.x + shape.width)},{number(shape.y + shape.height)})"
        )
        drawn = []
        if shape.fill is not None:
            drawn.append(rf"\fill[{_fill(shape.fill, shape.opacity)}] {corners};")
        if shape.stroke is not None:
            drawn.append(rf"\draw[{_stroke(shape.stroke, scale)}] {corners};")
        return drawn
    if isinstance(shape, Marker):
        return _marker(shape, scale)
    return [_text(shape, scale)]


def _marker(marker: Marker, scale: float) -> list[str]:
    x, y, r = marker.x, marker.y, marker.size
    color = _name(marker.color)
    if marker.shape in ("cross", "plus"):
        width = f"line width={number(1.6 * scale)}bp"
        if marker.shape == "plus":
            segments = [((x - r, y), (x + r, y)), ((x, y - r), (x, y + r))]
        else:
            d = r * math.sqrt(0.5)
            segments = [((x - d, y - d), (x + d, y + d)), ((x - d, y + d), (x + d, y - d))]
        return [rf"\draw[draw={color},{width}] {_path((start, end))};" for start, end in segments]
    command = (
        rf"\fill[fill={color}]"
        if marker.filled
        else rf"\draw[draw={color},line width={number(1.2 * scale)}bp]"
    )
    if marker.shape == "circle":
        return [rf"{command} ({number(x)},{number(y)}) circle[radius={number(r)}];"]
    return [rf"{command} {_path(marker_outline(marker), closed=True)};"]


def _text(text: Text, scale: float) -> str:
    size = text.size * scale
    font = next((command for limit, command in _SIZES if size <= limit), r"\normalsize")
    if text.bold:
        font += r"\bfseries"
    if text.italic:
        font += r"\itshape"
    options = [
        f"anchor={_ANCHORS[text.anchor]}",
        "inner sep=0pt",
        f"text={_name(text.color)}",
        f"font={font}",
    ]
    if text.vertical:
        options.append("rotate=90")
    return (
        rf"\node[{','.join(options)}] at ({number(text.x)},{number(text.y)}) "
        rf"{{{tex_escape(text.text)}}};"
    )
