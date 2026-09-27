"""A griffe extension that renders the reStructuredText markup of the docstrings.

The docstrings of DispatchAtlas mark references with Sphinx roles, such as
:class:`~dispatchatlas.core.model.Problem` or :func:`solve`, introduce literal
blocks with ``::``, and hold one simple table. mkdocstrings reads docstrings as
Markdown, where none of these has a meaning, so this extension rewrites them
before rendering:

- a role becomes an optional cross-reference of mkdocs-autorefs: a link when
  the target is documented, plain code otherwise. A relative name resolves in
  the scope of the documented object, as Sphinx resolves it, and a leading
  ``~`` shows only the last component;
- a paragraph ending in ``::`` ends in ``:``, and the indented block after it
  becomes a fenced code block;
- a simple table, framed by rows of ``=``, becomes a fenced text block.
"""

from __future__ import annotations

import html
import re
import textwrap
from typing import Any

import griffe

ROLE = re.compile(r":(?P<role>class|func|meth|attr|data|mod|exc|obj|const):`(?P<body>[^`]+)`")
EXPLICIT = re.compile(r"^(?P<title>.+?)\s*<(?P<target>[^<>]+)>$")
BORDER = re.compile(r"^\s*=+(?:\s+=+)+\s*$")


class SphinxMarkup(griffe.Extension):
    """Rewrite roles, literal blocks, and simple tables in every docstring."""

    def on_package(self, *, pkg: griffe.Module, **kwargs: Any) -> None:
        """Rewrite the docstring of the package itself."""
        _rewrite(pkg)

    def on_object(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        """Rewrite the docstring of every member of the package."""
        _rewrite(obj)


def _rewrite(obj: griffe.Object) -> None:
    if obj.docstring is None:
        return
    text = _tables(_literal_blocks(obj.docstring.value))
    obj.docstring.value = ROLE.sub(lambda match: _reference(obj, match), text)


def _reference(obj: griffe.Object, match: re.Match[str]) -> str:
    body = match["body"].strip()
    explicit = EXPLICIT.match(body)
    target = (explicit["target"] if explicit else body).strip()
    short = target.startswith("~")
    target = target.lstrip("~.")
    if explicit:
        title = explicit["title"]
    else:
        title = target.rsplit(".", 1)[-1] if short else target
        if match["role"] in ("func", "meth"):
            title += "()"
    code = f"<code>{html.escape(title)}</code>"
    identifier = _resolve(obj, target)
    if identifier is None:
        return code
    return f'<autoref identifier="{html.escape(identifier)}" optional>{code}</autoref>'


def _resolve(obj: griffe.Object, target: str) -> str | None:
    """Return the full path of ``target`` as seen from ``obj``, or ``None``."""
    head, _, rest = target.partition(".")
    if head == "dispatchatlas":
        path = target
    else:
        try:
            path = obj.resolve(head)
        except griffe.NameResolutionError:
            return None
        if rest:
            path = f"{path}.{rest}"
    try:
        found = obj.modules_collection[path]
    except (KeyError, ValueError, griffe.AliasResolutionError):
        return path
    try:
        return str(found.final_target.path) if found.is_alias else str(found.path)
    except griffe.AliasResolutionError:
        return path


def _literal_blocks(text: str) -> str:
    """Turn ``text::`` and the indented block after it into ``text:`` and a fenced block."""
    lines = text.split("\n")
    result: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.rstrip()
        indent = len(line) - len(line.lstrip())
        if stripped.endswith("::") and stripped.strip() != "::" and _blank(lines, index + 1):
            end = index + 2
            while end < len(lines) and (
                not lines[end].strip() or len(lines[end]) - len(lines[end].lstrip()) > indent
            ):
                end += 1
            block = lines[index + 2 : end]
            while block and not block[-1].strip():
                block.pop()
            if block:
                prefix = " " * indent
                code = textwrap.dedent("\n".join(block)).split("\n")
                result += [stripped[:-1], "", f"{prefix}```{_language(code)}"]
                result += [f"{prefix}{item}" if item else "" for item in code]
                result += [f"{prefix}```", ""]
                index = index + 2 + len(block)
                while index < len(lines) and not lines[index].strip():
                    index += 1
                continue
        result.append(line)
        index += 1
    return "\n".join(result).rstrip("\n")


def _blank(lines: list[str], index: int) -> bool:
    return index < len(lines) and not lines[index].strip()


def _language(code: list[str]) -> str:
    first = next((line for line in code if line.strip()), "")
    if first.startswith(("[", "schema_version")):
        return "toml"
    if first.startswith("dispatchatlas"):
        return "text"
    return "python"


def _tables(text: str) -> str:
    """Wrap each simple table, from its top border to its bottom border, in a text block."""
    lines = text.split("\n")
    borders = [index for index, line in enumerate(lines) if BORDER.match(line)]
    borders = borders[: len(borders) - len(borders) % 3]
    triples = zip(borders[::3], borders[1::3], borders[2::3], strict=True)
    for top, _, bottom in reversed(list(triples)):
        indent = lines[top][: len(lines[top]) - len(lines[top].lstrip())]
        lines[top : bottom + 1] = [f"{indent}```text", *lines[top : bottom + 1], f"{indent}```"]
    return "\n".join(lines)
