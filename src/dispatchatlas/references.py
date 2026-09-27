"""The bibliography of the methods, benchmarks, and statistics DispatchAtlas implements.

Solvers and benchmark families cite their sources by key, for example
``SolverInfo.references = ("kolisch1996serial",)``. The entries live in TOML
files under ``dispatchatlas/data/references``, one file per area, and are
merged here. Keys follow ``<first author><year><first title word>``.
:func:`to_bibtex` exports entries as BibTeX, so users can cite what they use.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from dispatchatlas.core.errors import ConfigError

__all__ = ["Reference", "bibliography", "cite", "reference", "to_bibtex"]

_ARTICLE = re.compile(
    r"^(?P<journal>[^,]+?) (?P<volume>\d+)(?:\((?P<number>[^()]+)\))?:(?P<pages>[^\s:,]+)$"
)
"""A journal venue as the bibliography writes it: ``Journal volume(number):pages``."""

_PROCEEDINGS = re.compile(r"\b(?:proceedings|conference|congress|symposium|workshop)\b", re.I)
_EDITORS = re.compile(r"\s*\(eds?\.\)\s*$")
_NAME_SEPARATOR = re.compile(r",\s*|\s+and\s+")
_LATEX_SPECIAL = str.maketrans(
    {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
)


@dataclass(frozen=True, slots=True)
class Reference:
    """One bibliography entry.

    Attributes:
        key: Citation key.
        authors: Author list as printed, for example ``"R. Kolisch"``.
        year: Publication year.
        title: Title of the work.
        venue: Journal, proceedings, or publisher, with volume and pages.
        doi: Digital object identifier without the ``https://doi.org/`` prefix.
        url: Stable URL, for works without a DOI.
    """

    key: str
    authors: str
    year: int
    title: str
    venue: str
    doi: str | None = None
    url: str | None = None

    @property
    def link(self) -> str | None:
        """A resolvable link to the work."""
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return self.url

    def __str__(self) -> str:
        text = f"{self.authors} ({self.year}). {self.title}. {self.venue}."
        return f"{text} {self.link}" if self.link else text

    def bibtex(self) -> str:
        """Return the entry as one BibTeX entry, keyed by its citation key.

        A venue of the form ``Journal volume(number):pages``, which the
        bibliography uses for journal articles, gives an ``@article`` with
        those fields. Any other venue, such as proceedings, a book, or a
        report, has no common structure and becomes the ``howpublished``
        field of a ``@misc``. Authors become ``Surname, Initials`` names
        joined by ``and``, with a surname of several words braced so BibTeX
        keeps it whole; a list ending in ``(eds.)`` becomes the ``editor``
        field. The title is braced to keep its capitalisation, and the LaTeX
        special characters of the text fields are escaped.
        """
        authors, role = self.authors, "author"
        if _EDITORS.search(authors):
            authors, role = _EDITORS.sub("", authors), "editor"
        fields = [(role, _bibtex_names(authors)), ("title", "{" + _latex(self.title) + "}")]
        article = _ARTICLE.match(self.venue)
        if article is not None and not _PROCEEDINGS.search(article["journal"]):
            kind = "article"
            fields += [("journal", _latex(article["journal"])), ("volume", article["volume"])]
            if article["number"]:
                fields.append(("number", article["number"]))
            fields.append(("pages", article["pages"].replace("-", "--")))
        else:
            kind = "misc"
            fields.append(("howpublished", _latex(self.venue)))
        fields.append(("year", str(self.year)))
        if self.doi:
            fields.append(("doi", self.doi))
        if self.url:
            fields.append(("url", self.url))
        body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
        return f"@{kind}{{{self.key},\n{body}\n}}"


def _load() -> dict[str, Reference]:
    merged: dict[str, Reference] = {}
    folder = files("dispatchatlas.data").joinpath("references")
    for item in sorted(folder.iterdir(), key=lambda entry: entry.name):
        if not item.name.endswith(".toml"):
            continue
        document: dict[str, Any] = tomllib.loads(item.read_text(encoding="utf-8"))
        for key, fields in document.items():
            entry = Reference(key=key, **fields)
            if key in merged and merged[key] != entry:
                raise ConfigError(f"reference {key!r} is defined differently in {item.name}")
            merged[key] = entry
    return merged


@cache
def bibliography() -> MappingProxyType[str, Reference]:
    """Return every bibliography entry, keyed by citation key."""
    return MappingProxyType(dict(sorted(_load().items())))


def reference(key: str) -> Reference:
    """Return the entry with the given key.

    Raises:
        ConfigError: If the key is unknown.
    """
    try:
        return bibliography()[key]
    except KeyError:
        raise ConfigError(f"unknown reference key {key!r}") from None


def cite(source: str) -> str:
    """Render a citation from a key, a DOI, or a URL.

    Keys resolve to the full entry. DOIs (``doi:...`` or ``10....``) and URLs
    are returned as links.
    """
    if source in bibliography():
        return str(bibliography()[source])
    if source.startswith("doi:"):
        return f"https://doi.org/{source[4:]}"
    if source.startswith("10."):
        return f"https://doi.org/{source}"
    return source


def to_bibtex(keys: Iterable[str] | None = None) -> str:
    """Return BibTeX entries, each followed by a blank line.

    Args:
        keys: Citation keys, in the order to write them. Defaults to the
            whole bibliography, sorted by key.

    Raises:
        ConfigError: If a key is unknown.
    """
    entries = bibliography().values() if keys is None else [reference(key) for key in keys]
    return "".join(f"{entry.bibtex()}\n\n" for entry in entries)


def _bibtex_names(authors: str) -> str:
    """Convert ``"A. B. Surname, C. Other and D. Last"`` to BibTeX's ``and``-joined names.

    The leading words that end in a period are the initials and the rest is
    the surname, braced when it has several words: ``"C. A. Coello Coello"``
    gives ``"{Coello Coello}, C. A."``.
    """
    names = []
    for name in _NAME_SEPARATOR.split(authors.strip()):
        words = _latex(name).split()
        if not words:
            continue
        count = 0
        while count < len(words) - 1 and words[count].endswith("."):
            count += 1
        surname = " ".join(words[count:])
        if len(words) - count > 1:
            surname = "{" + surname + "}"
        names.append(f"{surname}, {' '.join(words[:count])}" if count else surname)
    return " and ".join(names)


def _latex(text: str) -> str:
    """Escape the characters LaTeX treats specially."""
    return text.translate(_LATEX_SPECIAL)
