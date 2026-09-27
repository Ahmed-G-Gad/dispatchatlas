"""``dispatchatlas references``: the bibliography, as text or BibTeX, to cite what you use."""

from __future__ import annotations

import argparse
from typing import Any

from dispatchatlas.benchmarks.registry import default_registry as default_families
from dispatchatlas.benchmarks.suites import suite
from dispatchatlas.cli._common import (
    PROG,
    CommandError,
    Commands,
    UsageError,
    add_command,
    emit_json,
)
from dispatchatlas.references import bibliography, cite
from dispatchatlas.solvers import default_registry as default_solvers

__all__ = ["register"]


def register(commands: Commands) -> None:
    """Add the ``references`` command."""
    parser = add_command(
        commands,
        "references",
        _references,
        "Export the bibliography of the implemented methods, as text or BibTeX.",
        description="Print the bibliography of the methods, benchmarks, and statistics "
        "DispatchAtlas implements: every entry, the entries with the given keys, or the "
        "sources of given solvers, benchmark families, and suites, so you can cite what "
        "you use.",
        epilog="examples:\n"
        f"  {PROG} references\n"
        f"  {PROG} references kolisch1996serial --format bibtex\n"
        f"  {PROG} references --solver tabu-search --suite taillard-flow-shop --format bibtex "
        "> sources.bib",
    )
    parser.add_argument("keys", nargs="*", metavar="KEY", help="citation key")
    parser.add_argument(
        "--solver", action="append", metavar="ID", help="add the sources of this solver"
    )
    parser.add_argument(
        "--family", action="append", metavar="ID", help="add the sources of this benchmark family"
    )
    parser.add_argument(
        "--suite", action="append", metavar="ID", help="add the sources of this suite"
    )
    parser.add_argument(
        "--format",
        dest="style",
        choices=("text", "bibtex"),
        default="text",
        help="one citation per line, or BibTeX entries (default: text)",
    )


def _references(arguments: argparse.Namespace) -> int:
    if arguments.json and arguments.style == "bibtex":
        raise UsageError("--json and --format bibtex exclude each other")
    sources = _selected(arguments)
    known = bibliography()
    if arguments.json:
        emit_json([_document(source) for source in sources])
        return 0
    for source in sources:
        if arguments.style == "bibtex":
            # A DOI or URL that a solver cites without a key has no entry to write.
            print(known[source].bibtex() if source in known else f"% {cite(source)}")
            print()
        else:
            print(f"[{source}] {cite(source)}" if source in known else cite(source))
    return 0


def _selected(arguments: argparse.Namespace) -> list[str]:
    """Return the requested sources, in the order given, without repeats.

    Keys given on the command line must be in the bibliography. Solvers,
    families, and suites may also cite DOIs or URLs, which are kept as they
    are. With nothing requested, every entry is returned, sorted by key.

    Raises:
        CommandError: If a key is not in the bibliography.
        ConfigError: If a solver, family, or suite is unknown.
    """
    known = bibliography()
    for key in arguments.keys:
        if key not in known:
            raise CommandError(
                f"unknown reference key {key!r}",
                hint=f"'{PROG} references' lists every key with its citation",
            )
    sources = list(arguments.keys)
    if arguments.solver:
        solvers = default_solvers()
        sources += [key for item in arguments.solver for key in solvers.info(item).references]
    if arguments.family:
        families = default_families()
        sources += [key for item in arguments.family for key in families.info(item).references]
    sources += [key for item in arguments.suite or () for key in suite(item).references]
    if not (arguments.keys or arguments.solver or arguments.family or arguments.suite):
        return list(known)
    return list(dict.fromkeys(sources))


def _document(source: str) -> dict[str, Any]:
    """Describe a source; one that is not a key has only its link."""
    entry = bibliography().get(source)
    fields = ("key", "authors", "year", "title", "venue", "doi", "url")
    if entry is None:
        return {**dict.fromkeys(fields), "link": cite(source), "citation": cite(source)}
    return {
        **{name: getattr(entry, name) for name in fields},
        "link": entry.link,
        "citation": str(entry),
    }
