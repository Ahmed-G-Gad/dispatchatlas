"""``dispatchatlas suites``: published benchmark suites and their instances."""

from __future__ import annotations

import argparse
from typing import Any

from dispatchatlas.benchmarks.suites import Suite, suite, suites
from dispatchatlas.cli._common import (
    PROG,
    CommandError,
    Commands,
    add_command,
    add_group,
    emit_fields,
    emit_json,
    emit_table,
    write_problem,
)
from dispatchatlas.references import cite

__all__ = ["register"]


def register(commands: Commands) -> None:
    """Add the ``suites list``, ``show``, and ``load`` commands."""
    group = add_group(
        commands,
        "suites",
        "List the published benchmark suites and load their instances.",
        "List the published benchmark suites and load their instances. Generated suites "
        "(Taillard's) are rebuilt from their published seeds; file suites are read from "
        "files you download from the suite's source.",
    )
    add_command(group, "list", _list, "List the published suites.")
    show = add_command(
        group,
        "show",
        _show,
        "Show a suite's instances, source, files to download, and citations.",
        epilog=f"example:\n  {PROG} suites show taillard-flow-shop",
    )
    show.add_argument("suite", metavar="ID", help="suite id")
    load = add_command(
        group,
        "load",
        _load,
        "Load one instance of a suite as a problem document.",
        description="Load one instance of a suite and print its problem document, or write "
        "it to a file.",
        epilog="examples:\n"
        f"  {PROG} suites load taillard-flow-shop ta001 --output ta001.json\n"
        f"  {PROG} suites load orlib-job-shop ft06 --directory downloads/orlib",
    )
    load.add_argument("suite", metavar="ID", help="suite id")
    load.add_argument("instance", metavar="INSTANCE", help="instance name, in any case")
    load.add_argument(
        "--directory",
        metavar="D",
        help="directory holding the downloaded files of a file suite, searched recursively",
    )
    load.add_argument("--output", metavar="FILE", help="write the problem to this file")


def _describe(item: Suite) -> dict[str, Any]:
    """Return a suite's metadata as a JSON-ready dictionary, with citations resolved."""
    return {
        "id": item.id,
        "name": item.name,
        "problem_class": item.problem_class,
        "generated": item.generated,
        "instances": list(item.instances),
        "source": item.source,
        "files": item.files,
        "bounds": {
            name: {
                "lower": bound.lower,
                "upper": bound.upper,
                "optimal": bound.optimal,
                "objective": bound.objective,
            }
            for name, bound in item.bounds.items()
        },
        "notes": item.notes,
        "references": list(item.references),
        "citations": [cite(source) for source in item.references],
    }


def _list(arguments: argparse.Namespace) -> int:
    items = suites()
    if arguments.json:
        emit_json([_describe(item) for item in items])
        return 0
    emit_table(
        ("ID", "CLASS", "INSTANCES", "SOURCE"),
        [(item.id, item.problem_class, len(item.instances), _kind(item)) for item in items],
    )
    return 0


def _show(arguments: argparse.Namespace) -> int:
    item = suite(arguments.suite)
    document = _describe(item)
    if arguments.json:
        emit_json(document)
        return 0
    count = len(item.instances)
    emit_fields(
        [
            ("id", item.id),
            ("name", item.name),
            ("class", item.problem_class),
            ("instances", f"{count}: {item.instances[0]} to {item.instances[-1]}"),
            ("source", item.source),
            ("download", item.files if item.files is not None else "nothing: " + _kind(item)),
            ("bounds", f"published for {len(item.bounds)} instances" if item.bounds else "-"),
            ("notes", item.notes or "-"),
            ("references", "\n".join(document["citations"]) or "-"),
        ]
    )
    return 0


def _load(arguments: argparse.Namespace) -> int:
    item = suite(arguments.suite)
    if not item.generated and arguments.directory is None:
        raise CommandError(
            f"suite {item.id!r} is read from files: download {item.files} from {item.source}",
            hint="pass --directory D, the directory that holds the downloaded files",
        )
    problem = item.load(arguments.instance, arguments.directory)
    write_problem(problem, arguments.output, as_json=arguments.json)
    return 0


def _kind(item: Suite) -> str:
    return "generated offline" if item.generated else "downloaded files"
