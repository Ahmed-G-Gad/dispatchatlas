"""``dispatchatlas benchmarks``: the benchmark families and their generators."""

from __future__ import annotations

import argparse
from typing import Any

from dispatchatlas.benchmarks.registry import FamilyInfo, default_registry
from dispatchatlas.cli._common import (
    PROG,
    Commands,
    add_command,
    add_group,
    assignment,
    emit_fields,
    emit_json,
    emit_table,
    jsonable,
    non_negative_integer,
    parameter_lines,
    parameters,
    write_problem,
)
from dispatchatlas.references import cite

__all__ = ["register"]


def register(commands: Commands) -> None:
    """Add the ``benchmarks list``, ``show``, and ``generate`` commands."""
    group = add_group(
        commands,
        "benchmarks",
        "List the benchmark families and generate problems.",
        "List the registered benchmark families, built in and from plugins, show their "
        "parameters, and generate problems. Equal parameters and seed give the same problem.",
    )
    add_command(group, "list", _list, "List the benchmark families.")
    show = add_command(
        group,
        "show",
        _show,
        "Show a family's summary, parameters, objectives, and citations.",
        epilog=f"example:\n  {PROG} benchmarks show job-shop",
    )
    show.add_argument("family", metavar="ID", help="family id")
    generate = add_command(
        group,
        "generate",
        _generate,
        "Generate one problem of a family as a problem document.",
        description="Generate one problem of a family and print its problem document, or "
        "write it to a file.",
        epilog="example:\n"
        f"  {PROG} benchmarks generate job-shop --seed 1 --set jobs=6 --set machines=4 "
        "--output js.json",
    )
    generate.add_argument("family", metavar="ID", help="family id")
    generate.add_argument(
        "--seed",
        type=non_negative_integer,
        default=0,
        metavar="N",
        help="generator seed (default: 0)",
    )
    generate.add_argument(
        "--set",
        type=assignment,
        action="append",
        metavar="KEY=VALUE",
        help="set a family parameter; the value is read as JSON when it parses, else as text",
    )
    generate.add_argument("--output", metavar="FILE", help="write the problem to this file")


def _describe(info: FamilyInfo) -> dict[str, Any]:
    """Return a family's metadata as a JSON-ready dictionary, with citations resolved."""
    return {
        "id": info.id,
        "name": info.name,
        "problem_class": info.problem_class,
        "summary": info.summary,
        "objectives": list(info.objectives),
        "parameters": jsonable(info.parameters),
        "references": list(info.references),
        "citations": [cite(source) for source in info.references],
    }


def _list(arguments: argparse.Namespace) -> int:
    infos = default_registry().infos()
    if arguments.json:
        emit_json([_describe(info) for info in infos])
        return 0
    emit_table(
        ("ID", "CLASS", "OBJECTIVES"),
        [(info.id, info.problem_class, info.objectives) for info in infos],
    )
    return 0


def _show(arguments: argparse.Namespace) -> int:
    info = default_registry().info(arguments.family)
    document = _describe(info)
    if arguments.json:
        emit_json(document)
        return 0
    emit_fields(
        [
            ("id", info.id),
            ("name", info.name),
            ("class", info.problem_class),
            ("summary", info.summary),
            ("objectives", info.objectives),
            ("parameters", parameter_lines(info.parameters) or "none"),
            ("references", "\n".join(document["citations"]) or "-"),
        ]
    )
    return 0


def _generate(arguments: argparse.Namespace) -> int:
    values = parameters(arguments.set)
    problem = default_registry().generate(arguments.family, arguments.seed, **values)
    write_problem(problem, arguments.output, as_json=arguments.json)
    return 0
