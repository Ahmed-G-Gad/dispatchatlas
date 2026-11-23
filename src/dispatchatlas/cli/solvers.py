"""``dispatchatlas solvers``: the registered solvers and their metadata."""

from __future__ import annotations

import argparse
from typing import Any

from dispatchatlas.cli._common import (
    PROG,
    Commands,
    add_command,
    add_group,
    emit_fields,
    emit_json,
    emit_table,
    jsonable,
    parameter_lines,
)
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.model import FEATURES, SHAPES
from dispatchatlas.core.solver import SOLVER_FAMILIES, SolverInfo
from dispatchatlas.references import cite
from dispatchatlas.solvers import default_registry

__all__ = ["register"]


def register(commands: Commands) -> None:
    """Add the ``solvers list`` and ``solvers show`` commands."""
    group = add_group(
        commands,
        "solvers",
        "List the registered solvers and show their metadata.",
        "List the registered solvers, built in and from plugins, and show what each one "
        "does, its parameters, the problem features it models, and its sources.",
    )
    listing = add_command(
        group,
        "list",
        _list,
        "List the registered solvers.",
        epilog=f"example:\n  {PROG} solvers list --family local-search",
    )
    listing.add_argument("--family", metavar="F", help="list only the solvers of this family")
    show = add_command(
        group,
        "show",
        _show,
        "Show a solver's metadata, parameters, supported features, and citations.",
        epilog=f"example:\n  {PROG} solvers show simulated-annealing",
    )
    show.add_argument("solver", metavar="ID", help="solver id")


def _describe(info: SolverInfo) -> dict[str, Any]:
    """Return a solver's metadata as a JSON-ready dictionary, with citations resolved.

    The keys ``shape`` and ``shape_description`` are present only for a solver that
    requires a problem shape.
    """
    document: dict[str, Any] = {
        "id": info.id,
        "name": info.name,
        "family": info.family,
        "family_description": SOLVER_FAMILIES.get(info.family),
        "summary": info.summary,
        "stochastic": info.stochastic,
        "iterative": info.iterative,
        "multi_objective": info.multi_objective,
        "requires": list(info.requires),
        "parameters": jsonable(info.parameters),
        "supports": sorted(info.supports),
        "unsupported": sorted(set(FEATURES) - info.supports),
        "references": list(info.references),
        "citations": [cite(source) for source in info.references],
    }
    if info.shape is not None:
        document["shape"] = info.shape
        document["shape_description"] = SHAPES[info.shape]
    return document


def _list(arguments: argparse.Namespace) -> int:
    infos = default_registry().infos()
    family: str | None = arguments.family
    if family is not None:
        known = sorted({info.family for info in infos} | set(SOLVER_FAMILIES))
        if family not in known:
            raise ConfigError(f"unknown solver family {family!r}; known: {', '.join(known)}")
        infos = tuple(info for info in infos if info.family == family)
    if arguments.json:
        emit_json([_describe(info) for info in infos])
        return 0
    emit_table(
        ("ID", "FAMILY", "TRAITS"), [(info.id, info.family, _traits(info)) for info in infos]
    )
    return 0


def _show(arguments: argparse.Namespace) -> int:
    info = default_registry().info(arguments.solver)
    document = _describe(info)
    if arguments.json:
        emit_json(document)
        return 0
    iterative = (
        "yes: set --time-limit, --max-evaluations, or --max-iterations"
        if info.iterative
        else "no: it stops on its own"
    )
    family = document["family_description"] or "a family defined by a plugin"
    requires = ", ".join(f"the {extra} extra" for extra in info.requires)
    emit_fields(
        [
            ("id", info.id),
            ("name", info.name),
            ("family", f"{info.family} ({family})"),
            ("summary", info.summary),
            ("stochastic", info.stochastic),
            ("iterative", iterative),
            ("multi-objective", info.multi_objective),
            ("requires", requires or "nothing beyond the base install"),
            ("parameters", parameter_lines(info.parameters) or "none"),
            ("models", ", ".join(document["supports"]) or "none of the optional features"),
            ("does not model", ", ".join(document["unsupported"]) or "-"),
            *(
                [("shape", f"{info.shape}: {document['shape_description']}")]
                if info.shape is not None
                else []
            ),
            ("references", "\n".join(document["citations"]) or "-"),
        ]
    )
    return 0


def _traits(info: SolverInfo) -> str:
    traits = ["stochastic" if info.stochastic else "deterministic"]
    if info.iterative:
        traits.append("iterative")
    if info.multi_objective:
        traits.append("multi-objective")
    if info.shape is not None:
        traits.append(f"needs the {info.shape} shape")
    traits.extend(f"needs [{extra}]" for extra in info.requires)
    return ", ".join(traits)
