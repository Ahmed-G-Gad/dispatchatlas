"""``dispatchatlas catalogs``: named instance sets, written to disk with a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from dispatchatlas.benchmarks.catalog import Catalog, build_catalog, catalogs, materialize
from dispatchatlas.cli._common import (
    PROG,
    CommandError,
    Commands,
    add_command,
    add_group,
    emit_fields,
    emit_json,
    emit_table,
    jsonable,
    non_negative_integer,
)
from dispatchatlas.core.errors import ConfigError

__all__ = ["register"]


def register(commands: Commands) -> None:
    """Add the ``catalogs list``, ``show``, and ``build`` commands."""
    group = add_group(
        commands,
        "catalogs",
        "List the named instance catalogs and write them to disk.",
        "List the built-in instance catalogs, show their entries, and write every instance "
        "of one to a directory, with a manifest that records each file's SHA-256 digest, "
        "family, parameters, and seed.",
    )
    add_command(group, "list", _list, "List the built-in catalogs.")
    show = add_command(group, "show", _show, "Show the families and parameters of a catalog.")
    show.add_argument("catalog", metavar="NAME", help="catalog name")
    build = add_command(
        group,
        "build",
        _build,
        "Generate every instance of a catalog and write it with a manifest.",
        description="Generate every instance of a catalog and write it to a directory as "
        "problems/<id>.json with a manifest.json. Instance seeds derive from one root seed, "
        "so equal seeds give identical files.",
        epilog=f"example:\n  {PROG} catalogs build smoke --output instances/smoke --seed 1",
    )
    build.add_argument("catalog", metavar="NAME", help="catalog name")
    build.add_argument("--output", metavar="DIR", required=True, help="directory to write")
    build.add_argument(
        "--seed", type=non_negative_integer, default=0, metavar="N", help="root seed (default: 0)"
    )
    build.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the problems of a directory that is not empty",
    )


def _describe(catalog: Catalog) -> dict[str, Any]:
    """Return a catalog as a JSON-ready dictionary."""
    return {
        "name": catalog.name,
        "description": catalog.description,
        "size": catalog.size,
        "families": list(catalog.families),
        "entries": [
            {
                "name": entry.label,
                "family": entry.family,
                "instances": entry.instances,
                "parameters": jsonable(entry.parameters),
            }
            for entry in catalog.entries
        ],
    }


def _list(arguments: argparse.Namespace) -> int:
    items = list(catalogs().values())
    if arguments.json:
        emit_json([_describe(item) for item in items])
        return 0
    emit_table(
        ("NAME", "INSTANCES", "DESCRIPTION"),
        [(item.name, item.size, item.description) for item in items],
    )
    return 0


def _show(arguments: argparse.Namespace) -> int:
    catalog = _catalog(arguments.catalog)
    if arguments.json:
        emit_json(_describe(catalog))
        return 0
    emit_fields(
        [
            ("name", catalog.name),
            ("description", catalog.description),
            ("instances", catalog.size),
            ("families", catalog.families),
        ]
    )
    print()
    emit_table(
        ("ENTRY", "FAMILY", "INSTANCES", "PARAMETERS"),
        [
            (
                entry.label,
                entry.family,
                entry.instances,
                " ".join(f"{key}={value}" for key, value in entry.parameters.items()),
            )
            for entry in catalog.entries
        ],
    )
    return 0


def _build(arguments: argparse.Namespace) -> int:
    catalog = _catalog(arguments.catalog)
    target = Path(arguments.output)
    if target.is_dir() and any(target.iterdir()) and not arguments.overwrite:
        raise CommandError(
            f"{target} is not empty",
            hint="pass --overwrite to replace the problems it holds, or choose another directory",
        )
    problems = build_catalog(catalog, seed=arguments.seed)
    directory = materialize(problems, target, overwrite=arguments.overwrite)
    report = {
        "catalog": catalog.name,
        "seed": arguments.seed,
        "directory": directory,
        "manifest": directory / "manifest.json",
        "problems": [problem.id for problem in problems],
    }
    if arguments.json:
        emit_json(report)
        return 0
    print(
        f"wrote {len(problems)} problems of catalog {catalog.name} (seed {arguments.seed}) "
        f"to {directory}, listed in {report['manifest']}"
    )
    return 0


def _catalog(name: str) -> Catalog:
    known = catalogs()
    if name not in known:
        raise ConfigError(f"unknown catalog {name!r}; known: {', '.join(known)}")
    return known[name]
