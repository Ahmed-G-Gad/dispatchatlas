"""``dispatchatlas version``: the versions of the package, Python, and optional backends."""

from __future__ import annotations

import argparse
import platform
import re
from importlib.metadata import PackageNotFoundError, entry_points, metadata, version
from typing import Any

from dispatchatlas import __version__
from dispatchatlas.benchmarks.registry import PLUGIN_GROUP as BENCHMARK_PLUGINS
from dispatchatlas.cli._common import (
    PROG,
    Commands,
    add_command,
    emit_fields,
    emit_json,
    emit_table,
)
from dispatchatlas.solvers.registry import PLUGIN_GROUP as SOLVER_PLUGINS

__all__ = ["register"]

_NAME = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_EXTRA = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")


def register(commands: Commands) -> None:
    """Add the ``version`` command."""
    add_command(
        commands,
        "version",
        _show,
        "Show the versions of DispatchAtlas, Python, and the optional backends.",
        description="Show the versions of DispatchAtlas, Python, and the dependencies; which "
        "optional backends are installed, with the extra that provides each; and the "
        "installed solver and benchmark plugins.",
    )


def _versions() -> dict[str, Any]:
    """Describe the installed software.

    The optional dependencies are those the installed package metadata
    declares, each with the smallest extra that provides it.

    Returns:
        A JSON-ready dictionary with the keys ``dispatchatlas``, ``python``,
        ``platform``, ``dependencies``, ``optional``, and ``plugins``.
    """
    required, optional = _declared()
    return {
        "dispatchatlas": __version__,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "dependencies": [{"package": name, "version": _installed(name)} for name in required],
        "optional": [
            {"package": name, "extra": extra, "version": _installed(name)}
            for name, extra in optional
        ],
        "plugins": {
            group: sorted(entry.name for entry in entry_points(group=group))
            for group in (SOLVER_PLUGINS, BENCHMARK_PLUGINS)
        },
    }


def _show(arguments: argparse.Namespace) -> int:
    report = _versions()
    if arguments.json:
        emit_json(report)
        return 0
    python = report["python"]
    plugins = report["plugins"]
    emit_fields(
        [
            (PROG, report["dispatchatlas"]),
            ("python", f"{python['version']} ({python['implementation']})"),
            ("platform", report["platform"]),
            *(
                (item["package"], item["version"] or "not installed")
                for item in report["dependencies"]
            ),
            ("solver plugins", plugins[SOLVER_PLUGINS] or "none"),
            ("benchmark plugins", plugins[BENCHMARK_PLUGINS] or "none"),
        ]
    )
    if report["optional"]:
        print()
        emit_table(
            ("OPTIONAL", "VERSION", "EXTRA"),
            [
                (
                    item["package"],
                    item["version"] or "not installed",
                    item["extra"],
                )
                for item in report["optional"]
            ],
        )
    return 0


def _declared() -> tuple[list[str], list[tuple[str, str]]]:
    """Return the required distributions, and each optional one with its smallest extra."""
    try:
        lines = metadata(PROG).get_all("Requires-Dist") or []
    except PackageNotFoundError:
        return [], []
    required: list[str] = []
    extras: dict[str, set[str]] = {}
    for line in lines:
        requirement, _, marker = line.partition(";")
        named = _NAME.match(requirement)
        if named is None or named.group(1).lower() == PROG:
            continue
        condition = _EXTRA.search(marker)
        if condition is None:
            required.append(named.group(1))
        else:
            extras.setdefault(condition.group(1), set()).add(named.group(1))
    smallest: dict[str, str] = {}
    for extra, names in sorted(extras.items(), key=lambda item: (len(item[1]), item[0])):
        for name in sorted(names):
            smallest.setdefault(name, extra)
    return required, sorted(smallest.items(), key=lambda item: (item[1], item[0]))


def _installed(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
