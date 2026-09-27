"""Run the command line in-process and capture what it prints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.cli import main


@dataclass(frozen=True)
class Result:
    """What one call of the command line returned and printed."""

    code: int
    out: str
    err: str

    def json(self) -> Any:
        return json.loads(self.out)


class Cli:
    """Calls :func:`dispatchatlas.cli.main` with the arguments given, as strings."""

    def __init__(self, capsys: pytest.CaptureFixture[str]) -> None:
        self._capsys = capsys

    def __call__(self, *arguments: str | Path) -> Result:
        self._capsys.readouterr()
        code = main([str(item) for item in arguments])
        captured = self._capsys.readouterr()
        return Result(code, captured.out, captured.err)


def campaign_file(
    directory: Path,
    *,
    problem: Path | None = None,
    solvers: str = 'id = "input-order"\n\n[[solvers]]\nid = "random-search"',
    execution: str = 'backend = "sequential"',
    campaign_id: str = "tiny",
    seeds: str = "[1, 2]",
) -> Path:
    """Write a small campaign: flow shops, and optionally one problem file."""
    source = f'\n[[instances]]\nfile = "{problem.as_posix()}"\n' if problem is not None else ""
    text = f"""schema_version = 1
id = "{campaign_id}"
seed = 7
replicates = 2
{source}
[[instances]]
family = "flow-shop"
parameters = {{ jobs = 4, machines = 3 }}
seeds = {seeds}

[[solvers]]
{solvers}

[[protocols]]
name = "evals"
stop = {{ max_evaluations = 10 }}

[execution]
{execution}
"""
    path = directory / f"{campaign_id}.toml"
    path.write_text(text, encoding="utf-8")
    return path
