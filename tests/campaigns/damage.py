"""Ways to damage a campaign directory in a test, one at a time.

A record that a test changes is written back with its hashes computed anew
(:func:`rewrite`) unless the test is about the hashes (:func:`edit`), so
that only the change under test is found.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from dispatchatlas.campaigns.records import (
    Category,
    ErrorKind,
    FailureRecord,
    RunError,
    RunRecord,
)
from dispatchatlas.campaigns.store import failure_path


def edit(path: Path, change: Callable[[dict[str, Any]], object]) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read(path: Path) -> RunRecord:
    return RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))


def rewrite(path: Path, change: Callable[[RunRecord], RunRecord]) -> RunRecord:
    """Replace a record by a changed one whose hashes are computed anew."""
    record = change(read(path))
    record = replace(
        record,
        result_hash=record.outcome.result_hash(),
        identity_hash=record.identity_digest(),
    )
    path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n", "utf-8")
    return record


def fail(
    directory: Path,
    record_path: Path,
    kind: ErrorKind = ErrorKind.TIMEOUT,
    category: Category = Category.TERMINAL,
    *,
    keep: bool = False,
) -> Path:
    """Write the record of a failed first attempt of a run, in place of its record."""
    record = read(record_path)
    failure = FailureRecord(
        run_id=record.run_id,
        campaign=record.campaign,
        config_hash=record.config_hash,
        problem=record.problem,
        solver=record.solver,
        protocol=record.protocol,
        replicate=record.replicate,
        seed=record.seed,
        category=category,
        error=RunError(kind, "scripted failure"),
        provenance=record.provenance,
    )
    path = failure_path(
        directory,
        record.solver.name,
        record.problem.id,
        record.replicate,
        record.protocol.name,
        attempt=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failure.to_dict(), indent=2, sort_keys=True) + "\n", "utf-8")
    if not keep:
        record_path.unlink()
    return path
