"""The error hierarchy, logging helpers, and atomic file writes."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import pytest

from dispatchatlas.core.errors import (
    ConfigError,
    DispatchAtlasError,
    MissingDependencyError,
    ModelError,
    SerializationError,
    SolverError,
    UnsupportedProblemError,
)
from dispatchatlas.core.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    read_json,
)
from dispatchatlas.core.logs import ROOT_LOGGER, configure_logging, get_logger

# -------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("error", "builtin"),
    [
        (ModelError, ValueError),
        (SerializationError, ValueError),
        (ConfigError, ValueError),
        (SolverError, RuntimeError),
        (UnsupportedProblemError, RuntimeError),
    ],
)
def test_errors_belong_to_the_library_and_to_a_builtin_family(
    error: type[Exception], builtin: type[Exception]
) -> None:
    assert issubclass(error, DispatchAtlasError)
    assert issubclass(error, builtin)
    assert issubclass(UnsupportedProblemError, SolverError)


def test_model_errors_list_their_issues() -> None:
    error = ModelError("problem 'p' is invalid", ("first defect", "second defect"))
    assert error.issues == ("first defect", "second defect")
    assert str(error) == "problem 'p' is invalid\n  - first defect\n  - second defect"
    assert str(ModelError("plain")) == "plain"
    assert ModelError("plain").issues == ()


def test_missing_dependencies_name_the_extra_to_install() -> None:
    error = MissingDependencyError("ortools", "exact", "The CP-SAT solver")
    assert isinstance(error, ImportError)
    assert isinstance(error, DispatchAtlasError)
    assert (error.package, error.extra) == ("ortools", "exact")
    assert str(error) == (
        "The CP-SAT solver requires the optional package 'ortools'. "
        'Install it with: pip install "dispatchatlas[exact]"'
    )


# ------------------------------------------------------------------- logging


def test_loggers_nest_under_the_library_logger() -> None:
    assert get_logger("dispatchatlas.solvers.search").name == "dispatchatlas.solvers.search"
    assert get_logger(ROOT_LOGGER).name == "dispatchatlas"
    assert get_logger("my_plugin").name == "dispatchatlas.my_plugin"
    assert get_logger("dispatchatlasx").name == "dispatchatlas.dispatchatlasx"
    root = logging.getLogger(ROOT_LOGGER)
    assert any(isinstance(handler, logging.NullHandler) for handler in root.handlers)


def test_configure_logging_replaces_its_own_handler() -> None:
    root = logging.getLogger(ROOT_LOGGER)
    before, level = list(root.handlers), root.level
    try:
        configure_logging("DEBUG")
        configure_logging(logging.WARNING)
        added = [handler for handler in root.handlers if handler not in before]
        assert len(added) == 1
        assert isinstance(added[0], logging.StreamHandler)
        assert root.level == logging.WARNING
    finally:
        root.handlers[:] = before
        root.setLevel(level)


# ------------------------------------------------------------------------ io


def test_atomic_writes_create_parents_and_leave_no_temporary_files(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "data.bin"
    assert atomic_write_bytes(target, b"\x00\x01") == target
    assert target.read_bytes() == b"\x00\x01"
    atomic_write_text(target, "é\n")
    assert target.read_bytes() == "é\n".encode()
    assert [item.name for item in target.parent.iterdir()] == ["data.bin"]


def test_json_documents_are_sorted_and_end_with_a_newline(tmp_path: Path) -> None:
    path = atomic_write_json(tmp_path / "doc.json", {"b": [1, 2.5], "a": None}, indent=None)
    assert path.read_text(encoding="utf-8") == '{"a": null, "b": [1, 2.5]}\n'
    assert read_json(path) == {"a": None, "b": [1, 2.5]}
    with pytest.raises(SerializationError, match="cannot write"):
        atomic_write_json(tmp_path / "nan.json", {"x": math.nan})
    with pytest.raises(SerializationError, match="cannot write"):
        atomic_write_json(tmp_path / "set.json", {1, 2})
    assert sorted(item.name for item in tmp_path.iterdir()) == ["doc.json"]
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(SerializationError, match="invalid JSON"):
        read_json(tmp_path / "bad.json")


def test_a_failed_write_keeps_the_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "keep.txt"
    atomic_write_text(target, "old\n")

    def refuse(descriptor: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", refuse)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(target, "new\n")
    assert target.read_text(encoding="utf-8") == "old\n"
    assert [item.name for item in tmp_path.iterdir()] == ["keep.txt"]
