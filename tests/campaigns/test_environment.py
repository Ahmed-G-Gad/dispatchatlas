"""Environment capture: interpreter, platform, hardware, packages, and git state."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

from dispatchatlas.campaigns import capture_environment, read_sessions, run
from dispatchatlas.campaigns.environment import git_state
from tests.campaigns.solvers import registry, tiny_campaign


def test_records_interpreter_platform_hardware_and_packages(tmp_path: Path) -> None:
    environment = capture_environment(tmp_path)
    assert environment["python"]["version"] == platform.python_version()
    assert environment["platform"]["system"] == platform.system()
    assert environment["cpu"]["logical"] == os.cpu_count()
    assert environment["cpu"]["available"] >= 1
    assert environment["packages"]["dispatchatlas"]
    assert environment["packages"]["numpy"]
    assert environment["cpu"]["model"] is None or isinstance(environment["cpu"]["model"], str)
    if platform.system() == "Linux":
        assert environment["memory_bytes"] > 0
    assert "working_directory" not in environment["git"]
    assert set(environment) == {"python", "platform", "cpu", "memory_bytes", "packages", "git"}


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_records_the_commit_of_a_git_checkout(tmp_path: Path) -> None:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", *arguments],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    git("init", "-q")
    (tmp_path / "notes.txt").write_text("first\n")
    git("add", "notes.txt")
    git("commit", "-q", "-m", "first")
    state = git_state(tmp_path)
    assert state == {"commit": git("rev-parse", "HEAD"), "dirty": False}
    (tmp_path / "notes.txt").write_text("changed\n")
    assert git_state(tmp_path) == {"commit": git("rev-parse", "HEAD"), "dirty": True}
    assert capture_environment(tmp_path)["git"]["working_directory"]["dirty"] is True
    assert git_state(tmp_path.parent / "not-a-checkout") is None


def test_every_session_records_its_environment(tmp_path: Path) -> None:
    summary = run(tiny_campaign(), tmp_path, registry=registry())
    (session,) = read_sessions(summary.directory)
    assert session.environment["python"]["version"] == platform.python_version()
    assert session.workers == 1
    assert session.execution.backend == "sequential"
