"""Environment capture: interpreter, platform, hardware, packages, origins, and git state."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dispatchatlas.campaigns import capture_environment, environment, read_sessions, run
from dispatchatlas.campaigns.environment import (
    distribution_state,
    git_state,
    plugin_states,
    sample_load,
    source_digest,
)
from tests.campaigns.solvers import registry, tiny_campaign

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

COMMIT = "0123456789abcdef0123456789abcdef01234567"


class _Distribution:
    """An installed distribution with a scripted installer record."""

    def __init__(self, name: str, version: str, record: dict[str, Any] | None = None) -> None:
        self.name = name
        self.version = version
        self._record = record

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return None if self._record is None else json.dumps(self._record)


def _install(monkeypatch: pytest.MonkeyPatch, *distributions: _Distribution) -> None:
    """Make the given distributions the only ones the environment module finds."""
    known = {item.name: item for item in distributions}

    def lookup(name: str) -> _Distribution:
        if name not in known:
            raise PackageNotFoundError(name)
        return known[name]

    monkeypatch.setattr(environment, "distribution", lookup)


def _repository(folder: Path) -> Callable[..., str]:
    """Create a git checkout with one commit in ``folder`` and return a way to query it."""

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", *arguments],
            cwd=folder,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    folder.mkdir(parents=True, exist_ok=True)
    git("init", "-q")
    (folder / "notes.txt").write_text("first\n")
    git("add", "notes.txt")
    git("commit", "-q", "-m", "first")
    return git


def test_records_interpreter_platform_hardware_and_packages(tmp_path: Path) -> None:
    captured = capture_environment(tmp_path)
    assert captured["python"]["version"] == platform.python_version()
    assert captured["platform"]["system"] == platform.system()
    cpu = captured["cpu"]
    assert set(cpu) == {"model", "sockets", "cores", "logical", "available", "allowed"}
    assert cpu["logical"] == os.cpu_count()
    assert cpu["available"] >= 1
    assert cpu["model"] is None or isinstance(cpu["model"], str)
    for key in ("sockets", "cores"):
        assert cpu[key] is None or 1 <= cpu[key] <= cpu["logical"]
    assert captured["packages"]["dispatchatlas"]
    assert captured["packages"]["numpy"]
    if platform.system() == "Linux":
        assert captured["memory_bytes"] > 0
    if platform.system() in ("Linux", "Windows"):
        assert cpu["model"]
        assert cpu["cores"] >= cpu["sockets"] >= 1
    assert "working_directory" not in captured["git"]
    assert set(captured) == {
        "python",
        "platform",
        "cpu",
        "threads",
        "memory_bytes",
        "packages",
        "library",
        "plugins",
        "git",
    }
    json.dumps(captured, allow_nan=False)


@needs_git
def test_records_the_commit_of_a_git_checkout(tmp_path: Path) -> None:
    git = _repository(tmp_path)
    commit = git("rev-parse", "HEAD")
    assert git_state(tmp_path) == {
        "commit": commit,
        "dirty": False,
        "untracked": 0,
        "digest": None,
    }
    (tmp_path / "notes.txt").write_text("changed\n")
    state = git_state(tmp_path)
    assert state is not None
    assert (state["commit"], state["dirty"], state["untracked"]) == (commit, True, 0)
    assert len(state["digest"]) == 64
    assert capture_environment(tmp_path)["git"]["working_directory"] == state
    assert git_state(tmp_path.parent / "not-a-checkout") is None


@needs_git
def test_two_different_edits_of_one_commit_have_different_digests(tmp_path: Path) -> None:
    _repository(tmp_path)
    digests = []
    for text in ("one edit\n", "another edit\n", "one edit\n"):
        (tmp_path / "notes.txt").write_text(text)
        state = git_state(tmp_path)
        assert state is not None
        digests.append(state["digest"])
    assert digests[0] != digests[1]
    assert digests[0] == digests[2]


@needs_git
def test_an_untracked_file_makes_the_state_dirty(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored/\n")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "cache.bin").write_text("not counted")
    (tmp_path / "new.py").write_text("VALUE = 1\n")
    first = git_state(tmp_path)
    assert first is not None
    assert (first["dirty"], first["untracked"]) == (True, 2)  # new.py and .gitignore
    (tmp_path / "new.py").write_text("VALUE = 2\n")
    second = git_state(tmp_path)
    assert second is not None
    assert second["untracked"] == 2
    assert second["digest"] != first["digest"]


@needs_git
def test_an_excluded_directory_is_left_out_of_the_state(tmp_path: Path) -> None:
    git = _repository(tmp_path)
    results = tmp_path / "results" / "campaign"
    results.mkdir(parents=True)
    (results / "record.json").write_text("{}")
    assert git_state(tmp_path)["untracked"] == 1  # type: ignore[index]
    clean = {"commit": git("rev-parse", "HEAD"), "dirty": False, "untracked": 0, "digest": None}
    assert git_state(tmp_path, exclude=[results]) == clean
    assert git_state(tmp_path, exclude=[results, tmp_path.parent / "elsewhere"]) == clean
    assert capture_environment(tmp_path, exclude=[results])["git"]["working_directory"] == clean


@needs_git
def test_many_untracked_files_enter_the_digest_by_path_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repository(tmp_path)
    monkeypatch.setattr(environment, "UNTRACKED_HASHED", 1)
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    before = git_state(tmp_path)
    (tmp_path / "b.txt").write_text("other content")
    assert git_state(tmp_path) == before
    (tmp_path / "c.txt").write_text("c")
    assert git_state(tmp_path)["digest"] != before["digest"]  # type: ignore[index]


def test_the_origin_of_an_installed_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    vcs = {"url": "https://example.org/x.git", "vcs_info": {"vcs": "git", "commit_id": COMMIT}}
    archive = {"url": "https://example.org/x.whl", "archive_info": {"hashes": {"sha256": "ab12"}}}
    legacy = {"url": "https://example.org/x.whl", "archive_info": {"hash": "sha256=cd34"}}
    built = {"url": "file:///somewhere/x", "dir_info": {}}
    _install(
        monkeypatch,
        _Distribution("from-vcs", "1.0", vcs),
        _Distribution("from-archive", "1.1", archive),
        _Distribution("from-legacy-archive", "1.2", legacy),
        _Distribution("from-directory", "1.3", built),
        _Distribution("from-index", "1.4"),
    )
    assert distribution_state("from-vcs") == {
        "version": "1.0",
        "origin": "vcs",
        "editable": False,
        "revision": COMMIT,
    }
    assert distribution_state("from-archive") == {
        "version": "1.1",
        "origin": "archive",
        "editable": False,
        "revision": "sha256=ab12",
    }
    assert distribution_state("from-legacy-archive")["revision"] == "sha256=cd34"  # type: ignore[index]
    assert distribution_state("from-directory") == {
        "version": "1.3",
        "origin": "directory",
        "editable": False,
        "revision": None,
    }
    assert distribution_state("from-index") == {
        "version": "1.4",
        "origin": "index",
        "editable": False,
        "revision": None,
    }
    assert distribution_state("not-installed") is None


@needs_git
def test_an_editable_distribution_has_the_revision_of_its_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git = _repository(tmp_path / "plugin")
    record = {"url": (tmp_path / "plugin").as_uri(), "dir_info": {"editable": True}}
    _install(monkeypatch, _Distribution("plugin", "0.3", record))
    state = distribution_state("plugin")
    assert state is not None
    assert (state["origin"], state["editable"]) == ("directory", True)
    assert state["revision"] == git("rev-parse", "HEAD")
    assert state["git"] == git_state(tmp_path / "plugin")
    (tmp_path / "plugin" / "notes.txt").write_text("edited\n")
    assert distribution_state("plugin")["git"]["dirty"] is True  # type: ignore[index]


def test_an_installed_copy_of_the_library_records_the_revision_it_was_installed_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = {"url": "https://example.org/d.git", "vcs_info": {"vcs": "git", "commit_id": COMMIT}}
    _install(monkeypatch, _Distribution("dispatchatlas", "0.1.0", record))
    # An installed copy is no git checkout: every git query is unanswered.
    monkeypatch.setattr(environment, "_git", lambda arguments, directory: None)
    captured = capture_environment(tmp_path)
    assert captured["library"] == {
        "version": "0.1.0",
        "origin": "vcs",
        "editable": False,
        "revision": COMMIT,
        "source_digest": source_digest(),
    }
    assert captured["git"] == {}
    assert captured["packages"] == {"dispatchatlas": "0.1.0"}


@needs_git
def test_a_checkout_of_the_library_records_the_revision_of_its_source(tmp_path: Path) -> None:
    source = Path(environment.__file__).resolve().parent.parent
    if git_state(source) is None:
        pytest.skip("the library source is not in a git checkout")
    captured = capture_environment(tmp_path)
    assert captured["library"]["revision"] == captured["git"]["dispatchatlas"]["commit"]
    assert captured["library"]["version"] == captured["packages"]["dispatchatlas"]
    assert len(captured["library"]["source_digest"]) == 64
    assert set(captured["git"]["dispatchatlas"]) == {"commit", "dirty", "untracked", "digest"}


def test_every_plugin_distribution_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    vcs = {"url": "https://example.org/p.git", "vcs_info": {"vcs": "git", "commit_id": COMMIT}}
    _install(
        monkeypatch, _Distribution("solver-pack", "2.0", vcs), _Distribution("family-pack", "0.9")
    )
    entries = {
        "dispatchatlas.solvers": [
            SimpleNamespace(dist=SimpleNamespace(name="solver-pack")),
            SimpleNamespace(dist=SimpleNamespace(name="dispatchatlas")),
            SimpleNamespace(dist=None),
        ],
        "dispatchatlas.benchmarks": [SimpleNamespace(dist=SimpleNamespace(name="family-pack"))],
    }
    monkeypatch.setattr(environment, "entry_points", lambda group: entries[group])
    assert plugin_states() == {
        "family-pack": {"version": "0.9", "origin": "index", "editable": False, "revision": None},
        "solver-pack": {"version": "2.0", "origin": "vcs", "editable": False, "revision": COMMIT},
    }


def test_the_source_digest_follows_the_files_and_not_their_line_endings(tmp_path: Path) -> None:
    package = tmp_path / "package"
    (package / "inner" / "__pycache__").mkdir(parents=True)
    (package / "module.py").write_bytes(b"VALUE = 1\nOTHER = 2\n")
    (package / "inner" / "data.toml").write_bytes(b"key = 1\n")
    first = source_digest(package)
    assert first is not None
    assert len(first) == 64
    (package / "inner" / "__pycache__" / "module.cpython-312.pyc").write_bytes(b"compiled")
    (package / "module.py").write_bytes(b"VALUE = 1\r\nOTHER = 2\r\n")
    assert source_digest(package) == first
    (package / "module.py").write_bytes(b"VALUE = 1\nOTHER = 3\n")
    assert source_digest(package) != first
    (package / "module.py").write_bytes(b"VALUE = 1\nOTHER = 2\n")
    (package / "inner" / "data.toml").rename(package / "inner" / "moved.toml")
    assert source_digest(package) != first
    assert source_digest() == source_digest()


def test_a_load_sample_states_the_busy_share_of_the_processors() -> None:
    sample = sample_load(0.05)
    assert set(sample) == {"at", "seconds", "busy", "load_average"}
    assert sample["seconds"] == 0.05
    assert sample["at"].endswith("Z")
    if platform.system() in ("Linux", "Windows"):
        assert 0.0 <= sample["busy"] <= 1.0
    if platform.system() == "Windows":
        assert sample["load_average"] is None
    else:
        assert sample["load_average"] >= 0.0


def test_every_session_records_its_environment(tmp_path: Path) -> None:
    summary = run(tiny_campaign(), tmp_path, registry=registry())
    (session,) = read_sessions(summary.directory)
    assert session.environment["python"]["version"] == platform.python_version()
    assert session.workers == 1
    assert session.execution.backend == "sequential"
