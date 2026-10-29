"""Environment capture: interpreter, platform, hardware, packages, origins, and git state."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
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
from dispatchatlas.core import ConfigError
from tests.campaigns.solvers import registry, tiny_campaign

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

COMMIT = "0123456789abcdef0123456789abcdef01234567"

PACKAGE = Path(environment.__file__).resolve().parent.parent
"""The directory of the package that runs."""


class _Distribution:
    """An installed distribution with a scripted installer record.

    Its files are installed under ``location``, which defaults to the
    directory that holds the package that runs.
    """

    def __init__(
        self,
        name: str,
        version: str,
        record: dict[str, Any] | None = None,
        location: Path | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self._record = record
        self._location = PACKAGE.parent if location is None else location

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return None if self._record is None else json.dumps(self._record)

    def locate_file(self, path: str) -> Path:
        return self._location / path


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
    # Windows states the model, the sockets, and the cores. On Linux they are read from
    # /proc/cpuinfo, which names them on x86 and not on every other architecture.
    x86 = platform.machine().lower() in ("x86_64", "amd64", "i686", "i386")
    if platform.system() == "Windows" or (platform.system() == "Linux" and x86):
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


@needs_git
def test_an_untracked_file_past_the_byte_budget_enters_by_size_and_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repository(tmp_path)
    monkeypatch.setattr(environment, "UNTRACKED_BYTES", 8)
    small, large = tmp_path / "a.txt", tmp_path / "b.bin"
    small.write_bytes(b"small")  # 5 of the 8 bytes
    large.write_bytes(b"0123456789")  # more than the 3 bytes that remain
    read: list[str] = []
    digest = environment._file_digest

    def reading(path: Path) -> bytes:
        read.append(path.name)
        return digest(path)

    monkeypatch.setattr(environment, "_file_digest", reading)
    before = git_state(tmp_path)
    assert before is not None
    assert read == ["a.txt"]
    # Other content of the same size and time is not seen; another time is.
    stamp = large.stat()
    large.write_bytes(b"9876543210")
    os.utime(large, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert git_state(tmp_path) == before
    os.utime(large, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 2_000_000_000))
    assert git_state(tmp_path)["digest"] != before["digest"]  # type: ignore[index]
    os.utime(large, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    large.write_bytes(b"01234567890")
    os.utime(large, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert git_state(tmp_path)["digest"] != before["digest"]  # type: ignore[index]
    # The content of the file within the budget is.
    large.write_bytes(b"9876543210")
    os.utime(large, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    small.write_bytes(b"SMALL")
    assert git_state(tmp_path)["digest"] != before["digest"]  # type: ignore[index]


@needs_git
def test_a_git_state_does_not_write_the_index(tmp_path: Path) -> None:
    _repository(tmp_path)
    index = tmp_path / ".git" / "index"
    # A tracked file with another time and the same content: git refreshes its index.
    later = time.time() + 5
    os.utime(tmp_path / "notes.txt", (later, later))
    before = index.read_bytes()
    state = git_state(tmp_path)
    assert state is not None
    assert state["dirty"] is False
    assert index.read_bytes() == before
    (tmp_path / "new.txt").write_text("untracked\n")
    (tmp_path / "notes.txt").write_text("changed\n")
    assert git_state(tmp_path)["dirty"] is True  # type: ignore[index]
    assert index.read_bytes() == before


@needs_git
def test_an_excluded_directory_is_left_out_of_the_library_and_plugin_checkouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library, plugin = _checkouts(tmp_path, monkeypatch)
    results = [library / "out" / "campaign", plugin / "out" / "campaign"]
    for folder in results:
        folder.mkdir(parents=True)
        (folder / "record.json").write_text("{}")
    counted = capture_environment(tmp_path)
    assert counted["git"]["dispatchatlas"]["untracked"] == 1
    assert counted["plugins"]["plugin"]["git"]["untracked"] == 1
    left_out = capture_environment(tmp_path, exclude=results)
    for state in (left_out["git"]["dispatchatlas"], left_out["plugins"]["plugin"]["git"]):
        assert (state["dirty"], state["untracked"], state["digest"]) == (False, 0, None)
    assert plugin_states(exclude=results) == left_out["plugins"]
    assert distribution_state("plugin", exclude=results) == left_out["plugins"]["plugin"]


@needs_git
def test_a_campaign_directory_inside_the_checkouts_is_no_change_of_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One repository holds the library source, an editable plugin, and the results.
    repository, _ = _checkouts(tmp_path, monkeypatch, plugin="library")
    monkeypatch.chdir(repository)
    campaign = tiny_campaign(on_changed_code="refuse")
    for _ in range(3):
        summary = run(campaign, repository / "out", registry=registry(), max_runs=4)
    sessions = read_sessions(summary.directory)
    assert [session.changes for session in sessions] == [(), (), ()]
    for session in sessions:
        states = session.environment["git"]
        checkouts = (
            states["dispatchatlas"],
            states["working_directory"],
            session.environment["plugins"]["plugin"]["git"],
        )
        assert [state["dirty"] for state in checkouts] == [False, False, False]
    # A file beside the campaign directory is a change of the code.
    (repository / "out" / "notes.txt").write_text("not a result\n")
    with pytest.raises(ConfigError, match=r"environment\.git\.dispatchatlas\.untracked 0 -> 1"):
        run(campaign, repository / "out", registry=registry())


def _checkouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plugin: str = "plugin"
) -> tuple[Path, Path]:
    """Make the library source and an editable plugin git checkouts under ``tmp_path``.

    The library source becomes ``library/src/dispatchatlas`` and the plugin
    is installed from the directory ``plugin``, which may be ``library``.
    """
    library = tmp_path / "library"
    git = _repository(library)
    package = library / "src" / "dispatchatlas"
    (package / "campaigns").mkdir(parents=True)
    for module in (package / "__init__.py", package / "campaigns" / "environment.py"):
        module.write_text("")
    git("add", "src")
    git("commit", "-q", "-m", "package")
    monkeypatch.setattr(environment, "__file__", str(package / "campaigns" / "environment.py"))
    if plugin != "library":
        _repository(tmp_path / plugin)
    record = {"url": (tmp_path / plugin).as_uri(), "dir_info": {"editable": True}}
    _install(monkeypatch, _Distribution("plugin", "0.3", record))
    entry = SimpleNamespace(dist=SimpleNamespace(name="plugin"))
    monkeypatch.setattr(environment, "entry_points", lambda group: [entry])
    return library, tmp_path / plugin


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


def test_a_package_the_installed_distribution_does_not_provide_has_the_origin_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(environment, "_git", lambda arguments, directory: None)
    unknown = {
        "version": None,
        "origin": "path",
        "editable": None,
        "revision": None,
        "source_digest": source_digest(),
    }
    # The installed distribution is another copy: the package that runs is not its file.
    record = {"url": "https://example.org/d.git", "vcs_info": {"vcs": "git", "commit_id": COMMIT}}
    elsewhere = tmp_path / "site-packages"
    (elsewhere / "dispatchatlas").mkdir(parents=True)
    _install(monkeypatch, _Distribution("dispatchatlas", "9.9.9", record, location=elsewhere))
    captured = capture_environment(tmp_path)
    assert captured["library"] == unknown
    assert captured["packages"] == {"dispatchatlas": "9.9.9"}
    missing = _Distribution("dispatchatlas", "9.9.9", record, location=tmp_path / "nowhere")
    _install(monkeypatch, missing)
    assert capture_environment(tmp_path)["library"] == unknown
    # An editable distribution provides the packages under its directory, and no other.
    other = {"url": (tmp_path / "another-checkout").as_uri(), "dir_info": {"editable": True}}
    _install(monkeypatch, _Distribution("dispatchatlas", "9.9.9", other, location=elsewhere))
    assert capture_environment(tmp_path)["library"] == unknown
    this = {"url": PACKAGE.parent.as_uri(), "dir_info": {"editable": True}}
    _install(monkeypatch, _Distribution("dispatchatlas", "0.1.0", this, location=elsewhere))
    assert capture_environment(tmp_path)["library"] == {
        "version": "0.1.0",
        "origin": "directory",
        "editable": True,
        "revision": None,
        "source_digest": source_digest(),
    }


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


def test_the_processor_time_of_a_guest_is_counted_once() -> None:
    # user nice system idle iowait irq softirq steal guest guest_nice; the kernel
    # counts guest time in user time as well. Between the readings the machine is
    # busy for 100 ticks, 80 of them in a guest, and idle for 100.
    first = environment._stat_times("cpu  1000 0 500 8000 100 0 50 0 400 0")
    second = environment._stat_times("cpu  1100 0 500 8100 100 0 50 0 480 0")
    assert first == (8100.0, 9650.0)
    assert second == (8200.0, 9850.0)
    assert environment._stat_times("cpu 1 2 3 4") == (4.0, 10.0)
    for line in ("", "cpu 1 2 3", "cpu0 1 2 3 4 5", "cpu 1 2 three 4"):
        assert environment._stat_times(line) is None


def test_every_session_records_its_environment(tmp_path: Path) -> None:
    summary = run(tiny_campaign(), tmp_path, registry=registry())
    (session,) = read_sessions(summary.directory)
    assert session.environment["python"]["version"] == platform.python_version()
    assert session.workers == 1
    assert session.execution.backend == "sequential"


@needs_git
def test_dirtiness_can_be_judged_within_one_folder(tmp_path: Path) -> None:
    git = _repository(tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "driver.py").write_text("x = 1\n")
    git("add", "scripts/driver.py")
    git("commit", "-q", "-m", "scripts")
    (tmp_path / "notes.txt").write_text("changed outside\n")
    inside = environment.git_state(tmp_path, within="scripts")
    assert inside is not None
    assert (inside["commit"], inside["dirty"]) == (git("rev-parse", "HEAD"), False)
    assert environment.git_state(tmp_path)["dirty"] is True  # type: ignore[index]
    (tmp_path / "scripts" / "driver.py").write_text("x = 2\n")
    assert environment.git_state(tmp_path, within=tmp_path / "scripts")["dirty"] is True  # type: ignore[index]
    assert environment.git_state(tmp_path, within=tmp_path.parent) is None


def test_trees_are_compared_by_path_and_digest(tmp_path: Path) -> None:
    source, installed = tmp_path / "source", tmp_path / "installed"
    for root in (source, installed):
        (root / "sub").mkdir(parents=True)
        (root / "a.py").write_bytes(b"one\n")
        (root / "sub" / "b.py").write_bytes(b"two\n")
    (installed / "sub" / "__pycache__").mkdir()
    (installed / "sub" / "__pycache__" / "b.cpython-313.pyc").write_bytes(b"\0")
    assert environment.compare_trees(installed, source).identical
    (source / "a.py").write_bytes(b"one\r\n")
    assert environment.compare_trees(installed, source).identical
    assert environment.compare_trees(installed, source, normalize_newlines=False).differing == (
        "a.py",
    )
    (installed / "sub" / "b.py").write_bytes(b"changed\n")
    (installed / "extra.py").write_bytes(b"")
    (source / "missing.py").write_bytes(b"")
    found = environment.compare_trees(installed, source)
    assert (found.matched, found.differing, found.extra, found.missing) == (
        1,
        ("sub/b.py",),
        ("extra.py",),
        ("missing.py",),
    )
    assert not found.identical


def test_an_installed_package_is_located_without_importing_it() -> None:
    found = environment.installed_package_dir("dispatchatlas")
    assert found is not None
    assert (found / "processes" / "calls.py").is_file()
    assert environment.installed_package_dir("no_such_package_here") is None
