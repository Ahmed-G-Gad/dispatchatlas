"""The runner's lock tells a holder from a file system without flock, and falls back to a file."""

from __future__ import annotations

import errno
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.core import ConfigError
from dispatchatlas.processes.affinity import pinning_refusal_text
from dispatchatlas.processes.calls import Call, _alive, _held, _Lock, _try_lock, run_calls
from dispatchatlas.processes.rows import read_rows
from tests.processes.helpers import ROOT, slot_cpus

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

EXCLUSIVE = "rows.lock.excl"
HOLDER = ["EAGAIN", "EWOULDBLOCK", "EACCES"]
UNSUPPORTED = ["ENOSYS", "ENOTSUP", "EOPNOTSUPP"]
OTHER = ["EINVAL", "ENOLCK", "EIO", "EBADF", None]

RUNNER = """
import errno
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    import msvcrt

    def refuse(*arguments):
        raise OSError(errno.ENOSYS, "simulated")

    msvcrt.locking = refuse
else:
    import fcntl

    def refuse(*arguments):
        raise OSError(errno.ENOSYS, "simulated")

    fcntl.flock = refuse

from dispatchatlas.processes.calls import _held

signal = Path(sys.argv[3])
with _held(Path(sys.argv[1]), Path(sys.argv[2])):
    deadline = time.monotonic() + 120
    while not signal.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    # A crash: the exit runs no clause that releases the lock.
    os._exit(3)
"""
"""A runner on a file system without flock, which holds its rows file until a signal file
appears and then ends as a crash does."""


def _fail_with(monkeypatch: pytest.MonkeyPatch, name: str | None) -> None:
    """Make the operating system's lock call fail with the errno ``name``, or with none."""
    code = None if name is None else getattr(errno, name)

    def fail(*arguments: Any) -> None:
        if code is None:
            raise OSError("a failure without an errno")
        raise OSError(code, "simulated")

    if sys.platform == "win32":
        monkeypatch.setattr(msvcrt, "locking", fail)
    else:
        monkeypatch.setattr(fcntl, "flock", fail)


def _record(host: object, pid: object, since: str = "2026-11-20T10:00:00.000000Z") -> str:
    return json.dumps({"host": host, "pid": pid, "time": since}, sort_keys=True) + "\n"


def _wait_for_lock_file(path: Path, runner: subprocess.Popen[bytes]) -> None:
    """Wait until the runner has written its lock file, or fail when it ends or takes too long."""
    deadline = time.monotonic() + 60.0
    while not (path.is_file() and path.stat().st_size > 0):
        assert runner.poll() is None, "the runner ended before it took the lock"
        assert time.monotonic() < deadline, "the runner did not take the lock in time"
        time.sleep(0.02)


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    path = tmp_path / "results"
    path.mkdir()
    return path


@pytest.fixture
def lock(folder: Path) -> Path:
    """The path of the lock file of the rows file ``folder / "rows.jsonl"``."""
    return folder / "rows.lock"


@pytest.fixture
def rows(folder: Path) -> Path:
    return folder / "rows.jsonl"


@pytest.fixture
def dead_pid() -> Iterator[int]:
    """The id of a process of this host that has ended.

    The test keeps the ended process's handle, so that no other process can be given its id.
    """
    with subprocess.Popen([sys.executable, "-c", "pass"]) as child:
        child.wait()
        yield child.pid


# ------------------------------------------------------------ the errno decides


@pytest.mark.parametrize("name", HOLDER)
def test_a_refusal_by_a_holder_means_in_use(
    name: str, folder: Path, lock: Path, rows: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_with(monkeypatch, name)
    with lock.open("a+b") as handle:
        assert _try_lock(handle) is _Lock.HELD
    with pytest.raises(ConfigError, match="in use by another runner"), _held(lock, rows):
        pytest.fail("a run entered a rows file that another holds")
    assert not (folder / EXCLUSIVE).exists()


@pytest.mark.parametrize("name", UNSUPPORTED)
def test_a_file_system_without_flock_takes_the_exclusive_file_instead(
    name: str, folder: Path, lock: Path, rows: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_with(monkeypatch, name)
    with lock.open("a+b") as handle:
        assert _try_lock(handle) is _Lock.UNSUPPORTED
    with _held(lock, rows):
        assert (folder / EXCLUSIVE).is_file()
    assert not (folder / EXCLUSIVE).exists()


@pytest.mark.parametrize("name", OTHER)
def test_any_other_errno_still_means_in_use(
    name: str | None, folder: Path, lock: Path, rows: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_with(monkeypatch, name)
    with lock.open("a+b") as handle:
        assert _try_lock(handle) is _Lock.HELD
    with pytest.raises(ConfigError, match="in use by another runner"), _held(lock, rows):
        pytest.fail("a run entered a rows file that another holds")
    assert not (folder / EXCLUSIVE).exists()


def test_a_file_system_that_locks_never_takes_the_exclusive_file(
    folder: Path, lock: Path, rows: Path
) -> None:
    with _held(lock, rows):
        assert not (folder / EXCLUSIVE).exists()
        with (
            pytest.raises(ConfigError, match="in use by another runner") as refused,
            _held(lock, rows),
        ):
            pytest.fail("a second run entered the rows file")
        assert f"pid {os.getpid()} since" in str(refused.value)
        assert not (folder / EXCLUSIVE).exists()
    with _held(lock, rows):
        pass


# ------------------------------------------------------------ the exclusive file


def test_the_exclusive_file_holds_the_host_the_process_and_the_time(
    folder: Path, lock: Path, rows: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_with(monkeypatch, "ENOSYS")
    with _held(lock, rows):
        text = (folder / EXCLUSIVE).read_bytes().decode("ascii")
    assert text.endswith("\n")
    owner = json.loads(text)
    assert list(owner) == ["host", "pid", "time"]
    assert owner["host"] == socket.gethostname()
    assert owner["pid"] == os.getpid()
    assert owner["time"].endswith("Z")
    assert datetime.fromisoformat(owner["time"]).utcoffset() == timedelta(0)


def test_the_exclusive_file_refuses_a_second_run_and_is_released_for_a_third(
    folder: Path, lock: Path, rows: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_with(monkeypatch, "ENOSYS")
    with _held(lock, rows):
        written = (folder / EXCLUSIVE).read_bytes()
        with (
            pytest.raises(ConfigError, match="in use by another runner") as refused,
            _held(lock, rows),
        ):
            pytest.fail("a second run entered the rows file")
        assert f"process {os.getpid()} on {socket.gethostname()}" in str(refused.value)
        # The refusal leaves the holder's file as it was.
        assert (folder / EXCLUSIVE).read_bytes() == written
    assert not (folder / EXCLUSIVE).exists()
    with _held(lock, rows):
        assert (folder / EXCLUSIVE).is_file()


def test_the_exclusive_file_is_released_when_the_run_fails(
    folder: Path, lock: Path, rows: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_with(monkeypatch, "ENOSYS")
    with pytest.raises(RuntimeError, match="boom"), _held(lock, rows):
        raise RuntimeError("boom")
    assert not (folder / EXCLUSIVE).exists()


def test_a_run_does_not_remove_a_lock_file_that_another_process_replaced(
    folder: Path, lock: Path, rows: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_with(monkeypatch, "ENOSYS")
    with _held(lock, rows):
        (folder / EXCLUSIVE).write_text(_record("elsewhere", 5))
    assert (folder / EXCLUSIVE).read_text() == _record("elsewhere", 5)


def _echo(index: int) -> Call:
    return Call(
        key=["e", index],
        target="tests.processes.child_targets:echo",
        arguments={"key": ["e", index]},
    )


def test_calls_run_without_flock_and_have_one_runner_at_a_time(
    folder: Path, lock: Path, rows: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_with(monkeypatch, "ENOSYS")
    settings: dict[str, Any] = {
        "key_of": lambda row: row["key"],
        "cpus": slot_cpus(1),
        "pin": pinning_refusal_text() is None,
        "cwd": ROOT,
    }
    with _held(lock, rows), pytest.raises(ConfigError, match="in use by another runner"):
        run_calls([_echo(0)], rows, **settings)
    assert not rows.exists()
    assert not (folder / "rows.children.log").exists()
    report = run_calls([_echo(0), _echo(1)], rows, **settings)
    assert report.rows_written == 2
    assert sorted(row["key"][1] for row in read_rows(rows)) == [0, 1]
    assert not (folder / EXCLUSIVE).exists()


# ------------------------------------------------------------ a file left behind


def test_a_file_of_a_process_that_has_ended_on_this_host_is_reported_as_stale(
    folder: Path, lock: Path, rows: Path, dead_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_with(monkeypatch, "ENOSYS")
    left = folder / EXCLUSIVE
    left.write_text(_record(socket.gethostname(), dead_pid))
    with pytest.raises(ConfigError, match="stale") as refused, _held(lock, rows):
        pytest.fail("a run entered a rows file with a stale lock file")
    message = str(refused.value)
    assert str(left) in message
    assert f"process {dead_pid} on {socket.gethostname()}, since 2026-11-20T10:00:00" in message
    assert "Remove the file" in message
    # Nothing removes the file but its user, who then runs again.
    assert left.read_text() == _record(socket.gethostname(), dead_pid)
    left.unlink()
    with _held(lock, rows):
        assert left.is_file()


def test_a_file_of_a_running_process_on_this_host_is_in_use_not_stale(
    folder: Path, lock: Path, rows: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_with(monkeypatch, "ENOSYS")
    left = folder / EXCLUSIVE
    left.write_text(_record(socket.gethostname(), os.getpid()))
    with pytest.raises(ConfigError, match="in use by another runner") as refused, _held(lock, rows):
        pytest.fail("a run entered a rows file that a running process holds")
    assert "stale" not in str(refused.value)
    assert left.read_text() == _record(socket.gethostname(), os.getpid())


def test_a_file_of_another_host_is_in_use_whatever_its_process_is(
    folder: Path, lock: Path, rows: Path, dead_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_with(monkeypatch, "ENOSYS")
    left = folder / EXCLUSIVE
    other = socket.gethostname() + "-other"
    left.write_text(_record(other, dead_pid))
    with pytest.raises(ConfigError, match="in use by another runner") as refused, _held(lock, rows):
        pytest.fail("a run entered a rows file that another host holds")
    message = str(refused.value)
    assert "stale" not in message
    assert f"process {dead_pid} on {other}" in message
    assert f"remove the lock file {left}" in message
    assert left.read_text() == _record(other, dead_pid)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("not json", id="not-json"),
        pytest.param("[]", id="list"),
        pytest.param('"text"', id="string"),
        pytest.param('{"host": "elsewhere"}', id="no-pid-or-time"),
        pytest.param(_record("elsewhere", 0), id="pid-zero"),
        pytest.param(_record("elsewhere", -3), id="pid-negative"),
        pytest.param(_record("elsewhere", True), id="pid-bool"),
        pytest.param(_record("elsewhere", "7"), id="pid-text"),
        pytest.param(_record("elsewhere", 7.5), id="pid-float"),
        pytest.param(_record(5, 7), id="host-number"),
    ],
)
def test_a_file_without_a_readable_owner_is_in_use_and_stays(
    text: str, folder: Path, lock: Path, rows: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_with(monkeypatch, "ENOSYS")
    left = folder / EXCLUSIVE
    left.write_text(text)
    with pytest.raises(ConfigError, match="in use by another runner") as refused, _held(lock, rows):
        pytest.fail("a run entered a rows file that a lock file holds")
    assert f"its lock file {left} holds no readable owner" in str(refused.value)
    assert left.read_text() == text


# ------------------------------------------------------------ a runner that crashes


def test_a_runner_that_crashes_leaves_a_file_that_the_next_run_reports_as_stale(
    folder: Path, lock: Path, rows: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_with(monkeypatch, "ENOSYS")
    left, signal = folder / EXCLUSIVE, folder.parent / "crash"
    command = [sys.executable, "-c", RUNNER, str(lock), str(rows), str(signal)]
    # The interpreter of a virtual environment may start another process on Windows, so the
    # process that took the lock is the one its file names, not the one started here.
    with subprocess.Popen(command) as runner:
        try:
            _wait_for_lock_file(left, runner)
            owner = json.loads(left.read_bytes())
            assert owner["host"] == socket.gethostname()
            assert owner["pid"] != os.getpid()
            # While it runs, its rows file is in use.
            with (
                pytest.raises(ConfigError, match="in use by another runner") as busy,
                _held(lock, rows),
            ):
                pytest.fail("a run entered the rows file of a running runner")
            assert f"process {owner['pid']} on {socket.gethostname()}" in str(busy.value)
            assert "stale" not in str(busy.value)
            signal.write_text("")
            runner.wait(timeout=60)
            # The crash left the file behind, and the next run says so and names it.
            assert left.is_file()
            with pytest.raises(ConfigError, match="stale") as stale, _held(lock, rows):
                pytest.fail("a run entered a rows file with a stale lock file")
            assert str(left) in str(stale.value)
            assert f"process {owner['pid']} on {socket.gethostname()}" in str(stale.value)
            assert left.is_file()
        finally:
            signal.write_text("")
            runner.kill()


# ------------------------------------------------------------ the process check


def test_a_process_that_runs_is_alive_and_one_that_has_ended_is_not(dead_pid: int) -> None:
    assert _alive(os.getpid())
    assert not _alive(dead_pid)
    assert not _alive(2**31 - 1)
