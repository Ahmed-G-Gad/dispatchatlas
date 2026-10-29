from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from dispatchatlas.campaigns.processors import allowed_cpus, confined
from dispatchatlas.processes.memory import (
    available_memory_bytes,
    peak_memory_bytes,
    total_memory_bytes,
)
from dispatchatlas.processes.snapshots import (
    ProcessInfo,
    descendants,
    end_process,
    interpreters_on,
    is_running,
    snapshot,
)
from tests.processes.helpers import gone, needs_pinning, sleeper, wait_for

SUPPORTED = sys.platform == "win32" or Path("/proc/self/stat").exists()
needs_snapshot = pytest.mark.skipif(not SUPPORTED, reason="no process snapshot here")


@needs_snapshot
def test_the_snapshot_holds_this_process_and_its_tree(tmp_path: Path) -> None:
    marker = tmp_path / "pids"
    script = (
        "import os, subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(marker)!r}, 'w').write(f'{{os.getpid()}} {{g.pid}}')\n"
        "time.sleep(60)\n"
    )
    child = subprocess.Popen([sys.executable, "-c", script])
    unrelated = sleeper()
    try:
        interpreter, grandchild = map(int, wait_for(marker).split())
        listed = snapshot()
        mine = next(info for info in listed if info.pid == os.getpid())
        assert mine.parent_pid == os.getppid()
        assert mine.name.lower().startswith("python")
        assert mine.started_utc is not None
        tree = {info.pid for info in descendants(os.getpid(), listed)}
        assert {child.pid, interpreter, grandchild} <= tree
        assert unrelated.pid in tree  # a child of this process too
        assert unrelated.pid not in {info.pid for info in descendants(child.pid, listed)}
    finally:
        for process in (child, unrelated):
            process.kill()
            process.wait(30)


def test_a_reused_parent_id_is_not_followed() -> None:
    parent = ProcessInfo(1, None, "a", None, None, "2026-01-01T00:00:10.000000+00:00")
    older = ProcessInfo(2, 1, "b", None, None, "2026-01-01T00:00:05.000000+00:00")
    newer = ProcessInfo(3, 1, "c", None, None, "2026-01-01T00:00:15.000000+00:00")
    assert [info.pid for info in descendants(1, [parent, older, newer])] == [3]


@needs_pinning
@needs_snapshot
def test_interpreters_on_given_processors_are_found() -> None:
    cpu = allowed_cpus()[-1]
    with confined(cpu):
        pinned = sleeper()
    try:
        tree = {info.pid for info in descendants(pinned.pid)} | {pinned.pid}
        found = {info.pid for info in interpreters_on([cpu])}
        assert tree & found
        assert not tree & {info.pid for info in interpreters_on([cpu], exclude=tree)}
    finally:
        pinned.kill()
        pinned.wait(30)


def test_end_process_ends_a_process_and_reports_what_it_cannot_end() -> None:
    process = sleeper()
    assert is_running(process.pid)
    assert end_process(process.pid) is None
    process.wait(30)
    assert gone(process.pid)
    assert not is_running(process.pid)
    assert end_process(process.pid) is not None


def test_memory_is_read_where_the_platform_tells() -> None:
    peak = peak_memory_bytes()
    available, total = available_memory_bytes(), total_memory_bytes()
    if sys.platform in ("win32", "linux"):
        assert peak is not None
        assert peak > 0
        assert available is not None
        assert total is not None
        assert 0 < available <= total
        block = b"x" * (64 << 20)
        after = peak_memory_bytes() or 0
        assert after >= max(peak, len(block))
        del block
