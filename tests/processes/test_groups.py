from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dispatchatlas.processes.child import SCHEMA
from dispatchatlas.processes.groups import ProcessGroup, group_name, join_group
from dispatchatlas.processes.snapshots import end_process
from tests.processes.helpers import ROOT, gone, posix_only, wait_for, windows_only


def _start(
    group: ProcessGroup, marker: Path, *, name: str | None
) -> tuple[subprocess.Popen[bytes], bytes]:
    """Start a child that spawns a grandchild, through the child entry."""
    process = subprocess.Popen(
        [sys.executable, "-m", "dispatchatlas.processes.child"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=ROOT,
        **group.popen_options(),
    )
    return process, json.dumps(
        {
            "schema": SCHEMA,
            "key": "g",
            "target": "tests.processes.child_targets:spawn_grandchild",
            "arguments": {"marker": str(marker), "seconds": 120},
            "cpu": None,
            "group": name,
            "attempt": 1,
        }
    ).encode()


def test_group_names_carry_the_owner_and_the_slot() -> None:
    name = group_name(42, 3)
    assert name.endswith("dispatchatlas-42-3")
    assert name.startswith("Local\\") == (sys.platform == "win32")


@windows_only
def test_a_terminated_job_ends_the_child_and_what_it_started(tmp_path: Path) -> None:
    marker = tmp_path / "pids"
    with ProcessGroup.create(f"{group_name(os.getpid(), 0)}-t1") as group:
        assert group.platform == "job"
        process, payload = _start(group, marker, name=group.name)
        assert group.assign(process.pid) is None
        assert process.stdin is not None
        process.stdin.write(payload)
        process.stdin.close()
        child, grandchild = map(int, wait_for(marker).split())
        group.terminate()
        assert gone(child)
        assert gone(grandchild)
        process.wait(30)


@windows_only
def test_a_child_joins_its_job_by_name_alone(tmp_path: Path) -> None:
    marker = tmp_path / "pids"
    with ProcessGroup.create(f"{group_name(os.getpid(), 0)}-t2") as group:
        process = subprocess.Popen(
            [sys.executable, "-m", "dispatchatlas.processes.child"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=ROOT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        document = {
            "schema": SCHEMA,
            "key": "g",
            "target": "tests.processes.child_targets:spawn_grandchild",
            "arguments": {"marker": str(marker), "seconds": 120},
            "cpu": None,
            "group": group.name,
            "attempt": 1,
        }
        assert process.stdin is not None
        process.stdin.write(json.dumps(document).encode())
        process.stdin.close()
        child, grandchild = map(int, wait_for(marker).split())
        group.terminate()  # the parent never assigned it: only the child's own join reaches it
        assert gone(child)
        assert gone(grandchild)
        process.wait(30)


@windows_only
def test_joining_a_job_twice_is_not_an_error() -> None:
    name = f"{group_name(os.getpid(), 0)}-t3"
    script = (
        "from dispatchatlas.processes.groups import join_group\n"
        f"print(join_group({name!r}), join_group({name!r}))\n"
    )
    with ProcessGroup.create(name):
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=True
        )
    assert completed.stdout.split()[-2:] in (["job", "(already)"], ["(already)", "job"])
    assert "job (already)" in completed.stdout


@windows_only
def test_a_job_of_a_name_in_use_is_refused() -> None:
    name = f"{group_name(os.getpid(), 0)}-t4"
    with ProcessGroup.create(name) as first:
        second = ProcessGroup.create(name)
        assert first.platform == "job"
        assert (second.platform, second.handle) == ("none", None)
        assert "already exists" in (second.error or "")
        assert second.assign(os.getpid()) is not None


@posix_only
def test_a_member_leads_its_own_session_and_a_kill_reaches_its_tree(tmp_path: Path) -> None:
    marker = tmp_path / "pids"
    group = ProcessGroup.create(group_name(os.getpid(), 0))
    assert group.platform == "session"
    process, payload = _start(group, marker, name=group.name)
    assert group.assign(process.pid) is None
    if sys.platform != "win32":
        assert os.getsid(process.pid) == process.pid
    assert process.stdin is not None
    process.stdin.write(payload)
    process.stdin.close()
    _, grandchild = map(int, wait_for(marker).split())
    group.terminate()
    process.wait(30)
    assert gone(grandchild)
    assert join_group(group.name) == "session"


@pytest.mark.slow
def test_children_end_when_the_runner_is_killed(tmp_path: Path) -> None:
    runner = subprocess.Popen(
        [sys.executable, "-m", "tests.processes.runner_script", str(tmp_path)], cwd=ROOT
    )
    try:
        child, grandchild = map(int, wait_for(tmp_path / "pids", within=120).split())
        runner_pid = int(wait_for(tmp_path / "runner.pid"))
        # The runner's own interpreter, not a launcher in front of it: the job handle is there.
        assert end_process(runner_pid) is None
        assert gone(child, within=20)
        assert gone(grandchild, within=20)
    finally:
        runner.kill()
        runner.wait(30)
