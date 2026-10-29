from __future__ import annotations

import os
import sys
from pathlib import Path

from dispatchatlas.processes.snapshots import end_process
from tests.cli.runner import Cli
from tests.processes.helpers import sleeper, wait_for


def test_rows_check_counts_rows_or_reports_the_torn_line(cli: Cli, tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b'{"a":1}\n{"a":2}\n')
    result = cli("processes", "rows-check", rows)
    assert (result.code, result.out.strip()) == (0, "2 row(s)")
    rows.write_bytes(rows.read_bytes() + b'{"a":')
    torn = cli("processes", "rows-check", rows)
    assert torn.code == 1
    assert "line 3 at byte 16" in torn.err
    assert "rows-repair" in torn.err
    assert cli("processes", "rows-check", rows, "--json").json()["torn_line"] == 3


def test_rows_repair_moves_the_torn_tail_aside(cli: Cli, tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b'{"a":1}\n{"a":')
    result = cli("processes", "rows-repair", rows)
    assert result.code == 0
    assert "moved 5 byte(s)" in result.out
    assert rows.read_bytes() == b'{"a":1}\n'
    assert len(list(tmp_path.glob("rows.jsonl.torn-*.bin"))) == 1
    assert "nothing to repair" in cli("processes", "rows-repair", rows).out


def test_list_shows_the_children_of_a_process(cli: Cli) -> None:
    child = sleeper()
    try:
        result = cli("processes", "list", "--children-of", str(os.getpid()), "--json")
        if sys.platform == "win32" or Path("/proc/self/stat").exists():
            assert child.pid in {item["pid"] for item in result.json()}
        table = cli("processes", "list", "--children-of", str(os.getpid()))
        assert table.out.split()[:5] == ["pid", "parent", "name", "mask", "started"]
    finally:
        child.kill()
        child.wait(30)


def test_end_ends_a_process_on_request(cli: Cli) -> None:
    child = sleeper()
    result = cli("processes", "end", str(child.pid))
    child.wait(30)
    assert (result.code, result.out.strip()) == (0, f"ended {child.pid}")
    assert end_process(child.pid) is not None


def test_launch_starts_a_detached_command(cli: Cli, tmp_path: Path) -> None:
    marker = tmp_path / "pid"
    script = f"import os; open({str(marker)!r}, 'w').write(str(os.getpid()))"
    result = cli(
        "processes",
        "launch",
        "--log",
        tmp_path / "run.log",
        "--cwd",
        tmp_path,
        "--",
        sys.executable,
        "-c",
        script,
    )
    assert result.code == 0
    assert result.out.startswith("started pid ")
    assert int(wait_for(marker)) > 0
    assert cli("processes", "launch", "--log", tmp_path / "x.log").code == 2
