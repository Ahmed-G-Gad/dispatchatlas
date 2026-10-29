from __future__ import annotations

import threading
import time
from pathlib import Path

from dispatchatlas.processes.watching import format_status, watch
from tests.processes.helpers import sleeper, slot_cpus


def test_the_watch_follows_the_runner_and_its_rows_until_it_ends(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b'{"a":1}\n')
    runner = sleeper(6.0)

    def append() -> None:
        time.sleep(1.0)
        with rows.open("ab") as stream:
            stream.write(b'{"a":2}\n{"a":3}\n')

    writer = threading.Thread(target=append)
    writer.start()
    statuses = list(watch(runner.pid, rows=rows, cpus=slot_cpus(2), interval=0.5, until=120))
    writer.join()
    runner.wait(30)
    assert statuses[0].runner_alive
    assert statuses[0].rows_lines == 1
    assert not statuses[-1].runner_alive
    assert statuses[-1].rows_lines == 3
    assert statuses[-1].rows_bytes == rows.stat().st_size
    assert format_status(statuses[-1]).startswith("RUNNER-DEAD ")
    assert not format_status(statuses[0]).startswith("RUNNER-DEAD")
    assert all(
        status.available_memory_bytes is None or status.available_memory_bytes > 0
        for status in statuses
    )


def test_a_repaired_rows_file_is_counted_again(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b"a\nb\nc\n")
    runner = sleeper(60.0)
    try:
        polls = watch(runner.pid, rows=rows, cpus=slot_cpus(1), interval=0.1, until=2)
        assert next(polls).rows_lines == 3
        rows.write_bytes(b"a\n")
        assert next(polls).rows_lines == 1
    finally:
        runner.kill()
        runner.wait(30)
