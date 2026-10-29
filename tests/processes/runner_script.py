"""A runner of calls for the tests that kill it: ``python -m tests.processes.runner_script DIR``.

It writes its own process id to ``DIR/runner.pid`` and runs one call that
starts a sleeping grandchild and writes both process ids to ``DIR/pids``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dispatchatlas.processes.affinity import pinning_refusal_text
from dispatchatlas.processes.calls import Call, run_calls
from tests.processes.helpers import ROOT, slot_cpus


def main(directory: str) -> int:
    folder = Path(directory)
    (folder / "runner.pid").write_text(str(os.getpid()), encoding="ascii")
    call = Call(
        key="tree",
        target="tests.processes.child_targets:spawn_grandchild",
        arguments={"marker": str(folder / "pids"), "seconds": 120},
    )
    run_calls(
        [call],
        folder / "rows.jsonl",
        key_of=lambda row: "tree",
        cpus=slot_cpus(1),
        pin=pinning_refusal_text() is None,
        cwd=ROOT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
