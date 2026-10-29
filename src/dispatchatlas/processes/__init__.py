"""The operating system's view of a run: pinned child processes and the tools around them.

- :mod:`~dispatchatlas.processes.calls`: :func:`run_calls`, which runs
  caller-defined functions, each once in a child process of its own on a
  processor of its own, and appends the rows they return to a rows file.
- :mod:`~dispatchatlas.processes.child`: the child side of a call.
- :mod:`~dispatchatlas.processes.rows`: the rows file, its writer, its strict
  reader, resume by keys, and the repair of a torn last line.
- :mod:`~dispatchatlas.processes.affinity`: processor sets as text and masks,
  and a process pinning itself with a read-back.
- :mod:`~dispatchatlas.processes.groups`: kill-on-close process groups
  (Windows job objects, POSIX sessions).
- :mod:`~dispatchatlas.processes.snapshots`: the machine's processes with
  parent, executable, affinity mask, and start time.
- :mod:`~dispatchatlas.processes.memory`: peak, available, and total memory.
- :mod:`~dispatchatlas.processes.launching`: starting a command detached from
  its shell.
- :mod:`~dispatchatlas.processes.watching`: a status line per interval about
  a running run.

The package sits between :mod:`dispatchatlas.core` and
:mod:`dispatchatlas.campaigns`, which uses its affinity, group, and memory
primitives. Its leaf modules use the standard library only.

Unlike the other packages, this one re-exports its names lazily: a child
process started for every call runs ``python -m
dispatchatlas.processes.child``, which imports this package first, and
importing the runner's dependencies there would add a substantial part of a
second to every call. ``from dispatchatlas.processes import run_calls``
works as usual.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dispatchatlas.processes.affinity import (
        PinReport,
        cpus_of_mask,
        current_cpus,
        current_mask,
        mask_of_cpus,
        parse_cpus,
        pin_self,
    )
    from dispatchatlas.processes.calls import (
        Call,
        CallError,
        CallFacts,
        CallReport,
        CallsStoppedError,
        interleaved,
        run_calls,
    )
    from dispatchatlas.processes.groups import ProcessGroup, join_group
    from dispatchatlas.processes.launching import Launched, launch_detached
    from dispatchatlas.processes.memory import (
        available_memory_bytes,
        peak_memory_bytes,
        total_memory_bytes,
    )
    from dispatchatlas.processes.rows import (
        RepairReport,
        RowWriter,
        TornRowsError,
        canonical_key,
        done_keys,
        duplicate_keys,
        read_rows,
        repair_rows,
    )
    from dispatchatlas.processes.snapshots import (
        ProcessInfo,
        descendants,
        end_process,
        interpreters_on,
        is_running,
        snapshot,
    )
    from dispatchatlas.processes.watching import Status, format_status, watch

_EXPORTS = {
    "affinity": (
        "PinReport",
        "cpus_of_mask",
        "current_cpus",
        "current_mask",
        "mask_of_cpus",
        "parse_cpus",
        "pin_self",
    ),
    "calls": (
        "Call",
        "CallError",
        "CallFacts",
        "CallReport",
        "CallsStoppedError",
        "interleaved",
        "run_calls",
    ),
    "groups": ("ProcessGroup", "join_group"),
    "launching": ("Launched", "launch_detached"),
    "memory": ("available_memory_bytes", "peak_memory_bytes", "total_memory_bytes"),
    "rows": (
        "RepairReport",
        "RowWriter",
        "TornRowsError",
        "canonical_key",
        "done_keys",
        "duplicate_keys",
        "read_rows",
        "repair_rows",
    ),
    "snapshots": (
        "ProcessInfo",
        "descendants",
        "end_process",
        "interpreters_on",
        "is_running",
        "snapshot",
    ),
    "watching": ("Status", "format_status", "watch"),
}
_MODULE_OF = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = [
    "Call",
    "CallError",
    "CallFacts",
    "CallReport",
    "CallsStoppedError",
    "Launched",
    "PinReport",
    "ProcessGroup",
    "ProcessInfo",
    "RepairReport",
    "RowWriter",
    "Status",
    "TornRowsError",
    "available_memory_bytes",
    "canonical_key",
    "cpus_of_mask",
    "current_cpus",
    "current_mask",
    "descendants",
    "done_keys",
    "duplicate_keys",
    "end_process",
    "format_status",
    "interleaved",
    "interpreters_on",
    "is_running",
    "join_group",
    "launch_detached",
    "mask_of_cpus",
    "parse_cpus",
    "peak_memory_bytes",
    "pin_self",
    "read_rows",
    "repair_rows",
    "run_calls",
    "snapshot",
    "total_memory_bytes",
    "watch",
]


def __getattr__(name: str) -> Any:
    module = _MODULE_OF.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
