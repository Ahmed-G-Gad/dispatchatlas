# Pinned child processes

A campaign runs the library's solvers on a plan. Some experiments instead
need to run *their own* functions, thousands of times, each on a processor of
its own, with one record per run that survives a crash: a timing study of a
custom algorithm, a measurement harness around an external tool, a sweep that
does not fit a campaign file. `dispatchatlas.processes` provides that:

- `run_calls` runs a list of calls, each a function called once in a fresh
  child process, and appends the mapping each returns as one line of a rows
  file;
- the rows file is the record of what is done, so a second invocation
  resumes;
- every child is pinned to its slot's processor, proves it, and lives in a
  process group that ends with the runner;
- failures go to a sidecar file, never to the rows file;
- tools to launch a run detached from its shell, watch it, list processes,
  and end them on request.

Use a [campaign](campaigns.md) when the work is solvers on instances under
protocols: it plans, checks fairness, and replays. Use `run_calls` when the
work is your function and the row is yours to define.

## Calls, keys, and the rows file

```python
from dispatchatlas.processes import Call, run_calls

calls = [
    Call(
        key=["sweep", size, seed],
        target="mystudy.measure:run_one",
        arguments={"size": size, "seed": seed},
    )
    for size in (100, 200)
    for seed in range(10)
]
report = run_calls(
    calls,
    "results/rows.jsonl",
    key_of=lambda row: ["sweep", row["size"], row["seed"]],
    cpus="4-7",
)
```

A **call** names a function as `"package.module:function"`, importable in
the child (the child's working directory is `cwd`, which is on its import
path), and keyword arguments that are JSON-able. Its **key** is a string, an
integer, or a list or tuple of keys. The function returns a mapping, the
**row**. The runner checks that `key_of(row)` gives back the call's key, so
the row marks its call as done; keys are compared through `canonical_key`
(compact JSON with sorted object keys), so a tuple and the list JSON returns
for it are one key.

The rows file is written by the runner alone (`RowWriter`): one JSON object
per line, UTF-8 without a byte-order mark, line feeds only, written in one
call and followed by `flush` and `fsync`. A crash therefore loses at most the
row being written, and leaves at most a torn last line. Before it starts
anything, `run_calls` reads the file back (`done_keys`) and skips every call
whose key has a row. A torn last line is refused (`TornRowsError`, with the
line number and byte offset); `dispatchatlas processes rows-repair FILE`
(`repair_rows`) moves the bytes after the last line feed to
`FILE.torn-<UTC time>.bin` and truncates the file there. A malformed line that
is not last means another writer touched the file, and is left for you to
inspect.

One rows file has one runner. For the whole run, `run_calls` holds an
operating-system lock on `<rows stem>.lock` (whose text names the holder's
process id and start time); a second `run_calls` on the same file, in this
process or another, raises `ConfigError` before it starts anything. The lock
ends with the process that holds it, so a killed runner leaves none behind. A
rows file that holds one key twice is refused too (`duplicate_keys` lists
the keys): move the extra rows aside before resuming.

Each invocation appends a line to `<rows stem>.runs.jsonl`: when it started,
its `stamp`, its number of workers, and its processors. A resume whose stamp,
workers, or processors differ from the first line's raises `ConfigError`
naming each difference, so one file never mixes code revisions or load
conditions by accident; `allow_changed_run=True` resumes anyway. A rows file
written before the runs sidecar existed is checked against the stamp fields
of its rows.

By default the runner adds what it knows about the call under `row["run"]`
(see [Load](#load) and [Processors](#processors)). A `finish(row, facts)`
function shapes the row instead, for example to flatten the fields a reader
wants; `stamp` adds constant fields to every row, such as the provenance of
the code (a stamp field the row already has is refused).

## Processors

`cpus` is the run's processor set: indices, a mask (`0xF0`), or text that
`parse_cpus` reads (`"0xF0"`, `"4-7,12"`). Every processor must be one the
calling process may use. Slot `i` owns `cpus[i]`; `workers` (at most
`len(cpus)`) sets how many slots run.

- The runner pins itself to the whole set, the spare processors beyond
  `workers` included, for the run and returns to its earlier set afterwards.
- It creates each child while confined to the slot's processor, so the child
  and everything it starts begin there.
- The child then pins **itself** to that processor and reads its mask back;
  a mismatch is a failure (`PinningRefused`), never a row.

The self-pin matters where `python` is a launcher executable that starts the
real interpreter as its child, as in some virtual-environment layouts on
Windows: a mask set on the launcher's handle from outside reaches the
launcher alone. Each row records `cpu`, `affinity_inherited` (the mask the
child started with) and `affinity_mask` (the mask it read back).

Each child's environment sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and
`NUMEXPR_NUM_THREADS` to 1, with `PYTHONUNBUFFERED=1` and
`PYTHONIOENCODING=utf-8`; `environment` adds or overrides variables. The row
records what the child saw (`threads`).

`pin=False` runs unpinned (`cpu` is `None`), for platforms that cannot pin.

## Process groups

Each slot has a kill-on-close `ProcessGroup` for the whole run:

- On **Windows** it is a named job object. A child is started suspended,
  assigned to the job, then resumed, so no process it starts can escape the
  job; the child also joins the job by name (`join_group`) and records how
  (`group`: `job` or `job (already)`). The runner holds the job's only
  lasting handle: when the runner dies, for any reason, the kernel ends every
  process in every slot's job.
- On **POSIX** systems each child leads a session of its own, and a thread in
  the child ends its process group when the runner's process is gone. A
  child whose runner died while the child's interpreter was still starting
  ends at once, since the runner's process id travels in the call's document.

When a call ends, its slot's group is ended too, so nothing a call started
keeps running beside the next call. A per-call `timeout_seconds` (or the
runner's `timeout_seconds`) ends the group of a call that passes it.

## Failures

The child's reply, the row or an error object, travels as the last line on a
private copy of its standard output descriptor. Everything the call, an exit
handler, a late thread, or a helper process writes to standard output, before
or after the reply, goes to standard error instead, which the runner appends
to `<rows stem>.children.log` with every traceback; it cannot take the
reply's place. A failure is one line of `<rows stem>.errors.jsonl`
(`CallError`): the key, the attempt, the type and a message with the words of
`clean_tokens` removed (from both, since a module or class name can hold
one), the message also without file paths and traceback frames, the return
code, whether the child replied, the call's facts, the processor, and the
times.

Paths are recognised by their roots (`C:\`, `\\server\`, `/home/...`, and
quoted paths as `OSError` prints them); an unquoted path may hold spaces, so
it is taken to run to the last separator on its line, and text after it on
that line can go with it. `clean_tokens` is the dependable filter for names
that must never reach the sidecar: it removes them wherever they occur.

Rows are standard JSON: a NaN, an infinity, or a value JSON has no form for
(a set, a NumPy scalar, a date) makes the call fail with
`RowNotSerializable`, and the message names the field (`field 'x.y' is inf`).
Write `None` or a string instead, and convert NumPy scalars with `.item()`.

Types:

| Type | Meaning |
| --- | --- |
| the exception's class | the function raised (`ValueError`, `mypkg.errors.Bad`); tokens removed |
| `TargetNotFound`, `RowNotMapping`, `BadDocument` | the call could not be made, or its result is not a mapping |
| `RowNotSerializable` | the row holds a NaN, an infinity, or a value that is not JSON; the message names the field |
| `PinningRefused`, `GroupRefused` | the child could not pin itself or join its group |
| `ProcessDied` | the child ended without a reply (exit code or signal) |
| `Timeout` | the call passed its time limit and was ended |
| `KeyMismatch` | `key_of` on the final row does not give the call's key |
| `RowRefused` | `finish` failed, a stamp field collides, or the row is not JSON |
| `StartFailed` | the child process could not be started |
| `AttendFailed` | the runner's thread attending the child failed; the child was ended |

After `max_failures` failures in one invocation (0 for no limit) the run
stops at once: nothing more starts, every group is ended, the report is
written, and `CallsStoppedError` is raised, so a driver that ignores return
values still exits with an error. Ctrl-C does the same and re-raises
`KeyboardInterrupt`. A failed call is retried by the next invocation, with
its `attempt` counted from the sidecar, unless `skip_failed=True`.

## Load

Each row records `workers`, `workers_live_start` (busy slots once the runner
had filled every slot it could when the call started) and `workers_live_min`
(the least busy slots during the call's life). The counts are of slots, not
processes: a slot whose call ends and that the runner refills at once counts
as busy throughout, although for a few milliseconds no process of it runs
while the next child starts. In the tail of a run, slots
fall idle and the last calls run on a quieter machine. `ballast=True` keeps
every slot busy until the last real call ends, with repeats of the calls in
a seeded order (`ballast_seed`, by default derived from the rows file's name),
or with the calls you pass as `ballast`, repeated in the order you give;
their rows are discarded and their facts go to `<rows stem>.ballast.jsonl`.
Ballast costs processor time and nothing else. Ballast failures do not count
towards `max_failures`, and after three ballast calls in a row fail, ballast
stops for the rest of the run with a warning.

## Order

Calls start in the order given. `interleaved(calls, seed=...)` shuffles them
reproducibly; with `classes=` it orders the calls by class (lower first) and
shuffles within each class, for example to start the longest calls first.

## Launching and watching

A run of days should not depend on the shell that started it:

```bash
dispatchatlas processes launch --log results/driver.log --cwd . -- python driver.py
dispatchatlas processes watch --pid <printed pid> --rows results/rows.jsonl --cpus 4-7 --status results/status.log
```

`launch_detached` starts the command with no window and no standard input,
its output appended to the log, in a process group of its own; on Windows it
first asks to leave the job of the calling process (which an agent session
or a CI step may hold with kill-on-close), and says `breakaway no` when the
job forbids it, since the command then ends with that job. Any other failure
to start, such as a missing command, is raised.

The pid `launch` prints is that of the process it started. Where `python` is
a virtual environment's launcher executable (on Windows), that process is the
launcher, and the interpreter running the driver is its child: read masks
from the child, and give either pid to `watch`.

`watch` prints one line per interval: whether the runner is alive
(`RUNNER-DEAD` first when it is not), its live descendants (`children`, every
process at any depth: about two per busy slot with a Windows launcher, plus
console hosts), how many of them are Python interpreters (`interpreters`),
the other Python interpreters whose masks meet the run's processors (with
their ids), the rows file's line count and size, and the available memory.
The interpreters in an unbroken line above the watched process and above the
watcher itself, such as a Windows launcher, are not counted as other
interpreters. It never starts, restarts, or ends anything.
`dispatchatlas processes list` shows processes with parent, mask, and start
time (`--children-of PID`, `--interpreters-on CPUS`), and
`dispatchatlas processes end PID` ends a process on request.

A process is a Python interpreter when its name or the file name of its
executable is an interpreter's, such as `python`, `python3`, `python3.13`,
or `pythonw`, with or without `.exe`. On Linux, a script run through its
`#!` line, such as the console script `pytest` or `dispatchatlas`, is named
after the script, and its executable, the interpreter, tells. Linux shows
the executable of another user's process only to a privileged caller such as
root; to any other caller, such a script of another user is not counted.

## Provenance

A driver that refuses to start on unrecorded code can use:

- `git_state(directory, within="scripts")`: the commit of the checkout, and
  whether files *within that folder* differ from it;
- `compare_trees(installed_package_dir("mypkg"), source)`: whether the
  installed copy of a package holds the same files with the same SHA-256
  digests as a source tree (`__pycache__` left out; CRLF read as LF unless
  `normalize_newlines=False`, so pass `False` to claim byte identity; the
  result's `normalized_newlines` records which);
- `peak_memory_bytes()` and `available_memory_bytes()`.

Their results go into `stamp`, so every row carries them.

## Platforms

| | Windows | Linux | macOS |
| --- | --- | --- | --- |
| pinning and read-back | `SetProcessAffinityMask` (one processor group) | `sched_setaffinity` | refused; use `pin=False` |
| process group | named kill-on-close job object | session, ended with the parent | session, ended with the parent |
| snapshot | Toolhelp | `/proc` | empty |
| whether a process runs | `GetExitCodeProcess` | `kill`, then the `/proc` state | `kill`, then the `sysctl` state |
| peak memory | peak working set | `VmHWM` | `ru_maxrss` |
| available memory | `GlobalMemoryStatusEx` | `MemAvailable` | `None` |

## From Python

```python
import sys
from pathlib import Path

from dispatchatlas.campaigns.environment import git_state
from dispatchatlas.processes import Call, CallsStoppedError, interleaved, run_calls

calls = [
    Call(
        key=[method, seed],
        target="mystudy.measure:run_one",
        arguments={"method": method, "seed": seed},
    )
    for method in ("a", "b")
    for seed in range(50)
]
state = git_state(Path("."), within="mystudy")
if state is None or state["dirty"]:
    sys.exit("commit the study's code first")
try:
    report = run_calls(
        interleaved(calls, seed=1),
        "results/rows.jsonl",
        key_of=lambda row: [row["method"], row["seed"]],
        cpus="0xF0",
        stamp={"revision": state["commit"]},
        max_failures=3,
        ballast=True,
        clean_tokens=("mystudy",),
    )
except CallsStoppedError as stop:
    sys.exit(f"stopped: {stop.report.stopped}")
print(report.rows_written, "rows")
```

The entry points `python -m dispatchatlas.processes.child`,
`python -m dispatchatlas.processes.launching`, and
`python -m dispatchatlas.processes.watching` work without the command line.

## Command line

See [`dispatchatlas processes`](../cli.md#processes) for every option.
