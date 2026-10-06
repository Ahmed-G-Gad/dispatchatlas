# Campaigns

A campaign is a reproducible experiment: every solver of a roster runs on
every problem of its instance sources, under every stopping protocol, with a
number of replicates. It is described in a TOML or JSON file, planned and
checked before anything runs, executed in worker processes, recorded one JSON
file per run, resumed after an interruption, and replayed to verify that its
runs reproduce.

```bash
dispatchatlas campaign validate campaign.toml --check-compatibility
dispatchatlas campaign estimate campaign.toml
dispatchatlas campaign run campaign.toml --output results --workers auto
dispatchatlas campaign replay results/<campaign id>
dispatchatlas analyze results/<campaign id> --output analysis
```

## The campaign file

A complete file, with every key explained. It is
`examples/campaigns/reference.toml` in the repository, and it validates as it
is:

```toml
--8<-- "examples/campaigns/reference.toml"
```

The reader is strict. An unknown key, a value of the wrong type, or a missing
required key is an error that names the entry, for example
`instances[1].seeds[0] must be an integer, got the string '3'`. The file must
declare `schema_version = 1`. Relative paths refer to the file's directory. In
JSON, `null` stands for an absent optional value.

### Top-level keys

| Key | Required | Meaning |
| --- | --- | --- |
| `schema_version` | yes | the file format version, 1 |
| `id` | yes | the campaign name; results go to `<output>/<id>/` |
| `seed` | yes | the root of every run seed, a non-negative integer |
| `instances` | yes | one or more instance sources |
| `solvers` | yes | one or more solvers |
| `protocols` | yes | one or more stopping protocols |
| `replicates` | no | runs per stochastic solver, problem, and protocol; default 30 |
| `objectives` | no | objectives, primary first, that replace those every problem declares |
| `execution` | no | how to run, see [Execution](#execution) |
| `metadata` | no | free annotations kept with the results |

Names that become file names, the campaign id, solver names, protocol names,
and problem ids, start with a letter or digit and hold only letters, digits,
`.`, `_`, and `-`. Solver and protocol names must be unique, ignoring case.

### Instance sources

| Source | Keys | Problems |
| --- | --- | --- |
| family | `family`, `seeds`, `parameters` | one problem per seed of a registered family |
| suite | `suite`, `instances`, `directory` | named instances of a published suite; file suites read `directory` |
| catalog | `catalog`, `seed` | every instance of a built-in catalog |
| directory | `directory` | the problems a catalog build or `materialize` wrote, each checked against its digest |
| file | `file`, `format` | one problem document or instance file |

Every problem id must be unique across the sources. Parameters of a family
source are the family's own; the others keep their defaults.

### Solvers and protocols

A solver entry has an `id` from the solver registry, optional `parameters`,
and an optional `label`. The label names the entry in results and tables, so
two settings of one solver can compete in one campaign. Parameters are checked
against the solver when the campaign is planned.

A protocol has a `name` and a `stop` table with the fields of `StopCriteria`:
`time_limit`, `max_evaluations`, `max_iterations`, `target`, and `patience`.
Every protocol must set a time, evaluation, or iteration limit.

### The config hash

The **config hash** identifies an experiment: the SHA-256 digest of the
canonical JSON of the fields that determine the results, `seed`, `instances`,
`solvers`, `protocols`, `replicates`, and `objectives`, as written, so
reordering a list changes it too. The campaign `id` only names the directory,
and `execution` and `metadata` never change a result, so they stay outside
the hash. The locations of instance files and directories stay outside it
too: a problem read from a file is identified by the digest of its content,
which enters the id of every run on it. A campaign therefore keeps its
identity when its files move, or live elsewhere on another machine.

## Planning

`campaign validate` reads the file, loads every problem, checks every solver
id and parameter, and expands the grid problem × solver × protocol ×
replicate into planned runs, without running any.

- **Replicates.** A stochastic solver runs `replicates` times per problem and
  protocol, as replicates 0, 1, and so on. A deterministic solver runs once,
  as replicate 0.
- **Run ids.** `run-` and the first 16 hexadecimal digits of the SHA-256 of
  the config hash, the problem id and digest, the solver id, label, and
  parameters, the protocol name, and the replicate. Equal campaigns give
  equal run ids on every machine, and a problem whose content changes gets new
  ones.
- **Compatibility.** A solver is never paired with a problem that has a
  feature the solver does not model. The pair is excluded, and the plan
  records why. `--check-compatibility` lists every exclusion and every
  caveat, such as a single-objective solver on a problem with several
  objectives, which optimises the primary objective only.

## Seeds and paired comparisons

Seeds are common random numbers. Replicate `r` of problem `p` receives

```text
derive_seed(seed, "problem:" + p, r)
```

under every solver and every protocol. The results are therefore paired: the
first replicate of every solver on a problem shares one seed, the second
another, and so on. Adding, removing, or reordering solvers never changes a
seed, so a campaign can grow a solver without rerunning the others.

## Fair comparison

Solvers are compared fairly when, under each protocol, every one of them runs
with identical stop criteria, on the same problems, with the same seeds. The
planner builds runs that way, and the plan is then checked rather than assumed
to be fair (`Fairness`, recorded in `plan.json`):

- every protocol bounds its runs with a time, evaluation, or iteration limit;
- every run refers to one of the plan's protocols, so all runs under a
  protocol share its stop criteria;
- every solver has its full set of replicates under every protocol on every
  problem it is not excluded from, and no run is planned twice;
- all runs of one problem and replicate share one seed.

Each protocol states the budgets it equalises, its **parities**, read off its
stop criteria: `equal-wall-clock` (`time_limit`), `equal-evaluations`
(`max_evaluations`), `equal-iterations` (`max_iterations`), and `fixed-target`
(`target`). Equal evaluations is the budget to prefer for comparing
algorithms: it does not depend on the machine, and runs stopped by it replay
exactly. Wall-clock budgets depend on the speed and load of the machine, and
runs they stop cannot be replayed. An iteration means a different amount of
work in different solvers.

## Estimating

`campaign estimate` counts the runs, by protocol and by solver, and bounds how
long they can take. A run whose protocol sets `time_limit` takes at most its
hard limit, and every attempt may take that long. A run bounded only by
evaluations or iterations has no duration bound, so those runs are counted
apart. With `--output`, runs already completed there are left out.

## Execution

The `[execution]` table, or the options of `campaign run`, choose how to run.
None of these settings changes a result, so they may differ between sessions
of one campaign.

- **Backends.** `process`, the default, runs each attempt in one of `workers`
  worker processes, started with the `spawn` method so that every platform
  behaves alike, and supervised by the calling process. `sequential` runs one
  attempt after another in the calling process; it suits debugging and
  solvers that cannot be sent to another process, but a solver that ignores
  its time limit, or a crash in native code, stops the campaign. A Python
  script that runs a campaign with the process backend must do so under
  `if __name__ == "__main__":`, since every worker imports the main module.
- **Processors.** The engine counts the processors its process is allowed to
  use, not those of the machine: the scheduler affinity on Linux, which
  `taskset` and the cpuset of a container set, and the process affinity mask
  on Windows. A limit on processor time is not an affinity: under the CPU
  quota of a container (`docker run --cpus`, a Kubernetes CPU limit) every
  processor of the host is still counted, so set `workers` by hand there.
  Where the set cannot be read, on macOS and on a Windows machine whose
  processors form several processor groups, every processor of the machine
  is counted, the session records `cpu.allowed` as null, and a campaign with
  a time limit logs a warning. `workers = "auto"` uses the count minus one.
  `affinity` chooses where the workers run. `"inherit"`, the default, leaves
  every worker on all the allowed processors. `"pin"` confines worker `k` to
  the `k`-th allowed processor, in index order. It does not look at cores:
  on a machine with simultaneous multithreading that numbers the two
  hardware threads of a core side by side, two pinned workers share one
  core. A list such as `affinity = [2, 4, 6]` confines worker `k` to its
  `k`-th entry, so name one logical processor of each core there. A pinned
  worker is created on its processor, where every thread it starts also
  runs, and it reports its processors to the supervisor, which stops the
  campaign when they are not the one it was given. A pinned campaign with
  runs to attempt is refused, before it discards a record or creates its
  directory, when the workers outnumber the processors, when a listed
  processor is not an allowed one, and on a system that cannot pin a
  process: macOS, and a Windows machine whose processors form several
  processor groups. The sequential backend accepts only `"inherit"`.
- **Threads.** Every worker process starts with the thread pools of
  numerical libraries limited to `threads_per_worker` threads, 1 by default.
  The engine sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and
  `NUMEXPR_NUM_THREADS` in the environment each worker is created with, so
  the limit holds for any number of workers, whatever the launching
  environment sets, and also when the launching script imports NumPy first.
  The sequential backend has no worker processes and leaves the environment
  of the calling process alone, so it accepts only the default
  `threads_per_worker`; under a time limit the runner names every one of
  these variables that is unset, since a numerical library may then use
  every processor, and they take effect when set before Python starts. A
  solver's own `threads` parameter is separate. When a protocol sets a time
  limit and the workers, each with the larger of `threads_per_worker` and
  the largest `threads` among the solvers, need more processors than are
  allowed, the runner warns that the budgets differ between runs. A pinned
  worker has one processor, so the runner warns whenever that thread count
  is above 1. The plan records the `threads` every solver holds in its
  solver entry, the solver's default when the campaign sets none. When a
  protocol with a time limit compares solvers whose `threads` differ, the
  plan keeps a note of it in its fairness assessment and stays fair.
- **Hard time limits.** Stop criteria are cooperative, so the process
  backend gives every run whose protocol sets `time_limit` a hard limit of
  `timeout_factor * time_limit + grace` seconds, counted from the moment the
  run is handed to its worker. When it passes, the supervisor kills the worker
  and records a timeout, and a fresh worker takes the next run. On POSIX
  systems the kill reaches every process the solver started.
- **Retries.** A failure is **terminal** when another attempt would fail the
  same way: a typed error of the library, such as an invalid configuration or
  a missing optional dependency, or a hard timeout. Any other exception, and a
  worker that dies, is **recoverable**. A run gets up to `1 + retries`
  attempts, until a terminal failure or the last attempt settles it.
- **What each attempt checks.** Every schedule a solver returns is checked
  with `check_schedule`, and every objective is computed again from the
  schedule, so a record never relies on the solver's own claims. The
  measurements cover the solve alone: wall time, CPU time, and on Linux the
  peak resident memory. Each attempt also records the processor its worker
  was pinned to, and how many runs were executing when it was handed to its
  worker, itself included.

## Storage layout

```text
<output>/<campaign id>/
    campaign.json      the campaign document, its base directory, and its config hash
    plan.json          the plan, with the parities and the fairness assessment
    sessions/<n>.json  one record per invocation of the runner, with its environment
    runs/<solver>/<problem>/r<replicate:03d>-<protocol>.json
    failures/<solver>/<problem>/r<replicate:03d>-<protocol>.attempt-<n>.json
    .lock              held while a runner works on the campaign
```

Every document is written atomically, with sorted keys, two-space
indentation, ASCII text, and a final newline.

- A **run record** (`dispatchatlas.run/1`) names the run, its campaign and
  config hash, problem id and digest, solver, protocol, replicate, and seed;
  holds the outcome, with the status, stop reason, objective values, bound,
  feasibility and violation codes as checked, evaluations, iterations,
  elapsed time, trace, and front; the best schedule as `[task, mode, start,
  end]` rows, unless `record_schedules` is false; and the provenance of the
  attempt: session, attempt number, worker, times, CPU seconds, peak memory,
  hard limit, the processor the worker was pinned to (`cpu`, null when it was
  not pinned), and the runs executing when the run was handed to its worker
  (`concurrent`). A record that states neither reads as null for both.
- Its **result hash** is the SHA-256 of the outcome without the fields that
  measure wall time, and without the solver-specific extras. Nothing about the
  machine or the session enters it, so a replay on any machine compares
  results directly, and a record edited after it was written fails the
  comparison with its hash.
- A **failure record** (`dispatchatlas.failure/1`) holds the error, its kind,
  and whether it is terminal or recoverable.
- A **session record** (`dispatchatlas.session/1`) holds the execution
  settings, the counts, and the environment: Python, the operating system and
  machine, the processor and memory, the indices of the processors the
  session was allowed to use (`cpu.allowed`, null when the system does not
  tell, in which case `cpu.available` counts the machine), the value of each thread
  variable in the launching environment (`threads`, null when unset), the
  versions of DispatchAtlas, NumPy, and the installed backends, and the git
  commit when the source or the working directory is a checkout. It records
  no host or user name.

`read_campaign`, `read_plan`, `read_runs`, `read_failures`, and
`read_sessions` read these files back as typed records.

## Resuming

Running `campaign run` again with the same file and output resumes the
campaign. A record already written is kept when it belongs to the same config
hash and still matches its result hash, and only the runs that are neither
completed nor settled by failures are attempted. An interrupted session
therefore loses at most the runs that were in progress.

- `--retry-failed` gives every run that failures settled one more attempt.
- `--max-runs N` attempts at most `N` runs in this session.
- `--workers`, `--backend`, and `--affinity` override the file's execution
  settings; they do not change the config hash. `--affinity` takes
  `inherit`, `pin`, or processor indices such as `0,2,4`. `--backend
  sequential` implies `--affinity inherit` and the default
  `threads_per_worker`. `campaign replay` takes the same three options, so a
  campaign pinned on one machine resumes and replays on another. A campaign
  with nothing left to attempt is not refused for its pinning.
- A directory that holds results of another configuration is refused;
  `--fresh` discards its records first. Give a changed experiment a new `id`
  to keep both.

## Replaying

`campaign replay` runs completed runs again from their recorded seeds, with
the same solver parameters and stop criteria, and compares result hashes,
without writing anything. It first checks every record against its own result
hash, so a record edited after it was written is caught even when its run
reproduces. Each run gets a verdict: `match`, `mismatch` with the outcome
fields that differ, `skipped`, or `failed`. Runs stopped by their wall-clock
limit are skipped, because how far they got depends on the machine. `--run`
replays selected runs only, and the exit code is 1 when a run does not
reproduce.

## From Python

Every command has a function behind it:

```python
from dispatchatlas.campaigns import estimate, load_campaign, plan, read_runs, replay, run


def main() -> None:
    campaign = load_campaign("campaign.toml")
    checked = plan(campaign)  # what `campaign validate` shows
    print(estimate(campaign).remaining)  # what `campaign estimate` shows
    summary = run(campaign, "results")  # resumes when called again
    records = read_runs(summary.directory)
    report = replay(campaign, "results")
    print(report.ok)


if __name__ == "__main__":
    main()
```

A campaign can also be built in Python from `Campaign`, `FamilySource`,
`SuiteSource`, `SolverSpec`, `Protocol`, and `Execution`, as the example
`examples/campaign_analysis.py` does, and written to a file with
`campaign_to_dict`.
