# Campaigns

A campaign is a reproducible experiment: every solver of a roster runs on
every problem of its instance sources, under every stopping protocol, with a
number of replicates. It is described in a TOML or JSON file, planned and
checked before anything runs, executed in worker processes, recorded one JSON
file per run, resumed after an interruption, checked as a directory before
its results are used, and replayed to verify that its runs reproduce.

```bash
dispatchatlas campaign validate campaign.toml --check-compatibility
dispatchatlas campaign estimate campaign.toml
dispatchatlas campaign run campaign.toml --output results --workers auto
dispatchatlas campaign check results/<campaign id>
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
| `replicate_timed` | no | `true` gives a solver declared deterministic every replicate under a protocol that sets a time limit; default `false` |
| `seed_namespace` | no | a name that separates the run seeds of this campaign from those of campaigns with the same `seed`; default none |
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
an optional `label`, and an optional `replicates`. The label names the entry
in results and tables, so two settings of one solver can compete in one
campaign. Parameters are checked against the solver when the campaign is
planned. `replicates`, an integer of at least 1, is the number of runs of
this solver per problem and protocol, in place of the count the campaign
would give it (see [Planning](#planning)).

A protocol has a `name` and a `stop` table with the fields of `StopCriteria`:
`time_limit`, `max_evaluations`, `max_iterations`, `target`, `patience`, and
`schedule`. Every protocol must set a time, evaluation, or iteration limit.

`schedule` ends no run. It states which clock a solver's parameter schedule
follows, such as an exploration rate that falls over the run. Such a solver
asks how far through its budget the run is, and the answer is the largest
spent fraction among the limits the setting counts:

| `schedule` | Limits counted | Consequence |
| --- | --- | --- |
| `"deterministic"`, the default | the evaluation and iteration limits; the time limit only when it is the only limit | under an evaluation or iteration limit the schedule is the same on every machine and replays exactly, and it stops short of its end when the time limit ends the run first |
| `"binding"` | every limit that is set | the schedule reaches its end at whichever limit ends the run, and under a time limit its position at a given evaluation depends on the speed and load of the machine, so no run of an iterative solver replays |

With the time limit as the only limit, both settings follow the elapsed time.
A solver that follows a schedule records in the `extras` of its outcome
`schedule_clock`, the limit that places the end of the run on the schedule
(`evaluations`, `iterations`, or `time`), and `schedule_progress`, that place
between 0 and 1. A value below 1 says the schedule did not reach its end.

### The config hash

The **config hash** identifies an experiment: the SHA-256 digest of the
canonical JSON of the fields that determine the results, `seed`, `instances`,
`solvers`, `protocols`, `replicates`, and `objectives`, as written, so
reordering a list changes it too. The optional settings `replicate_timed`,
`seed_namespace`, a solver's `replicates`, and a protocol's `schedule` enter
it only when they are set to other than their default, so a campaign file
that sets none of them has the config hash, the seeds, and the run ids it has
without them. The campaign `id` only names the directory,
and `execution` and `metadata` never change a result, so they stay outside
the hash. The locations of instance files and directories stay outside it
too: a problem read from a file is identified by the digest of its content,
which enters the id of every run on it. A campaign therefore keeps its
identity when its files move, or live elsewhere on another machine.

## Planning

`campaign validate` reads the file, loads every problem, checks every solver
id and parameter, and expands the grid problem × solver × protocol ×
replicate into planned runs, without running any. With `--runs` it lists every
planned run with its id, problem, solver, protocol, replicate, and seed, the
ones `campaign run` then writes to `plan.json`; with `--json` as well, `runs`
is that list in place of the number of runs.

- **Replicates.** A stochastic solver runs `replicates` times per problem and
  protocol, as replicates 0, 1, and so on. A solver declared deterministic
  runs once, as replicate 0. Where a time limit ends its run, though, the
  result depends on the speed and load of the machine, and one run does not
  show by how much. With `replicate_timed = true`, such a solver runs every
  replicate under each protocol that sets a time limit, and still once under
  the others. A solver that ignores the time limit then repeats one result.
  Without the key, the plan notes the iterative solvers it concerns, and
  states the others in a conditional note (see
  [Fair comparison](#fair-comparison)).
- **A solver's own count.** A solver entry with `replicates` runs that many
  times under every protocol, whatever it declares. It is the way to give
  several runs to a solver that is declared deterministic and is not under
  its parameters, such as an exact solver with several threads, and to give
  one run to a solver that ignores the clock in a campaign with
  `replicate_timed`.
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

Two campaigns with the same `seed` give the same seed to a problem they
share. To keep their random streams apart, as for a tuning study and the
comparison that follows it, give each a `seed_namespace`. The root of the
derivation is then

```text
derive_seed(seed, "namespace:" + seed_namespace)
```

in place of `seed`, so two campaigns that differ only in namespace share no
run seed, and each stays paired within itself. The namespace is part of the
config hash. Without one, every seed is the one above.

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
runs they stop cannot be replayed; neither can any run of an iterative
solver under a protocol whose only bound is its time limit, or whose
`schedule` is `"binding"` beside a time limit. An iteration means a different
amount of work in different solvers.

The assessment also carries **notes**: facts that break none of these rules
and that a reader of the comparison needs. The planner logs each as a
warning, `campaign validate` prints them, `plan.json` stores them, and the
analysis reports those of the analysed protocol as limitations. Both concern
protocols with a time limit: solvers whose `threads` differ (see
[Execution](#execution)), and iterative solvers declared deterministic that
run once although the campaign has more replicates, unless the solver entry
states its own `replicates`.

A solver declared deterministic that is not iterative ends when its algorithm
does. A constructive rule such as `neh` never reads the clock, and an exact
solver such as `milp` can be ended by its time limit; the plan cannot tell
the two apart. Such a solver that runs once under a time limit is therefore a
**conditional note** (`conditional_notes` in `plan.json`): it holds only if
the time limit ends one of the solver's runs. The planner logs it as
information, `campaign validate` prints it beside the notes, and the analysis
reports it as a limitation only for the solvers whose completed runs the time
limit ended.

A fair plan does not make the runs comparable by itself. Under one protocol, a
solver that stops for lack of improvement spends less than the budget another
uses up to its time limit, and a solver that honours no deadline spends more.
`dispatchatlas.campaigns.fairness.assess_outcomes(plan, records)` compares the
records of the completed runs with the plan, protocol by protocol: how many
runs of each iterative solver ended for each stop reason, whether those
reasons diverge, which solvers' runs took more wall time than the time limit
allows, and which solvers of a conditional note had a run that the time limit
ended. A run counts as over its time limit by the rule of the engine's note
on a run: its wall time passes the limit by more than the tolerance, 5 % of
the limit by default, and by more than 0.05 seconds. A divergence qualifies a
comparison and does not refuse it, since a reached target or a proof of
optimality is a legitimate end. The analysis reports these as limitations
(see [Analyzing campaigns](analyzing.md)).

## Estimating

`campaign estimate` counts the runs, by protocol and by solver, and bounds how
long their solves can take. The solve of a run whose protocol sets
`time_limit` takes at most its hard limit, and every attempt may take that
long; the time a worker needs to begin a solve is not counted. A run bounded
only by evaluations or iterations has no duration bound, so those runs are
counted apart. With `--output`, runs already completed there are left out.

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
  `timeout_factor * time_limit + grace` seconds, counted from the instant its
  worker states as the start of the solve. What comes before that instant is
  not charged to the run: the start of the worker's interpreter and the
  loading of plugins, for a worker that started for this run, the
  rebuilding of the solver and the problem, and the warm-up of the
  solver. The provenance of the attempt
  records it as `waited_seconds`, counted from the hand-over, before a fresh
  worker exists. When the hard limit passes, the supervisor kills the worker
  and records a timeout, and a fresh worker takes the next run. The
  supervisor never waits for a worker to start: a starting worker gets its
  run when it reports that it is ready, so the limits of the other runs are
  enforced meanwhile.
- **One clock per run.** The engine fixes the origin of a run's clock at the
  start of each attempt, and every solver counts its elapsed time and its
  time limit from that origin. It is one instant: the wall time of the
  attempt counts from it, and under the process backend it is the instant
  the worker states as the start of the solve, from which the hard limit
  counts. The `elapsed` of an outcome and the times of its trace are
  therefore seconds since the start of the attempt, for every solver alike:
  they include what a solver does in every run before its search starts,
  such as building its evaluator or its model, and the time limit covers
  that work too. A solver called directly, outside a campaign, counts from
  the instant it creates its clock, as
  [`dispatchatlas.core.clock`](../api/core.md) describes; a solver from a
  plugin that keeps a clock of its own starts it at `clock_start()` to
  follow the same origin. The provenance of an attempt states that its
  solver was timed this way (`shared_clock`).
- **What a solver does once in a process.** The first call of a solver in a
  process may pay what no later call pays, such as the import of a backend.
  Inside the time limit, that cost would shorten the search of the first run
  each worker executes, and of the first run of every worker that replaces
  a killed one, so replicates of one solver would search for unequal times
  according to the order of the runs and the number of workers. The engine
  therefore warms up the solver before the start of each attempt: it loads
  NumPy's random module and calls the solver's `warm_up` method, which
  `cp-sat`, `milp`, and `restricted-neighbourhood-search` use to import
  their backend. The warm-up is outside the time limit and the wall
  time of every run; under the process backend it counts toward the
  start-up limit and is part of `waited_seconds`. A warm-up that raises
  fails the attempt as an `exception`. A solver from a plugin that pays such
  a cost inside `solve`, without a `warm_up` method, is charged for it in
  the first run of each process; see
  [Extending](../extending.md#a-solver).
- **What the time limit cannot stop.** A solver checks its time limit
  between evaluations and never before its first one, so a run passes the
  limit by the evaluation in progress and by any work between two checks,
  and a heuristic that builds one schedule, such as a priority rule or a
  mapping heuristic, checks no limit at all. The sequential backend has no
  hard limit, and under the process backend a run that ends late but before
  its hard limit is an ordinary success. The engine therefore compares the
  wall time of every completed attempt with the time limit and keeps a note
  when it passes the limit by more than 5 percent and by more than 0.05
  seconds, and the [analysis](analyzing.md#runs-failures-and-denominators)
  reports these runs.
- **Start-up limit.** From the hand-over of a run, its worker has
  `startup_limit` seconds, 120 by default, to begin the solve, whatever the
  size of the problem it is sent. A worker that takes longer, or never
  starts, is killed, and the attempt is a start-up failure (`startup`): a
  kind of its own, recoverable, and no timeout of the solver. The limit
  applies to every run of the process backend, with or without a time limit.
  With `retries = 0` a start-up failure settles its run like any other
  failure; `--retry-failed` gives the run another attempt, and the
  [analysis](analyzing.md#runs-failures-and-denominators) counts these runs
  apart.
- **What a kill reaches.** A kill ends the worker and every process the
  solver started: on POSIX systems through the worker's process group, on
  Windows through a job object that the supervisor creates for each worker.
  Where Windows refuses the job object, for example on a version without
  nested jobs when the supervisor is in a job itself, a warning says so and
  a kill reaches the worker alone. A worker also ends itself when its
  supervising process is gone, so no worker of a killed session computes
  beside a resumed one. A worker inside a native extension that keeps the
  interpreter lock ends when the extension returns; on Windows the job
  object ends it at once.
- **What a killed run keeps.** While a run with a hard limit executes, its
  worker forwards the incumbents of a solver that runs inside `Search`:
  every objective value, the feasibility, and the evaluations, iteration, and
  elapsed time at which the search found the incumbent. A timeout failure
  holds two of them. `last_incumbent` is the last one the run reported; a
  solver that searches past its time limit may have found it after the
  limit. `limit_incumbent` is what the run held when its time limit passed:
  the last one found within `time_limit` seconds of the start of the solve.
  The two are equal when the search did not improve after the limit. The
  values are the solver's own: no schedule comes with them, so they are not
  checked as the result of a completed run is. An incumbent leaves the
  worker at once, from the thread of the search, when at least 0.25 seconds
  have passed since the last report, so it is kept whatever the solver does
  next. One found sooner after a report waits for the rest of the 0.25
  seconds, and is missing from the record if the worker is killed before
  then, or if the solver, right after finding it, enters a call that keeps
  the interpreter lock until the kill. Reporting costs the search about half
  a microsecond per improvement and, at most once every 0.25 seconds, one
  computation of the incumbent's objective values and one message on the
  pipe, however often the solver improves; the measured run time of a search
  whose every evaluation improves is the same with and without it, within 2
  percent. A solver that does not run inside `Search` reports nothing, and
  its timeout failure holds no incumbent. The
  [analysis](analyzing.md#runs-failures-and-denominators) scores a killed
  run as the worst value unless its setting `timeout_score` asks for the
  incumbent, and then by `limit_incumbent` alone.
- **Retries.** A failure is **terminal** when another attempt would fail the
  same way: a typed error of the library, such as an invalid configuration or
  a missing optional dependency, or a hard timeout. Any other exception, a
  worker that dies, and a worker that does not begin the solve within the
  start-up limit are **recoverable**. A run gets up to `1 + retries`
  attempts, until a terminal failure or the last attempt settles it.
- **What each attempt checks.** Every schedule a solver returns is checked
  with `check_schedule`, and every objective is computed again from the
  schedule, so a record never relies on the solver's own claims. The
  measurements cover the solve alone: wall time, CPU time, and on Linux the
  peak resident memory. Each attempt also records the processor its worker
  was pinned to, how many runs were executing when it was handed to its
  worker, itself included, and how long it then waited for its solve to
  begin.
- **Notes.** What the engine notices about a completed attempt is logged as
  a warning and stored in the run record, in `notes`; the run stays an
  ordinary completed run. A note is kept when the solver reports an
  objective value its schedule does not give, calls an infeasible schedule
  feasible, or returns a malformed schedule; when the wall time passes the
  time limit by more than 5 percent of the limit and by more than 0.05
  seconds, which the scheduler of a machine can add to any run; when the
  solver stops for its time limit while the wall time is below the limit by
  more than 5 percent of it and by more than 0.05 seconds, as a solver does
  that gives a search a share of its limit counted from the start of the
  attempt; when the elapsed time the solver reports differs from the wall
  time the engine measured by more than 0.05 seconds and by more than 5
  percent of that wall time; when the trace holds a negative time, times
  that decrease, or a time more than a microsecond beyond the elapsed time
  the solver reports; when the evaluation counts of the trace decrease; when
  the last value of the trace differs from the value of the outcome; and
  when an iterative solver returns a schedule and no trace. A result whose
  elapsed time, or whose trace holds a time or a value, that is not a finite
  number cannot be compared or stored: the attempt fails with an
  `exception`, and the session continues.

## Storage layout

```text
<output>/<campaign id>/
    campaign.json      the campaign document, its base directory, and its config hash
    plan.json          the plan, with the parities, the fairness assessment, and its digest
    sessions/<n>.json  one record per invocation of the runner, with its environment
                       and the runs that left or entered the plan
    runs/<solver>/<problem>/r<replicate:03d>-<protocol>.json
    failures/<solver>/<problem>/r<replicate:03d>-<protocol>.attempt-<n>.json
    superseded/runs/<solver>/<problem>/r<replicate:03d>-<protocol>.<run id>.json
    superseded/failures/<solver>/<problem>/r<replicate:03d>-<protocol>.attempt-<n>.<run id>.json
                       files moved aside when new records took their places
    logs/session-<n>.log  what each session logged, with UTC time stamps
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
  not pinned), the runs executing when the run was handed to its worker
  (`concurrent`), and the seconds from that hand-over to the start of the
  solve (`waited_seconds`, null for the sequential backend), which hold the
  start of a worker that started for the run and the warm-up of the
  solver, and whether the solver's clocks started at the instant the wall
  time counts from (`shared_clock`, true for an attempt whose solver
  returned or raised, null for one the engine ended or that never reached
  its solver). A record that lacks one of the four reads as null for it. The CPU
  seconds are those of the executing process, with the threads the solver
  started, and on POSIX systems those of the child processes it started and
  waited for; on Windows the time of a child process, such as a backend
  that runs as a program of its own, is not in them. The record also holds the
  `notes` of the engine on the attempt, a list of sentences that is empty
  when the engine noticed nothing; a record without the field reads as one
  with no notes. The `elapsed` of the outcome is the solver's report, in
  seconds since the start of the attempt, and the `wall_seconds` of the
  provenance is the engine's own measurement from the same instant until
  the solver returned.
- Its **result hash** is the SHA-256 of the outcome without the fields that
  measure wall time, and without the solver-specific extras. Nothing about the
  machine or the session enters it, so a replay on any machine compares
  results directly, and a record edited after it was written fails the
  comparison with its hash.
- Its **identity hash** (`identity_hash`) is the SHA-256 of what names the
  run: the digest of the problem, the id and the parameters of the solver,
  the protocol with its stop criteria, the replicate, and the seed. A
  record whose seed, or another of these fields, was edited fails the
  comparison as a record with an edited outcome does: `read_runs` refuses
  it, and a resume keeps it aside and runs its run again. The times an
  attempt measured, the solver-specific extras, and the provenance are
  under neither hash. A record without the field is not checked for it.
- A record is also compared with **what the plan states for its run**: the
  id and the digest of the problem, the name, the id, and the parameters of
  the solver, the protocol, the replicate, and the seed. The comparison
  needs no hash, so it holds for a record that states no identity hash as
  well. `read_runs` refuses a record that states another value, and a
  resume runs its run again. A seed that the run had in an earlier plan of
  the directory is accepted, as the session records name it in
  `plan_reseeded`; when a session record cannot be read, the earlier seeds
  are not known and the seed is not compared.
- A **failure record** (`dispatchatlas.failure/1`) holds the error, its kind
  (`exception`, `timeout`, `startup`, `worker-exit`, or `transfer`), and
  whether it is terminal or recoverable. It says whether the solve had begun
  when the attempt ended (`solve_started`), and the record of a run killed
  at its hard limit holds the last incumbent its search had reported
  (`last_incumbent`, null when it had reported none) and the last one it had
  found within its time limit (`limit_incumbent`, null when it had found
  none by then). A record that lacks one of the three fields reads as null
  for it. The `wall_seconds` of an attempt that
  the engine ended count from the start of the solve, or from the hand-over
  when the solve never began.
- A **session record** (`dispatchatlas.session/1`) holds the execution
  settings, the counts, and the environment: Python, the operating system and
  machine, the processor and memory, the indices of the processors the
  session was allowed to use (`cpu.allowed`, null when the system does not
  tell, in which case `cpu.available` counts the machine), the value of each thread
  variable in the launching environment (`threads`, null when unset), the
  versions of DispatchAtlas, NumPy, and the installed backends, and what
  identifies the code. `library` states where the DispatchAtlas
  distribution was installed from (`origin`: `vcs`, `archive`, `directory`,
  or `index`), whether it is `editable`, its `revision`, and
  `source_digest`, a SHA-256 over the files of the package. The revision is
  the commit of a checkout, the commit of the version-control URL an
  installed copy came from, or the hash of an archive; it is null for a
  copy built from a local directory, whose record does not say which state
  was built, and the source digest identifies that copy. When the package
  that runs is not the one the installed distribution provides, such as a
  copy that comes first on the module search path, the installer's record
  does not describe it: `origin` is `path`, the version is null, and the
  revision is that of the copy's checkout or null. `plugins` states
  the same facts for every distribution that provides a solver or
  benchmark entry point. `git` states, for the source and for the working
  directory when either is a checkout, the `commit`, whether the checkout
  is `dirty` (a tracked file differs from the commit, or there is an
  untracked file that git does not ignore), the number of `untracked`
  files, and a `digest` of the uncommitted changes, so two different sets
  of edits of one commit have different digests; an editable plugin states
  the same for its checkout. The directory of the campaign is left out of
  every one of these states, so results kept inside a checkout are no
  change of its code. Untracked files enter the digest by their content up
  to 16 MiB in all, in path order; a file past that total enters by its
  size and modification time, and more than 1000 untracked files enter by
  their paths only. The queries do not write to the checkout: git's index
  stays as it is. The
  processor is recorded by its model name, with its `sockets` and physical
  `cores` where the system gives them. The record holds
  no host or user name. Under `load` it holds two samples of the machine,
  one taken before the first run (`start`) and one after the last (`end`,
  for a session that finished): the share of the processor time of the
  whole machine that was busy over a tenth of a second (`busy`), and the
  one-minute load average where the system has one. A session with nothing
  to attempt takes no sample. Under `changes` it holds how the session
  differs from the session before it (see [Resuming](#resuming)). It also holds `plan_digest`, the digest of the run
  ids and seeds of the plan the session executed, which `plan.json` states as
  `runs_digest`, and `plan_left`, `plan_entered`, `plan_reseeded`, and
  `plan_solvers`: how that plan differs from the one the directory held
  before under the same config hash (see [Resuming](#resuming)). A record
  without these fields reads as one that states no digest, no change, and
  no load.

`read_campaign`, `read_plan`, `read_runs`, `read_failures`, and
`read_sessions` read these files back as typed records. `read_runs` returns
the records of the plan, and warns about every other file under `runs/`:
records of runs the plan does not hold, files that cannot be read (one that
is not UTF-8 text or not JSON, one that is no run record, one the system
does not let read), and files that are no records. `read_strays` returns
those files. No session replaces or removes them.

The log file of a session holds one record per line, with its UTC time to
the millisecond, its level, and its logger:

- what the engine itself logs while the session runs, from the level
  `INFO`: the start of the session, its progress, its warnings, and its
  summary;
- what the other loggers of the library let through, among them those of
  solvers and plugins that log through
  `dispatchatlas.core.logs.get_logger`: `WARNING` and higher unless the
  application lowered a level. A worker process starts with the default
  level, so it sends `WARNING` and higher, each line marked with the name
  of the worker.

The execution key `log_level` asks for more: `"info"` or `"debug"` lowers
the library's loggers to that level for the session, in the calling process
and in every worker; the default `"inherit"` leaves them as they are. The
file is written beside whatever the application shows. Under every setting
the lines on standard error are the same with and without it, in the calling
process and in the workers, for the library's loggers and for the
application's own. Only a handler that is attached while a session runs is
not held to that: it receives what the lowered loggers record.

Logging inside a run costs the run time. A record below the level of its
logger is not made, and the call that would log it costs next to nothing,
so a solver that logs details in its loop is not slowed under the default
setting. A record that is made costs its making, its formatting, and a
buffered write or, in a worker, a place in a queue: tens of microseconds
each. A solver that logs every iteration of a short loop therefore does a
fraction of its work in the same time limit once `log_level` lets those
records through, and its results under a time limit are not comparable
with those of a session that did not. The thread that logs in a worker
never waits for the calling process: the queue holds 10,000 records, a
record beyond them is dropped, and the log states how many were. The log
file is buffered, so the lines of a burst below `WARNING` reach the disk
with the next warning, with the next record after a second, or when the
session ends. It grows to about 64 MiB (2^26 characters) at most: later
records are counted, and its last line states how many were left out.

## Resuming

Running `campaign run` again with the same file and output resumes the
campaign. A record already written is kept when it belongs to the same config
hash, still matches its result hash and its identity hash, and states what
the plan states for its run, and only the runs that are neither
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
- A session compares its execution settings and its environment with those
  of the last session the directory recorded. It logs one warning that
  names each difference with both values, such as `execution.workers 1 ->
  2` or `environment.library.source_digest`, and its record stores all of
  them in `changes`, each with `field`, `earlier`, `now`, and `kind`:
  `execution`, `code` (the Python version, the packages, the library, the
  plugins, and the git states), or `machine` (the rest of the
  environment). A value the earlier record does not state is not compared.
  With the execution key `on_changed_code = "refuse"`, a session whose
  code differs is refused before it writes anything; the default `"warn"`
  resumes. A difference of execution settings or of the machine is never
  refused. The directory of the campaign itself is left out of every git
  state, so it may lie inside the working directory, the library's
  checkout, or the checkout of an editable plugin. Keep other files that
  change between sessions, such as the results of other campaigns, out of
  those states, by an ignore rule or a place outside the checkouts.
- A worker process that starts after a source file of the library changed
  runs on other code than the session recorded. The runner warns about
  such a worker, and every run it completes carries a note with both
  source digests.
- A file at the path of a planned run that is not the valid record of that
  run does not count, and is never replaced. This holds for a record made
  on another version of its problem, whose content changed under the same
  id, for a record that no longer matches one of its hashes or the plan,
  and for a file that cannot be read. The session names such files in a
  warning when it starts and runs their runs again. Each file stays where
  it is until its run is repeated, and moves to `superseded/` when the
  repeated run gets its first record, of its result or of a failed
  attempt: under `runs/` and the same solver and problem folders, with the
  run id it states before `.json`. A session that repeats no run, because
  of `--max-runs` or an interruption, moves nothing, and when the problem
  is what it was again, its records count again without a rerun. The
  record of a failed attempt on another version of a problem is kept the
  same way, under `superseded/failures/`, when a new attempt takes its
  path.
- A record under `superseded/` that is the valid record of a planned run
  again, because the problem was restored after some of its runs were
  repeated, moves back to its path when a session starts, and the record
  that stood there moves aside in its place. Nothing else reads
  `superseded/`, and nothing replaces a file in it. `campaign estimate`
  counts such a run as complete and moves nothing.
- A file that cannot be moved aside, for example because another program
  holds it open, ends the session as failed with an error that names the
  file. The new record is not written, the file stays, and the next
  session attempts the run again.
- With the execution key `on_changed_problem = "refuse"`, a directory that
  holds a record, or the record of a failed attempt, of another version of
  a problem is refused before anything moves; the default `"rerun"` runs
  the runs again. `--fresh` also discards `superseded/` and `logs/`.
- A directory that holds results of another configuration is refused;
  `--fresh` discards its records first. Give a changed experiment a new `id`
  to keep both.
- The plan is built again in every session, and it follows more than the
  campaign file: the metadata the registered solvers declare, the content of
  the problems, and the library that plans. When the new plan differs from
  the one the directory holds under the same config hash, the runner warns
  with the number of runs that left the plan and that entered it and names
  the first of each, and the session record stores all of them in
  `plan_left` and `plan_entered`, beside the `plan_digest` of the new plan.
  The records of runs that left stay in the directory and are no longer
  read. The warning also names the runs whose seed changed and the fields of
  the solver entries that changed, such as a solver's `iterative` flag, and
  the session record stores them in `plan_reseeded` (each run with its
  `earlier_seed`) and `plan_solvers` (each with `solver`, `field`,
  `earlier`, and `now`). The session record is written before `plan.json` is
  replaced, so a session that ends before its record leaves the comparison
  to the next session. Two sessions with equal `plan_digest` executed the
  same runs with the same seeds. The digest does not cover the solver
  entries.

## Checking a campaign directory

`campaign check` reads a campaign directory before its results are used and
reports what qualifies them. It writes nothing and runs nothing.

```bash
dispatchatlas campaign check results/<campaign id>
dispatchatlas campaign check results/<campaign id> --replicate-floor 10 --json
```

The readers of results work from `plan.json` and pass over what does not fit
it, so a directory can look complete and not be. The check asks what they do
not:

- **The campaign file.** Whether `campaign.json` is there and still matches
  the config hash recorded with it.
- **The plan.** Whether `plan.json` lists the problem, the solver, and the
  protocol of each of its runs and names nothing twice; a plan that does not
  ends the check. Whether it has the campaign's config hash, and whether it
  is the plan that the stored campaign gives when it is planned again on
  this installation. Every field is compared, and each difference names its
  field: the campaign id, the seed and its namespace, the replicates, each
  field of each problem, solver, and protocol, the problem, solver,
  protocol, replicate, and seed of each run, the order of each list, the
  excluded pairs with their features, and the caveats. Whether it holds the
  runs and seeds the last session executed (`plan_digest`), and whether the
  plan changed between sessions. The entries that the library derives from a
  plan and writes for readers of the file, such as `runs_digest`, the
  `parities` of a protocol, `fairness`, and the `reason` of an exclusion,
  are compared with what the file's own plan gives. The library derives them
  again wherever it reads a plan, so a difference there, and another wording
  of a caveat, is a note.
- **The planned runs.** Every run needs a valid record or failures that
  settle it. A run with neither was never attempted, and one whose failures
  the campaign's `retries` do not settle is owed another attempt: both block.
  Runs settled by failures are counted by solver and by the kind of error of
  their last attempt. A cell, one solver on one problem under one protocol,
  that has records for some of its planned runs only is named, and so is a
  solver without any record. The pairs of problem and solver that the plan
  excludes have no run and are not missing: they are listed apart, and a
  solver excluded from every problem is named.
- **The files.** A file at a planned run's path that is no valid record of
  the run: one that cannot be read, a record of another configuration or of
  another version of its problem, a record that fails its result hash, its
  schedule digest, or its identity hash, and a record that states another
  problem, solver, protocol, replicate, or seed than the plan. A valid record
  that lies under `superseded/`. Records of runs the plan does not hold,
  files that cannot be read, and files that are no records.
- **The records.** A trace that cannot be read against time, because it
  holds a negative time, times or evaluation counts that decrease, or a time
  beyond the elapsed time of its run; a trace whose last value differs from
  the value of the outcome; and a record of an iterative solver with a
  schedule and no trace. The check derives these facts from each record, so
  it covers records that hold no notes of the engine.
- **The budgets.** Under each protocol, the solvers whose completed runs
  took more wall time than the time limit and the tolerance allow, with the
  largest ratio (`--overrun-tolerance`, by default the margin of the
  engine's note on a run); hard limits that differ among the attempts; and
  stop reasons that differ among the runs of the iterative solvers.
- **The sessions.** The sessions that executed the attempts the directory
  counts are compared with each other. Each field that differs is named
  once, with its values and the sessions that have each. Differences in the
  Python version, the packages, the library, or a git state, in the plugins,
  in the execution settings, and in the machine are four findings. A field
  is compared among the sessions whose records state it, and a note names
  the sessions whose records state less than another's
  (`sessions-fields-unstated`), such as a record without the library, the
  plugins, the thread settings, or the allowed processors. A difference in
  the machine is a note, except for what a time limit is measured on: when
  the sessions that executed the attempts of a protocol with a time limit
  differ in the processor, the processors they were allowed, or the thread
  settings, that is a limitation of the protocol
  (`sessions-differ-processors`). A session that executed nothing that
  counts is not compared. An attempt of a session without a record is
  reported.
- **The replication.** With `--replicate-floor N`, the cells of replicated
  solvers that hold fewer than `N` completed replicates. A seed that serves
  two replicates of one problem blocks, and a replicate of a problem that ran
  with two seeds, as after a plan gave its runs other seeds, is a limitation.

Every finding has a code, a severity, a message, the items it concerns, and,
when it concerns the runs of one protocol, that protocol. The severities are:

- **blocking**: the directory does not hold what it claims, so its results
  are not ready to be used;
- **limitation**: the results stand, with a caveat that a report of them has
  to state;
- **note**: a fact that qualifies nothing, such as an optional field that
  the records of the directory do not state. Such a directory is checked as
  far as its files allow: records without an `identity_hash` are compared
  with the plan field by field, and without a `plan_digest` in the session
  records the plan file is not compared with the last session.

The exit code is 1 when a finding is blocking, and 0 otherwise. The text form
names the first six items of a finding; `--json` prints the whole report,
with `ok`, the planned runs by what the directory holds for them (`counts`),
the number of findings of each severity (`severities`), and every finding
with all of its items. Planning the campaign again loads every problem, so it
needs the campaign's instance files and solvers; `--no-replan` leaves that
comparison out and says so in a note.

<!-- dispatchatlas:check-findings -->

`dispatchatlas analyze` and `load_dataset` run the same check, without
planning the campaign again, and the analysis states the blocking findings
and the limitations for the analysed protocol among its own limitations. It
leaves out the five that it states itself, over the solvers and problems it
analyses: `runs-unattempted`, `runs-failed`, `stop-reasons-diverge`,
`budget-overrun`, and `replicates-below-floor`.

## Replaying

`campaign replay` runs completed runs again from their recorded seeds, with
the same solver parameters and stop criteria, and compares result hashes,
without writing anything. It first checks every record against its own result
hash, so a record edited after it was written is caught even when its run
reproduces. Each run gets a verdict: `match`, `mismatch` with the outcome
fields that differ, `skipped`, or `failed`. Runs stopped by their wall-clock
limit are skipped, because how far they got depends on the machine. So is
every run of an iterative solver under a protocol whose only bound is its
time limit, whatever stopped it: without an evaluation or iteration limit a
search places itself on its schedule by the elapsed time, so a run that
stopped at a target or for lack of improvement also depended on the
machine. The same holds under any time limit when the protocol's `schedule`
is `"binding"`. A solver that is not iterative, such as a priority rule, follows
no schedule, and its completed run is replayed under any protocol. `--run`
replays selected runs only, and the exit code is 1 when a run does not
reproduce.

## From Python

Every command has a function behind it:

```python
from dispatchatlas.campaigns import (
    check_campaign,
    estimate,
    load_campaign,
    plan,
    read_runs,
    replay,
    run,
)


def main() -> None:
    campaign = load_campaign("campaign.toml")
    checked = plan(campaign)  # what `campaign validate` shows
    print(estimate(campaign).remaining)  # what `campaign estimate` shows
    summary = run(campaign, "results")  # resumes when called again
    findings = check_campaign(summary.directory)  # what `campaign check` shows
    for finding in findings.of("blocking"):
        print(finding.code, finding.message, finding.items)
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
