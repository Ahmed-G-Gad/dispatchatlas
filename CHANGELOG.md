# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- A scheduling data model: tasks with alternative modes, renewable resources
  of any capacity, dependencies with time lags and network transfers, release
  times, due dates, hard deadlines, sequence-dependent setups, and permutation
  schedules. Problems are validated when built, and `check_schedule` checks
  any schedule against every constraint.
- An objective registry with makespan, completion, flow, response, tardiness,
  lateness, energy, and cost objectives, open to plugins.
- Benchmark families for shop, parallel-machine, project, task-graph, and
  edge-fog-cloud continuum problems; Taillard's suites rebuilt from their
  published seeds; readers for OR-Library, FJS, PSPLIB, WfCommons, and Pegasus
  DAX files; a streaming reader for the batch jobs of the Alibaba 2018 cluster
  trace, which decodes each job's task graph and reports every job it skips
  with a reason; instance catalogs with SHA-256 manifests; and descriptive
  instance features.
- Three of the readers map a workload onto a continuum platform:
  `read_wfcommons` for WfFormat 1.4, 1.5, and 1.6 instances,
  `read_pegasus_dax` for Pegasus DAX files, and `iter_alibaba_batch` for the
  Alibaba trace. They choose nodes from the same `tiers` argument and refuse
  the same values of it, refuse an objective that needs due dates, and report
  every refused argument as a `ConfigError`, a source that is not a usable
  path included. A number in a DAX file or in the table of the trace is
  written in ASCII digits with an optional minus sign, fraction, and
  exponent; a WfFormat file holds JSON numbers, which also need a digit on
  both sides of the decimal point and a whole part without a leading zero. A
  problem read from one file is named after the file, without its extension,
  an empty `problem_id` is refused, and an error in the content of the file
  names the file.
- `read_wfcommons` names a parsed document after its instance name and keeps
  the instance name in the problem's metadata. A document that contradicts
  itself is refused: a task or an execution entry listed twice, an execution
  entry for a task the specification does not list, or a file in which a JSON
  object repeats a key.
- Solvers from dispatching rules, list scheduling, and constructive
  heuristics to local search, population and swarm methods, multi-objective
  evolutionary algorithms, learning methods, hyper-heuristics, exact CP-SAT,
  MILP, and enumeration solvers, and a matheuristic, all under one contract
  that enforces stop criteria and counts evaluations exactly. Solver and
  benchmark plugins are discovered through entry points.
- Campaigns described in TOML or JSON, with paired seeds, verified fair
  comparisons, worker processes with hard time limits and retries, one record
  per run, resumption, estimates, and replay.
- Campaigns count the processors their process is allowed to use, on Linux
  and on Windows, for `workers = "auto"`, for the warning about shared
  processors, and in the session record, which lists their indices. The
  execution key `affinity` pins each worker to a processor of its own, and
  `threads_per_worker` limits the threads of numerical libraries in every
  worker from the moment its process is created. A run record states the
  processor its worker was pinned to and how many runs were executing when
  it was handed over, and a plan notes a time limit that compares solvers
  with different thread counts, which it reads from the `threads` every
  solver holds and keeps in its solver entries. A pinned worker is created
  on its processor and reports it; a campaign that cannot be pinned is
  refused before it discards a record or creates its directory;
  `campaign run` and `campaign replay` take `--affinity`; and the sequential
  backend, which has no workers, accepts neither a pinning nor a
  `threads_per_worker` other than 1.
- The hard limit of a run counts from the instant its worker states as the
  start of the solve. Until then the run is under the execution key
  `startup_limit`, 120 seconds by default, counted from the hand-over of the
  run, before a fresh worker exists; its expiry is a recoverable failure of
  the kind `startup`, and the provenance of an attempt states the wait as
  `waited_seconds`. The supervisor never waits for a worker to start, so it
  enforces the limits of the other runs meanwhile. A run killed at its hard
  limit keeps, in its failure record, the last incumbent its search reported
  (`last_incumbent`) and the last one it found within its time limit
  (`limit_incumbent`); the record also says whether the solve had begun
  (`solve_started`). The analysis setting `timeout_score` scores such a run
  as the worst value or by the incumbent it held at its time limit, never by
  one found after it; the runs table says what every run is scored by, and
  the limitations count the killed runs, those that reported an incumbent
  after their time limit, and the runs that ended at the start-up limit. The
  reports of a run end with its solve and never hold back its result. A
  kill ends every process the solver started, on Windows through a job
  object for each worker, and a worker ends itself when its supervising
  process is gone.
- One clock per run: `dispatchatlas.core.clock` holds the origin of solver
  clocks, and the engine sets it at the start of every attempt, so the
  elapsed time and the time limit of every solver in a campaign cover what
  the solver does before its search starts, such as building an evaluator
  or a model. The same instant is the start of the hard limit and of the
  wall time, and the provenance of an attempt states it in the optional
  field `shared_clock`. What a solver does once in a process, such as
  importing a backend, the engine has it do before that instant, through
  the solver's optional `warm_up` method, so that it shortens the search of
  no run. A solver called outside a campaign counts from the instant it
  creates its clock.
- The engine compares what every completed attempt took with its budget and
  keeps what it notices in the optional field `notes` of the run record: a
  wall time beyond the time limit, a stop for the time limit well before
  it, an elapsed time its own clock contradicts, a trace whose times or
  evaluation counts are out of order or that ends at another value than the
  outcome, and an iterative solver that returns no trace, beside the claims
  of a solver that the checks of its result contradict. A result whose
  times are not finite numbers fails its attempt. The CPU seconds of an
  attempt include, on POSIX systems, the child processes the solver waited
  for. The runs table holds `time_limit`, `wall_seconds`,
  `overrun_seconds`, `cpu_share`, `concurrent`, `cpu`, and `attempts`; the
  limitations name the solvers whose completed runs passed their time limit
  by more than the analysis setting `overrun_tolerance` (`budget`), state
  the processor share and the concurrency the runs of a timed protocol
  executed under (`timing`), report records that do not all state a shared
  clock (`clock`), and count the failed runs by solver and kind of error,
  with what each kind means. A run of an iterative solver under a protocol
  whose only bound is its time limit is not replayed, whatever stopped it.
- Plans replicate what depends on the clock. The campaign key
  `replicate_timed` gives a solver declared deterministic every replicate
  under a protocol that sets a time limit, and a solver entry may carry its
  own `replicates`. Without the key, the plan notes the iterative solvers
  declared deterministic that run once under a time limit, and states those
  that are not iterative, which the clock ends only if they honour the time
  limit, in a conditional note that the planner logs as information. The
  campaign key
  `seed_namespace` separates the run seeds of campaigns that share a seed.
  The stop criteria state in the field `schedule` which clock a parameter
  schedule follows, `"deterministic"` or `"binding"`, and a solver that
  follows one, the swarm family included, records `schedule_clock` and
  `schedule_progress` in the extras of its result; under a time limit with
  `"binding"`, a run of an iterative solver is not replayed. Each of these
  settings enters the config hash only
  when it is set to other than its default, so a campaign file without them
  keeps its config hash, seeds, and run ids.
- `dispatchatlas.campaigns.fairness.assess_outcomes` compares the completed
  runs of a campaign with its plan: for each protocol, the stop reasons of
  the iterative solvers' runs, whether they diverge, the solvers whose runs
  took more wall time than the time limit allows, and the solvers of a
  conditional note whose runs the time limit ended. It counts an overrun by
  the rule of the engine's note on a run
  (`dispatchatlas.campaigns.execution.overran`), which the `budget`
  limitation of the analysis follows too. The analysis reports divergent
  stop reasons (`stop-reasons`), the notes of the plan on the analysed
  protocol, and the conditional notes that the outcomes confirm (`plan`) as
  limitations.
- `campaign validate --runs` lists every planned run with its id and seed
  before anything runs. A plan written over one of the same config hash
  names, in a warning, the runs that left it, the runs that entered it, the
  runs whose seed changed, and the fields of the solver entries that
  changed, and the session record keeps them (`plan_left`, `plan_entered`,
  `plan_reseeded`, `plan_solvers`) with the digest of the plan's run ids and
  seeds (`plan_digest`), which `plan.json` states as `runs_digest`. The
  session record is written before `plan.json` is replaced
  (`Store.plan_change`, then `Store.write_plan`). The manifest of an
  analysis bundle states the `runs_digest` of the analysed plan and, when
  the campaign sets them, `seed_namespace`, `replicate_timed`, and the
  solvers' own replicate counts (`solver_replicates`).
- A `processes` package for runs of caller-defined functions outside a
  campaign. `run_calls` runs every `Call` of a list in a child process of its
  own, started with the running interpreter on a processor of its own from an
  allowed set given as indices, a mask, or ranges, and appends the mapping
  each call returns as one JSON line to a rows file: UTF-8 without a
  byte-order mark, line feeds only, one `fsync` per row, one writer. A call is
  identified by a caller-defined key, which the rows file gives back through
  the caller's key function, so a second invocation skips the calls that have
  a row and refuses a torn last line with its line number and byte offset;
  `repair_rows` and `dispatchatlas processes rows-repair` move the torn bytes
  aside. A lock file held for the run refuses a second runner on the same
  rows file, a file holding a key twice is refused (`duplicate_keys`), and a
  resume whose stamp, worker count, or processors differ from the file's
  first invocation, recorded in a runs sidecar, is refused unless the caller
  allows it. A row holding a NaN, an infinity, or a value JSON has no form
  for is a failure that names the field. Every row carries the processor, the affinity mask the child read
  back after pinning itself and the one it inherited, the process id, the
  thread limits, the peak working set, the times, the number of worker
  slots, and how many were busy when the call started and at the least
  during its life; an optional ballast mode keeps every slot busy with
  discarded repeat calls until the last real call ends. A call that raises,
  dies, returns something that is not a row, or passes its time limit is
  recorded in a sidecar file, with a message cleared of file paths (with or
  without spaces) and a message and type cleared of caller-named words,
  never in the rows file, and is retried on the next
  invocation unless skipped; after a given number of failures the run stops
  at once, ends every child through its process group, writes its report,
  and raises `CallsStoppedError`. `interleaved` orders calls by a seeded
  shuffle, optionally class by class.
- Each child pins itself to its slot's processor first thing and refuses to
  run when the mask it reads back differs, which matters where the
  environment's `python` executable is a launcher that starts the real
  interpreter. Each slot has a kill-on-close process group, a Windows job
  object that a child enters before its first instruction and also joins by
  name, or a session on POSIX systems, so a dead runner takes its children
  with it and the end of a call ends what it started. The child sends its
  reply on a private copy of its standard output and points standard output
  at the children's log for good, so nothing the call or a helper prints,
  before or after the reply, can take the reply's place. The child entry,
  `python -m dispatchatlas.processes.child`, loads neither NumPy nor the
  solvers.
- Process tools without extra dependencies: a process snapshot with parent,
  executable, affinity mask, and start time (Toolhelp on Windows, `/proc` on
  Linux), the descendants of a process, the Python interpreters running on
  given processors (told by name or executable, so that a console script
  counts on Linux), `is_running`, `end_process`, `launch_detached` for a
  command that must outlive its shell (no window, its own process group, out
  of the caller's job where allowed), and `watch`, which reports once per
  interval whether a runner is alive, its live children, other interpreters
  on its processors, the rows file's line count, and the available memory,
  and never restarts anything; on the command line as
  `dispatchatlas processes launch | watch | list | end | rows-check | rows-repair`.
  The campaign engine's affinity, job-object, and memory primitives are the
  same ones.
- Provenance helpers: `git_state` can judge dirtiness within one folder of a
  checkout, `compare_trees` tells whether an installed package holds the
  same files with the same SHA-256 digests as a source tree (CRLF read as LF
  unless `normalize_newlines=False`, and the result records which), and
  `peak_memory_bytes` reads the peak memory of the process (the peak working
  set on Windows, `VmHWM` on Linux, `ru_maxrss` on macOS), and
  `available_memory_bytes` reads the available memory of the machine on
  Windows and Linux and returns `None` on macOS.
- Analysis of campaigns: rank tests, post-hoc tests, corrections, effect sizes
  with bootstrap intervals, Bayesian tests, power, relative deviation, and
  multi-objective indicators; tables in Markdown, CSV, and LaTeX; figures in
  SVG and TikZ; and a byte-reproducible bundle with a manifest.
- `dispatchatlas campaign check DIR` and
  `dispatchatlas.campaigns.check_campaign` check a campaign directory before
  its results are used, and write nothing. The report lists findings, each
  with a code, a severity (`blocking`, `limitation`, or `note`), a message,
  the items concerned, and the protocol when it concerns one: a campaign
  file that is missing or no longer matches its config hash; a plan whose
  runs name a problem, a solver, or a protocol it does not list, a plan of
  another config hash, one that differs in any field from the plan the
  stored campaign gives when it is planned again, with each difference
  naming its field, and one that differs from the plan the last session
  executed; derived entries of the plan file that its own plan does not
  give, as a note; planned runs without a record and without a failed
  attempt, runs whose failures the retries do not settle, runs settled by
  failures by solver and kind of error, cells with records for only some of
  their runs, and solvers without a record, with the pairs the plan excludes
  listed apart; files at the paths of planned runs that are no valid records,
  valid records under `superseded/`, records of runs the plan does not hold,
  unreadable files, and a file that the operating system does not let read,
  which is a blocking finding and no error; traces that cannot be read
  against time or that end at another value than the outcome, and records
  of iterative solvers without a trace; wall times beyond the time limit by
  solver with the largest ratio, hard limits that differ within a protocol,
  and divergent stop reasons; sessions that executed counted attempts on
  other code, plugins, execution settings, or machines, each field that
  differs named once with its values and their sessions, with a limitation
  for a protocol with a time limit whose sessions differ in the processor,
  the allowed processors, or the thread settings, and a note for session
  records that state less of the environment than another's; cells below a
  replicate floor; and a seed that serves two replicates of one problem.
  The command exits with 0 when no finding is blocking, with 1 when a
  finding is blocking or the path is no directory, and with 2 for a command
  line that is not valid, and `--json` prints the whole report. A directory
  whose records lack optional fields, such as `identity_hash` or the
  `plan_digest` of a session, is checked as far as its files allow, with a
  note for each comparison that cannot be made.
  `dispatchatlas.campaigns.execution.observations` returns the facts of a
  completed attempt's time and trace with a kind each, from which the check
  derives them again for any record; `Store.holdings` returns the ledger of
  a plan with the valid records it counts; and the ledger lists the failure
  files that cannot be read.
- `load_dataset` runs the same check, without planning the campaign again
  and before it reads the records itself, keeps the report in
  `Dataset.check`, and logs a blocking finding as a warning;
  `load_dataset(directory, check=False)` skips it. The analysis states each
  blocking finding and limitation of the check for the analysed protocol as
  a limitation under the code of the finding, and leaves out the five that
  it states itself, the partial cells that its `campaign` limitation names,
  and the solvers without a record whose runs its `failures` limitation
  counts. Its `campaign` limitation names the cells of solver and problem
  whose runs have no record.
- The Wilcoxon signed-rank test is exact up to 50 pairs with tied
  magnitudes too: the p-value comes from the distribution of the rank sum
  over every assignment of signs to the midranks, and the result names the
  method `exact-conditional`. The normal approximation is left to larger
  samples. The tie term of its variance, and of the Mann-Whitney and
  Friedman variances, counts the groups that share a midrank, so values
  equal within the tolerance of the ranks tie there too; `tie_sum` takes the
  `sense` in which the ranks are formed. The power analysis uses the exact
  rejection region up to the same 50 pairs.
- A bibliography of every implemented method, exported as text or BibTeX.
- The `dispatchatlas` command line, documentation, and runnable examples.
- The environment of a session identifies the code and the machine further:
  where the library's distribution was installed from and at which
  revision, for a checkout and for an installed copy (`library`), a digest
  of the library's source files (`library.source_digest`), every plugin
  distribution with its version and revision (`plugins`), a git state that
  counts untracked files and digests the uncommitted changes (`dirty`,
  `untracked`, `digest`), and the processor's model name, sockets, and
  cores, on Windows from the registry. A package that the installed
  distribution does not provide, such as a copy first on the module search
  path, is recorded with the origin `path`. The git states of the working
  directory, of the library's checkout, and of every editable plugin leave
  the campaign's directory out, are read without writing git's index, and
  read at most 16 MiB of untracked files. A session record holds two samples
  of the machine's load, before its first run and after its last (`load`).
- A session compares its execution settings and environment with those of
  the last recorded session of its directory, warns with each difference,
  and stores them in its record (`changes`). The execution key
  `on_changed_code`, `"warn"` by default, refuses a session on other code
  when it is `"refuse"`. A worker process that starts on other library
  source than its session is warned about, and its runs carry a note.
- A file at the path of a planned run that is no valid record of the run
  is never replaced: a record made on another version of its problem, one
  that fails a hash or differs from the plan, or an unreadable file. It
  moves under `superseded/` when the new record of its run is written, so
  a session that repeats no run moves nothing and a restored problem needs
  no rerun; the record of a failed attempt is kept the same way. A record
  under `superseded/` that is valid for a planned run again moves back
  when a session starts. The execution key `on_changed_problem`, `"rerun"`
  by default, refuses a directory with a record of another problem version
  when it is `"refuse"`. `read_runs` warns about the other files under
  `runs/`, which `read_strays` lists, among them files that cannot be read.
- A run record holds `identity_hash`, a hash of the problem digest, the
  solver and its parameters, the protocol, the replicate, and the seed,
  which `RunRecord.verify` checks beside the result hash. The store and
  `read_runs` also compare every record with what the plan states for its
  run, which needs no hash.
- Every session writes `logs/session-<n>.log` in its campaign directory,
  with UTC time stamps (`dispatchatlas.core.logs.SessionLog`): the engine's
  records from `INFO`, and what the other loggers of the library let
  through, in the calling process and in the worker processes. The
  execution key `log_level`, `"inherit"` by default, lowers the library's
  loggers to `"info"` or `"debug"` for a session
  (`dispatchatlas.core.logs.Lowered`). The handlers of the application
  show the same lines with and without the file. A worker hands its
  records over without waiting, the file is buffered and limited in size,
  and dropped records are counted in it.
- A solver can require a structural shape of its problems. `Problem.shapes`
  lists the shapes a problem has, named in `SHAPES`: `"job-shop"` when its
  dependencies join the tasks into disjoint chains and every task has one
  mode that holds one machine whole. `SolverInfo.shape` names the shape a
  solver requires, and `Search` and a campaign's exclusions refuse a problem
  without it as they refuse a feature the solver does not model.
  `Search.assess_built` counts and scores a schedule the solver builds
  itself, after the budget check, so a run at its evaluation limit never
  builds a schedule it cannot count, and `Decoder.repairs` counts the decodes
  that pass an order through the precedence repair. The new built-in module
  `dispatchatlas.solvers.shop` holds the job-shop solvers, and the
  bibliography gains their sources and those of iterated racing and of
  equivalence testing.
- A job-shop view of the problems with the job-shop shape, which refuses every
  feature its builders do not model; semi-active, active (Giffler and
  Thompson) and non-delay builders that decode job sequences without repair,
  with random keys sorted into a job sequence by `sequence_from_keys`; and the
  disjunctive graph of a schedule: machine orders, heads and tails, and
  critical blocks.
