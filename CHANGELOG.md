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
  after their time limit, and the runs that ended at the start-up limit. A
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
- Analysis of campaigns: rank tests, post-hoc tests, corrections, effect sizes
  with bootstrap intervals, Bayesian tests, power, relative deviation, and
  multi-objective indicators; tables in Markdown, CSV, and LaTeX; figures in
  SVG and TikZ; and a byte-reproducible bundle with a manifest.
- A bibliography of every implemented method, exported as text or BibTeX.
- The `dispatchatlas` command line, documentation, and runnable examples.
