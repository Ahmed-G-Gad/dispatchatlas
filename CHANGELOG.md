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
- `read_wfcommons` reads WfFormat 1.4, 1.5, and 1.6 instances onto a continuum
  platform. A problem read from a file is named after the file, without its
  extension, a parsed document after its instance name, and the instance name
  is kept in the problem's metadata. A document that contradicts itself is
  refused: a task or an execution entry listed twice, an execution entry for
  a task the specification does not list, or a file in which a JSON object
  repeats a key.
- Solvers from dispatching rules, list scheduling, and constructive
  heuristics to local search, population and swarm methods, multi-objective
  evolutionary algorithms, learning methods, hyper-heuristics, exact CP-SAT,
  MILP, and enumeration solvers, and a matheuristic, all under one contract
  that enforces stop criteria and counts evaluations exactly. Solver and
  benchmark plugins are discovered through entry points.
- Campaigns described in TOML or JSON, with paired seeds, verified fair
  comparisons, worker processes with hard time limits and retries, one record
  per run, resumption, estimates, and replay.
- Analysis of campaigns: rank tests, post-hoc tests, corrections, effect sizes
  with bootstrap intervals, Bayesian tests, power, relative deviation, and
  multi-objective indicators; tables in Markdown, CSV, and LaTeX; figures in
  SVG and TikZ; and a byte-reproducible bundle with a manifest.
- A bibliography of every implemented method, exported as text or BibTeX.
- The `dispatchatlas` command line, documentation, and runnable examples.
