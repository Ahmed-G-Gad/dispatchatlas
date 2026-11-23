# Data model

One data model describes every problem DispatchAtlas generates, reads, and
solves: machine scheduling, project scheduling, and task placement across a
computing continuum. This page defines each element and the exact meaning of
the constraints, since solvers, the feasibility checker, and the exact models
all implement the same definitions. The classes live in
[`dispatchatlas.core`](../api/core.md).

## Elements

- A **resource** (`Resource`) has a capacity: the units available at any
  instant. A machine is a resource of capacity 1; a compute node with eight
  cores is a resource of capacity 8.
- A **task** (`Task`) runs exactly once, in one of its **modes** (`Mode`). A
  mode fixes the duration and the units of each resource the task holds for
  that whole duration. Alternative machines of a flexible job shop,
  unrelated parallel machines, and the nodes a task may be placed on are
  alternative modes.
- A **dependency** (`Dependency`) makes one task wait for another, with an
  optional time lag and an optional volume of data to transfer.
- A **network** (`Network`) of **links** (`Link`) prices those transfers.
- **Setup times** (`SetupTimes`) make a machine wait between two tasks.
- A **problem** (`Problem`) bundles the tasks, resources, dependencies,
  objectives, and the optional network, setup times, and permutation
  constraint.
- A **schedule** (`Schedule`) assigns every task a mode, a start, and an end
  (`Assignment`). It records decisions only; whether they respect the problem
  is checked separately.

A problem validates its whole structure when it is built and raises
`ModelError` listing every defect: unknown ids, demands above capacity, a
dependency cycle, duplicate declarations, and the inconsistencies described
below. Every object is immutable, so code that receives a problem can rely on
it being consistent.

```python
from dispatchatlas.core import Dependency, Mode, Problem, Resource, Task

problem = Problem(
    "two-jobs-two-machines",
    tasks=(
        Task.on("j0-o0", "m0", 3.0, group="j0"),
        Task.on("j0-o1", "m1", 2.0, group="j0"),
        Task("j1-o0", (Mode(2.0, {"m0": 1.0}), Mode(4.0, {"m1": 1.0})), group="j1"),
        Task.on("j1-o1", "m0", 1.0, group="j1", deadline=9.0),
    ),
    resources=(Resource("m0"), Resource("m1")),
    dependencies=(Dependency("j0-o0", "j0-o1"), Dependency("j1-o0", "j1-o1")),
)
```

`Task.on(id, resource, duration)` builds a task with one mode that holds one
unit of one resource. Task `j1-o0` has two modes: 2 time units on `m0` or 4
on `m1`.

## Resources and capacity

At every instant, the units that the running tasks hold of a resource must not
exceed its capacity. A task holds its units from its start to its end,
excluding the end, so a task may start on a machine at the moment another
finishes.

A resource is **exclusive** in a problem when every demand on it takes its
whole capacity, as on a machine that runs one task at a time
(`Problem.exclusive_resources`). No two tasks of positive duration ever overlap
on an exclusive resource. The distinction matters for zero-duration tasks.

## Zero-duration tasks

A mode of zero duration occupies an instant:

- On an **exclusive** resource, no other task may run across that instant: a
  zero-duration operation may sit at the start or the end of another task on
  the same machine, never inside it. This is the disjunctive-graph semantics
  of shop scheduling, where an operation of zero processing time still takes
  its place in the machine's sequence.
- On any **other** resource, a zero-duration task holds nothing, as the dummy
  start and end activities of project scheduling do.
- In a **permutation** problem, it also takes its place in each machine's
  order of groups (see below). This reproduces the completion-time recurrence
  of the permutation flow shop on instances with zero processing times.
- A zero-duration mode may not use a machine that has setup times.

On a machine `m` that runs task `a` from 0 to 4, a zero-duration task `z` on
`m` may be placed at time 0 or 4, not at 2. On a two-core pool where each task
holds one core, `z` may be placed anywhere:

```python
from dispatchatlas.core import Assignment, Problem, Resource, Schedule, Task, check_schedule

machine = Problem("instant", (Task.on("a", "m", 4.0), Task.on("z", "m", 0.0)), (Resource("m"),))
inside = Schedule.of([Assignment("a", 0, 0.0, 4.0), Assignment("z", 0, 2.0, 2.0)])
print([violation.code for violation in check_schedule(machine, inside)])  # ['capacity']
```

## Dependencies, lags, and transfers

A dependency from `before` to `after` is finish-to-start:

```text
start(after) >= end(before) + lag + transfer
```

The transfer time is zero unless the problem has a network, the dependency
carries data, and the two tasks run at different locations. A mode's location
is the resource it names in `location`, by default the one resource it
demands. The network joins **endpoints**: a resource's `tier`, such as
`"edge"` or `"cloud"`, or the resource's own id when it has no tier. The link
between the endpoints of the two locations gives the transfer time
`latency + data / bandwidth`; a link from an endpoint to itself prices
transfers between two resources of the same tier. Two tasks on the same
resource exchange data instantly. The problem refuses to build when a
dependency that carries data can join two endpoints without a link.

Dependencies must form an acyclic graph. Links also carry an energy and a
monetary cost per unit of data, which the `energy` and `cost` objectives
count.

## Release times, due dates, and deadlines

- A task never starts before its `release` time.
- A `due` date is soft: finishing later is allowed, and the tardiness
  objectives measure by how much.
- A `deadline` is hard: a schedule that finishes the task later is
  infeasible. Solvers still return such a schedule when they find nothing
  better, with the status `violated` rather than `feasible`, and
  `deadline_excess` measures the total overrun.

## Jobs and groups

Tasks that share a `group` form a **job**, such as the operations of a job in
a shop or the tasks of one application. A task without a group is a job of
its own. Completion-time objectives are computed per job: a job completes when
its last task ends, and its release is the earliest release of its tasks.
Tasks of one group must share a `weight`. Due-date objectives are computed per
task, so the due date of a job belongs on its final task, as the built-in
families place it.

## Setup times

`SetupTimes` maps `(machine, previous, following)` to the idle time the
machine needs after finishing `previous` before it can start `following`;
`previous = None` gives the setup before the first task on the machine.
Missing entries mean no setup. Setup times apply to machines of capacity 1
that every task holds whole. The sequence of a machine is its tasks ordered by
start, and each task must start at least its setup time after the previous
one ends.

## Permutation problems

With `permutation=True`, every machine must process the groups (jobs) in one
common order, as in the permutation flow shop. A task without a group forms a
group of its own. A permutation problem uses capacity-1 machines that every
task holds whole, and each group uses a machine in at most one of its tasks,
so each machine's sequence is an order of groups. The checker derives the
order each machine implies between groups and reports a violation when two
machines contradict each other. Zero-duration tasks that share an instant may
appear in either order.

## Features

The structural features a problem exhibits are computed from its content
(`Problem.features`). Each solver declares the features it models
(`SolverInfo.supports`); a solver refuses a problem with a feature outside that
set by raising `UnsupportedProblemError`, and a campaign never pairs the two.

<!-- dispatchatlas:problem-features -->

## Objectives

A problem lists the names of its objectives, primary first. Single-objective
solvers optimise the primary one; multi-objective solvers optimise all of them.
Objectives are looked up by name in a registry, which plugins can extend (see
[Extending](../extending.md#objectives)).

<!-- dispatchatlas:objectives -->

The definitions, for a schedule with finish times `C`:

- `makespan`: the largest finish time.
- `total_completion_time`: the sum over jobs of the job weight times the job's
  completion time. `total_flow_time` subtracts each job's release first, and
  `mean_response_time` is the unweighted mean of completion minus release.
- `total_tardiness`: the sum over tasks with a due date of
  `weight * max(0, C - due)`. `max_lateness` is the largest `C - due`, which
  can be negative, and `tardy_tasks` counts the tasks that finish more than
  `1e-9` after their due date.
- `energy`: every resource's idle power over the makespan, plus, for every
  task, `(busy_power - idle_power) * units / capacity` times its duration on
  each resource it holds, plus the energy per unit of data of every transfer.
- `cost`: for every task, `cost_rate * units / capacity` times its duration on
  each resource it holds, plus the cost per unit of data of every transfer.

`evaluate(problem, schedule)` computes the problem's objectives for a schedule.

## Feasibility checking

`check_schedule(problem, schedule)` returns every constraint a schedule
breaks, as `Violation` records with a machine-readable `code`, a message, and
the size of the violation. It checks that every task is assigned exactly once
in a valid mode and runs for its mode's duration, release times, dependencies
with lags and transfers, resource capacities including the instants of
zero-duration tasks on exclusive resources, setup times, hard deadlines, and
the permutation constraint. Comparisons allow an absolute tolerance of
`1e-6`. `is_feasible` returns whether the list is empty.

The checker shares no code with the schedule builder the solvers use. Tests
use it as an oracle for every solver, and campaigns check every recorded
schedule with it rather than trusting the solver's status.

## How solvers compare schedules

Most solvers search over task orders and mode choices and turn each candidate
into a schedule with a shared builder, the serial schedule-generation scheme.
The builder always respects releases, dependencies, capacities, setup times,
and the permutation constraint. It does not enforce hard deadlines: a
candidate may overrun them.

Candidates are therefore compared by a lexicographic **score** of two numbers,
smaller being better:

1. the total time by which tasks overrun their hard deadlines, zero for a
   feasible schedule;
2. the primary objective, negated when it is maximised.

Any schedule that meets every deadline beats any schedule that does not, and
among schedules with equal overrun, the better objective wins. For example,
with a weight-10 task `long` of duration 5 and a task `short` of duration 1
due by the deadline 2 on one machine, minimising `total_completion_time`:

```python
from dispatchatlas.core import Problem, Resource, Task
from dispatchatlas.solvers import Evaluator

problem = Problem(
    "deadline",
    tasks=(Task.on("long", "m", 5.0, weight=10.0), Task.on("short", "m", 1.0, deadline=2.0)),
    resources=(Resource("m"),),
    objectives=("total_completion_time",),
)
evaluator = Evaluator(problem)
print(evaluator.evaluate([0, 1])[0])  # Score(violation=4.0, value=56.0)
print(evaluator.evaluate([1, 0])[0])  # Score(violation=0.0, value=61.0)
```

The first order has the better objective but overruns the deadline by 4, so
the second order has the better score. A run whose best schedule still
overruns a deadline reports the status `violated`.

## Documents and digests

Problems, schedules, and results have JSON documents, written and read by
`save_problem`, `load_problem`, `save_schedule`, `load_schedule`,
`save_result`, and `load_result`. Fields equal to their default are omitted,
so small problems stay readable:

```json
--8<-- "examples/campaigns/problem.json"
```

A schedule document lists `[task, mode, start, end]` rows. A result document
holds every field of a solver result, with its schedules as schedule documents
and its trace as rows. A result that has bounds over time (see
[Results](solvers.md#results)) also holds them under `bounds`, one
`[elapsed, value]` row for each improvement of the proven bound, such as
`[[0.07, 361.0]]`. A result without them has no `bounds` key, so the documents
of the solvers that report none are as they were. A run record of a campaign
holds them the same way (see [Campaigns](campaigns.md#storage-layout)).

`canonical_json` writes JSON with sorted keys, no spaces, and ASCII text, and
`digest` is the SHA-256 of it. `problem_digest(problem)` therefore identifies
a problem by its content: campaigns record it with every run.

## Seeds

`derive_seed(root, namespace, index)` derives independent seeds from one root
seed: the first eight bytes, read big-endian, of the SHA-256 digest of
`"dispatchatlas.seed.v1\0{root}\0{namespace}\0{index}"`, reduced modulo
`2**63`. Equal arguments give the same seed on every platform, and different
namespaces or indices give unrelated seeds, so adding a stream never shifts
another.
