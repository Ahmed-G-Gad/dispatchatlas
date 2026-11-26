# Online scheduling

A static solver sees every task at once and has as long as its budget
allows. An online scheduler sees each workflow only when it arrives, cannot
move a task that has started, loses the work running on a node that fails,
and pays for its own computation in the time it schedules.
`dispatchatlas.online` runs that loop as a deterministic discrete-event
simulation and keeps every raw record of it.

```python
from dispatchatlas.benchmarks import generate
from dispatchatlas.online import (
    AlternatingRenewal,
    Budget,
    Exponential,
    OnlineEft,
    PlanningClock,
    WarmRestart,
    availability_events,
    check_run,
    online_problem,
    poisson_arrivals,
    rate_for_load,
    run_online,
    workflows_from_problem,
    write_records,
)

static = generate("continuum-workflow", seed=3, applications=8, tasks=5)
workflows = [workflow for workflow, _ in workflows_from_problem(static)]
arrivals = poisson_arrivals(workflows, rate_for_load(workflows, static.resources, 0.7), seed=1)
span = arrivals.events[-1].time
nodes = [node.id for node in static.resources if node.tier != "device"]
outages = AlternatingRenewal(Exponential(span), Exponential(span / 5))
trace = arrivals.merged(availability_events(nodes, outages, seed=1, horizon=10 * span))

clock = PlanningClock("counted", rate=1e4)  # 10,000 placements per second
runs = [
    run_online(static, trace, OnlineEft(), clock=clock, seed=1),
    run_online(static, trace, WarmRestart("tabu-search"), clock=clock, budget=Budget(2000), seed=1),
]
for run in runs:
    assert check_run(online_problem(static, trace), trace, run) == ()
    print(run.config["planner"]["planner"], run.summary["mean_response"])
write_records("results/loop.jsonl.gz", runs, {"trace": trace.digest})
```

## Traces

A `Trace` is the sorted, immutable list of a run's events: `Arrival` of a
`Workflow` (a graph of tasks whose modes name the platform's nodes, with
release, due, and deadline offsets from the arrival), `Failure` of a node,
and `Repair`. At one instant repairs come before failures and failures
before arrivals. A trace is built before any method runs, so every method on
it meets the same events, and its `digest`, the SHA-256 digest of its
canonical JSON, identifies it in the records. `trace.save(path)` and
`Trace.load(path)` keep it beside the results.

The generators draw from streams derived from one seed, one per purpose, so
the arrivals do not change with the availability model and one node's
outages do not change with the others:

- `poisson_arrivals(workflows, rate, seed)`, with `rate_for_load` giving the
  rate of an offered load: the rate times the mean work of the workflows
  (each task on its cheapest mode) over the capacity of the nodes they use;
- `submitted_arrivals(workflows, times)` for recorded submit times, and
  `workflows_from_problem` to cut a static problem, such as one read from
  WfCommons or the Alibaba trace, into workflows and submit times;
- `availability_events(nodes, model, seed, horizon)` with an
  `AlternatingRenewal` of time-to-failure and time-to-repair distributions,
  `Exponential`, `Weibull` and `LogNormal` (the parameters of the underlying
  normal, as the Failure Trace Archive fits them) provided; another published
  model plugs in as any object with `sample` and `describe`.

## The rules of the loop

A task **commits** when it starts. Its node, start, and finish are then a
reservation nothing moves; a failure of its node ends it early, the attempt
is **lost**, and the task restarts from scratch. Finished work is never
lost.

One scheduler thread does two kinds of work, both charged in simulated time:

- a **pass** calls the planner at each decision point, which is every event
  or set of events at one instant. Its plan is adopted only when its charged
  time has elapsed; the runner then places the plan's tasks, in its order and
  on its nodes, on the committed schedule;
- an **answer** decides one ready task by the earliest-finish rule, at one
  placement per node that can run it.

The runner's rules, each stated in `dispatchatlas.online.runner`:

1. **Freshness**: an event withdraws the plan in force, so no task starts on
   a plan that a pass began before the last event.
2. **Answering comes first**: while no fresh plan is in force, ready tasks
   are answered as they fall due, before the pass in progress, which finishes
   that much later. Without a planner (`planner=None`) every task is decided
   this way.
3. **Every uncommitted task is planned again** at each event.
4. **Overtaking**: an event before a pass is adopted, or at the same instant,
   overtakes it; it adopts nothing.
5. **Overrun**: a pass that exceeds its `Budget` loses its window and holds
   the scheduler for the whole budget.

## Planners and the clock

A planner has a `name`, a `describe()` for the records, and a
`plan(context)` that returns a `Plan`, an order of `context.tasks` and a mode
for each (a negative mode lets the builder choose the earliest-finishing
one), or `None` to leave the tasks to the dispatcher. The `PlanningContext`
gives the state, the allowed modes, the plan in force before the event, a
seeded random generator, and three ways to spend placements: `seed_plan()`
(the greedy plan), `evaluate(plan)`, and `charge(n)` for work done outside,
such as a static solver's. A `BudgetExhausted` exception refuses what the
budget cannot pay; `context.remaining` tells what is left.

The `PlanningClock` turns placements into time. `counted` charges a pass
`placements / rate` seconds, so runs reproduce exactly on any machine and in
unit tests; `measured` charges the planner's wall time, for runs on the
machine the results describe. An infinite rate makes planning free.

`OnlineEft` is the greedy reference: at every event it ranks every task again
in the order the tasks fall due, each on the node that finishes it earliest.
A planner that seeds with the greedy plan and never searches commits exactly
where it does; on a trace of arrivals with free planning, both commit where
the dispatcher alone would. `WarmRestart(solver)` re-runs a registered static
solver at every event on the residual problem, the uncommitted tasks with the
running tasks and finished predecessors frozen, and adopts its plan only
when it beats the better of the greedy plan and the plan in force.

## Records and recovery

`run_online` returns an `OnlineResult`: the configuration, the seed, the
digests of the trace and of the run's problem, every event, every pass with
its placements, charge, outcome, estimate before and after, and its anytime
curve, every answer, every attempt with its outcome, every workflow's
response time, the forecasts at each decision point, the recovery after each
event, and a summary. `write_records` and `RecordWriter` write one
gzip-compressed JSON-lines file per job, a header line then one line per run,
atomically; `read_records` refuses a torn line.

Recovery is measured on the forecasts: for each event, the cohort of
workflows open just before it, the time until the cohort's mean planned
response returns within a tolerance of its value just before the event, and
the excess response integrated over that time. `dispatchatlas.online.metrics`
defines both precisely; `recovery(run, tolerance)` recomputes them for any
tolerance from the raw records.
