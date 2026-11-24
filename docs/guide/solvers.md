# Solvers

Every solver implements one contract: it has metadata, `info`, and a method
`solve(problem, stop, seed)` that returns a `SolveResult`. The library's own
solvers and those of plugins are interchangeable everywhere. The
[solver catalogue](../catalogue/solvers.md) describes every registered solver.

## Running a solver

`solve` runs a registered solver by id, with its stop criteria and parameters
as keyword arguments:

```python
from dispatchatlas.benchmarks import generate
from dispatchatlas.solvers import solve

problem = generate("rcpsp", seed=3, activities=30)
result = solve(problem, "tabu-search", max_evaluations=5000, seed=1, tenure=10)
```

The registry separates building a solver from running it, which a script that
runs one solver many times prefers:

```python
from dispatchatlas.core import StopCriteria
from dispatchatlas.solvers import default_registry

registry = default_registry()
solver = registry.create("tabu-search", tenure=10)
result = solver.solve(problem, StopCriteria(max_evaluations=5000), seed=1)
```

`registry.create` refuses an unknown id or parameter, and a parameter value
the solver's checks reject, with a `ConfigError` that names what is known.
`registry.info(id)` returns the metadata: the family, a one-sentence summary
of the mechanism, the parameters and their defaults, whether the solver is
stochastic, iterative, or multi-objective, the problem features it models, the
shape it requires, the extras it needs, and its sources. From the shell:

```bash
dispatchatlas solvers list --family local-search
dispatchatlas solvers show tabu-search
dispatchatlas solve problem.json --solver tabu-search --set tenure=10 --max-evaluations 5000 --seed 1
```

A solver refuses a problem with a feature it does not model, such as setup
times for a solver that ignores them, or without a shape it requires (see
[Job-shop methods](#job-shop-methods)), with an `UnsupportedProblemError`
before it starts; `info.accepts(problem)` tells in advance.

## Solver families

<!-- dispatchatlas:solver-families -->

## Results

A `SolveResult` holds:

- `status`: what is known about the schedule. `optimal`: proven optimal for
  the primary objective. `feasible`: meets every constraint, optimality not
  proven. `violated`: a schedule was built but misses a hard deadline.
  `infeasible`: proven that no feasible schedule exists. `unknown`: no
  schedule found, and infeasibility not proven.
- `schedule`, and `objectives`: the value of every objective of the problem
  for that schedule. `value` is the primary objective's value.
- `stop_reason`: `completed` (the algorithm finished on its own),
  `time-limit`, `evaluation-limit`, `iteration-limit`, `target-reached`, or
  `stagnation` (no improvement for `patience` iterations).
- `elapsed`, `evaluations`, and `iterations`: the effort spent.
- `trace`: a point each time the best schedule improved, with the elapsed
  time, the evaluations and iteration so far, the value, and whether the
  schedule was feasible, plus a final point.
- `bound` and `gap`: a proven bound on the primary objective and the relative
  distance `|value - bound| / |value|`, for exact solvers.
- `bounds`: how that bound rose over the run, for exact solvers that report it:
  a `BoundPoint` with the `elapsed` seconds and the `value` of the best bound at
  each improvement, so the values never fall for a minimised objective and the
  last equals `bound`. `cp-sat` records every improvement its bound callback
  reports where the installed OR-Tools has one, and otherwise the bound it holds
  with each solution, and says which in `extras["bound_callback"]`. The tuple is
  empty for a solver that reports none.
- `front`: the non-dominated schedules of a multi-objective solver.
- `extras`: solver-specific facts, such as a backend's version and raw status.

For an exact solver, `bounds` reads as the bound over time:

```python
from dispatchatlas.benchmarks import generate
from dispatchatlas.solvers import solve

problem = generate("job-shop", seed=1, jobs=6, machines=4)
result = solve(problem, "cp-sat", time_limit=10, seed=1)  # needs the exact extra
for point in result.bounds:
    print(f"{point.elapsed:.2f} s: bound {point.value}")
assert result.bounds[-1].value == result.bound
```

## Stop criteria

`StopCriteria` holds five criteria; the first one met ends the run:

| Criterion | Stops the run |
| --- | --- |
| `time_limit` | after that many wall-clock seconds |
| `max_evaluations` | after that many evaluated schedules |
| `max_iterations` | after that many main-loop iterations |
| `target` | once a feasible schedule reaches that value of the primary objective or better |
| `patience` | after that many iterations without improvement |

A sixth field, `schedule`, ends no run: it states which clock a parameter
schedule follows, `"deterministic"` by default or `"binding"` (see
[Determinism and seeds](#determinism-and-seeds)).

**Iterative** solvers, those that run until a criterion fires, need at least
one of the three limits `time_limit`, `max_evaluations`, and `max_iterations`,
and raise `SolverError` without one. Each of them works with any single one of
the three. Solvers that finish on their own, such as the priority rules, NEH,
and the enumeration, need no criterion but honour those they are given.

What an iteration is depends on the solver, as its documentation states: one
move of a local search, one generation of a genetic algorithm, one update of a
swarm. Iteration limits are therefore not comparable across solvers, while
evaluation limits are.

## Budgets and exact evaluation counts

Order-based solvers, the large majority, run inside a `Search`, the run-time
that owns the random generator, the evaluator, the stop criteria, the best
schedule found, and the trace. Every candidate schedule a solver scores goes
through it, and counts as one **evaluation**:

- `Search.evaluate(order, modes)` decodes a task order into a schedule with
  the shared builder and scores it;
- `Search.submit(decoded)` scores a schedule a constructive heuristic built
  task by task;
- `Search.probe(order, modes)` scores a partial or tentative order, for
  example an insertion position, without making it the best schedule.

The evaluation limit is checked before each evaluation, so a run limited to
`N` evaluations scores exactly `N` schedules when the limit ends it, never
`N + 1`. The iteration limit is checked when an iteration starts, and the time
limit before every evaluation and iteration, and inside long computations that
evaluate nothing. The counts in the result are the search's own counts, and
the score of every candidate comes from the builder that produces the reported
schedule, so what a solver reports is what it evaluated.

Exact solvers count what they compute: `enumeration` counts each complete
schedule it decodes, `cp-sat` counts each improving solution the backend
reports, and `milp` counts the one schedule it reads back. Their time limit
covers building the model. See the
[exact solvers](../api/solvers-exact.md) for the guarantees of their statuses
and bounds.

The time limit is cooperative: a solver checks it itself. Campaigns add a hard
limit that kills a run that overruns (see [Campaigns](campaigns.md#execution)).

A solver counts its `elapsed` time and its time limit from the instant it
creates its clock: for an order-based solver, after it has built its
evaluator; for `cp-sat`, `milp`, and `restricted-neighbourhood-search`, after
the backend is imported. A caller that wants one origin for every solver sets
it with `clock_origin` of
[`dispatchatlas.core.clock`](../api/core.md), and the clocks created inside
the block start there. A campaign does so at the start of every attempt, so
in a campaign `elapsed` and the time limit cover everything a solver does
from that start on. What a solver does once in a process, the import of its
backend, a campaign has it do before that start, through the solver's
`warm_up` method (see [Extending](../extending.md#a-solver)).

Inside a `clock_origin` block a clock counts from the origin, not from its
own creation, and so does the time limit of a search created there. Code
that runs several searches in turn inside one block opens a block of its own
around each, `with clock_origin():`, to give each its whole limit.

## How candidates are compared

Solvers compare schedules by a lexicographic score: first the total overrun
of hard deadlines, then the primary objective in minimisation form. A schedule
that meets every deadline always beats one that does not
([details](model.md#how-solvers-compare-schedules)). The best schedule of a
run is the one with the smallest score; the result reports it with the status
`feasible`, or `violated` if even the best one overruns a deadline.

Multi-objective solvers keep, in addition, an archive of every non-dominated
feasible schedule they evaluate, over all the problem's objectives in
minimisation form, and return it as the front. Single-objective solvers
optimise the primary objective only; the result still reports every
objective's value.

## Determinism and seeds

Every random choice of a solver comes from the generators its search is
seeded with: `Search.rng`, a `random.Random(seed)`, and
`Search.numpy_rng()`, a `numpy.random.default_rng(seed)`. No solver uses global
random state. Hence:

- Equal problem, parameters, stop criteria, and seed give the same schedule,
  objective values, counts, and trace (apart from its elapsed times) on every
  run, as long as no time limit ends it. A run stopped by its time limit stops
  at a point that depends on the speed and load of the machine.
- Deterministic solvers ignore the seed.
- Solvers whose parameters follow a schedule over the run, such as a falling
  inertia weight or a shrinking population, read `Search.progress()`: the
  fraction of the binding limit spent. Which limits count is the `schedule`
  field of the stop criteria. With `"deterministic"`, the default, the time
  limit plays no part when an evaluation or iteration limit is set, so the
  schedule replays exactly, and it stops short of its end when the time limit
  ends the run first. With `"binding"`, every set limit counts: the schedule
  reaches its end at whichever limit ends the run, and under a time limit the
  run depends on the machine. With the time limit as the only limit, both
  follow the elapsed time. The swarm family counts the same limits and
  measures them from the end of its initial population (`Swarm.progress`).
  Such a solver records in the `extras` of its
  result `schedule_clock`, the limit that places the end of the run on the
  schedule (`evaluations`, `iterations`, or `time`), and `schedule_progress`,
  that place between 0 and 1.
- `cp-sat` is deterministic with its default single worker and without a time
  limit.

Draw the seeds of repeated runs from one root seed with `derive_seed`, so
that runs are independent and reproducible:

```python
from dispatchatlas.core import derive_seed

values = []
for k in range(10):
    seed = derive_seed(2024, "sa", k)
    values.append(solve(problem, "simulated-annealing", max_evaluations=5000, seed=seed).value)
```

Campaigns do this for you, and share the seeds across solvers so that their
results are paired (see [Campaigns](campaigns.md#seeds-and-paired-comparisons)).

Transcendental functions of the C library, such as `exp` and `log`, may differ
in their last bit between platforms, which can steer a stochastic search
elsewhere. Results therefore reproduce exactly on the same platform with the
same versions of Python and NumPy, and usually, but not always, elsewhere.

## Job-shop methods

Nine solvers work on job shops only, and say so in their metadata. Four are
**search families**, each written once for two representations of a schedule:
`shop-random-sampling`, `shop-simulated-annealing`, `shop-steady-state-ga`, and
`shop-particle-swarm`. Five give them context: `shop-tabu-search`,
`shop-brkga`, the dispatching rules `gt-shortest-processing-time` and
`gt-most-work-remaining`, and the insertion heuristic
`werner-winkler-insertion`.

### The job-shop shape

A problem has the **job-shop shape** when its dependencies join its tasks into
chains, the jobs, and every task has one mode that holds one machine whole.
`Problem.shapes` lists the shapes of a problem, and `SolverInfo.shape` names
the shape a solver requires. A problem without it is unsupported, as one with a
feature the solver does not model is: the solver raises
`UnsupportedProblemError` before it starts, with the missing shape among the
names it gives, and a campaign excludes the pairing and records why.
`solvers list`, `solvers show`, and the catalogue state the shape. Release
times, due dates, and groups are accepted; lags, deadlines, setup times, a
network, and permutation schedules are refused, since the builders model none
of them.

```python
from dispatchatlas.benchmarks import generate, taillard_job_shop
from dispatchatlas.solvers import default_registry

ta01 = taillard_job_shop("ta01")
print(sorted(ta01.shapes))  # ['job-shop']
print(default_registry().info("shop-tabu-search").shape)  # job-shop
print(sorted(generate("rcpsp", seed=3, activities=30).shapes))  # []
```

### Representations and builders

A **job sequence** lists every job once per operation: the `k`-th occurrence of
job `j` stands for its `k`-th operation, so every permutation is feasible and
none needs repair. **Random keys** give one number in [0, 1] per operation, and
sorting the operations by key gives a job sequence. The parameter
`representation`, `"keys"` or `"sequence"`, names what a family searches, and
both reach one **builder** through the job sequence, named by the parameter
`builder`:

- `"semi-active"`: each operation, in sequence order, starts as early as its job
  predecessor, its machine, and its release time allow; none fills an idle
  interval left earlier on its machine;
- `"active"`, the default: the procedure of Giffler and Thompson, which
  schedules, among the operations that can start before the earliest completion
  time, the one that comes first in the sequence;
- `"non-delay"`: the same procedure, scheduling at each step the operation that
  can start earliest, so no machine idles while an operation that needs it is
  ready.

No builder repairs a sequence. A run that decoded an order through the
precedence repair all the same raises `SolverError`, and every result states
`repairs`, 0, among its extras.

### Matched families

The control logic of a family is written once and runs unchanged on either
representation: for annealing, the walk, the calibration of its temperatures,
the cooling schedule, and the acceptance test. The two arms differ in the
genotype and its operators only:

- A random solution is one uniform number per operation, kept as the keys or
  sorted into a job sequence, so both arms draw the same job sequence.
- Every move changes the genotype. One key redrawn answers an insert
  (`key_move="redraw"` and `sequence_move="insert"`, the defaults), and two keys
  perturbed by the step `key_step` answer a swap (`"perturb"` and `"swap"`);
  `calibrate_step` of `dispatchatlas.solvers.shop.distance` sets the step at
  which the median distance of the keys' move equals the swap's.
- Recombination takes each key, or each position of the child, from the fitter
  parent with probability `inheritance`: the parameterised uniform crossover
  for keys, and for job sequences the precedence preservative crossover
  (`sequence_crossover="ppx"`) or the job-based order crossover (`"jox"`).

The particle swarm is a weaker match. A velocity has no natural form for a
permutation, so the arms share the swarm's size, topology, memory of bests, and
stop, while keys move by a constriction swarm and job sequences by the discrete
swarm of Pan, Tasgetiren, and Liang (2007).

### Common random numbers

A family draws from two generators that `derive_seed` derives from the seed of
the run, each created once. The **control stream** serves the control logic:
selection, replacement, acceptance, and the ties they break. The **operator
stream** serves the operators: random solutions, moves, and recombination. Both
depend on the seed alone, not on the representation or the builder, and no
control draw depends on an outcome: annealing draws its acceptance number at
every step, not only for a worse move. Two runs with one seed therefore read the
same control numbers at the same step, and the first draw of the operator
stream is a random solution, so they start from the same schedule. Annealing
calibrates its temperatures on random solutions, which count toward the
evaluation limit, the same number in both arms. A run records
`first_schedule_digest` among its extras, the digest of the first schedule it
evaluated, so that two strings tell whether two arms started alike:

```python
from dispatchatlas.benchmarks import taillard_job_shop
from dispatchatlas.solvers import solve

ta01 = taillard_job_shop("ta01")
keys, sequence = (
    solve(ta01, "shop-random-sampling", max_evaluations=300, seed=7, representation=name)
    for name in ("keys", "sequence")
)
assert keys.extras["first_schedule_digest"] == sequence.extras["first_schedule_digest"]
assert keys.schedule == sequence.schedule  # random sampling has no control logic
assert keys.evaluations == sequence.evaluations == 300
```

A campaign gives every solver of a problem and replicate one seed (see
[Campaigns](campaigns.md#seeds-and-paired-comparisons)), so the arms of a
family, entered as two solvers of one id with a label each, are paired runs
without another setting, and the records of the two arms state one
`first_schedule_digest` in the `extras` of their outcomes, for every problem and
replicate.

### Observers

A family reports every evaluation outside its calibration to an **observer**.
The `TrajectoryObserver` reads schedules only: it draws no random number and
evaluates nothing, and the run times it apart from the decoding, so a run with
it makes the same decisions as one without. It adds to the extras of the result:

- `revisit_share`, the share of evaluations that rebuild a schedule evaluated
  before, and `unchanged_move_share`, the share of moves whose schedule is
  their parent's;
- `phenotype_distance_mean` and `phenotype_distance_median`, the Kendall tau
  distance between the machine orders of a move and of its parent, which counts
  the pairs of operations on one machine that the two schedules process in
  different orders;
- the counts these are taken over, `observed_evaluations`, `observed_moves`, and
  `measured_moves`, and the time the observer took, `observer_seconds`.

The distance takes time linear in the pairs of operations on a machine, so the
parameter `distance_every`, 1 by default, measures it on the first move and on
every `n`-th move after it, while the two shares stay per evaluation. It changes
no decision of the search. Random sampling, which makes no move, accepts it so
that the four families take the same settings.

```python
result = solve(ta01, "shop-simulated-annealing", max_evaluations=500, seed=7, distance_every=10)
print(result.extras["observed_moves"], result.extras["measured_moves"])
```

`shop-steady-state-ga` with `local_search=True` is a Baldwinian memetic
algorithm, a matched pair like the plain one. Every evaluated genotype is
decoded, and the schedule is improved by `critical_block_descent`, a
best-improvement descent over the N5 swaps of a critical path. The descent works
on the schedule, never on the genotype, so it is the same in both
representations. The individual takes the improved schedule's score as its
fitness and keeps its genotype unchanged. Every neighbour the descent evaluates
exactly counts as one evaluation, so an evaluation limit bounds the decodes and
the local search together. The extras add `descent_evaluations` and
`descent_moves`. The option is off by default, and a run without it is the
plain algorithm.

### Context methods

Five solvers stand beside the families, as context for what the families
reach. None is a matched pair.

- `shop-tabu-search` searches the machine orders of a schedule. It applies the
  best admissible move of the blocks of a critical path, the swaps at their
  ends (`neighbourhood="n5"`) or those and insertions (`"n7"`), even when the
  makespan worsens, forbids the reversal of recent arcs, and restarts from the
  best orders after a stagnation. Only an applied move counts as an evaluation,
  so an evaluation limit does not bound the work between two evaluations: set a
  time limit too.
- `shop-brkga` is the biased random-key genetic algorithm. It decodes each
  chromosome with a builder, improves the schedule by `critical_block_descent`,
  and with `write_back` rearranges the keys to sort into the improved schedule.
  Every decode and every swap of the descent counts as one evaluation.
- `gt-shortest-processing-time` and `gt-most-work-remaining` are dispatching
  rules run by the Giffler and Thompson procedure from one rank per operation,
  since a job sequence cannot rank an operation ahead of its predecessor. Each
  builds one schedule, evaluates it once, and ignores the seed.
- `werner-winkler-insertion` is the basic insertion heuristic of Werner and
  Winkler (1995), without beam search or the iterative phase.

```python
rule = solve(ta01, "gt-most-work-remaining")
print(rule.evaluations, rule.stop_reason.value)  # 1 completed
```

## Writing a solver

A new solver is a dataclass with `info` and `solve`. [Extending](../extending.md)
walks through one, and the
[contributing guide](https://github.com/ahmed-g-gad/dispatchatlas/blob/main/CONTRIBUTING.md#the-solver-contract)
lists the contract every built-in solver follows.
