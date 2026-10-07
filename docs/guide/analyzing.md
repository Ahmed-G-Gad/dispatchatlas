# Analysis

The analysis reads a campaign directory and compares its solvers over its
problems with the non-parametric protocol for comparing algorithms (Demšar
2006; Derrac et al. 2011). It writes every table, figure, and caveat to one
bundle, and records its settings so that the bundle can be produced again
from them.

```bash
dispatchatlas analyze results/quickstart --output results/quickstart-analysis
```

```python
from dispatchatlas.analysis import AnalysisConfig, load_dataset, summarize, write_bundle

dataset = load_dataset("results/quickstart")
summary = summarize(dataset, AnalysisConfig(correction="holm"))
write_bundle("results/quickstart", "results/quickstart-analysis")
```

## Runs, failures, and denominators

`load_dataset` returns one run for every run of the plan, in plan order,
whether or not it completed. Each has one of four statuses:

- `succeeded`: the run completed with a feasible schedule;
- `infeasible`: it completed, but its schedule breaks a constraint, such as a
  hard deadline, or the solver returned none;
- `failed`: every attempt of the run failed;
- `missing`: the run has no record yet.

The last three count as failures and are never dropped. Success rates divide
by the planned runs, and value statistics score a failure as the worst
possible value, infinity for a minimised objective. Denominators are fixed by
the plan, so a solver cannot look better by failing on hard problems.

A failed run whose last attempt was killed at its hard limit may hold the
incumbent its search had reported when its time limit passed (see
[What a killed run keeps](campaigns.md#execution)). The setting
`timeout_score` chooses its score. With `"worst"`, the default, it scores as
the worst value, like any other failure. With `"incumbent"`, a killed run
that held a feasible incumbent at its time limit scores that incumbent's
value in the cell scores, and so in the ranks, the omnibus and pairwise
tests, and the comparison with baselines. Only that incumbent is scored
(`limit_incumbent` in the failure record): what a search found after the
time limit, in the seconds before the kill, is never a score, so a solver
that ignores its time limit is compared on the same seconds as one that
stops on time. The run stays a failed run: the success rates, the
descriptive statistics of a cell, the deviation table, the indicators, and
the figures count it as before. The value is the solver's own, with no
schedule to check it against. Under either choice the limitations state how
many runs were killed, how many of them held a feasible incumbent when their
time limit passed, and how many reported an incumbent after it. `runs.csv`
names the last failure of every failed run (`failure`) and what every run is
scored by (`scored_by`: `result`, `incumbent`, or `worst`); the row of a run
scored by its incumbent holds that incumbent's objective values, and in
`evaluations` and `elapsed` when its search found it.

A failed run whose last attempt ended at the start-up limit says nothing
about its solver: its worker did not begin the solve in time. It counts as
the worst value like any other failure, and a limitation of its own
(`startup`) states how many runs it concerns. Running the campaign again
with `--retry-failed` gives them another attempt.

The `failures` limitation counts the failed runs of each solver by the kind
of error of their last attempt, and states what each kind that occurs says
about who ended the attempt:

| Kind | Meaning |
| --- | --- |
| `exception` | the solver, its warm-up, or the checks of its result raised an exception |
| `timeout` | the engine killed the worker at the hard limit of the run |
| `startup` | the engine killed a worker that had not begun the solve at the start-up limit |
| `worker-exit` | the worker process ended during the run, as after a crash in the solver's native code or when memory runs out; the engine does not know the cause |
| `transfer` | the run could not be sent to a worker or rebuilt in it, as when its solver cannot be pickled |

Only `timeout` and `startup` are the engine's doing. A `worker-exit` is as
likely the solver's as the machine's, and the traceback or the message of
the failure record is where to look.

### Realised time against the budget

A time limit is cooperative: a solver checks it between evaluations, a
heuristic that builds one schedule checks none, and a run that ends late but
before its hard limit is recorded as an ordinary success (see
[What the time limit cannot stop](campaigns.md#execution)). The analysis
therefore compares the wall time the engine measured for every run with the
time limit of its protocol. `runs.csv` holds, for every run:

| Column | Meaning |
| --- | --- |
| `time_limit` | the time limit of the run's protocol, empty without one |
| `wall_seconds` | the wall time the engine measured for the last attempt of the run, the completing one or the last failed one; `elapsed` is the solver's own report |
| `overrun_seconds` | `wall_seconds` minus `time_limit`, negative for a run that ended early; empty for a run that did not complete |
| `cpu_share` | the processor seconds of the completing attempt over its wall seconds: below 1 when the executing process was not computing for part of the run, above 1 when the solver ran threads; empty for a run that did not complete |
| `concurrent` | the runs executing when the attempt started, itself included |
| `cpu` | the processor the attempt's worker was pinned to, empty when it was not pinned |
| `attempts` | the number of the last attempt of the run |

For a run that did not complete, `cpu_seconds`, `wall_seconds`,
`concurrent`, `cpu`, and `attempts` all describe its last failed attempt,
whose kind is in the `failure` column.

A `cpu_share` below 1 does not say why the process was not computing: it
waited for a processor, it slept, or it left the work to another process
whose time the record does not hold. The processor seconds are those of the
worker process and, on POSIX systems, of the child processes it waited for;
on Windows a solver whose backend runs as a program of its own, as `milp`
does with CBC, shows a share near 0 while that program computes.

Three limitations summarise the timing of the analysed protocol when it
sets a time limit. `budget` names every solver whose completed runs took
more wall time than `time_limit + max(0.05, overrun_tolerance * time_limit)`
seconds, with how many of its completed runs did and the largest ratio of
wall time to limit. The rule is the one of the engine's note on a run, and
its 0.05 seconds are the margin within which the engine's clock cannot tell
a run from its limit. Such a
run searched for longer than the protocol states; it is neither excluded nor
rescored. `timing` states the load that the completed runs which lasted at
least half of the time limit executed under: the range and median of their
processor share, the lowest and highest median share of a solver, and the
range and median of the number of runs executing at once. When the time
limit is the protocol's only bound, every run has the same wall-clock time
and searches as far as the processor time it received allows, so solvers
whose shares differ for want of a processor were not given the same budget;
when the protocol also bounds evaluations or iterations, that holds for the
runs the time limit stopped. Shorter runs are left out: the clock did not
bound them, and processor time advances in steps, of about 16 milliseconds
on Windows, so the `cpu_share` of a run shorter than a few tenths of a
second is coarse. `clock` appears when only some of the completed runs
state, in their provenance, that their solver's clock started at the start
of the attempt (`shared_clock`): a solver whose clock starts after its
set-up has the whole time limit for its search, so the time limit and the
elapsed times of the two groups may not measure the same thing. A directory
whose records all state it, or none, has no such limitation.

### The plan's notes and the stop reasons

Two more limitations come from the fairness assessment of the campaign (see
[Fair comparison](campaigns.md#fair-comparison)). Each `plan` limitation is a
note of the plan on the analysed protocol and solvers: solvers whose `threads`
differ under a time limit, and iterative solvers declared deterministic that
run once under a time limit although the campaign has more replicates. The
result of such a run depends on the machine wherever the time limit ends it,
and its single run cannot show the spread that the replicates of the other
solvers show; `replicate_timed = true` in the campaign file gives it every
replicate. A solver declared deterministic that is not iterative, such as a
constructive rule or an exact solver, is reported the same way only when the
time limit ended one of its completed runs, with its runs counted by stop
reason, for example `milp 3 completed, 2 time-limit`. A constructive rule,
which the clock cannot end, is never reported.

`stop-reasons` appears when the completed runs of the iterative solvers ended
for more than one reason under the analysed protocol, and counts each
solver's runs by reason, for example `hill-climbing 14 stagnation, 6
time-limit; random-search 20 time-limit`. A run that stopped for lack of
improvement did not use the time or the evaluations that a run stopped at its
limit used, so the protocol did not bound the two alike. The patience of a
protocol counts iterations, which each solver defines, and exact solvers
ignore it. The runs are compared as they are. The `budget` counts and these
come from one function,
`dispatchatlas.campaigns.fairness.assess_outcomes(plan, records)`, which
returns them for every protocol of a campaign.

## The statistical comparison

`summarize` analyses one protocol and one objective, the primary objective
unless the settings name another:

1. **Cells.** The replicates of each solver on each problem become one score,
   the mean (or median) of the planned runs, a failure counting as the worst
   value. Descriptive statistics and a bootstrap interval of the mean describe
   the succeeded runs.
2. **Reliability.** Each solver's succeeded, infeasible, failed, and missing
   runs, over its planned runs.
3. **Ranks and omnibus test.** On the problems every solver is planned on,
   the scores are ranked per problem, rank 1 best, ties sharing their mean
   rank. With at least three solvers and two problems, the Friedman test and
   its Iman-Davenport form test whether the solvers differ, and Kendall's W,
   with a bootstrap interval, measures by how much.
4. **Post-hoc test.** When the omnibus test rejects, the Nemenyi critical
   difference tells which average ranks differ.
5. **Pairwise tests.** Every pair of solvers is compared over the problems
   both are planned on with the Wilcoxon signed-rank test, and the p-values are
   adjusted over the family with the chosen correction. Beside every p-value
   stand the paired win share with a bootstrap interval, the mean advantage
   with a bootstrap interval, and Cohen's d_z (Carrasco et al. 2020).
6. **Limitations.** What qualifies the results: failed or missing runs,
   with the failed ones by solver and kind of error,
   runs killed at their hard limit and how they are scored,
   what the plan notes on the protocol, stop reasons that differ among the
   runs of the iterative solvers,
   completed runs whose wall time passed their time limit, the processor
   share and the concurrency the runs of a timed protocol executed under,
   problems left out of the ranks, too few shared problems for a paired test
   to reach significance, cells below a replicate floor, and tests that could
   not run.

The pairwise tests have one pair per problem, so their sample size is the
number of problems, not the number of runs. With `n` pairs the smallest
two-sided exact p-value of the Wilcoxon test is `2 / 2^n`: fewer than six
problems can never reach `p <= 0.05`, whatever the difference, and the
limitations say so. Up to 25 pairs the p-value is exact, also when
differences tie in magnitude, as integer objective values often make them:
the tied differences share their mean rank, and the p-value comes from the
distribution of the rank sum over every assignment of signs to those ranks.
The pairwise table says under *Null distribution*, the column `method` of its
CSV form, how each p-value was obtained:
`exact`, `exact-conditional` for the exact value under ties, `normal` for the
normal approximation above 25 pairs, and `degenerate` when no difference is
other than zero. The manifest of a bundle also states, for each protocol,
the smallest standardized effect the paired tests detect at the configured
power (Campelo and Takahashi 2019).

`compare_to_baselines(dataset, focal)` answers the question a new method
raises: is it better than each established one? It compares the focal solver
with each baseline by the same paired test and adjusts the p-values over those
comparisons only, the many-to-one family (Demšar 2006).

### Reference values and quality indicators

With reference values for the problems, such as best-known or proven optimal
values, the analysis reports each solver's **relative percentage deviation**,
`100 (value - reference) / reference` for a minimised objective, averaged over
its succeeded runs (ARPD; Vallada, Ruiz and Framinan 2015).

On problems with two or more objectives, each run's front is scored by the
**hypervolume** it dominates up to a reference point shared by every solver
on the problem, computed exactly, and by the **IGD** and **IGD+** distances to
the non-dominated union of all runs. Objectives are normalised to the ideal
and nadir of that union by default. A run without a feasible result has
hypervolume 0 and infinite distances. A reference point derived from the data
moves when solvers are added or removed, so hypervolumes are comparable only
between results that share their reference points; set `reference_points` to
fix them.

### Methods

Every statistical method the analysis implements, with its sources. The
bundle's `index.md` lists the ones an analysis used.

<!-- dispatchatlas:methods -->

## Settings

`AnalysisConfig` holds every choice that changes a result. Every field has a
default:

| Setting | Default | Meaning |
| --- | --- | --- |
| `protocol` | every protocol | the protocol to analyse |
| `objective` | the primary objective | the objective to compare on, which the analysed problems must share |
| `sense` | the objective's | `"min"` or `"max"` |
| `solvers`, `problems` | all | the solver names and problem ids to analyse |
| `alpha` | `0.05` | significance level of the tests and corrections |
| `confidence` | `0.95` | coverage of the confidence intervals |
| `resamples` | `1000` | bootstrap resamples per interval |
| `seed` | `0` | seed of the bootstrap and of the power simulation |
| `interval` | `"bca"` | bootstrap interval: `"bca"` or `"percentile"` |
| `correction` | `"holm"` | `"none"`, `"bonferroni"`, `"holm"`, `"hochberg"`, `"shaffer"`, or `"finner"` |
| `omnibus` | `"iman-davenport"` | the test that gates the Nemenyi test, or `"friedman"` |
| `statistic` | `"mean"` | how replicates become one score; with `"median"`, failures decide the cell only when they reach half of the runs |
| `timeout_score` | `"worst"` | how a run killed at its hard limit is scored: as the worst value, or with `"incumbent"` by the feasible incumbent it held when its time limit passed |
| `overrun_tolerance` | `0.05` | share of the time limit by which the wall time of a completed run may pass the limit before the `budget` limitation names it, beside the engine's clock margin of 0.05 seconds, which always applies; nothing is excluded |
| `replicate_floor` | none | fewest replicates per cell of a stochastic solver; smaller cells are reported, not excluded |
| `power` | `0.8` | target power of the detectable-effect statement |
| `references` | none | a reference value per problem, for the relative deviation |
| `reference_source` | none | where the reference values come from, shown with the table |
| `reference_points` | none | a hypervolume reference point per problem, in minimisation form |
| `reference_rule` | `"margin"` | derive the point a margin beyond the nadir, or by the rule of Ishibuchi et al. (2018) with `"ishibuchi"` |
| `reference_parameter` | `0.1` | the margin, or `h` of the Ishibuchi rule |
| `normalize` | `true` | compute indicators on objectives scaled to the reference front |
| `focal`, `baselines` | none | a solver under study, and its baselines (default: every other solver) |

On the command line the settings come from `--config FILE`, a TOML or JSON
file of these fields, overridden by the options. The quickstart campaign's
flow shops have optimal makespans of 591, 554, and 581, proven by the
`enumeration` solver, so they make exact references:

```toml
correction = "holm"
resamples = 2000
replicate_floor = 5
reference_source = "optimal makespans proven by the enumeration solver"

[references]
flow-shop-8x4-1 = 591.0
flow-shop-8x4-2 = 554.0
flow-shop-8x4-3 = 581.0
```

```bash
dispatchatlas analyze results/quickstart --output results/rpd --config settings.toml --focal simulated-annealing
```

The bundle then also holds the deviation table and the table of the focal
solver against each baseline. A bundle's manifest can serve as `--config`:
it repeats the analysis with the same settings.

## Tables

Tables are written in Markdown and LaTeX, with six significant digits (three
for p-values), and in CSV at full precision, an interval as two columns. Every
renderer refuses a table that reports a p-value without an effect size and a
confidence interval beside it, as Carrasco et al. (2020) recommend. A bundle
holds, per protocol:

| Table | Content |
| --- | --- |
| `solvers` | runs by status, success rate, and average rank of each solver |
| `cells` | score, mean with interval, median, standard deviation, and best of each solver on each problem |
| `pairwise` | the pairwise comparisons |
| `omnibus` | the Friedman and Iman-Davenport tests with Kendall's W, and the Nemenyi critical difference when the post-hoc test runs |
| `indicators` | hypervolume, IGD, and IGD+, when a problem has several objectives |
| `deviation` | relative percentage deviations, when reference values are given |
| `baselines` | the focal solver against each baseline, when one is named |

## Figures

Figures are drawn without a plotting library, in SVG and in TikZ from one
layout, so both formats agree to the coordinate. They read the same runs,
objective, and settings as the tables, and keep failures in every
denominator. The series palette stays distinguishable under the common forms
of colour-vision deficiency, and every series also gets a marker shape.

<!-- dispatchatlas:figures -->

`render_figures(dataset, directory)` writes every figure that applies, and
each family's functions compute its data and draw it separately (see the
[figures API](../api/analysis-figures.md)).

## The bundle

```text
<output>/
    index.md                  what the bundle holds: tables, figures, methods, and sources
    limitations.md            the caveats of every analysis
    runs.csv                  one row per analysed run, at full precision
    tables/<protocol>/<table>.md, .csv, .tex
    figures/<protocol>/<figure>.svg, .tex
    manifest.json             settings, versions, environment, and file digests
```

The manifest records the campaign's id, config hash, seed, replicates, the
digest of the run ids and seeds of the analysed plan (`runs_digest`), and
run counts by status, and, when the campaign sets them, its `seed_namespace`,
`replicate_timed`, and the solvers' own replicate counts
(`solver_replicates`); the full settings; the DispatchAtlas version and the
environment; for each protocol, the replicates behind each score and the
problems behind each paired test, with the effect those tests detect; and
the SHA-256 digest of every other file. It holds no timestamp. The bundle is
byte-reproducible: equal inputs, settings, and software give identical files.
