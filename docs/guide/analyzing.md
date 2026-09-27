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
   problems left out of the ranks, too few shared problems for a paired test
   to reach significance, cells below a replicate floor, and tests that could
   not run.

The pairwise tests have one pair per problem, so their sample size is the
number of problems, not the number of runs. With `n` pairs the smallest
two-sided exact p-value of the Wilcoxon test is `2 / 2^n`: fewer than six
problems can never reach `p <= 0.05`, whatever the difference, and the
limitations say so. The manifest of a bundle also states, for each protocol,
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

The manifest records the campaign's id, config hash, seed, replicates, and
run counts by status; the full settings; the DispatchAtlas version and the
environment; for each protocol, the replicates behind each score and the
problems behind each paired test, with the effect those tests detect; and
the SHA-256 digest of every other file. It holds no timestamp. The bundle is
byte-reproducible: equal inputs, settings, and software give identical files.
