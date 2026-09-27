"""Run a small campaign and analyse it, all through the Python API.

The campaign compares four solvers on six generated flow shops under one
protocol, a budget of 300 evaluated schedules. Stochastic solvers run three
replicates per problem; the deterministic ``neh`` runs once. Seeds are shared
across solvers, so the results are paired.

The script then:

1. runs the campaign, writing one JSON record per run (running it again
   resumes it and runs nothing new);
2. loads the records as an analysis dataset and prints the solver table and
   the pairwise comparisons, each p-value with its effect size and interval;
3. compares one solver with the others as baselines;
4. writes the analysis bundle of tables, figures, and manifest;
5. replays every run from its recorded seed and checks it reproduces.

Run it from the repository root; by default the files go to a temporary
directory that is removed at the end::

    python examples/campaign_analysis.py
    python examples/campaign_analysis.py --output results/example
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from dispatchatlas.analysis import (
    AnalysisConfig,
    compare_to_baselines,
    load_dataset,
    summarize,
    write_bundle,
)
from dispatchatlas.campaigns import (
    Campaign,
    Execution,
    FamilySource,
    Protocol,
    SolverSpec,
    Verdict,
    replay,
    run,
)
from dispatchatlas.core import StopCriteria

CAMPAIGN = Campaign(
    id="flow-shop-example",
    seed=2024,
    instances=(FamilySource("flow-shop", seeds=(1, 2, 3, 4, 5, 6), parameters={"jobs": 6}),),
    solvers=(
        SolverSpec("neh"),
        SolverSpec("iterated-greedy"),
        SolverSpec("simulated-annealing"),
        SolverSpec("random-search"),
    ),
    protocols=(Protocol("evaluations-300", StopCriteria(max_evaluations=300)),),
    replicates=3,
    execution=Execution(backend="sequential"),
)
"""The campaign. ``Execution`` only says how to run it: it never changes a result."""


def analyse(root: Path) -> None:
    """Run, analyse, and replay the campaign under ``root``."""
    summary = run(CAMPAIGN, root)
    print(f"campaign {summary.campaign}: {summary.completed} of {summary.planned} runs completed")
    print(f"records in {summary.directory}\n")

    dataset = load_dataset(summary.directory)
    config = AnalysisConfig(resamples=500)
    result = summarize(dataset, config)
    print(f"{'solver':<22} {'succeeded':>9} {'average rank':>13}")
    for row in result.solver_summaries:
        print(f"{row.solver:<22} {row.succeeded:>4}/{row.planned:<4} {row.average_rank:>13.2f}")

    print("\npairwise comparisons over the six problems (Wilcoxon signed-rank, Holm):")
    for pair in result.pairwise:
        print(
            f"  {pair.left} vs {pair.right}: win share {pair.win_share:.2f} "
            f"[{pair.win_share_low:.2f}, {pair.win_share_high:.2f}], "
            f"adjusted p {pair.adjusted_p_value:.3f}"
        )
    for limitation in result.limitations:
        print(f"  limitation: {limitation.message}")

    baselines = compare_to_baselines(dataset, "iterated-greedy", config=config)
    print("\niterated-greedy against each baseline (Holm over these comparisons only):")
    for pair in baselines.rows:
        print(
            f"  {pair.right}: {pair.wins} wins, {pair.losses} losses, {pair.ties} ties, "
            f"adjusted p {pair.adjusted_p_value:.3f}"
        )
    print(f"  better than every baseline: {baselines.better_than_all}")

    bundle = write_bundle(summary.directory, root / "analysis", config)
    print(f"\nanalysis bundle: {bundle}, start with {bundle / 'index.md'}")

    report = replay(CAMPAIGN, root)
    matched = report.count(Verdict.MATCH)
    print(f"replay: {matched} of {len(report.entries)} runs reproduced exactly")


def main() -> None:
    """Parse the output directory and run the example."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="directory to keep the results in")
    arguments = parser.parse_args()
    if arguments.output is not None:
        analyse(arguments.output)
        return
    with tempfile.TemporaryDirectory() as directory:
        analyse(Path(directory))


if __name__ == "__main__":
    main()
