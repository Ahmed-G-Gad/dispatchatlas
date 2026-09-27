"""The summary pipeline, on a real tiny campaign and on hand-computed synthetic runs."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.dataset import load_dataset
from dispatchatlas.analysis.stats.nonparametric import friedman_test, wilcoxon_signed_rank
from dispatchatlas.analysis.summary import summarize
from dispatchatlas.core.errors import ConfigError
from tests.analysis.analysis_data import Entry, campaign_dir, make_dataset

__all__ = ["campaign_dir"]

type Cells = dict[str, dict[str, list[Entry]]]

# Every problem ranks a < b < c, with distinct gaps so the exact null applies:
# b - a = 1 + 0.1 i, c - a = 3 + 0.3 i, and c - b = 2 + 0.2 i on problem i.
ORDERED = {
    "a": {f"p{i}": [10.0 + i] for i in range(8)},
    "b": {f"p{i}": [11.0 + 1.1 * i] for i in range(8)},
    "c": {f"p{i}": [13.0 + 1.3 * i] for i in range(8)},
}
MEAN_GAPS = {("a", "b"): 1.35, ("a", "c"): 4.05, ("b", "c"): 2.7}


def test_summary_of_a_real_campaign(campaign_dir: Path) -> None:
    dataset = load_dataset(campaign_dir)
    summary = summarize(dataset)
    assert (summary.protocol, summary.objective, summary.sense) == ("evals", "makespan", "min")
    assert len(summary.cells) == 4 * 7
    assert {cell.planned for cell in summary.cells if cell.solver in {"hc", "random-search"}} == {3}
    assert {cell.planned for cell in summary.cells if cell.solver == "input-order"} == {1}
    assert all(row.success_rate == 1.0 for row in summary.solver_summaries)
    ranks = [row.average_rank for row in summary.solver_summaries]
    assert sum(rank for rank in ranks if rank is not None) == pytest.approx(10.0)
    assert len(summary.pairwise) == 6
    assert summary.omnibus is not None
    assert summary.blocks == summary.problems
    scores = {(cell.solver, cell.problem): cell.score for cell in summary.cells}
    blocks = [{solver: scores[(solver, p)] for solver in summary.solvers} for p in summary.blocks]
    assert summary.omnibus.friedman_statistic == pytest.approx(friedman_test(blocks).statistic)
    for row in summary.pairwise:
        assert row.wins + row.losses + row.ties == row.pairs == 7
        assert row.win_share == pytest.approx((row.wins + 0.5 * row.ties) / row.pairs)
        assert row.win_share_low <= row.win_share <= row.win_share_high
        assert row.p_value <= row.adjusted_p_value <= 1.0
        advantages = [scores[(row.right, p)] - scores[(row.left, p)] for p in summary.problems]
        assert row.p_value == pytest.approx(wilcoxon_signed_rank(advantages).p_value)
        assert row.mean_advantage == pytest.approx(sum(advantages) / len(advantages))
    assert summary.limitations == ()


def test_a_consistent_ordering_is_significant_with_hand_computed_values() -> None:
    summary = summarize(make_dataset(ORDERED, deterministic="abc"))
    omnibus = summary.omnibus
    assert omnibus is not None
    # Q = 12 / (8 * 3 * 4) * (8^2 + 16^2 + 24^2) - 3 * 8 * 4 = 16.
    assert omnibus.friedman_statistic == pytest.approx(16.0)
    assert omnibus.friedman_p_value == pytest.approx(math.exp(-8.0))
    assert (omnibus.iman_davenport_statistic, omnibus.iman_davenport_p_value) == (math.inf, 0.0)
    assert (omnibus.kendalls_w, omnibus.kendalls_w_low, omnibus.kendalls_w_high) == (1.0, 1.0, 1.0)
    assert omnibus.significant
    critical = summary.critical_difference
    assert critical is not None
    assert critical.value == pytest.approx(2.343 * 0.5)
    assert critical.average_ranks == (("a", 1.0), ("b", 2.0), ("c", 3.0))
    assert critical.significant_pairs == (("a", "c"),)
    assert critical.cliques == (("a", "b"), ("b", "c"))
    for row in summary.pairwise:
        assert (row.wins, row.losses) == (8, 0)
        assert row.p_value == pytest.approx(2 / 256)
        assert row.adjusted_p_value == pytest.approx(3 * 2 / 256)
        assert row.win_share == 1.0
        assert row.mean_advantage == pytest.approx(MEAN_GAPS[(row.left, row.right)])
    assert [row.average_rank for row in summary.solver_summaries] == [1.0, 2.0, 3.0]


def test_failures_count_as_the_worst_value_with_fixed_denominators() -> None:
    dataset = make_dataset(
        {
            "a": {"p1": [1.0, 2.0], "p2": ["failed", 3.0], "p3": [1.0, 1.0]},
            "b": {"p1": [2.0, 3.0], "p2": [4.0, 4.0], "p3": [2.0, "infeasible"]},
        }
    )
    summary = summarize(dataset)
    a, b = summary.solver_summaries
    assert (a.planned, a.succeeded, a.failed) == (6, 5, 1)
    assert a.success_rate == pytest.approx(5 / 6)
    assert (b.succeeded, b.infeasible) == (5, 1)
    cell = next(c for c in summary.cells if (c.solver, c.problem) == ("a", "p2"))
    assert (cell.planned, cell.succeeded, cell.score, cell.mean) == (2, 1, math.inf, 3.0)
    (row,) = summary.pairwise
    assert (row.wins, row.losses, row.ties) == (2, 1, 0)
    assert row.win_share == pytest.approx(2 / 3)
    assert row.median_advantage == 1.0
    assert (row.mean_advantage, row.advantage_low, row.cohens_dz) == (None, None, None)
    # Advantages 1, -inf, +inf rank as 1, 2.5, 2.5: W- = 2.5 under the normal approximation.
    assert (row.statistic, row.method) == (2.5, "normal")
    assert any(item.topic == "failures" for item in summary.limitations)


def test_the_median_tolerates_a_minority_of_failures() -> None:
    cells: Cells = {"a": {"p1": ["failed", 3.0, 3.0]}, "b": {"p1": [4.0, 4.0, 4.0]}}
    median = summarize(make_dataset(cells), AnalysisConfig(statistic="median"))
    mean = summarize(make_dataset(cells))
    assert median.cells[0].score == 3.0
    assert mean.cells[0].score == math.inf
    assert median.pairwise[0].wins == 1
    assert mean.pairwise[0].losses == 1


def test_design_limitations_are_reported() -> None:
    cells: Cells = {
        "a": {"p1": [1.0, 1.5], "p2": [2.0, None], "p3": [1.0, 1.0]},
        "b": {"p1": [2.0, 2.5], "p2": [3.0, 3.0], "p3": [2.0, 2.0]},
        "c": {"p1": [3.0]},
    }
    summary = summarize(make_dataset(cells, deterministic="c"), AnalysisConfig(replicate_floor=5))
    topics = {item.topic: item.message for item in summary.limitations}
    assert "1 of 13 planned runs have no record yet" in topics["campaign"]
    assert "p2, p3" in topics["ranking"]
    assert "fewer than 6 shared problems" in topics["pairwise"]
    assert "a (2), b (2)" in topics["replicates"]
    assert "at least two problems" in topics["omnibus"]
    assert summary.blocks == ("p1",)
    assert summary.omnibus is None


def test_a_maximised_objective_flips_the_orientation() -> None:
    summary = summarize(make_dataset(ORDERED, deterministic="abc"), AnalysisConfig(sense="max"))
    assert summary.sense == "max"
    assert summary.comparison("a", "c").losses == 8
    assert summary.comparison("c", "a").wins == 8
    assert summary.comparison("c", "a").mean_advantage == pytest.approx(4.05)
    assert [row.average_rank for row in summary.solver_summaries] == [3.0, 2.0, 1.0]


def test_the_scope_must_be_unambiguous_and_known() -> None:
    two = make_dataset(ORDERED, protocols=("p", "q"))
    with pytest.raises(ConfigError, match="2 protocols"):
        summarize(two)
    assert summarize(two, AnalysisConfig(protocol="q")).protocol == "q"
    with pytest.raises(ConfigError, match="unknown protocol"):
        summarize(two, AnalysisConfig(protocol="z"))
    unregistered = make_dataset({"a": {"p1": [1.0]}}, objectives=("latency",))
    with pytest.raises(ConfigError, match="sense"):
        summarize(unregistered)
    assert summarize(unregistered, AnalysisConfig(sense="min")).objective == "latency"
    with pytest.raises(ConfigError, match="does not declare"):
        summarize(make_dataset(ORDERED), AnalysisConfig(objective="energy"))
    energy = make_dataset(ORDERED, objectives=("makespan", "energy"))
    assert summarize(energy, AnalysisConfig(objective="energy")).objective == "energy"


def test_shaffer_needs_every_pair_compared() -> None:
    cells: Cells = {"a": {"p1": [1.0], "p2": [2.0]}, "b": {"p1": [2.0]}, "c": {"p2": [3.0]}}
    dataset = make_dataset(cells)
    summary = summarize(dataset)
    assert [(row.left, row.right) for row in summary.pairwise] == [("a", "b"), ("a", "c")]
    assert "b and c" in {item.topic: item.message for item in summary.limitations}["pairwise"]
    with pytest.raises(ConfigError, match="Shaffer"):
        summarize(dataset, AnalysisConfig(correction="shaffer"))


def test_comparisons_can_be_read_from_either_side() -> None:
    summary = summarize(make_dataset(ORDERED, deterministic="abc"))
    forward = summary.comparison("a", "b")
    backward = summary.comparison("b", "a")
    assert backward == forward.swapped()
    assert (backward.wins, backward.win_share) == (0, 0.0)
    assert backward.mean_advantage == pytest.approx(-1.35)
    with pytest.raises(ConfigError, match="not compared"):
        summary.comparison("a", "z")
