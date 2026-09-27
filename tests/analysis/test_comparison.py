"""One solver against baselines: orientation, the many-to-one family, and refusals."""

from __future__ import annotations

from pathlib import Path

import pytest

from dispatchatlas.analysis.comparison import compare_to_baselines
from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.dataset import load_dataset
from dispatchatlas.analysis.summary import summarize
from dispatchatlas.core.errors import ConfigError
from tests.analysis.analysis_data import campaign_dir, make_dataset

__all__ = ["campaign_dir"]

# On problem i, a scores 10 + i, b 11 + 1.1 i, and c 13 + 1.3 i: a beats both.
CELLS = {
    "a": {f"p{i}": [10.0 + i] for i in range(8)},
    "b": {f"p{i}": [11.0 + 1.1 * i] for i in range(8)},
    "c": {f"p{i}": [13.0 + 1.3 * i] for i in range(8)},
}


def test_rows_are_oriented_towards_the_focal_solver() -> None:
    comparison = compare_to_baselines(make_dataset(CELLS), "c")
    assert comparison.baselines == ("a", "b")
    assert [(row.left, row.right) for row in comparison.rows] == [("c", "a"), ("c", "b")]
    for row in comparison.rows:
        assert (row.wins, row.losses, row.win_share) == (0, 8, 0.0)
        assert row.mean_advantage is not None
        assert row.mean_advantage < 0.0
    assert not comparison.better_than_all


def test_p_values_are_adjusted_over_the_baselines_only() -> None:
    comparison = compare_to_baselines(make_dataset(CELLS), "a", ["b", "c"])
    assert comparison.better_than_all
    for row in comparison.rows:
        assert row.p_value == pytest.approx(2 / 256)
        # Holm over two comparisons, not over the three pairs of the summary.
        assert row.adjusted_p_value == pytest.approx(2 * 2 / 256)
    summary = summarize(make_dataset(CELLS))
    assert summary.comparison("a", "b").adjusted_p_value == pytest.approx(3 * 2 / 256)


def test_missing_or_invalid_baselines_are_refused() -> None:
    dataset = make_dataset(CELLS)
    with pytest.raises(ConfigError, match="unknown baseline 'z'"):
        compare_to_baselines(dataset, "a", ["b", "z"])
    with pytest.raises(ConfigError, match="unknown focal solver"):
        compare_to_baselines(dataset, "z", ["a"])
    with pytest.raises(ConfigError, match="own baseline"):
        compare_to_baselines(dataset, "a", ["a", "b"])
    with pytest.raises(ConfigError, match="twice"):
        compare_to_baselines(dataset, "a", ["b", "b"])
    with pytest.raises(ConfigError, match="at least one baseline"):
        compare_to_baselines(dataset, "a", [])
    with pytest.raises(ConfigError, match="Shaffer"):
        compare_to_baselines(dataset, "a", config=AnalysisConfig(correction="shaffer"))
    with pytest.raises(ConfigError, match="unknown baseline 'c'"):
        compare_to_baselines(dataset, "a", ["c"], AnalysisConfig(solvers=("a", "b")))


def test_a_baseline_without_shared_problems_is_refused() -> None:
    dataset = make_dataset({"a": {"p1": [1.0]}, "b": {"p2": [2.0]}, "c": {"p1": [3.0]}})
    with pytest.raises(ConfigError, match="shares no problem"):
        compare_to_baselines(dataset, "a", ["b", "c"])


def test_a_real_campaign_against_its_baselines(campaign_dir: Path) -> None:
    comparison = compare_to_baselines(load_dataset(campaign_dir), "hc")
    assert comparison.baselines == ("input-order", "random-search", "shortest-processing-time")
    assert all(row.left == "hc" and row.pairs == 7 for row in comparison.rows)
    assert all(row.win_share >= 0.5 for row in comparison.rows)
