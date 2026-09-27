"""Quality indicators: hand-computed hypervolume, IGD, and IGD+, and their reference frames."""

from __future__ import annotations

import math

import pytest

from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.indicators import (
    ensure_shared_reference_points,
    igd,
    igd_plus,
    indicator_report,
    reference_point,
)
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.pareto import hypervolume
from tests.analysis.analysis_data import make_dataset

POOL = [(1.0, 3.0), (3.0, 1.0), (2.0, 2.0)]
OBJECTIVES = ("makespan", "total_tardiness")


def test_reference_point_rules_and_hand_computed_hypervolumes():
    # Ideal (1, 1), nadir (3, 3), span 2.
    margin = reference_point(POOL)
    assert margin == pytest.approx((3.2, 3.2))
    assert hypervolume(POOL, margin) == pytest.approx(1.84)
    scaled = reference_point(POOL, "ishibuchi", 2)
    assert scaled == pytest.approx((4.0, 4.0))
    assert hypervolume(POOL, scaled) == pytest.approx(6.0)


def test_reference_point_on_an_axis_without_spread():
    assert reference_point([(1.0, 5.0), (2.0, 5.0)]) == pytest.approx((2.1, 5.5))
    assert reference_point([(1.0, 0.0), (2.0, 0.0)], "ishibuchi", 4) == pytest.approx((2.25, 0.25))
    with pytest.raises(ConfigError, match="integer"):
        reference_point(POOL, "ishibuchi", 1.5)
    with pytest.raises(ConfigError, match="positive"):
        reference_point(POOL, "margin", 0.0)
    with pytest.raises(ConfigError, match="at least one"):
        reference_point([])


def test_igd_and_igd_plus_hand_values():
    front = [(2.0, 2.0)]
    # Distances to (1, 3) and (3, 1) are sqrt(2); to (2, 2) it is 0.
    assert igd(front, POOL) == pytest.approx(2.0 * math.sqrt(2.0) / 3.0)
    # IGD+ counts only the objectives where (2, 2) is worse: 1, 1, and 0.
    assert igd_plus(front, POOL) == pytest.approx(2.0 / 3.0)
    assert igd(POOL, POOL) == 0.0
    assert igd([], POOL) == math.inf
    with pytest.raises(ConfigError, match="reference set"):
        igd(front, [])


def test_one_run_with_the_pool_as_front_reproduces_the_hand_values():
    dataset = make_dataset({"moea": {"p1": [list(POOL)]}}, objectives=OBJECTIVES)
    raw = indicator_report(dataset, AnalysisConfig(normalize=False))
    (row,) = raw.rows
    assert row.hypervolume == pytest.approx(1.84)
    assert (row.igd, row.igd_plus) == (0.0, 0.0)
    (frame,) = raw.reference_points
    assert (frame.ideal, frame.nadir) == ((1.0, 1.0), (3.0, 3.0))
    assert frame.point == pytest.approx((3.2, 3.2))
    ishibuchi = indicator_report(
        dataset,
        AnalysisConfig(normalize=False, reference_rule="ishibuchi", reference_parameter=2),
    )
    assert ishibuchi.rows[0].hypervolume == pytest.approx(6.0)
    normalized = indicator_report(dataset)
    # Scaling each axis by its span of 2 divides the area by 4.
    assert normalized.rows[0].hypervolume == pytest.approx(1.84 / 4.0)
    assert normalized.reference_points[0].point == pytest.approx((3.2, 3.2))


def test_failed_runs_count_with_zero_volume_and_infinite_distance():
    dataset = make_dataset(
        {"a": {"p1": [[(1.0, 3.0), (3.0, 1.0)], "failed"]}, "b": {"p1": [(2.0, 2.0), (2.0, 2.0)]}},
        objectives=OBJECTIVES,
    )
    report = indicator_report(dataset, AnalysisConfig(normalize=False))
    a, b = report.rows
    single = hypervolume([(1.0, 3.0), (3.0, 1.0)], (3.2, 3.2))
    assert (a.planned, a.succeeded) == (2, 1)
    assert a.hypervolume == pytest.approx(single / 2.0)
    assert a.igd == math.inf
    assert b.hypervolume == pytest.approx(1.44)
    median = indicator_report(dataset, AnalysisConfig(normalize=False, statistic="median"))
    assert median.rows[0].igd == math.inf


def test_fixed_reference_points_and_single_objective_problems():
    dataset = make_dataset({"a": {"p1": [list(POOL)]}}, objectives=OBJECTIVES)
    fixed = indicator_report(
        dataset, AnalysisConfig(normalize=False, reference_points={"p1": (4.0, 4.0)})
    )
    assert fixed.rows[0].hypervolume == pytest.approx(6.0)
    assert (fixed.reference_points[0].rule, fixed.reference_points[0].parameter) == ("fixed", None)
    with pytest.raises(ConfigError, match="3 values for 2 objectives"):
        indicator_report(dataset, AnalysisConfig(reference_points={"p1": (4.0, 4.0, 4.0)}))
    single = make_dataset({"a": {"p1": [1.0]}})
    assert indicator_report(single).rows == ()


def test_a_problem_without_succeeded_runs_is_a_limitation():
    dataset = make_dataset({"a": {"p1": ["failed"]}}, objectives=OBJECTIVES)
    report = indicator_report(dataset)
    assert report.rows == ()
    assert "p1" in report.limitations[0].message


def test_compared_reports_must_share_their_reference_points():
    dataset = make_dataset({"a": {"p1": [list(POOL)]}}, objectives=OBJECTIVES)
    margin = indicator_report(dataset)
    scaled = indicator_report(
        dataset, AnalysisConfig(reference_rule="ishibuchi", reference_parameter=2)
    )
    ensure_shared_reference_points([])
    ensure_shared_reference_points([margin, margin])
    with pytest.raises(ConfigError, match="different reference points"):
        ensure_shared_reference_points([margin, scaled])
