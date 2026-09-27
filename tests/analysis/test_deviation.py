"""Relative percentage deviation from reference values."""

from __future__ import annotations

import pytest

from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.deviation import arpd, relative_deviation
from dispatchatlas.core.errors import ConfigError
from tests.analysis.analysis_data import make_dataset


def test_relative_deviation_is_positive_when_worse():
    assert relative_deviation(110.0, 100.0) == pytest.approx(10.0)
    assert relative_deviation(95.0, 100.0) == pytest.approx(-5.0)
    assert relative_deviation(90.0, 100.0, "max") == pytest.approx(10.0)
    assert relative_deviation(120.0, 100.0, "max") == pytest.approx(-20.0)
    with pytest.raises(ConfigError, match="positive reference"):
        relative_deviation(1.0, 0.0)
    with pytest.raises(ConfigError, match="finite"):
        relative_deviation(float("inf"), 1.0)


def test_arpd_is_the_joint_mean_over_problems_and_runs():
    dataset = make_dataset(
        {
            "a": {"p1": [110.0, 120.0], "p2": [200.0, "failed"]},
            "b": {"p1": [100.0, 95.0], "p2": [210.0, 220.0]},
        }
    )
    config = AnalysisConfig(references={"p1": 100.0, "p2": 200.0})
    a, b = arpd(dataset, config)
    # a: 10, 20, and 0 percent over three succeeded runs of four planned.
    assert (a.solver, a.problems, a.runs, a.planned) == ("a", 2, 3, 4)
    assert (a.arpd, a.rpd_min, a.rpd_max) == pytest.approx((10.0, 0.0, 20.0))
    # b: 0, -5, 5, and 10 percent.
    assert (b.arpd, b.rpd_min, b.rpd_max) == pytest.approx((2.5, -5.0, 10.0))


def test_arpd_fails_closed_without_references():
    dataset = make_dataset({"a": {"p1": [1.0], "p2": [2.0]}})
    with pytest.raises(ConfigError, match="needs reference values"):
        arpd(dataset, AnalysisConfig())
    with pytest.raises(ConfigError, match="p2"):
        arpd(dataset, AnalysisConfig(references={"p1": 1.0}))
    with pytest.raises(ConfigError, match="positive"):
        AnalysisConfig(references={"p1": -1.0})


def test_a_solver_without_succeeded_runs_has_no_deviation():
    dataset = make_dataset({"a": {"p1": ["failed"]}, "b": {"p1": [2.0]}})
    a, _ = arpd(dataset, AnalysisConfig(references={"p1": 1.0}))
    assert (a.runs, a.planned, a.arpd, a.rpd_min) == (0, 1, None, None)
