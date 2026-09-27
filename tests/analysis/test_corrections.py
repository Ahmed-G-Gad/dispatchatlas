"""Multiple-comparison corrections on hand-computed vectors."""

from __future__ import annotations

import pytest

from dispatchatlas.analysis.stats.corrections import (
    CORRECTIONS,
    adjust_p_values,
    shaffer_hypothesis_counts,
)
from dispatchatlas.core.errors import ConfigError

PAIRS_OF_THREE = [("a", "b"), ("a", "c"), ("b", "c")]


def test_shaffer_counts_follow_the_recursion():
    assert shaffer_hypothesis_counts(3) == (0, 1, 3)
    assert shaffer_hypothesis_counts(4) == (0, 1, 2, 3, 6)
    assert shaffer_hypothesis_counts(1) == (0,)


def test_shaffer_uses_the_static_multipliers_of_four_groups():
    raw = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05]
    # Multipliers 6, 3, 3, 3, 2, 1 instead of Holm's 6, 5, 4, 3, 2, 1.
    assert adjust_p_values(raw, "shaffer", groups=4) == pytest.approx(
        [0.006, 0.03, 0.06, 0.09, 0.09, 0.09]
    )


def test_shaffer_is_less_conservative_than_holm_at_step_two():
    raw = [0.01, 0.02, 0.04]
    shaffer = adjust_p_values(raw, "shaffer", keys=PAIRS_OF_THREE)
    holm = adjust_p_values(raw, "holm")
    assert shaffer == pytest.approx([0.03, 0.03, 0.04])
    assert holm == pytest.approx([0.03, 0.04, 0.04])


def test_finner_hand_computed():
    assert adjust_p_values([0.01, 0.02, 0.04], "finner") == pytest.approx(
        [0.0297010, 0.0298496, 0.04], abs=1e-6
    )


def test_bonferroni_scales_and_caps():
    assert adjust_p_values([0.01, 0.02, 0.40], "bonferroni") == pytest.approx([0.03, 0.06, 1.0])


def test_hochberg_steps_up_with_a_running_minimum():
    raw = [0.01, 0.02, 0.03]
    assert adjust_p_values(raw, "hochberg") == pytest.approx([0.03, 0.03, 0.03])
    assert adjust_p_values(raw, "holm") == pytest.approx([0.03, 0.04, 0.04])


def test_results_follow_the_input_order():
    assert adjust_p_values([0.04, 0.01, 0.02], "holm") == pytest.approx([0.04, 0.03, 0.04])
    assert adjust_p_values([0.3, 0.2], "none") == (0.3, 0.2)
    assert adjust_p_values([], "holm") == ()


def test_equal_p_values_break_ties_by_key():
    keys = [("c", "d"), ("a", "b")]
    assert adjust_p_values([0.01, 0.01], "holm", keys=keys) == (0.02, 0.02)
    assert adjust_p_values([0.01, 0.04], "finner", keys=keys) == adjust_p_values(
        [0.01, 0.04], "finner"
    )


@pytest.mark.parametrize("method", CORRECTIONS)
def test_every_adjustment_is_a_bounded_non_decreasing_map(method):
    raw = [0.001, 0.2, 0.03, 0.04, 0.9, 0.01]
    adjusted = adjust_p_values(raw, method, groups=4)
    assert all(p <= a <= 1.0 for p, a in zip(raw, adjusted, strict=True))
    ordered = sorted(range(len(raw)), key=lambda index: raw[index])
    values = [adjusted[index] for index in ordered]
    assert values == sorted(values)


def test_invalid_input_is_refused():
    with pytest.raises(ConfigError, match="unknown correction"):
        adjust_p_values([0.1], "sidak")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="all 3 pairs"):
        adjust_p_values([0.01, 0.02], "shaffer", keys=[("a", "b"), ("a", "c")])
    with pytest.raises(ConfigError, match="groups"):
        adjust_p_values([0.01, 0.02, 0.03], "shaffer")
    with pytest.raises(ConfigError, match="lie in"):
        adjust_p_values([1.5], "holm")
    with pytest.raises(ConfigError, match="one key"):
        adjust_p_values([0.1, 0.2], "holm", keys=[("a", "b")])
