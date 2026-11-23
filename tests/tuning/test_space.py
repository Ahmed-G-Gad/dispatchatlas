"""Parameter spaces: validation, the sampling distributions, and the distances.

The sampling distributions are checked without chance where they can be: a generator that returns
a chosen uniform number makes a draw a function of that number, so the quantile function of
the truncated normal distribution can be integrated over a grid and set against the closed form of
its mean, and the probabilities of the integers can be counted over a grid.
"""

from __future__ import annotations

import copy
import math
import pickle
from collections import Counter
from functools import partial
from typing import cast

import numpy as np
import pytest

from dispatchatlas.analysis.stats.distributions import normal_ppf
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.tuning import (
    DIGITS,
    Categorical,
    Integer,
    Ordinal,
    ParameterSpace,
    Real,
    SamplingModel,
    Value,
)
from dispatchatlas.tuning.space import _truncated_normal


class Fixed:
    """A generator whose every uniform draw is the same chosen number."""

    def __init__(self, u: float) -> None:
        self.u = u

    def random(self) -> float:
        return self.u


def fixed(u: float) -> np.random.Generator:
    return cast("np.random.Generator", Fixed(u))


def grid(count: int) -> list[float]:
    """The midpoints of ``count`` equal cells of the unit interval."""
    return [(i + 0.5) / count for i in range(count)]


def normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def normal_cdf_exact(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "build",
    [
        lambda: Real("", 0.0, 1.0),
        lambda: Real(" x", 0.0, 1.0),
        lambda: Real(3, 0.0, 1.0),  # type: ignore[arg-type]
        lambda: Real("x", 1.0, 1.0),
        lambda: Real("x", 2.0, 1.0),
        lambda: Real("x", 0.0, math.inf),
        lambda: Real("x", math.nan, 1.0),
        lambda: Real("x", True, 2.0),
        lambda: Real("x", 0.0, 1.0, digits=0),
        lambda: Real("x", 0.0, 1.0, digits=16),
        lambda: Real("x", 0.0, 1.0, digits=True),
        lambda: Real("x", 0.0, 1.0, digits=2.5),  # type: ignore[arg-type]
        lambda: Integer("n", 1, 1),
        lambda: Integer("n", 5, 1),
        lambda: Integer("n", 1.0, 5),  # type: ignore[arg-type]
        lambda: Integer("n", True, 5),
        lambda: Ordinal("o", [1]),
        lambda: Ordinal("o", [1, 1.0]),
        lambda: Ordinal("o", "abc"),
        lambda: Ordinal("o", [1, True]),
        lambda: Ordinal("o", [1, math.nan]),
        lambda: Categorical("c", ["a", ""]),
        lambda: Categorical("c", ["a", "a"]),
        lambda: Categorical("c", [None, "a"]),  # type: ignore[list-item]
        lambda: Categorical("c", 5),  # type: ignore[arg-type]
    ],
)
def test_a_parameter_refuses_what_is_not_one(build) -> None:
    with pytest.raises(ConfigError):
        build()


def test_whole_bounds_of_a_real_parameter_become_floats() -> None:
    parameter = Real("x", 0, 1)
    assert isinstance(parameter.low, float)
    assert isinstance(parameter.high, float)
    assert parameter.digits == DIGITS == 4


def test_levels_are_stored_as_a_tuple_and_may_mix_numbers_and_strings() -> None:
    assert Ordinal("o", [1, 2.5, 7]).levels == (1, 2.5, 7)
    assert Categorical("c", ["a", 3]).levels == ("a", 3)


def test_a_space_needs_distinct_named_parameters() -> None:
    with pytest.raises(ConfigError, match="at least one"):
        ParameterSpace([])
    with pytest.raises(ConfigError, match="differ"):
        ParameterSpace([Real("x", 0, 1), Integer("x", 1, 3)])
    with pytest.raises(ConfigError, match="not a parameter"):
        ParameterSpace([Real("x", 0, 1), "y"])  # type: ignore[list-item]
    space = ParameterSpace([Real("x", 0, 1), Integer("n", 1, 3)])
    assert space == ParameterSpace((Real("x", 0, 1), Integer("n", 1, 3)))
    assert space.names == ("x", "n")
    assert len(space) == 2
    assert isinstance(space.parameters, tuple)


# ------------------------------------------------------------ the sampling laws


def test_a_truncated_normal_draw_is_the_quantile_between_the_two_ends() -> None:
    # Half of the normal distribution, from its mean up: the quantile at u is that of the normal
    # distribution at 1/2 + u/2, and the median of the half is 0.6744897501960817.
    assert _truncated_normal(0.5, 0.0, 1.0, 0.0, 60.0) == pytest.approx(
        0.6744897501960817, rel=1e-7
    )
    for u in (0.1, 0.37, 0.9):
        assert _truncated_normal(u, 0.0, 1.0, 0.0, 60.0) == pytest.approx(
            normal_ppf(0.5 + 0.5 * u), rel=1e-7
        )
    # A symmetric range has its median at the mean, and its ends are the ends of the range.
    assert _truncated_normal(0.5, 0.5, 0.2, 0.0, 1.0) == pytest.approx(0.5, abs=1e-9)
    assert _truncated_normal(0.0, 0.5, 0.2, 0.0, 1.0) == pytest.approx(0.0, abs=1e-7)
    assert _truncated_normal(1.0 - 1e-12, 0.5, 0.2, 0.0, 1.0) == pytest.approx(1.0, abs=1e-6)
    # Even a draw of exactly 0 or past the last uniform number stays inside the range.
    assert 0.0 <= _truncated_normal(0.0, 0.0, 1e-3, 0.0, 1.0) <= 1.0
    assert 0.0 <= _truncated_normal(1.0, 1.0, 1e-3, 0.0, 1.0) <= 1.0


@pytest.mark.parametrize(
    ("mean", "spread", "low", "high"),
    [(0.2, 0.3, 0.0, 1.0), (0.0, 0.5, 0.0, 1.0), (1.0, 0.5, 0.0, 1.0), (3.0, 7.0, -2.0, 11.0)],
)
def test_the_truncated_normal_draws_have_the_mean_of_the_truncated_distribution(
    mean: float, spread: float, low: float, high: float
) -> None:
    # The mean of a distribution is the integral of its quantile function over (0, 1).
    alpha, beta = (low - mean) / spread, (high - mean) / spread
    mass = normal_cdf_exact(beta) - normal_cdf_exact(alpha)
    expected = mean + spread * (normal_pdf(alpha) - normal_pdf(beta)) / mass
    draws = [_truncated_normal(u, mean, spread, low, high) for u in grid(20000)]
    assert sum(draws) / len(draws) == pytest.approx(expected, abs=1e-4)
    assert all(low <= draw <= high for draw in draws)
    assert draws == sorted(draws)


def test_the_extreme_integers_get_as_much_probability_as_the_middle_ones() -> None:
    # Footnote 2 of López-Ibáñez et al. (2016): a wide normal distribution, truncated to the
    # range, rounded to the nearest integer, would give 1 and 3 a quarter each and 2 a half. The
    # draw from [1, 4] centred half a unit up and rounded down gives each a third.
    parameter = Integer("n", 1, 3)
    counts = Counter(parameter.sample_around(fixed(u), 2, 1e6) for u in grid(3000))
    assert set(counts) == {1, 2, 3}
    assert all(abs(count - 1000) <= 2 for count in counts.values())
    ordinal = Ordinal("o", ["low", "mid", "high"])
    levels = Counter(ordinal.sample_around(fixed(u), "mid", 1e6) for u in grid(3000))
    assert all(abs(count - 1000) <= 2 for count in levels.values())
    assert set(levels) == {"low", "mid", "high"}


def test_a_narrow_spread_keeps_an_integer_or_an_ordinal_at_its_parent() -> None:
    integer = Integer("n", 1, 9)
    assert {integer.sample_around(fixed(u), 5, 0.05) for u in grid(500)} == {5}
    ordinal = Ordinal("o", [10, 20, 30, 40])
    assert {ordinal.sample_around(fixed(u), 30, 0.05) for u in grid(500)} == {30}
    # With a spread of one index, a neighbour is likely and a far level is rare.
    counts = Counter(ordinal.sample_around(fixed(u), 10, 1.0) for u in grid(4000))
    assert counts[10] > counts[20] > counts[30] > counts[40] > 0


def test_a_real_draw_stays_in_the_range_and_is_rounded() -> None:
    parameter = Real("x", 0.0, 1.0, digits=2)
    for u in grid(400):
        value = parameter.sample(fixed(u))
        assert 0.0 <= value <= 1.0
        assert value == round(value, 2)
        around = parameter.sample_around(fixed(u), 0.0, 0.5)
        assert 0.0 <= around <= 1.0
        assert around == round(around, 2)
    # Rounding past an end of the range is cut back to it.
    odd = Real("y", 0.0, 0.12345, digits=4)
    assert max(odd.sample(fixed(u)) for u in grid(2000)) <= 0.12345
    assert Real("z", 0.0, 1.0).sample_around(fixed(0.5), 0.5, 1e-3) == pytest.approx(0.5, abs=1e-9)


def test_a_draw_around_a_parent_needs_a_positive_finite_spread() -> None:
    rng = np.random.default_rng(0)
    for spread in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ConfigError, match="spread"):
            Real("x", 0.0, 1.0).sample_around(rng, 0.5, spread)
        with pytest.raises(ConfigError, match="spread"):
            Integer("n", 1, 5).sample_around(rng, 3, spread)
        with pytest.raises(ConfigError, match="spread"):
            Ordinal("o", [1, 2, 3]).sample_around(rng, 2, spread)
    with pytest.raises(ConfigError, match="not a level"):
        Ordinal("o", [1, 2, 3]).sample_around(rng, 7, 1.0)


def test_uniform_draws_cover_every_value_alike() -> None:
    assert Counter(Integer("n", 3, 5).sample(fixed(u)) for u in grid(900)) == {
        3: 300,
        4: 300,
        5: 300,
    }
    levels = Counter(Categorical("c", ["a", "b", "c"]).sample(fixed(u)) for u in grid(900))
    assert levels == {"a": 300, "b": 300, "c": 300}
    assert Counter(Ordinal("o", [5, 6]).sample(fixed(u)) for u in grid(10)) == {5: 5, 6: 5}
    assert Real("x", 2.0, 4.0).sample(fixed(0.5)) == 3.0
    # A uniform draw of 1, which a generator never returns, still stays in the range.
    assert Integer("n", 3, 5).sample(fixed(1.0)) == 5
    assert Categorical("c", ["a", "b"]).sample(fixed(1.0)) == "b"


def test_weighted_draws_follow_the_weights() -> None:
    parameter = Categorical("c", ["a", "b", "c"])
    weights = (0.2, 0.3, 0.5)
    for u, expected in ((0.0, "a"), (0.19, "a"), (0.2, "b"), (0.49, "b"), (0.5, "c"), (0.999, "c")):
        assert parameter.sample_weighted(fixed(u), weights) == expected
        # Only the proportions count, and a level of weight zero is never drawn.
        assert parameter.sample_weighted(fixed(u), (2.0, 3.0, 5.0)) == expected
    assert {parameter.sample_weighted(fixed(u), (0.0, 1.0, 0.0)) for u in grid(100)} == {"b"}
    assert parameter.sample_weighted(fixed(0.999999), (1.0, 1.0, 0.0)) == "b"
    for bad in ((1.0, 1.0), (1.0, -1.0, 1.0), (0.0, 0.0, 0.0), (1.0, math.nan, 1.0)):
        with pytest.raises(ConfigError):
            parameter.sample_weighted(fixed(0.5), bad)


def test_every_draw_takes_one_uniform_number_whatever_its_outcome() -> None:
    space = ParameterSpace(
        [Real("x", 0, 1), Integer("n", 1, 9), Ordinal("o", [1, 2, 3]), Categorical("c", ["a", "b"])]
    )
    parent = space.sample(np.random.default_rng(0))[0]
    for narrow in (1e-3, 5.0):
        model = SamplingModel({"x": narrow, "n": narrow, "o": narrow}, {"c": (0.99, 0.01)})
        for draw in (space.sample, partial(space.sample_around, parent=parent, model=model)):
            used = np.random.default_rng(5)
            draw(used)
            expected = np.random.default_rng(5)
            expected.random(4)
            assert used.bit_generator.state == expected.bit_generator.state


# ----------------------------------------------------------------- the space


def test_sampling_a_space_gives_valid_read_only_configurations() -> None:
    space = ParameterSpace(
        [
            Real("x", -2, 2),
            Integer("n", 1, 9),
            Ordinal("o", [4, 8, 16]),
            Categorical("c", ["a", "b"]),
        ]
    )
    configurations = space.sample(np.random.default_rng(3), 200)
    assert len(configurations) == 200
    for configuration in configurations:
        assert list(configuration) == ["x", "n", "o", "c"]
        assert -2.0 <= configuration["x"] <= 2.0  # type: ignore[operator]
        assert configuration["n"] in range(1, 10)
        assert configuration["o"] in (4, 8, 16)
        assert configuration["c"] in ("a", "b")
    assert len(set(configurations)) > 150  # hashable, and mostly different
    with pytest.raises(TypeError):
        configurations[0]["x"] = 0.0  # type: ignore[index]
    # A configuration is sent to child processes, so it must pickle and copy.
    assert pickle.dumps(configurations[0])
    assert copy.deepcopy(configurations[0]) == configurations[0]
    assert space.sample(np.random.default_rng(3), 0) == ()
    again = space.sample(np.random.default_rng(3), 200)
    assert again == configurations
    for count in (-1, 1.5, True):
        with pytest.raises(ConfigError):
            space.sample(np.random.default_rng(3), count)  # type: ignore[arg-type]


def test_the_initial_model_has_the_initial_spreads_and_uniform_weights() -> None:
    space = ParameterSpace(
        [
            Real("x", 1.0, 5.0),
            Integer("n", 1, 10),
            Ordinal("o", [1, 2, 3, 4, 5]),
            Categorical("c", ["a", "b", "c", "d"]),
        ]
    )
    model = space.initial_model()
    # (upper - lower) / 2 for a real and an integer parameter, and (levels - 1) / 2 for an
    # ordinal one: López-Ibáñez et al. (2016, pp. 47-48).
    assert dict(model.spreads) == {"x": 2.0, "n": 4.5, "o": 2.0}
    assert dict(model.weights) == {"c": (0.25, 0.25, 0.25, 0.25)}
    with pytest.raises(TypeError):
        model.spreads["x"] = 1.0  # type: ignore[index]


def test_a_draw_around_a_parent_uses_its_values_and_the_model() -> None:
    space = ParameterSpace(
        [Real("x", 0.0, 1.0), Integer("n", 1, 100), Categorical("c", ["a", "b", "c"])]
    )
    parent: dict[str, Value] = {"x": 0.25, "n": 40, "c": "a"}
    model = SamplingModel({"x": 0.01, "n": 0.05}, {"c": (0.0, 1.0, 0.0)})
    rng = np.random.default_rng(1)
    draws = [space.sample_around(rng, parent, model) for _ in range(300)]
    assert {draw["c"] for draw in draws} == {"b"}  # the model, not the parent's level
    assert {draw["n"] for draw in draws} == {40}
    assert all(abs(draw["x"] - 0.25) < 0.06 for draw in draws)  # type: ignore[operator]
    with pytest.raises(ConfigError, match="lacks the parameter 'n'"):
        space.sample_around(rng, {"x": 0.5, "c": "a"}, model)
    with pytest.raises(ConfigError, match="model lacks the parameter 'x'"):
        space.sample_around(rng, parent, SamplingModel({"n": 1.0}, {"c": (1.0, 1.0, 1.0)}))
    with pytest.raises(ConfigError, match="needs a number"):
        space.sample_around(rng, {"x": "a", "n": 4, "c": "a"}, model)


def test_the_distance_is_the_largest_distance_of_a_parameter() -> None:
    space = ParameterSpace(
        [
            Real("x", 0.0, 10.0),
            Integer("n", 0, 100_000),
            Ordinal("o", [1, 2, 3]),
            Categorical("c", ["a", "b"]),
        ]
    )
    base: dict[str, Value] = {"x": 1.0, "n": 5, "o": 2, "c": "a"}
    assert space.distance(base, dict(base)) == 0.0
    # Normalised differences up to 10 ** -4 of the range are no difference.
    assert space.distance(base, {**base, "x": 1.0005}) == 0.0
    assert space.distance(base, {**base, "n": 6}) == 0.0
    assert space.distance(base, {**base, "x": 1.002}) == pytest.approx(0.0002)
    assert space.distance(base, {**base, "x": 6.0}) == pytest.approx(0.5)
    assert space.distance(base, {**base, "n": 50_005}) == pytest.approx(0.5)
    # An ordinal or a categorical value is either the same or at distance 1.
    assert space.distance(base, {**base, "o": 3}) == 1.0
    assert space.distance(base, {**base, "c": "b"}) == 1.0
    assert space.distance(base, {**base, "x": 6.0, "c": "b"}) == 1.0
    assert space.distance(base, {**base, "x": 6.0, "n": 25_005}) == pytest.approx(0.5)
    with pytest.raises(ConfigError, match="lacks"):
        space.distance(base, {"x": 1.0})
    narrow = Real("y", 0.0, 1.0, digits=2)
    assert narrow.distance(0.5, 0.505) == 0.0
    assert narrow.distance(0.5, 0.52) == pytest.approx(0.02)
