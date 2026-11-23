"""The normal, chi-square, F, and Student t distribution functions, in pure Python.

The rank tests read their p-values from these distributions: the normal for
large-sample Wilcoxon and Mann-Whitney statistics, the chi-square for the
Friedman statistic, and the F for the Iman-Davenport statistic. The Student t
serves the paired t test, its confidence interval, and the equivalence tests
built on them.

- The normal tails use :func:`math.erfc`, so a tail probability keeps its
  relative precision far into the tail instead of cancelling against one.
- The normal quantile is P. J. Acklam's rational approximation, with a
  relative error below 1.15e-9 on the open unit interval. It serves critical
  values and confidence levels, never the inversion of a reported p-value.
- The chi-square tail is the regularized upper incomplete gamma function
  ``Q(df/2, x/2)``: a power series below the transition ``x < a + 1`` and the
  modified Lentz continued fraction above it (Press et al. 2007, section 6.2).
- The F tail is the regularized incomplete beta function
  ``I_{d2/(d2 + d1 f)}(d2/2, d1/2)``, evaluated by the modified Lentz continued
  fraction on the side of its symmetry point where it converges (Press et al.
  2007, section 6.4).
- The Student t tail is half the same incomplete beta function,
  ``I_{df/(df + t^2)}(df/2, 1/2) / 2``, of the magnitude of the statistic. The
  t quantile inverts it by Newton's method on the logarithm of the tail,
  started from the normal quantile and held inside a bracket that every
  evaluation narrows, so a tail of 1e-12 is inverted as exactly as one of 0.05.

Both continued fractions and the series stop at relative convergence (1e-15
for the gamma function, 3e-14 for the beta function) and raise
:class:`ArithmeticError` rather than return a partial sum when an iteration
cap is reached. The caps grow with the square root of the shape parameters,
the rate at which the expansions converge near the distributions' centres.
"""

from __future__ import annotations

import math

from dispatchatlas.core.errors import ConfigError

__all__ = ["chi2_sf", "f_sf", "normal_cdf", "normal_ppf", "normal_sf", "t_ppf", "t_sf"]

_SQRT2 = math.sqrt(2.0)

# Acklam's coefficients: numerator and denominator of the central region (a, b)
# and of the tails (c, d).
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425
"""Probability below which (and above one minus which) the tail formula applies."""

_GAMMA_CONVERGENCE = 1e-15
_BETA_CONVERGENCE = 3e-14
_GAMMA_TINY = 1e-300
_BETA_TINY = 1e-30
_QUANTILE_CONVERGENCE = 1e-13
_QUANTILE_LIMIT = 1e150
_QUANTILE_STEPS = 200


def normal_cdf(z: float) -> float:
    """Return the standard normal distribution function ``P(Z <= z)``."""
    return 0.5 * math.erfc(-float(z) / _SQRT2)


def normal_sf(z: float) -> float:
    """Return the standard normal upper tail ``P(Z > z)``.

    Unlike ``1 - normal_cdf(z)``, it stays accurate when the tail is far
    below machine epsilon.
    """
    return 0.5 * math.erfc(float(z) / _SQRT2)


def normal_ppf(probability: float) -> float:
    """Return the standard normal quantile by Acklam's rational approximation.

    The central region ``[0.02425, 0.97575]`` uses a rational function of
    ``(p - 1/2)^2``, and the tails a rational function of
    ``sqrt(-2 log p)``. The relative error is below 1.15e-9.

    Args:
        probability: A probability strictly between 0 and 1.

    Returns:
        ``z`` such that ``normal_cdf(z)`` is ``probability``.

    Raises:
        ConfigError: If ``probability`` is not strictly between 0 and 1.
    """
    p = float(probability)
    if not 0.0 < p < 1.0:
        raise ConfigError(f"the normal quantile needs a probability in (0, 1), got {probability!r}")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        return _tail(q)
    if p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        a, b = _A, _B
        numerator = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        denominator = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        return numerator / denominator
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -_tail(q)


def _tail(q: float) -> float:
    c, d = _C, _D
    numerator = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    denominator = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    return numerator / denominator


def chi2_sf(statistic: float, df: float) -> float:
    """Return the chi-square upper tail ``P(X > statistic)``.

    Args:
        statistic: The observed statistic. Non-positive values give 1 and
            positive infinity gives 0.
        df: Degrees of freedom, positive.

    Returns:
        The tail probability.

    Raises:
        ConfigError: If ``df`` is not a positive finite number or the
            statistic is NaN.
        ArithmeticError: If the incomplete gamma function fails to converge.
    """
    shape = _positive(df, "chi-square degrees of freedom")
    x = _statistic(statistic, "chi-square statistic")
    if x <= 0.0:
        return 1.0
    if math.isinf(x):
        return 0.0
    return _gamma_q(shape / 2.0, x / 2.0)


def f_sf(statistic: float, df_numerator: float, df_denominator: float) -> float:
    """Return the F-distribution upper tail ``P(F > statistic)``.

    Args:
        statistic: The observed statistic. Non-positive values give 1 and
            positive infinity gives 0.
        df_numerator: Numerator degrees of freedom ``d1``, positive.
        df_denominator: Denominator degrees of freedom ``d2``, positive.

    Returns:
        ``I_x(d2/2, d1/2)`` at ``x = d2 / (d2 + d1 statistic)``.

    Raises:
        ConfigError: If a degree of freedom is not a positive finite number
            or the statistic is NaN.
        ArithmeticError: If the incomplete beta function fails to converge.
    """
    d1 = _positive(df_numerator, "F numerator degrees of freedom")
    d2 = _positive(df_denominator, "F denominator degrees of freedom")
    f = _statistic(statistic, "F statistic")
    if f <= 0.0:
        return 1.0
    if math.isinf(f):
        return 0.0
    total = d2 + d1 * f
    return _beta_i(d2 / 2.0, d1 / 2.0, d2 / total, d1 * f / total)


def t_sf(statistic: float, df: float) -> float:
    """Return the Student t upper tail ``P(T > statistic)``.

    For a positive statistic ``t`` the tail is ``I_x(df/2, 1/2) / 2`` at
    ``x = df / (df + t^2)``, evaluated like the F tail with the complement
    ``t^2 / (df + t^2)`` passed directly, so that a tail far below machine
    epsilon keeps its relative precision. The lower half of the distribution
    follows by symmetry.

    Args:
        statistic: The observed statistic. Positive infinity gives 0 and
            negative infinity gives 1.
        df: Degrees of freedom, positive. They need not be an integer.

    Returns:
        The tail probability. Its relative error is below 1e-12 up to 10^4
        degrees of freedom and below 1e-10 up to 10^6, for tails down to
        1e-15.

    Raises:
        ConfigError: If ``df`` is not a positive finite number or the
            statistic is NaN.
        ArithmeticError: If the incomplete beta function fails to converge.
    """
    nu = _positive(df, "Student t degrees of freedom")
    t = _statistic(statistic, "t statistic")
    if t == 0.0:
        return 0.5
    if math.isinf(t):
        return 0.0 if t > 0.0 else 1.0
    tail = _t_upper_tail(abs(t), nu)
    return tail if t > 0.0 else 1.0 - tail


def t_ppf(probability: float, df: float) -> float:
    """Return the Student t quantile ``t`` with ``P(T <= t)`` equal to ``probability``.

    The quantile is found from the tail on its own side of the median, the
    smaller of ``probability`` and ``1 - probability``, so a probability
    near 0 or 1 is inverted as exactly as :func:`t_sf` evaluates the tail
    and ``t_ppf(1 - p, df) == -t_ppf(p, df)`` up to rounding. Newton's method
    runs on the logarithm of the tail, whose slope is minus the hazard rate
    ``pdf / sf``, from the normal quantile ``z`` plus the first term of its
    expansion for the t quantile, ``z (z^2 + 1) / (4 df)``. A bracket
    around the root narrows with every evaluation: a step that leaves it is
    replaced by the midpoint, or by a doubling while the upper end is unknown,
    so a poor start costs steps and never diverges.

    Args:
        probability: A probability strictly between 0 and 1.
        df: Degrees of freedom, positive. They need not be an integer.

    Returns:
        The quantile. Up to 10^4 degrees of freedom and for tails down to
        1e-15, its relative error is below 1e-12 beyond 0.1 in magnitude and
        its absolute error about 1e-13 closer to 0.

    Raises:
        ConfigError: If ``probability`` is not strictly between 0 and 1, or
            ``df`` is not a positive finite number.
        ArithmeticError: If the quantile exceeds 1e150, the iteration does
            not converge, or the incomplete beta function does not.
    """
    p = float(probability)
    if not 0.0 < p < 1.0:
        raise ConfigError(
            f"the Student t quantile needs a probability in (0, 1), got {probability!r}"
        )
    nu = _positive(df, "Student t degrees of freedom")
    if p == 0.5:
        return 0.0
    magnitude = _t_upper_quantile(min(p, 1.0 - p), nu)
    return -magnitude if p < 0.5 else magnitude


def _positive(value: float, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{what} must be a number, got {value!r}")
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise ConfigError(f"{what} must be positive and finite, got {value!r}")
    return number


def _statistic(value: float, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"the {what} must be a number, got {value!r}")
    number = float(value)
    if math.isnan(number):
        raise ConfigError(f"the {what} is NaN")
    return number


def _cap(*shapes: float) -> int:
    """Return an iteration cap that the expansions reach only when they diverge."""
    return 1000 + int(20.0 * math.sqrt(max(shapes)))


def _gamma_q(a: float, x: float) -> float:
    """Return the regularized upper incomplete gamma function ``Q(a, x)``."""
    if x < a + 1.0:
        return 1.0 - _gamma_p_series(a, x)
    return _gamma_q_fraction(a, x)


def _gamma_prefactor(a: float, x: float) -> float:
    return math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_p_series(a: float, x: float) -> float:
    """Return ``P(a, x)`` from ``x^a e^-x / Gamma(a) * sum_n x^n / (a (a+1) ... (a+n))``."""
    term = 1.0 / a
    total = term
    denominator = a
    for _ in range(_cap(a)):
        denominator += 1.0
        term *= x / denominator
        total += term
        if abs(term) < abs(total) * _GAMMA_CONVERGENCE:
            return total * _gamma_prefactor(a, x)
    raise ArithmeticError(f"the incomplete gamma series did not converge for a={a!r}, x={x!r}")


def _gamma_q_fraction(a: float, x: float) -> float:
    """Return ``Q(a, x)`` from its continued fraction, by the modified Lentz method."""
    b = x + 1.0 - a
    c = 1.0 / _GAMMA_TINY
    d = 1.0 / b
    h = d
    for index in range(1, _cap(a)):
        an = -index * (index - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _GAMMA_TINY:
            d = _GAMMA_TINY
        c = b + an / c
        if abs(c) < _GAMMA_TINY:
            c = _GAMMA_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _GAMMA_CONVERGENCE:
            return h * _gamma_prefactor(a, x)
    raise ArithmeticError(
        f"the incomplete gamma continued fraction did not converge for a={a!r}, x={x!r}"
    )


def _beta_i(a: float, b: float, x: float, y: float) -> float:
    """Return the regularized incomplete beta function ``I_x(a, b)``, where ``y = 1 - x``.

    Taking ``y`` computed directly avoids the cancellation of ``1 - x`` when
    ``x`` is close to one.
    """
    if x <= 0.0:
        return 0.0
    if y <= 0.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(y)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_fraction(a, b, x) / a
    return 1.0 - front * _beta_fraction(b, a, y) / b


def _beta_fraction(a: float, b: float, x: float) -> float:
    """Return the continued fraction of the incomplete beta function (modified Lentz)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _BETA_TINY:
        d = _BETA_TINY
    d = 1.0 / d
    result = d
    for m in range(1, _cap(a, b) + 1):
        even = 2 * m
        numerator = m * (b - m) * x / ((qam + even) * (a + even))
        d = 1.0 + numerator * d
        if abs(d) < _BETA_TINY:
            d = _BETA_TINY
        c = 1.0 + numerator / c
        if abs(c) < _BETA_TINY:
            c = _BETA_TINY
        d = 1.0 / d
        result *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + even) * (qap + even))
        d = 1.0 + numerator * d
        if abs(d) < _BETA_TINY:
            d = _BETA_TINY
        c = 1.0 + numerator / c
        if abs(c) < _BETA_TINY:
            c = _BETA_TINY
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < _BETA_CONVERGENCE:
            return result
    raise ArithmeticError(
        f"the incomplete beta continued fraction did not converge for a={a!r}, b={b!r}"
    )


def _log_gamma_ratio(a: float) -> float:
    """Return ``ln Gamma(a + 1/2) - ln Gamma(a)``.

    Below 20 the difference of :func:`math.lgamma` values is accurate to about
    1e-14. Above it the difference would lose digits in proportion to ``a``, as
    both terms grow like ``a ln a``, so the Stirling series
    ``ln(a)/2 - 1/(8a) + 1/(192a^3) - 1/(640a^5) + 17/(14336a^7)`` takes over,
    its truncation error below 4e-15 from ``a = 20``.
    """
    if a < 20.0:
        return math.lgamma(a + 0.5) - math.lgamma(a)
    inverse = 1.0 / a
    square = inverse * inverse
    series = -inverse / 8.0 + inverse * square * (
        1.0 / 192.0 + square * (-1.0 / 640.0 + square * (17.0 / 14336.0))
    )
    return 0.5 * math.log(a) + series


def _t_upper_tail(magnitude: float, nu: float) -> float:
    """Return ``P(T > magnitude)`` of a Student t variable, for a non-negative magnitude.

    This is :func:`_beta_i` at ``b = 1/2``, halved, with its prefactor
    assembled from quantities that keep their precision as ``nu`` grows: the
    gamma ratio of :func:`_log_gamma_ratio` and ``ln x = -log1p(t^2 / nu)``,
    where ``ln x`` taken from a rounded ``x`` near one would lose digits in
    proportion to ``nu``.
    """
    square = magnitude * magnitude
    total = nu + square
    x, y = nu / total, square / total
    if y <= 0.0:
        return 0.5
    if x <= 0.0:
        return 0.0
    a = nu / 2.0
    front = math.exp(
        _log_gamma_ratio(a)
        - 0.5 * math.log(math.pi)
        - a * math.log1p(square / nu)
        + 0.5 * math.log(y)
    )
    if x < (a + 1.0) / (a + 2.5):
        return 0.5 * front * _beta_fraction(a, 0.5, x) / a
    return 0.5 * (1.0 - 2.0 * front * _beta_fraction(0.5, a, y))


def _t_log_density(t: float, nu: float) -> float:
    """Return the logarithm of the Student t density at ``t``."""
    return (
        _log_gamma_ratio(nu / 2.0)
        - 0.5 * math.log(nu * math.pi)
        - (nu + 1.0) / 2.0 * math.log1p(t * t / nu)
    )


def _t_upper_quantile(tail: float, nu: float) -> float:
    """Return the ``t > 0`` whose upper tail is ``tail``, for a ``tail`` below one half.

    ``low`` always has a tail above the target and ``high`` one below it. The
    Newton step on the logarithm of the tail is ``gap * value / density``, for
    the gap between the logarithms of the tail ``value`` at ``t`` and of the
    target. A step that leaves the bracket, or is undefined where the tail or
    the density underflows, gives way to a doubling or a bisection.
    """
    target = math.log(tail)
    z = -normal_ppf(tail)
    t = z + z * (z * z + 1.0) / (4.0 * nu)
    low, high = 0.0, math.inf
    for _ in range(_QUANTILE_STEPS):
        value = _t_upper_tail(t, nu)
        gap = math.log(value) - target if value > 0.0 else -math.inf
        if gap > 0.0:
            low = t
        else:
            high = t
        density = math.exp(_t_log_density(t, nu))
        candidate = t + gap * value / density if density > 0.0 else math.nan
        if not low < candidate < high:
            candidate = 2.0 * t if math.isinf(high) else 0.5 * (low + high)
        if candidate > _QUANTILE_LIMIT:
            raise ArithmeticError(
                f"the Student t quantile for tail={tail!r}, df={nu!r} exceeds {_QUANTILE_LIMIT!r}"
            )
        if abs(candidate - t) <= _QUANTILE_CONVERGENCE * max(1.0, candidate):
            return candidate
        t = candidate
    raise ArithmeticError(f"the Student t quantile did not converge for tail={tail!r}, df={nu!r}")
