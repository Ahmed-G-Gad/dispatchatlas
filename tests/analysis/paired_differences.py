"""Deterministic paired differences for the tests of the rank and equivalence procedures.

A dataset of fifty differences takes dozens of lines to write out, so the tests that compare the
library with SciPy build the large ones with :func:`congruential`. The reference values were
computed from the same expression, so a test and its constants agree on the data without storing
it.
"""

from __future__ import annotations


def congruential(
    size: int, multiplier: int, modulus: int, centre: int, scale: float, fraction: float = 0.37
) -> list[float]:
    """Return ``size`` differences built from the residues of a linear congruence.

    Difference ``i`` is ``((i * multiplier + 17) % modulus - centre + fraction) * scale``. With a
    prime ``modulus`` above ``size`` and a ``multiplier`` that it does not divide, the residues are
    distinct, and a ``fraction`` other than 0 and one half leaves no difference zero and no two
    magnitudes equal.
    """
    return [
        (((index * multiplier + 17) % modulus) - centre + fraction) * scale for index in range(size)
    ]
