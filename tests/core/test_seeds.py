"""Seed derivation: pinned values, input rules, and independence of streams."""

from __future__ import annotations

import hashlib
import random

import pytest

from dispatchatlas.core.seeds import MAX_SEED, SEED_DOMAIN, derive_seed

GOLDEN = [
    ((0, "x", 0), 7195940993135537051, 0.41268117423491835),
    ((42, "benchmarks", 7), 7756789700827681433, 0.4968818363897869),
    ((42, "benchmark.smoke", 3), 4290991338521208927, 0.7930745759381643),
    ((11, "campaign", 3), 2267286961556994774, 0.5087503231262709),
    ((12, "serialization", 0), 6955571933468269714, 0.0657806807825293),
    ((20260612, "exact", 0), 5335863566612045546, 0.734102160079022),
    ((1, "solver", 0), 5263492911602304428, 0.4780403664145422),
    ((0, "run:flow-shop-3:simulated-annealing", 4), 6281978563721240036, None),
    ((0, "é", 0), 2316290702254818277, None),
    ((10**30, "x", 0), 3729698716213707506, None),
]
"""``(root, namespace, index)``, the derived seed, and the first draw of
``random.Random(seed)``. Changing any of them changes every published experiment."""


@pytest.mark.parametrize(("arguments", "seed", "draw"), GOLDEN)
def test_derived_seeds_are_pinned(
    arguments: tuple[int, str, int], seed: int, draw: float | None
) -> None:
    assert derive_seed(*arguments) == seed
    if draw is not None:
        assert random.Random(seed).random() == draw


def test_the_derivation_follows_the_documented_formula() -> None:
    assert SEED_DOMAIN == "dispatchatlas.seed.v1"
    assert MAX_SEED == 2**63 - 1
    for root, namespace, index in ((0, "x", 0), (7, "a b", 3), (2**70, "ns", 2**40)):
        payload = f"{SEED_DOMAIN}\0{root}\0{namespace}\0{index}".encode()
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2**63
        assert derive_seed(root, namespace, index) == expected


def test_the_index_defaults_to_zero_and_whitespace_is_ignored() -> None:
    assert derive_seed(5, "stream") == derive_seed(5, "stream", 0)
    assert derive_seed(5, "  stream\n") == derive_seed(5, "stream")
    assert derive_seed(5, "a b") != derive_seed(5, "ab")


def test_different_coordinates_give_different_seeds_in_range() -> None:
    seeds = {
        derive_seed(root, namespace, index)
        for root in range(4)
        for namespace in ("a", "b", "run:x", "run:y")
        for index in range(25)
    }
    assert len(seeds) == 4 * 4 * 25
    assert all(0 <= seed <= MAX_SEED for seed in seeds)
    # The separators keep neighbouring coordinates apart.
    assert derive_seed(1, "23") != derive_seed(12, "3")
    assert derive_seed(1, "x", 23) != derive_seed(1, "x2", 3)


@pytest.mark.parametrize(
    ("root", "namespace", "index", "message"),
    [
        (-1, "x", 0, "root must be a non-negative integer"),
        (True, "x", 0, "root must be a non-negative integer"),
        (1.0, "x", 0, "root must be a non-negative integer"),
        ("1", "x", 0, "root must be a non-negative integer"),
        (0, "x", -1, "index must be a non-negative integer"),
        (0, "x", False, "index must be a non-negative integer"),
        (0, "x", 2.5, "index must be a non-negative integer"),
        (0, "", 0, "namespace must be a non-empty string"),
        (0, "   ", 0, "namespace must be a non-empty string"),
        (0, None, 0, "namespace must be a non-empty string"),
    ],
)
def test_invalid_coordinates_are_refused(
    root: object, namespace: object, index: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        derive_seed(root, namespace, index)  # type: ignore[arg-type]
