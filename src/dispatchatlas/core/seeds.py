"""Deterministic seed derivation.

Experiments draw every random stream from a single root seed. A stream is
addressed by a namespace, such as ``"run:flow-shop-3:simulated-annealing"``,
and an index, such as a repetition number. Equal coordinates always give the
same seed on every platform, and different coordinates give independent
seeds, so adding a solver to an experiment never shifts another solver's
randomness.
"""

from __future__ import annotations

import hashlib

__all__ = ["MAX_SEED", "SEED_DOMAIN", "derive_seed"]

SEED_DOMAIN = "dispatchatlas.seed.v1"
"""Domain separator mixed into every derivation."""

MAX_SEED = 2**63 - 1
"""Largest seed value :func:`derive_seed` returns."""


def derive_seed(root: int, namespace: str, index: int = 0) -> int:
    r"""Derive a seed from a root seed, a namespace, and an index.

    The seed is the first eight bytes, read big-endian, of the SHA-256 digest of
    ``"dispatchatlas.seed.v1\0{root}\0{namespace}\0{index}"``, reduced modulo
    ``2**63``.

    Args:
        root: Non-negative root seed of the experiment.
        namespace: Non-empty name of the stream. Surrounding whitespace is
            ignored.
        index: Non-negative position within the stream.

    Returns:
        A seed in ``[0, 2**63 - 1]``.

    Raises:
        ValueError: If ``root`` or ``index`` is negative or not an integer, or
            ``namespace`` is empty.
    """
    for name, value in (("root", root), ("index", index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("namespace must be a non-empty string")
    payload = f"{SEED_DOMAIN}\0{root}\0{namespace.strip()}\0{index}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (MAX_SEED + 1)
