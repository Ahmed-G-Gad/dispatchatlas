"""What the job-shop search families share: their representations by name, and their rules.

A **search family** of this package is one solver class whose control logic is
written once, against :class:`~dispatchatlas.solvers.shop.representation.Representation`,
and runs unchanged on either representation, which its ``representation``
parameter names (:data:`REPRESENTATIONS`):

- ``"keys"``: random keys,
  :class:`~dispatchatlas.solvers.shop.representation.KeyRepresentation`;
- ``"sequence"``: job sequences,
  :class:`~dispatchatlas.solvers.shop.representation.SequenceRepresentation`.

Two runs of a family with the same seed, one on each representation, then
differ in the genotype and its operators only. Every family of the package keeps
them in step by the same rules:

- it searches through a :class:`~dispatchatlas.solvers.shop.run.ShopRun`, whose
  control and operator streams depend on the seed alone;
- its first evaluation is a random solution, drawn with
  :meth:`~dispatchatlas.solvers.shop.representation.Representation.initial`
  from the operator stream before any other draw from it, so both
  representations start from the same schedule;
- its control logic draws from the control stream a number of times that no
  outcome decides, such as one uniform number for every acceptance test whether
  or not the move worsens, so both runs read the same control numbers at the
  same step;
- its observer draws from no stream;
- it calls :meth:`~dispatchatlas.solvers.shop.run.ShopRun.finish` once its search
  is exhausted, which refuses a run that repaired a sequence;
- a job shop of one job has a single job sequence, which the family evaluates
  before it stops.

:func:`make_representation` builds a representation by name, and
:func:`check_family` and :func:`check_moves` check the parameters the families
share, so that a solver refuses a bad value when it is created.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from dispatchatlas.solvers.shop.decoding import BUILDERS
from dispatchatlas.solvers.shop.instance import JobShop
from dispatchatlas.solvers.shop.representation import (
    KEY_MOVES,
    SEQUENCE_MOVES,
    KeyMove,
    KeyRepresentation,
    Representation,
    SequenceCrossover,
    SequenceMove,
    SequenceRepresentation,
)

__all__ = [
    "REPRESENTATIONS",
    "RepresentationName",
    "check_family",
    "check_moves",
    "make_representation",
]

type RepresentationName = Literal["keys", "sequence"]
"""The name of a representation of a job-shop search family, one of :data:`REPRESENTATIONS`."""

REPRESENTATIONS: tuple[RepresentationName, ...] = ("keys", "sequence")
"""The representations a job-shop search family searches: random keys and job sequences."""


def make_representation(
    shop: JobShop,
    representation: RepresentationName,
    *,
    key_move: KeyMove = "perturb",
    key_step: float = 0.1,
    sequence_move: SequenceMove = "swap",
    crossover: SequenceCrossover = "ppx",
) -> Representation[Any]:
    """Return the representation of ``shop`` that ``representation`` names.

    ``"keys"`` gives ``KeyRepresentation(shop, key_move, key_step)``, which
    changes as many keys as its kind of move does, and ``"sequence"`` gives
    ``SequenceRepresentation(shop, sequence_move, crossover)``. Each reads its
    own arguments only, and the defaults are those of the two classes.

    Args:
        shop: The job shop.
        representation: ``"keys"`` or ``"sequence"``.
        key_move: The key move, ``"perturb"`` or ``"redraw"``.
        key_step: The standard deviation of a key perturbation.
        sequence_move: The sequence move, ``"swap"``, ``"insert"`` or ``"mixed"``.
        crossover: The crossover of job sequences, ``"ppx"`` or ``"jox"``.

    Returns:
        The representation.

    Raises:
        ValueError: If a name is unknown or ``key_step`` is not positive and finite.
    """
    if representation == "keys":
        return KeyRepresentation(shop, key_move, key_step)
    if representation == "sequence":
        return SequenceRepresentation(shop, sequence_move, crossover)
    raise ValueError(_unknown("representation", representation, REPRESENTATIONS))


def check_family(representation: object, builder: object) -> None:
    """Raise :class:`ValueError` unless both name a representation and a schedule builder.

    Args:
        representation: One of :data:`REPRESENTATIONS`.
        builder: One of :data:`~dispatchatlas.solvers.shop.decoding.BUILDERS`.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(_unknown("representation", representation, REPRESENTATIONS))
    if builder not in BUILDERS:
        raise ValueError(_unknown("builder", builder, BUILDERS))


def check_moves(key_move: object, key_step: object, sequence_move: object) -> None:
    """Raise :class:`ValueError` unless the moves of both representations are valid.

    A family checks both, whichever representation it searches, so that one
    configuration is valid or invalid for both of its runs.

    Args:
        key_move: One of :data:`~dispatchatlas.solvers.shop.representation.KEY_MOVES`.
        key_step: A positive and finite number.
        sequence_move: One of
            :data:`~dispatchatlas.solvers.shop.representation.SEQUENCE_MOVES`.
    """
    if key_move not in KEY_MOVES:
        raise ValueError(_unknown("key move", key_move, KEY_MOVES))
    if (
        isinstance(key_step, bool)
        or not isinstance(key_step, int | float)
        or not (math.isfinite(key_step) and key_step > 0.0)
    ):
        raise ValueError(f"key_step must be positive and finite, got {key_step!r}")
    if sequence_move not in SEQUENCE_MOVES:
        raise ValueError(_unknown("sequence move", sequence_move, SEQUENCE_MOVES))


def _unknown(what: str, value: object, known: tuple[str, ...]) -> str:
    """Return the message that refuses an unknown name."""
    return f"unknown {what} {value!r}; expected one of {list(known)}"
