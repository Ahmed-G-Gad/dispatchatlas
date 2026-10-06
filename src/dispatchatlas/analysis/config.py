"""The settings of an analysis.

:class:`AnalysisConfig` holds every choice that changes an analysis result:
what to analyse (protocol, objective, solvers, problems), how the
per-problem values are formed, the significance and confidence levels, the
bootstrap, the multiple-comparison correction, the omnibus test, the
multi-objective and deviation references, and the focal solver compared with
its baselines. The bundle manifest records all of it, so an analysis can be
repeated from the manifest alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from dispatchatlas.analysis.stats.bootstrap import IntervalMethod
from dispatchatlas.analysis.stats.corrections import CORRECTIONS, Correction
from dispatchatlas.analysis.stats.ranks import Sense, check_sense
from dispatchatlas.core.errors import ConfigError

__all__ = ["AnalysisConfig", "CellStatistic", "Omnibus", "ReferenceRule", "TimeoutScore"]

type CellStatistic = Literal["mean", "median"]
"""How the replicates of one solver on one problem become one value."""

type Omnibus = Literal["iman-davenport", "friedman"]
"""The omnibus test whose p-value decides whether the post-hoc test runs."""

type ReferenceRule = Literal["margin", "ishibuchi"]
"""How a hypervolume reference point is derived from the reference front."""

type TimeoutScore = Literal["worst", "incumbent"]
"""How a run that was killed at its hard limit is scored."""


@dataclass(frozen=True)
class AnalysisConfig:
    """Settings of an analysis. Every field has a default.

    Attributes:
        protocol: Protocol to analyse. ``None`` analyses the campaign's only
            protocol; a bundle then analyses every protocol in turn.
        objective: Objective to compare solvers on. ``None`` takes the
            primary objective, which the analysed problems must share.
        sense: ``"min"`` or ``"max"``. ``None`` takes the sense the objective
            registry declares.
        solvers: Names of the solvers to analyse, or ``None`` for all.
        problems: Ids of the problems to analyse, or ``None`` for all.
        alpha: Significance level of the tests and corrections.
        confidence: Coverage of the confidence intervals.
        resamples: Bootstrap resamples per interval.
        seed: Seed of the bootstrap and of the power simulation.
        interval: Bootstrap interval, ``"bca"`` or ``"percentile"``.
        correction: Multiple-comparison correction of the pairwise tests.
        omnibus: Omnibus test that gates the Nemenyi post-hoc test.
        statistic: How the replicates of a cell become one value, where a
            failed run counts as the worst value: ``"mean"`` makes any
            failure decisive, ``"median"`` needs failures in half of the
            runs.
        replicate_floor: Fewest replicates per cell that the design should
            have for stochastic solvers. Cells below it are reported as a
            limitation; nothing is excluded.
        power: Target power of the detectable-effect statement.
        references: Reference value, such as a best-known objective value,
            of each problem for the relative percentage deviation.
        reference_source: Where the reference values come from, shown with
            the deviation table.
        reference_points: Hypervolume reference point of a problem, in
            objective units and minimisation form, overriding the rule.
        reference_rule: Rule deriving the reference point from the
            reference front: ``"margin"`` puts it ``reference_parameter``
            spans beyond the nadir, ``"ishibuchi"`` at
            ``ideal + (1 + 1/h) (nadir - ideal)`` with ``h =
            reference_parameter``.
        reference_parameter: The margin (positive) or ``h`` (a positive
            integer).
        normalize: Compute indicators on objectives scaled so that the
            reference front's ideal is 0 and its nadir is 1.
        focal: Name of a solver under study. A bundle then also compares it
            with each baseline, adjusting the p-values over these comparisons
            only (see :func:`~dispatchatlas.analysis.comparison.compare_to_baselines`).
        baselines: Names of the baselines of ``focal``, or ``None`` for every
            other analysed solver.
        timeout_score: How a run that never completed and was killed at its
            hard limit is scored. ``"worst"`` scores it as the worst value,
            like any other failure. ``"incumbent"`` scores it by the
            incumbent its search held when the time limit of its protocol
            passed, the last one it had reported by then, when that
            incumbent is feasible. An incumbent found after the time limit
            is never a score, so a run that overran is compared on the same
            seconds as one that stopped on time. The value is the solver's
            own, with no schedule to check it against. The limitations count
            these runs under either choice.
    """

    protocol: str | None = None
    objective: str | None = None
    sense: Sense | None = None
    solvers: tuple[str, ...] | None = None
    problems: tuple[str, ...] | None = None
    alpha: float = 0.05
    confidence: float = 0.95
    resamples: int = 1000
    seed: int = 0
    interval: IntervalMethod = "bca"
    correction: Correction = "holm"
    omnibus: Omnibus = "iman-davenport"
    statistic: CellStatistic = "mean"
    replicate_floor: int | None = None
    power: float = 0.8
    references: Mapping[str, float] | None = field(default=None, hash=False)
    reference_source: str | None = None
    reference_points: Mapping[str, tuple[float, ...]] | None = field(default=None, hash=False)
    reference_rule: ReferenceRule = "margin"
    reference_parameter: float = 0.1
    normalize: bool = True
    focal: str | None = None
    baselines: tuple[str, ...] | None = None
    timeout_score: TimeoutScore = "worst"

    def __post_init__(self) -> None:
        for name in ("protocol", "objective", "reference_source", "focal"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"{name} must be a non-empty string or None, got {value!r}")
        if self.sense is not None:
            object.__setattr__(self, "sense", check_sense(self.sense))
        for name in ("solvers", "problems", "baselines"):
            object.__setattr__(self, name, _names(getattr(self, name), name))
        if self.baselines is not None and self.focal is None:
            raise ConfigError("baselines need a focal solver to compare with them")
        if self.focal is not None and self.focal in (self.baselines or ()):
            raise ConfigError(f"solver {self.focal!r} cannot be its own baseline")
        for name in ("alpha", "confidence", "power"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{name} must be a number, got {value!r}")
            if not 0.0 < value < 1.0:
                raise ConfigError(f"{name} must lie strictly between 0 and 1, got {value!r}")
            object.__setattr__(self, name, float(value))
        _integer(self.resamples, "resamples", 0)
        _integer(self.seed, "seed", 0)
        if self.replicate_floor is not None:
            _integer(self.replicate_floor, "replicate_floor", 1)
        _choice(self.interval, "interval", ("bca", "percentile"))
        _choice(self.correction, "correction", CORRECTIONS)
        _choice(self.omnibus, "omnibus", ("iman-davenport", "friedman"))
        _choice(self.statistic, "statistic", ("mean", "median"))
        _choice(self.timeout_score, "timeout_score", ("worst", "incumbent"))
        _choice(self.reference_rule, "reference_rule", ("margin", "ishibuchi"))
        parameter = self.reference_parameter
        if isinstance(parameter, bool) or not isinstance(parameter, (int, float)):
            raise ConfigError(f"reference_parameter must be a number, got {parameter!r}")
        if not (math.isfinite(parameter) and parameter > 0.0):
            raise ConfigError(f"reference_parameter must be positive, got {parameter!r}")
        if self.reference_rule == "ishibuchi" and parameter != int(parameter):
            raise ConfigError(f"the ishibuchi rule needs an integer h >= 1, got {parameter!r}")
        object.__setattr__(self, "reference_parameter", float(parameter))
        if not isinstance(self.normalize, bool):
            raise ConfigError(f"normalize must be true or false, got {self.normalize!r}")
        object.__setattr__(self, "references", _references(self.references))
        object.__setattr__(self, "reference_points", _points(self.reference_points))

    def to_dict(self) -> dict[str, Any]:
        """Return every setting as a JSON-ready dictionary."""
        return {
            "protocol": self.protocol,
            "objective": self.objective,
            "sense": self.sense,
            "solvers": None if self.solvers is None else list(self.solvers),
            "problems": None if self.problems is None else list(self.problems),
            "alpha": self.alpha,
            "confidence": self.confidence,
            "resamples": self.resamples,
            "seed": self.seed,
            "interval": self.interval,
            "correction": self.correction,
            "omnibus": self.omnibus,
            "statistic": self.statistic,
            "replicate_floor": self.replicate_floor,
            "power": self.power,
            "references": None if self.references is None else dict(self.references),
            "reference_source": self.reference_source,
            "reference_points": None
            if self.reference_points is None
            else {key: list(point) for key, point in self.reference_points.items()},
            "reference_rule": self.reference_rule,
            "reference_parameter": self.reference_parameter,
            "normalize": self.normalize,
            "focal": self.focal,
            "baselines": None if self.baselines is None else list(self.baselines),
            "timeout_score": self.timeout_score,
        }


def _names(value: object, what: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigError(f"{what} must be a list of names or None")
    names = tuple(value)
    if not names:
        raise ConfigError(f"{what} must name at least one entry, or be None for all")
    for name in names:
        if not isinstance(name, str) or not name:
            raise ConfigError(f"{what} must hold non-empty strings, got {name!r}")
    if len(set(names)) != len(names):
        raise ConfigError(f"{what} names an entry twice")
    return names


def _integer(value: object, what: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{what} must be an integer of at least {minimum}, got {value!r}")


def _choice(value: object, what: str, allowed: Sequence[str]) -> None:
    if value not in allowed:
        raise ConfigError(f"{what} must be one of {list(allowed)}, got {value!r}")


def _references(value: object) -> Mapping[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigError("references must map problem ids to values")
    frozen: dict[str, float] = {}
    for key in sorted(value):
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ConfigError(f"the reference value of {key!r} must be a number")
        if not (math.isfinite(number) and number > 0.0):
            raise ConfigError(
                f"the reference value of {key!r} must be positive for a percentage deviation, "
                f"got {number!r}"
            )
        frozen[str(key)] = float(number)
    return MappingProxyType(frozen)


def _points(value: object) -> Mapping[str, tuple[float, ...]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigError("reference_points must map problem ids to points")
    frozen: dict[str, tuple[float, ...]] = {}
    for key in sorted(value):
        point = value[key]
        if isinstance(point, str) or not isinstance(point, Sequence) or len(point) < 2:
            raise ConfigError(f"the reference point of {key!r} needs one value per objective")
        coordinates = []
        for item in point:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ConfigError(f"the reference point of {key!r} must hold numbers")
            if not math.isfinite(item):
                raise ConfigError(f"the reference point of {key!r} must be finite")
            coordinates.append(float(item))
        frozen[str(key)] = tuple(coordinates)
    return MappingProxyType(frozen)
