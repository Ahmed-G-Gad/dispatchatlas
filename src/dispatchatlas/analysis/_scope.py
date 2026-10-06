"""The part of a dataset one analysis covers: a protocol, an objective, its sense."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from dispatchatlas.analysis.config import AnalysisConfig, TimeoutScore
from dispatchatlas.analysis.dataset import Dataset, Run
from dispatchatlas.analysis.stats.ranks import Sense, check_sense
from dispatchatlas.core.errors import ConfigError, ModelError
from dispatchatlas.core.objectives import get_objective

__all__ = ["Scope", "aggregate", "narrow", "resolve_scope", "run_scores", "worst_value"]


@dataclass(frozen=True)
class Scope:
    """A dataset narrowed to one protocol and the configured solvers and problems.

    Attributes:
        dataset: The narrowed dataset.
        protocol: The protocol.
        objective: The objective to compare on.
        sense: Its sense.
    """

    dataset: Dataset
    protocol: str
    objective: str
    sense: Sense


def narrow(dataset: Dataset, config: AnalysisConfig) -> tuple[Dataset, str]:
    """Return ``dataset`` narrowed to the configured protocol, solvers, and problems.

    Returns:
        The narrowed dataset and the protocol's name.

    Raises:
        ConfigError: If a name is unknown, the protocol is ambiguous, or
            nothing is planned in the selection.
    """
    if config.protocol is None:
        names = [protocol.name for protocol in dataset.protocols]
        if len(names) != 1:
            raise ConfigError(
                f"the campaign has {len(names)} protocols ({', '.join(names)}); choose one"
            )
        protocol = names[0]
    else:
        protocol = config.protocol
    narrowed = dataset.select(
        protocols=[protocol], solvers=config.solvers, problems=config.problems
    )
    if not narrowed.runs:
        raise ConfigError(f"no run is planned for protocol {protocol!r} and the selection")
    return narrowed, protocol


def resolve_scope(dataset: Dataset, config: AnalysisConfig) -> Scope:
    """Narrow ``dataset`` as ``config`` asks and settle the protocol, objective, and sense.

    Raises:
        ConfigError: If a name is unknown, the protocol or objective is
            ambiguous, a problem does not declare the objective, the sense
            cannot be determined, or nothing is planned in the scope.
    """
    narrowed, protocol = narrow(dataset, config)
    objective = config.objective
    if objective is None:
        primaries = sorted({problem.objectives[0] for problem in narrowed.problems})
        if len(primaries) != 1:
            raise ConfigError(
                f"the problems have different primary objectives ({', '.join(primaries)}); "
                "choose one"
            )
        objective = primaries[0]
    for problem in narrowed.problems:
        if objective not in problem.objectives:
            raise ConfigError(f"problem {problem.id!r} does not declare objective {objective!r}")
    if config.sense is not None:
        sense = config.sense
    else:
        try:
            sense = check_sense(get_objective(objective).sense)
        except ModelError as error:
            raise ConfigError(f"{error}; give its sense, 'min' or 'max'") from error
    return Scope(narrowed, protocol, objective, sense)


def worst_value(sense: Sense) -> float:
    """Return the score of a failed run: the worst value under ``sense``."""
    return math.inf if sense == "min" else -math.inf


def run_scores(
    runs: Sequence[Run], objective: str, sense: Sense, timeout_score: TimeoutScore = "worst"
) -> list[float]:
    """Return each planned run's value, a run that did not succeed scoring the worst value.

    With ``timeout_score`` ``"incumbent"``, a run killed at its hard limit
    scores the value of the feasible incumbent it held when its time limit
    passed, if it held one with a value for the objective. What its search
    found after the time limit is not scored.

    Raises:
        ConfigError: If a succeeded run has no value for the objective.
    """
    worst = worst_value(sense)
    scores = []
    for run in runs:
        if not run.succeeded:
            held = run.incumbent if timeout_score == "incumbent" else None
            kept = None if held is None else held.objectives.get(objective)
            scores.append(worst if kept is None else kept)
            continue
        value = run.value(objective)
        if value is None:
            raise ConfigError(f"run {run.run_id} records no value for objective {objective!r}")
        scores.append(value)
    return scores


def aggregate(scores: Sequence[float], statistic: str) -> float:
    """Return the mean or median of scores that may hold one kind of infinity."""
    if statistic == "median":
        return float(statistics.median(scores))
    if any(math.isinf(score) for score in scores):
        return next(score for score in scores if math.isinf(score))
    return math.fsum(scores) / len(scores)
