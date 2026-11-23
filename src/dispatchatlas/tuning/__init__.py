"""Automatic configuration of a solver's parameters by iterated racing.

- :mod:`~dispatchatlas.tuning.space`: parameter spaces of real, integer, ordinal, and
  categorical parameters, the configurations they hold, and the sampling distributions
  of iterated racing.
- :mod:`~dispatchatlas.tuning.racing`: :func:`race`, which evaluates configurations on
  instances one after another and eliminates the worse ones by the Friedman test and
  Conover's comparisons with the best, and :func:`compare_with_best`, the test itself.
- :mod:`~dispatchatlas.tuning.irace`: :class:`IteratedRacing`, the elitist iterated racing
  of the irace package (López-Ibáñez et al. 2016), and the formulas it applies.

The package never starts a solver. The caller passes ``run_batch``, a function that takes
the experiments of one step, each a configuration, an instance, and a seed, and returns
their costs, so the caller decides how the runs are made and where they are recorded.
The package depends on :mod:`dispatchatlas.core` and on the statistics of
:mod:`dispatchatlas.analysis.stats`.
"""

from dispatchatlas.tuning.irace import (
    IteratedRacing,
    TuningResult,
    candidate_count,
    default_iterations,
    default_survivors,
    iteration_budget,
    narrow_model,
    parent_probabilities,
    restart_model,
)
from dispatchatlas.tuning.racing import (
    Block,
    Comparison,
    Entry,
    Experiment,
    RaceResult,
    Record,
    RunBatch,
    StopReason,
    compare_with_best,
    race,
)
from dispatchatlas.tuning.space import (
    DIGITS,
    Categorical,
    Configuration,
    Integer,
    Ordinal,
    Parameter,
    ParameterSpace,
    Real,
    SamplingModel,
    Value,
)

__all__ = [
    "DIGITS",
    "Block",
    "Categorical",
    "Comparison",
    "Configuration",
    "Entry",
    "Experiment",
    "Integer",
    "IteratedRacing",
    "Ordinal",
    "Parameter",
    "ParameterSpace",
    "RaceResult",
    "Real",
    "Record",
    "RunBatch",
    "SamplingModel",
    "StopReason",
    "TuningResult",
    "Value",
    "candidate_count",
    "compare_with_best",
    "default_iterations",
    "default_survivors",
    "iteration_budget",
    "narrow_model",
    "parent_probabilities",
    "race",
    "restart_model",
]
