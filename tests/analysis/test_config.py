"""Validation and serialization of the analysis configuration."""

from __future__ import annotations

import json

import pytest

from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.core.errors import ConfigError


def test_defaults_and_their_dictionary():
    config = AnalysisConfig()
    assert (config.alpha, config.confidence, config.resamples, config.seed) == (0.05, 0.95, 1000, 0)
    assert (config.correction, config.omnibus, config.interval) == ("holm", "iman-davenport", "bca")
    document = config.to_dict()
    assert json.loads(json.dumps(document)) == document
    assert document["references"] is None


def test_values_are_normalised():
    config = AnalysisConfig(
        solvers=["b", "a"],  # type: ignore[arg-type]
        references={"p2": 2, "p1": 1.5},
        reference_points={"p1": [3, 4]},  # type: ignore[dict-item]
        alpha=0.1,
    )
    assert config.solvers == ("b", "a")
    assert list(config.references or {}) == ["p1", "p2"]
    assert config.reference_points == {"p1": (3.0, 4.0)}
    assert config.to_dict()["reference_points"] == {"p1": [3.0, 4.0]}


@pytest.mark.parametrize(
    "options",
    [
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"confidence": 1.5},
        {"power": 0.0},
        {"resamples": -1},
        {"seed": -1},
        {"replicate_floor": 0},
        {"interval": "basic"},
        {"correction": "sidak"},
        {"omnibus": "quade"},
        {"statistic": "best"},
        {"sense": "minimise"},
        {"protocol": ""},
        {"solvers": ()},
        {"solvers": ("a", "a")},
        {"problems": "p1"},
        {"references": {"p1": 0.0}},
        {"reference_points": {"p1": (1.0,)}},
        {"reference_points": {"p1": (1.0, float("nan"))}},
        {"reference_rule": "nadir"},
        {"reference_parameter": 0.0},
        {"reference_rule": "ishibuchi", "reference_parameter": 2.5},
        {"normalize": 1},
        {"focal": ""},
        {"baselines": ("a",)},
        {"focal": "a", "baselines": ("a", "b")},
        {"focal": "a", "baselines": ()},
    ],
)
def test_invalid_values_are_refused(options):
    with pytest.raises(ConfigError):
        AnalysisConfig(**options)
