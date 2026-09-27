"""The method catalog: every entry is implemented and cites known sources."""

from __future__ import annotations

import importlib

import pytest

from dispatchatlas.analysis.methods import METHOD_FAMILIES, implemented_methods, method
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.references import bibliography


@pytest.mark.parametrize("entry", implemented_methods(), ids=lambda entry: entry.id)
def test_every_method_resolves_its_function_and_references(entry):
    module_name, _, attribute = entry.function.rpartition(".")
    assert hasattr(importlib.import_module(module_name), attribute)
    assert entry.references
    assert all(key in bibliography() for key in entry.references)
    assert entry.family in METHOD_FAMILIES
    assert entry.summary.endswith(".")


def test_ids_are_unique_and_families_filter():
    identifiers = [entry.id for entry in implemented_methods()]
    assert len(identifiers) == len(set(identifiers))
    assert {entry.id for entry in implemented_methods("correction")} == {
        "bonferroni",
        "holm",
        "hochberg",
        "shaffer",
        "finner",
    }
    assert method("nemenyi").family == "post-hoc"
    with pytest.raises(ConfigError, match="unknown method"):
        method("t-test")
    with pytest.raises(ConfigError, match="unknown method family"):
        implemented_methods("parametric")  # type: ignore[arg-type]
