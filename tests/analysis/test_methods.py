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


def test_the_equivalence_and_zero_treatment_methods_are_catalogued_with_their_sources():
    tests = {entry.id for entry in implemented_methods("test")}
    assert {"tost-paired-t", "tost-signed-rank", "wilcoxon-pratt-zeros"} <= tests
    assert method("equivalence-report").family == "reporting"
    assert method("tost-paired-t").function.endswith(".equivalence.tost_paired_t")
    assert method("tost-signed-rank").function.endswith(".equivalence.tost_signed_rank")
    assert method("wilcoxon-pratt-zeros").function.endswith(".nonparametric.wilcoxon_signed_rank")
    for identifier in ("tost-paired-t", "tost-signed-rank", "equivalence-report"):
        assert "lakens2017equivalence" in method(identifier).references
    for identifier in ("tost-paired-t", "tost-signed-rank"):
        assert "schuirmann1987comparison" in method(identifier).references
    for identifier in ("wilcoxon-pratt-zeros", "tost-signed-rank"):
        assert "pratt1959remarks" in method(identifier).references
    # The entry that every bundle writes keeps its text.
    assert method("wilcoxon-signed-rank").references == (
        "wilcoxon1945individual",
        "derrac2011practical",
    )
