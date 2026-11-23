"""The reference bounds: the rules of the loader, and the values recorded for Taillard's job shops.

``PRINTED`` is a second transcription of the values of ta01 to ta50 as two published sources
print them: Peng, Lü and Cheng (2015, Table 8, p. 160, column UB(LB)) and Gonçalves and Resende
(2014, Table A7, pp. 243-244, columns UB and LB), each with the upper bound the source reports
as newly found (printed in bold) where it has one. The data file must agree with it value by
value, and its reference values must follow the rule of
:mod:`dispatchatlas.benchmarks.bounds` from the printed values alone.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from dispatchatlas.analysis import AnalysisConfig
from dispatchatlas.benchmarks import bounds, suites
from dispatchatlas.benchmarks.bounds import load_bounds, reference_values
from dispatchatlas.benchmarks.suites import Bound, BoundSource
from dispatchatlas.core import ConfigError
from dispatchatlas.references import bibliography

SUITE = "taillard-job-shop"
INSTANCES = suites.suite(SUITE).instances
PENG = "peng2015tabu"
GR = "goncalves2014extended"

PRINTED = """
ta01 1231 1231 - 1231 1231 - 243
ta02 1244 1244 - 1244 1244 - 243
ta03 1218 1218 - 1218 1218 - 243
ta04 1175 1175 - 1175 1175 - 243
ta05 1224 1224 - 1224 1224 - 243
ta06 1238 1238 - 1238 1238 - 243
ta07 1227 1227 - 1227 1227 - 243
ta08 1217 1217 - 1217 1217 - 243
ta09 1274 1274 - 1274 1274 - 243
ta10 1241 1241 - 1241 1241 - 243
ta11 1357 1323 - 1357 1323 - 243
ta12 1367 1351 - 1367 1351 - 243
ta13 1342 1282 - 1342 1282 - 243
ta14 1345 1345 - 1345 1345 - 243
ta15 1339 1304 - 1339 1304 - 243
ta16 1360 1304 - 1360 1302 - 244
ta17 1462 1462 - 1462 1462 - 244
ta18 1396 1369 - 1396 1369 - 244
ta19 1332 1304 - 1332 1297 - 244
ta20 1348 1318 - 1348 1318 - 244
ta21 1642 1573 - 1643 1539 1642 244
ta22 1600 1542 - 1600 1511 - 244
ta23 1557 1474 - 1557 1472 - 244
ta24 1644 1606 - 1646 1602 - 244
ta25 1595 1518 - 1595 1504 - 244
ta26 1643 1558 - 1645 1539 1643 244
ta27 1680 1617 - 1680 1616 - 244
ta28 1603 1591 - 1603 1591 - 244
ta29 1625 1525 - 1625 1514 - 244
ta30 1584 1485 - 1584 1473 - 244
ta31 1764 1764 - 1764 1764 - 244
ta32 1785 1774 - 1790 1774 1785 244
ta33 1791 1778 - 1791 1778 - 244
ta34 1829 1828 - 1829 1828 - 244
ta35 2007 2007 - 2007 2007 - 244
ta36 1819 1819 - 1819 1819 - 244
ta37 1771 1771 - 1771 1771 - 244
ta38 1673 1673 - 1673 1673 - 244
ta39 1795 1795 - 1795 1795 - 244
ta40 1669 1631 - 1673 1631 1669 244
ta41 2006 1874 - 2006 1859 - 244
ta42 1937 1867 - 1945 1867 1937 244
ta43 1848 1809 1846 1848 1809 - 244
ta44 1983 1927 1982 1983 1927 - 244
ta45 2000 1997 - 2000 1997 - 244
ta46 2004 1940 - 2008 1940 2004 244
ta47 1894 1789 1889 1897 1789 1894 244
ta48 1943 1912 - 1945 1912 1943 244
ta49 1964 1915 1963 1966 1915 1964 244
ta50 1924 1807 1923 1925 1807 - 244
"""
"""One row per instance: the name; then for Peng et al. the printed upper bound, the printed
lower bound and the bound it found ("-" for none); then the same three for Goncalves and
Resende; then the page of their Table A7."""


class Row(NamedTuple):
    peng_upper: int
    peng_lower: int
    peng_found: int | None
    gr_upper: int
    gr_lower: int
    gr_found: int | None
    gr_page: int


def _rows() -> dict[str, Row]:
    rows: dict[str, Row] = {}
    for line in PRINTED.strip().splitlines():
        name, *fields = line.split()
        peng_upper, peng_lower, peng_found, gr_upper, gr_lower, gr_found, page = fields
        rows[name] = Row(
            int(peng_upper),
            int(peng_lower),
            None if peng_found == "-" else int(peng_found),
            int(gr_upper),
            int(gr_lower),
            None if gr_found == "-" else int(gr_found),
            int(page),
        )
    return rows


ROWS = _rows()
OPTIMAL = {f"ta{number:02d}" for number in (*range(1, 11), 14, 17, 31, *range(35, 40))}
PENG_FOUND = {"ta43", "ta44", "ta47", "ta49", "ta50"}
GR_FOUND = {"ta21", "ta26", "ta32", "ta40", "ta42", "ta46", "ta47", "ta48", "ta49"}


def _smallest(row: Row) -> int:
    values = [row.peng_upper, row.gr_upper, row.peng_found, row.gr_found]
    return min(value for value in values if value is not None)


def _largest(row: Row) -> int:
    return max(row.peng_lower, row.gr_lower)


# ---------------------------------------------------------------- the values


def test_the_transcription_holds_ta01_to_ta50() -> None:
    assert list(ROWS) == [f"ta{number:02d}" for number in range(1, 51)]
    assert {name for name, row in ROWS.items() if row.peng_found is not None} == PENG_FOUND
    assert {name for name, row in ROWS.items() if row.gr_found is not None} == GR_FOUND
    assert {name for name, row in ROWS.items() if _smallest(row) == _largest(row)} == OPTIMAL


def test_the_recorded_instances_are_ta01_to_ta50() -> None:
    recorded = load_bounds(SUITE)
    assert list(recorded) == list(ROWS)
    assert len(recorded) == 50
    assert not {"ta51", "ta65", "ta80"} & set(recorded)


def test_every_value_matches_what_the_sources_print() -> None:
    recorded = load_bounds(SUITE)
    for name, row in ROWS.items():
        peng = "Table 8, p. 160"
        gr = f"Table A7, p. {row.gr_page}"
        uppers = {
            BoundSource(PENG, float(row.peng_upper), peng),
            BoundSource(GR, float(row.gr_upper), gr),
        }
        if row.peng_found is not None:
            uppers.add(BoundSource(PENG, float(row.peng_found), peng, found=True))
        if row.gr_found is not None:
            uppers.add(BoundSource(GR, float(row.gr_found), gr, found=True))
        lowers = {
            BoundSource(PENG, float(row.peng_lower), peng),
            BoundSource(GR, float(row.gr_lower), gr),
        }
        bound = recorded[name]
        assert set(bound.upper_sources) == uppers, name
        assert len(bound.upper_sources) == len(uppers), name
        assert set(bound.lower_sources) == lowers, name
        assert len(bound.lower_sources) == 2, name
        assert bound.objective == "makespan"


def test_the_reference_values_are_the_smallest_upper_and_the_largest_lower_bound() -> None:
    recorded = load_bounds(SUITE)
    for name, row in ROWS.items():
        bound = recorded[name]
        assert bound.upper == _smallest(row), name
        assert bound.lower == _largest(row), name
        assert bound.optimal is (_smallest(row) == _largest(row)), name
        assert bound.lower is not None
        assert bound.upper is not None
        assert bound.lower <= bound.upper
        assert bound.upper == min(item.value for item in bound.upper_sources)
        assert bound.lower == max(item.value for item in bound.lower_sources)
    assert dict(reference_values(SUITE, "upper")) == {
        n: float(_smallest(r)) for n, r in ROWS.items()
    }
    assert dict(reference_values(SUITE, "lower")) == {
        n: float(_largest(r)) for n, r in ROWS.items()
    }


def test_exactly_eighteen_instances_are_optimal() -> None:
    optimal = {name for name, bound in load_bounds(SUITE).items() if bound.optimal}
    assert optimal == OPTIMAL
    assert len(optimal) == 18
    for name in optimal:
        bound = load_bounds(SUITE)[name]
        assert bound.upper == bound.lower
        assert {item.value for item in bound.upper_sources} == {bound.upper}
        assert {item.value for item in bound.lower_sources} == {bound.lower}


def test_the_sources_compare_as_the_design_reads_them() -> None:
    same = {n for n, r in ROWS.items() if (r.peng_upper, r.peng_lower) == (r.gr_upper, r.gr_lower)}
    upper_only = {
        n for n, r in ROWS.items() if r.peng_upper != r.gr_upper and r.peng_lower == r.gr_lower
    }
    lower_only = {
        n for n, r in ROWS.items() if r.peng_upper == r.gr_upper and r.peng_lower != r.gr_lower
    }
    assert len(same) == 30
    assert len(upper_only) == 8
    assert len(lower_only) == 9
    assert len(ROWS) - len(same) - len(upper_only) - len(lower_only) == 3
    for row in ROWS.values():
        assert row.peng_upper <= row.gr_upper
        assert row.peng_lower >= row.gr_lower
    # The later source's own values give the reference values: its newly found bound if it has
    # one, else the upper bound it lists, and the lower bound it lists.
    recorded = load_bounds(SUITE)
    for name, row in ROWS.items():
        assert recorded[name].upper == (row.peng_found or row.peng_upper), name
        assert recorded[name].lower == row.peng_lower, name


def test_a_value_that_a_source_found_is_below_the_value_it_lists() -> None:
    for bound in load_bounds(SUITE).values():
        listed = {item.source: item.value for item in bound.upper_sources if not item.found}
        for item in bound.upper_sources:
            if item.found:
                assert item.value < listed[item.source]
        assert not any(item.found for item in bound.lower_sources)


def test_the_sources_are_entries_of_the_bibliography() -> None:
    assert bibliography()[PENG].venue.startswith("Computers & Operations Research 53:")
    assert bibliography()[GR].venue.startswith(
        "International Transactions in Operational Research 21(2):"
    )
    for bound in load_bounds(SUITE).values():
        assert {item.source for item in bound.upper_sources + bound.lower_sources} == {PENG, GR}


def test_the_suite_and_its_own_bounds_are_unchanged() -> None:
    assert suites.suite(SUITE).bounds == {}
    assert len(suites.suite(SUITE).instances) == 80
    assert suites.suite(SUITE).references == ("taillard1993benchmarks", "beasley1990orlibrary")


def test_a_bound_without_sources_is_what_it_was() -> None:
    assert Bound().upper_sources == () == Bound().lower_sources
    assert Bound(186, 193, True) == Bound(186, 193, True, "makespan", (), ())
    for bound in suites.suite("taillard-open-shop").bounds.values():
        assert bound.upper_sources == () == bound.lower_sources


def test_the_range_is_stated_in_the_docstring_and_in_the_data_file() -> None:
    assert bounds.__doc__ is not None
    assert "ta51 to ta80 are not included" in " ".join(bounds.__doc__.split())
    text = (
        files("dispatchatlas.data").joinpath("bounds", f"{SUITE}.toml").read_text(encoding="utf-8")
    )
    header = text[: text.index("\n[")]
    assert "ta51 to ta80 are not included" in " ".join(header.replace("#", " ").split())


def test_the_reference_values_feed_the_analysis_config() -> None:
    upper = reference_values(SUITE, "upper")
    config = AnalysisConfig(references=upper, reference_source="two sources")
    assert config.references is not None
    assert dict(config.references) == dict(upper)
    assert config.references["ta01"] == 1231.0
    assert config.references["ta43"] == 1846.0
    assert len(config.references) == 50
    assert AnalysisConfig(references=reference_values(SUITE, "lower")).references is not None


def test_the_values_are_read_only() -> None:
    recorded = load_bounds(SUITE)
    with pytest.raises(TypeError):
        recorded["ta01"] = Bound()  # type: ignore[index]
    with pytest.raises(TypeError):
        reference_values(SUITE, "upper")["ta01"] = 1.0  # type: ignore[index]
    assert load_bounds(SUITE) is recorded


def test_unknown_suites_kinds_and_suites_without_bounds_are_refused() -> None:
    with pytest.raises(ConfigError, match="unknown suite 'nobody'"):
        load_bounds("nobody")
    with pytest.raises(
        ConfigError, match="no reference bounds are recorded for suite 'taillard-flow-shop'"
    ):
        load_bounds("taillard-flow-shop")
    with pytest.raises(ConfigError, match=r"recorded: .*taillard-job-shop"):
        load_bounds("taillard-open-shop")
    for kind in ("middle", "", None, 1):
        with pytest.raises(ConfigError, match="kind must be 'upper' or 'lower'"):
            reference_values(SUITE, kind)  # type: ignore[arg-type]


# ----------------------------------------------------------- the rules of the loader


def _ta11() -> dict[str, Any]:
    """A valid table: ta11, whose two sources print the same values."""
    return {
        "upper": 1357,
        "lower": 1323,
        "optimal": False,
        PENG: {"upper": 1357, "lower": 1323, "where": "Table 8, p. 160"},
        GR: {"upper": 1357, "lower": 1323, "where": "Table A7, p. 243"},
    }


def _parse(document: Mapping[str, Any]) -> Mapping[str, Bound]:
    return bounds._parse(SUITE, document, INSTANCES)


def test_a_valid_document_loads() -> None:
    (bound,) = _parse({"ta11": _ta11()}).values()
    assert bound == Bound(
        1323.0,
        1357.0,
        False,
        "makespan",
        (BoundSource(PENG, 1357.0, "Table 8, p. 160"), BoundSource(GR, 1357.0, "Table A7, p. 243")),
        (BoundSource(PENG, 1323.0, "Table 8, p. 160"), BoundSource(GR, 1323.0, "Table A7, p. 243")),
    )


def test_a_source_that_found_a_bound_gives_a_second_upper_value() -> None:
    table = _ta11()
    table["upper"] = 1350
    table[GR]["found_upper"] = 1350
    bound = _parse({"ta11": table})["ta11"]
    assert bound.upper == 1350.0
    assert bound.upper_sources[-1] == BoundSource(GR, 1350.0, "Table A7, p. 243", found=True)
    assert len(bound.upper_sources) == 3
    assert len(bound.lower_sources) == 2


def test_a_lower_bound_above_the_upper_bound_is_refused() -> None:
    table = _ta11()
    table["lower"] = 1400
    table[PENG]["lower"] = table[GR]["lower"] = 1400
    with pytest.raises(ConfigError, match=r"ta11.*lower bound 1400 is above the upper bound 1357"):
        _parse({"ta11": table})


@pytest.mark.parametrize(
    ("upper", "lower", "optimal"),
    [(1357, 1357, False), (1357, 1323, True)],
    ids=["equal bounds not marked", "different bounds marked"],
)
def test_an_optimal_flag_that_disagrees_with_the_bounds_is_refused(
    upper: int, lower: int, optimal: bool
) -> None:
    table = _ta11()
    table["lower"] = table[PENG]["lower"] = table[GR]["lower"] = lower
    table["upper"] = table[PENG]["upper"] = table[GR]["upper"] = upper
    table["optimal"] = optimal
    with pytest.raises(ConfigError, match=r"ta11.*optimal is .*when they are equal"):
        _parse({"ta11": table})
    table["optimal"] = not optimal
    assert _parse({"ta11": table})["ta11"].optimal is (not optimal)


def test_an_optimal_flag_must_be_a_boolean() -> None:
    table = _ta11()
    table["optimal"] = 0
    with pytest.raises(ConfigError, match="optimal must be true or false"):
        _parse({"ta11": table})


def test_fewer_than_two_sources_are_refused() -> None:
    table = _ta11()
    del table[GR]
    with pytest.raises(ConfigError, match=r"ta11.*needs at least two sources, has 1"):
        _parse({"ta11": table})
    del table[PENG]
    with pytest.raises(ConfigError, match=r"ta11.*needs at least two sources, has 0"):
        _parse({"ta11": table})


def test_a_source_that_is_not_in_the_bibliography_is_refused() -> None:
    table = _ta11()
    table["nobody2000nothing"] = table.pop(GR)
    with pytest.raises(ConfigError, match=r"ta11.*unknown source 'nobody2000nothing'"):
        _parse({"ta11": table})


def test_a_source_counts_once_however_many_values_it_prints() -> None:
    table = _ta11()
    del table[GR]
    table[PENG]["found_upper"] = 1300
    table["upper"] = 1300
    with pytest.raises(ConfigError, match="needs at least two sources, has 1"):
        _parse({"ta11": table})


@pytest.mark.parametrize(
    "edit",
    [
        lambda document: document.update(colour="red"),
        lambda document: document.update(ta99=document["ta11"]),
        lambda document: document["ta11"].update(colour="red"),
        lambda document: document["ta11"][PENG].update(colour="red"),
        lambda document: document["ta11"][GR].update(upper_found=1300),
    ],
    ids=["file", "instance", "table", "source", "typo"],
)
def test_unknown_keys_are_refused(edit: Any) -> None:
    document = {"ta11": _ta11()}
    edit(document)
    with pytest.raises(ConfigError, match="unknown key"):
        _parse(document)


def test_a_document_may_record_any_instance_of_the_suite() -> None:
    assert list(_parse({"ta51": _ta11(), "ta11": _ta11()})) == ["ta51", "ta11"]


def test_an_upper_bound_that_is_not_the_smallest_value_is_refused() -> None:
    table = _ta11()
    table["upper"] = 1358
    with pytest.raises(
        ConfigError, match=r"ta11.*upper bound 1358 is not the smallest value.*1357"
    ):
        _parse({"ta11": table})
    table["upper"] = 1356
    with pytest.raises(
        ConfigError, match=r"ta11.*upper bound 1356 is not the smallest value.*1357"
    ):
        _parse({"ta11": table})
    # A value that a source found counts, so the listed value is no longer the smallest.
    table["upper"] = 1357
    table[GR]["found_upper"] = 1350
    with pytest.raises(ConfigError, match=r"upper bound 1357 is not the smallest value.*1350"):
        _parse({"ta11": table})


def test_a_lower_bound_that_is_not_the_largest_value_is_refused() -> None:
    table = _ta11()
    table[GR]["lower"] = 1330
    with pytest.raises(ConfigError, match=r"ta11.*lower bound 1323 is not the largest value.*1330"):
        _parse({"ta11": table})
    table["lower"] = 1330
    assert _parse({"ta11": table})["ta11"].lower == 1330.0
    table["lower"] = 1340
    with pytest.raises(ConfigError, match=r"lower bound 1340 is not the largest value.*1330"):
        _parse({"ta11": table})


def test_a_found_bound_must_improve_on_the_listed_one() -> None:
    for found in (1357, 1360):
        table = _ta11()
        table[PENG]["found_upper"] = found
        with pytest.raises(ConfigError, match=r"found_upper .* is not below the upper bound 1357"):
            _parse({"ta11": table})


@pytest.mark.parametrize("value", [True, "1357", 0, -3, float("nan"), float("inf"), None, [1357]])
def test_a_bound_must_be_a_positive_number(value: Any) -> None:
    for place in ("upper", "lower"):
        table = _ta11()
        table[place] = value
        with pytest.raises(ConfigError, match=f"{place} must be a positive number"):
            _parse({"ta11": table})
    table = _ta11()
    table[PENG]["upper"] = value
    with pytest.raises(ConfigError, match="upper must be a positive number"):
        _parse({"ta11": table})


@pytest.mark.parametrize("place", [None, "", " Table 8", 160, ["Table 8"]])
def test_a_source_names_the_table_and_page(place: Any) -> None:
    table = _ta11()
    table[PENG]["where"] = place
    with pytest.raises(ConfigError, match="where must name the table and page"):
        _parse({"ta11": table})
    del table[PENG]["where"]
    with pytest.raises(ConfigError, match=r"missing \['where'\]"):
        _parse({"ta11": table})


@pytest.mark.parametrize("key", ["upper", "lower", "optimal"])
def test_an_instance_table_lacks_nothing(key: str) -> None:
    table = _ta11()
    del table[key]
    with pytest.raises(ConfigError, match=rf"ta11.*missing \['{key}'\]"):
        _parse({"ta11": table})


@pytest.mark.parametrize("key", ["upper", "lower"])
def test_a_source_lacks_no_value(key: str) -> None:
    table = _ta11()
    del table[GR][key]
    with pytest.raises(ConfigError, match=rf"source '{GR}': missing \['{key}'\]"):
        _parse({"ta11": table})


# ------------------------------------------------------ the loader on a data folder


@pytest.fixture
def folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A data folder of the package that holds only the bounds files a test writes."""
    (tmp_path / "bounds").mkdir()
    monkeypatch.setattr(bounds, "files", lambda package: tmp_path)
    bounds._read.cache_clear()
    yield tmp_path / "bounds"
    bounds._read.cache_clear()


def test_a_file_loads_in_the_order_of_the_suite(folder: Path) -> None:
    peng = 'peng2015tabu = { upper = 1357, lower = 1323, where = "Table 8, p. 160" }'
    gr = 'goncalves2014extended = { upper = 1357, lower = 1323, where = "Table A7, p. 243" }'
    table = "upper = 1357\nlower = 1323\noptimal = false\n" + peng + "\n" + gr + "\n"
    (folder / f"{SUITE}.toml").write_text(
        f"# a file\n[ta12]\n{table}[ta11]\n{table}", encoding="utf-8"
    )
    assert list(load_bounds(SUITE)) == ["ta11", "ta12"]
    assert dict(reference_values(SUITE, "upper")) == {"ta11": 1357.0, "ta12": 1357.0}


def test_a_file_that_is_not_toml_is_refused(folder: Path) -> None:
    (folder / f"{SUITE}.toml").write_text("[ta11\nupper = ", encoding="utf-8")
    with pytest.raises(ConfigError, match="are not valid TOML"):
        load_bounds(SUITE)


def test_a_file_that_breaks_a_rule_is_refused_as_a_whole(folder: Path) -> None:
    good = (
        "[ta11]\nupper = 1357\nlower = 1323\noptimal = false\n"
        'peng2015tabu = { upper = 1357, lower = 1323, where = "Table 8, p. 160" }\n'
        'goncalves2014extended = { upper = 1357, lower = 1323, where = "Table A7, p. 243" }\n'
    )
    bad = good.replace("[ta11]", "[ta12]").replace("optimal = false", "optimal = true")
    (folder / f"{SUITE}.toml").write_text(good + "\n" + bad, encoding="utf-8")
    with pytest.raises(ConfigError, match=r"instance 'ta12': optimal is true"):
        load_bounds(SUITE)
