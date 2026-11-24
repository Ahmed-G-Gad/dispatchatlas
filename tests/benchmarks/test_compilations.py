"""The dated compilations of best known bounds, the trivial lower bound, and the loader.

The counts checked here were read from the two compilations when they were recorded: the
sources differ on 66 of the 160 instances, the ScheduleOpt compilation is never looser than
Optimizizer's, and the best known upper bound equals the trivial lower bound for 27 of the
Taillard job shops and 14 of the Demirkol job shops.
"""

from __future__ import annotations

import datetime
import tomllib
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

import pytest

from dispatchatlas.benchmarks import compilations as module
from dispatchatlas.benchmarks import suites
from dispatchatlas.benchmarks.compilations import (
    BestKnown,
    CompiledBound,
    best_known,
    compilation_bounds,
    compilations,
    trivial_lower_bound,
    upper_equals_trivial_lower_bound,
)
from dispatchatlas.benchmarks.shop import job_shop_problem, open_shop_problem
from dispatchatlas.core import ConfigError, Dependency, ModelError, Problem, Resource, Task

SCHEDULEOPT = "scheduleopt-2026-11-21"
OPTIMIZIZER = "optimizizer-2026-11-08"
TA = [f"ta{number:02d}" for number in range(1, 81)]
DMU = [f"dmu{number:02d}" for number in range(1, 81)]


def _closed(source: str, names: list[str]) -> int:
    return sum(compilation_bounds(source)[name].optimal for name in names)


# ----------------------------------------------------------------- the records


def test_the_compilations_keep_their_source_and_date() -> None:
    recorded = compilations()
    assert list(recorded) == [SCHEDULEOPT, OPTIMIZIZER]
    schedule, optimizizer = recorded[SCHEDULEOPT], recorded[OPTIMIZIZER]
    assert schedule.version == "af68a647c1d62a342b596d23dcb57011715f3ffd"
    assert schedule.date == datetime.date(2026, 11, 21)
    assert schedule.url == "https://github.com/ScheduleOpt/benchmarks"
    assert schedule.file == "jobshop/solutions/bks.json"
    assert schedule.licence == "CC-BY-SA-4.0"
    assert optimizizer.date == datetime.date(2026, 11, 8)
    assert optimizizer.version == "Last update: November 08, 2026"
    assert optimizizer.licence is None
    assert optimizizer.file == "TA.php and DMU.php"
    for item in recorded.values():
        assert item.retrieved == datetime.date(2026, 11, 24)
        assert item.date < item.retrieved
    with pytest.raises(TypeError):
        recorded["x"] = schedule  # type: ignore[index]


def test_each_compilation_covers_the_160_instances_with_its_own_date() -> None:
    for source in (SCHEDULEOPT, OPTIMIZIZER):
        table = compilation_bounds(source)
        assert list(table) == [*TA, *DMU]
        for name, item in table.items():
            assert item.compilation == source
            assert item.date == compilations()[source].date
            assert 0 < item.lower <= item.upper, name
    with pytest.raises(ConfigError, match="unknown compilation 'nobody'"):
        compilation_bounds("nobody")
    with pytest.raises(TypeError):
        compilation_bounds(SCHEDULEOPT)["ta01"] = None  # type: ignore[index]


def test_values_read_from_the_sources_are_recorded() -> None:
    schedule, optimizizer = compilation_bounds(SCHEDULEOPT), compilation_bounds(OPTIMIZIZER)
    assert (schedule["ta01"].lower, schedule["ta01"].upper) == (1231, 1231)
    assert (schedule["ta73"].lower, schedule["ta73"].upper) == (5568, 5568)
    assert (schedule["dmu01"].lower, schedule["dmu01"].upper) == (2563, 2563)
    assert (optimizizer["dmu04"].lower, optimizizer["dmu04"].upper) == (2658, 2669)
    assert (schedule["dmu04"].lower, schedule["dmu04"].upper) == (2669, 2669)
    assert (optimizizer["dmu69"].lower, optimizizer["dmu69"].upper) == (5500, 5682)
    assert (schedule["dmu69"].lower, schedule["dmu69"].upper) == (5511, 5682)
    # Optimizizer prints 1997 for both bounds of ta45 but does not mark it closed.
    assert (optimizizer["ta45"].lower, optimizizer["ta45"].upper) == (1997, 1997)


def test_the_sources_differ_on_66_instances_and_scheduleopt_is_never_looser() -> None:
    schedule, optimizizer = compilation_bounds(SCHEDULEOPT), compilation_bounds(OPTIMIZIZER)
    differ = [
        name
        for name in schedule
        if (schedule[name].lower, schedule[name].upper)
        != (optimizizer[name].lower, optimizizer[name].upper)
    ]
    assert len(differ) == 66
    assert len([name for name in differ if name.startswith("ta")]) == 18
    assert len([name for name in differ if name.startswith("dmu")]) == 48
    for name in schedule:
        assert schedule[name].lower >= optimizizer[name].lower, name
        assert schedule[name].upper <= optimizizer[name].upper, name
    assert sum(schedule[n].lower > optimizizer[n].lower for n in schedule) == 66
    assert sum(schedule[n].upper < optimizizer[n].upper for n in schedule) == 5


def test_the_numbers_of_closed_instances_per_source() -> None:
    expected = {
        SCHEDULEOPT: (38, 30, 34, 1),
        OPTIMIZIZER: (30, 30, 31, 0),
    }
    blocks = (TA[:50], TA[50:], DMU[:40], DMU[40:])
    for source, counts in expected.items():
        assert tuple(_closed(source, block) for block in blocks) == counts, source
    closed_only_in_scheduleopt = {
        name
        for name in TA + DMU
        if compilation_bounds(SCHEDULEOPT)[name].optimal
        and not compilation_bounds(OPTIMIZIZER)[name].optimal
    }
    assert closed_only_in_scheduleopt == {
        *(f"ta{n}" for n in (18, 22, 23, 25, 26, 27, 29, 33)),
        *(f"dmu0{n}" for n in (4, 6, 7)),
        "dmu43",
    }


# ------------------------------------------------------------------ best known


def test_best_known_is_the_tighter_bound_with_the_sources_that_list_it() -> None:
    for suite_id, names in (("taillard-job-shop", TA), ("demirkol-job-shop", DMU)):
        known = best_known(suite_id)
        assert list(known) == list(suites.suite(suite_id).instances) == names
        for name, entry in known.items():
            listed = entry.listed
            assert set(listed) == {SCHEDULEOPT, OPTIMIZIZER}
            assert entry.upper == min(item.upper for item in listed.values())
            assert entry.lower == max(item.lower for item in listed.values())
            assert entry.upper_sources == tuple(
                k for k, v in listed.items() if v.upper == entry.upper
            )
            assert entry.lower_sources == tuple(
                k for k, v in listed.items() if v.lower == entry.lower
            )
            assert entry.lower <= entry.upper
            assert entry.optimal is (entry.lower == entry.upper)
            assert entry.instance == name
    with pytest.raises(TypeError):
        best_known("demirkol-job-shop")["dmu01"] = None  # type: ignore[index]


def test_a_value_that_one_compilation_alone_lists_names_that_compilation() -> None:
    entry = best_known("demirkol-job-shop")["dmu43"]
    assert (entry.lower, entry.upper) == (3441, 3441)
    assert entry.lower_sources == (SCHEDULEOPT,)
    assert entry.upper_sources == (SCHEDULEOPT, OPTIMIZIZER)
    assert entry.optimal
    ta45 = best_known("taillard-job-shop")["ta45"]
    assert ta45.lower_sources == ta45.upper_sources == (SCHEDULEOPT, OPTIMIZIZER)
    assert best_known("demirkol-job-shop")["dmu69"].lower_sources == (SCHEDULEOPT,)
    assert best_known("demirkol-job-shop")["dmu69"].upper == 5682


def test_the_two_sources_are_never_merged_into_one_record() -> None:
    entry = best_known("demirkol-job-shop")["dmu04"]
    assert entry.listed[OPTIMIZIZER] == CompiledBound(
        OPTIMIZIZER, datetime.date(2026, 11, 8), 2658, 2669
    )
    assert entry.listed[SCHEDULEOPT] == CompiledBound(
        SCHEDULEOPT, datetime.date(2026, 11, 21), 2669, 2669
    )
    assert isinstance(entry, BestKnown)


def test_only_the_two_job_shop_suites_have_best_known_bounds() -> None:
    with pytest.raises(ConfigError, match="no compilations cover suite 'taillard-flow-shop'"):
        best_known("taillard-flow-shop")
    with pytest.raises(ConfigError, match="no compilations cover suite 'nobody'"):
        best_known("nobody")


# -------------------------------------------------------- the trivial lower bound


def test_the_trivial_lower_bound_is_the_larger_of_machine_load_and_job_length() -> None:
    machine_bound = job_shop_problem([[(0, 3.0), (1, 2.0)], [(1, 4.0), (0, 1.0)]], machines=2)
    assert trivial_lower_bound(machine_bound) == 6  # machine 1 carries 2 + 4
    job_bound = job_shop_problem([[(0, 9.0), (1, 2.0)], [(1, 1.0), (0, 1.0)]], machines=2)
    assert trivial_lower_bound(job_bound) == 11  # the first job takes 9 + 2
    assert trivial_lower_bound(job_shop_problem([[(0, 5.0)]], machines=1)) == 5


def test_the_trivial_lower_bound_of_known_instances() -> None:
    known = best_known("demirkol-job-shop")
    assert known["dmu01"].trivial_lower == 2179
    assert known["dmu43"].trivial_lower == 2523
    ta = best_known("taillard-job-shop")
    assert ta["ta01"].trivial_lower == 977
    assert ta["ta73"].trivial_lower == 5552
    for suite_id in ("taillard-job-shop", "demirkol-job-shop"):
        for name, entry in best_known(suite_id).items():
            assert entry.trivial_lower == trivial_lower_bound(suites.suite(suite_id).load(name))
            assert entry.trivial_lower <= entry.lower, name


def test_a_problem_that_is_not_a_job_shop_has_no_trivial_lower_bound() -> None:
    with pytest.raises(ModelError, match="is not a job shop"):
        trivial_lower_bound(open_shop_problem([[1.0, 2.0], [3.0, 4.0]]))
    ungrouped = Problem(
        "x",
        tasks=(Task.on("a0", "m0", 2.0), Task.on("a1", "m1", 3.0)),
        resources=(Resource("m0"), Resource("m1")),
        dependencies=(Dependency("a0", "a1"),),
    )
    with pytest.raises(ModelError, match="belong to no job group"):
        trivial_lower_bound(ungrouped)


def test_the_upper_bound_equals_the_trivial_bound_for_the_expected_instances() -> None:
    ta = {n for n, e in best_known("taillard-job-shop").items() if e.upper_equals_trivial_lower}
    dmu = {n for n, e in best_known("demirkol-job-shop").items() if e.upper_equals_trivial_lower}
    assert len(ta) == 27
    assert len(dmu) == 14
    assert {n for n in ta if int(n[2:]) <= 50} == {"ta31", "ta34", "ta37", "ta38"}
    assert {f"dmu{n}" for n in (14, 18, 25, 27, 29, 31, 32, *range(34, 41))} == dmu
    assert upper_equals_trivial_lower_bound("taillard-job-shop", "ta80")
    assert not upper_equals_trivial_lower_bound("taillard-job-shop", "ta01")
    assert not upper_equals_trivial_lower_bound("taillard-job-shop", "ta73")
    assert upper_equals_trivial_lower_bound("demirkol-job-shop", "dmu40")
    with pytest.raises(ConfigError, match="no instance 'ta99'"):
        upper_equals_trivial_lower_bound("taillard-job-shop", "ta99")
    with pytest.raises(ConfigError, match="no compilations cover"):
        upper_equals_trivial_lower_bound("orlib-job-shop", "ft06")


# ------------------------------------------------------ the rules of the loader


def _text(key: str) -> dict[str, Any]:
    path = files("dispatchatlas.data").joinpath("bounds", "compilations", f"{key}.toml")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _parse(document: Mapping[str, Any]) -> Any:
    return module._parse(SCHEDULEOPT, document)


def test_a_valid_document_parses() -> None:
    head, rows = _parse(_text(SCHEDULEOPT))
    assert head.id == SCHEDULEOPT
    assert len(rows) == 160


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d.update(extra={}), "the tables must be"),
        (lambda d: d["compilation"].update(colour="red"), "unknown keys"),
        (lambda d: d["compilation"].pop("url"), "missing keys"),
        (lambda d: d["compilation"].update(id="other"), "the id is 'other'"),
        (lambda d: d["compilation"].update(name=""), "must be text"),
        (lambda d: d["compilation"].update(date="21 Nov"), "date must be an ISO date"),
        (lambda d: d["compilation"].update(date=20261121), "ISO date in quotes"),
        (lambda d: d["compilation"].update(licence=""), "licence must be text"),
        (lambda d: d["bounds"].pop("ta01"), "must cover exactly the 160 instances"),
        (lambda d: d["bounds"].update(ta81={"lower": 1, "upper": 2}), "must cover exactly"),
        (lambda d: d["bounds"].update(ta01={"lower": 1}), "needs exactly lower and upper"),
        (lambda d: d["bounds"].update(ta01={"lower": 5, "upper": 4}), "lower bound 5 is above"),
        (lambda d: d["bounds"].update(ta01={"lower": 0, "upper": 4}), "positive whole number"),
        (lambda d: d["bounds"].update(ta01={"lower": 1.5, "upper": 4}), "positive whole number"),
        (lambda d: d["bounds"].update(ta01={"lower": True, "upper": 4}), "positive whole number"),
    ],
)
def test_a_document_that_breaks_a_rule_is_refused(edit: Any, message: str) -> None:
    document = _text(SCHEDULEOPT)
    edit(document)
    with pytest.raises(ConfigError, match=message):
        _parse(document)


def test_a_file_that_is_not_toml_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    folder = tmp_path / "bounds" / "compilations"
    folder.mkdir(parents=True)
    (folder / f"{SCHEDULEOPT}.toml").write_text("[compilation\n", encoding="utf-8")
    monkeypatch.setattr(module, "files", lambda package: tmp_path)
    with pytest.raises(ConfigError, match="is not valid TOML"):
        module._read(SCHEDULEOPT)
