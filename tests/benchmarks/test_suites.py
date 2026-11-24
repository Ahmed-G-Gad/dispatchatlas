from __future__ import annotations

import pytest

from dispatchatlas.benchmarks import suites
from dispatchatlas.benchmarks.taillard import open_shop_instances
from dispatchatlas.core import ConfigError
from dispatchatlas.references import bibliography
from tests.benchmarks.fixtures import FJS, FLOW_SHOP, FT06_ROUTE_0, ORLIB_JOB_SHOPS, PSPLIB_SM

EXPECTED = {
    "taillard-flow-shop": ("flow-shop", 120, True),
    "taillard-job-shop": ("job-shop", 80, True),
    "taillard-open-shop": ("open-shop", 60, True),
    "demirkol-job-shop": ("job-shop", 80, False),
    "orlib-job-shop": ("job-shop", 82, False),
    "orlib-flow-shop": ("flow-shop", 31, False),
    "brandimarte": ("flexible-job-shop", 10, False),
    "psplib-j30": ("rcpsp", 480, False),
    "psplib-j60": ("rcpsp", 480, False),
    "psplib-j90": ("rcpsp", 480, False),
    "psplib-j120": ("rcpsp", 600, False),
}


def test_the_registry_lists_every_suite():
    assert [item.id for item in suites.suites()] == list(EXPECTED)


@pytest.mark.parametrize("suite_id", list(EXPECTED))
def test_suite_records_are_complete(suite_id):
    item = suites.suite(suite_id)
    problem_class, count, generated = EXPECTED[suite_id]
    assert item.problem_class == problem_class
    assert len(item.instances) == count
    assert len(set(item.instances)) == count
    assert item.generated is generated
    assert item.source.startswith("https://")
    assert item.offline is (generated or suite_id == "demirkol-job-shop")
    assert item.bundled is (suite_id == "demirkol-job-shop")
    assert item.name
    assert item.references
    for key in item.references:
        assert key in bibliography()
    assert (item.files is None) is item.offline


def test_instance_names_follow_the_sources():
    assert suites.suite("orlib-job-shop").instances[5:8] == ("ft06", "ft10", "ft20")
    assert suites.suite("orlib-flow-shop").instances[-2:] == ("reC39", "reC41")
    assert suites.suite("brandimarte").instances[0] == "mk01"
    j30 = suites.suite("psplib-j30").instances
    assert (j30[0], j30[9], j30[-1]) == ("j301_1", "j301_10", "j3048_10")
    assert suites.suite("psplib-j120").instances[-1] == "j12060_10"


def test_open_shop_bounds_come_from_the_seed_table():
    bounds = suites.suite("taillard-open-shop").bounds
    assert len(bounds) == 60
    for item in open_shop_instances():
        bound = bounds[item.name]
        assert (bound.lower, bound.upper, bound.optimal) == (
            item.lower_bound,
            item.upper_bound,
            item.optimal,
        )
        assert bound.objective == "makespan"
    assert all(not suites.suite(s).bounds for s in EXPECTED if s != "taillard-open-shop")


@pytest.mark.parametrize(
    ("suite_id", "instance", "problem_id", "size"),
    [
        ("taillard-flow-shop", "ta001", "ta001", 100),
        ("taillard-flow-shop", "TA120", "ta120", 10_000),
        ("taillard-job-shop", "ta80", "ta80", 2000),
        ("taillard-open-shop", "tai7x7-4", "tai7x7-4", 49),
        ("demirkol-job-shop", "DMU80", "dmu80", 1000),
    ],
)
def test_generated_and_bundled_suites_load_offline(suite_id, instance, problem_id, size):
    problem = suites.load(suite_id, instance)
    assert (problem.id, problem.size) == (problem_id, size)


def test_file_suites_read_from_a_directory(tmp_path):
    (tmp_path / "jobshop1.txt").write_text(ORLIB_JOB_SHOPS)
    problem = suites.load("orlib-job-shop", "FT06", directory=tmp_path)
    assert problem.id == "ft06"
    assert [problem.tasks[i].modes[0].duration for i in problem.groups["j0"]] == [
        time for _, time in FT06_ROUTE_0
    ]

    nested = tmp_path / "TextData" / "Brandimarte_Data" / "Text"
    nested.mkdir(parents=True)
    (nested / "Mk01.fjs").write_text(FJS)
    assert suites.load("brandimarte", "mk01", directory=tmp_path).id == "mk01"

    (tmp_path / "j301_1.sm").write_text(PSPLIB_SM)
    assert suites.load("psplib-j30", "j301_1", directory=str(tmp_path)).size == 4

    flow = tmp_path / "flow"
    flow.mkdir()
    (flow / "flowshop1.txt").write_text(
        " +++\n instance car1\n +++\n" + FLOW_SHOP + " +++ EOF +++\n"
    )
    assert suites.load("orlib-flow-shop", "car1", directory=flow).permutation


def test_file_suites_need_a_directory_with_the_file(tmp_path):
    with pytest.raises(ConfigError, match=r"download jobshop1\.txt"):
        suites.load("orlib-job-shop", "ft06")
    with pytest.raises(ConfigError, match="not a directory"):
        suites.load("orlib-job-shop", "ft06", directory=tmp_path / "missing")
    with pytest.raises(ConfigError, match=r"no file 'mk02\.fjs'"):
        suites.load("brandimarte", "mk02", directory=tmp_path)
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "mk03.fjs").write_text(FJS)
    with pytest.raises(ConfigError, match="several files"):
        suites.load("brandimarte", "mk03", directory=tmp_path)


def test_unknown_suites_and_instances_are_refused():
    with pytest.raises(ConfigError, match="unknown suite"):
        suites.suite("lawrence")
    with pytest.raises(ConfigError, match="no instance 'ta121'"):
        suites.load("taillard-flow-shop", "ta121")
