"""Commands that list and show solvers, benchmarks, suites, catalogs, and references."""

from __future__ import annotations

from pathlib import Path

import pytest

from dispatchatlas.benchmarks.catalog import load_directory
from dispatchatlas.benchmarks.registry import default_registry as default_families
from dispatchatlas.core.serialization import load_problem, loads_problem, problem_to_dict
from dispatchatlas.references import bibliography, to_bibtex
from dispatchatlas.solvers import default_registry
from tests.cli.runner import Cli

# ------------------------------------------------------------------ solvers


def test_solvers_list_names_every_registered_solver(cli: Cli) -> None:
    result = cli("solvers", "list")
    assert result.code == 0
    assert result.out.splitlines()[0].split() == ["ID", "FAMILY", "TRAITS"]
    listed = [line.split()[0] for line in result.out.splitlines()[1:]]
    assert listed == list(default_registry().ids())
    documents = cli("solvers", "list", "--json").json()
    assert [item["id"] for item in documents] == listed
    assert {"parameters", "supports", "citations"} <= set(documents[0])


def test_solvers_list_filters_by_family(cli: Cli) -> None:
    exact = cli("solvers", "list", "--family", "exact", "--json").json()
    assert exact
    assert {item["family"] for item in exact} == {"exact"}
    unknown = cli("solvers", "list", "--family", "alchemy")
    assert unknown.code == 1
    assert "unknown solver family 'alchemy'; known: " in unknown.err


def test_solvers_show_gives_parameters_features_and_citations(cli: Cli) -> None:
    result = cli("solvers", "show", "tabu-search")
    assert result.code == 0
    assert "tenure = 7" in result.out
    assert "iterative        yes: set --time-limit" in result.out
    assert "F. Glover (1986)." in result.out
    document = cli("solvers", "show", "min-min", "--json").json()
    assert document["id"] == "min-min"
    assert "permutation" in document["unsupported"]
    assert len(document["citations"]) == len(document["references"]) >= 1


def test_solvers_show_names_the_extra_a_solver_needs(cli: Cli) -> None:
    result = cli("solvers", "show", "cp-sat")
    assert result.code == 0
    assert "the exact extra" in result.out


# --------------------------------------------------------------- benchmarks


def test_benchmarks_list_and_show(cli: Cli) -> None:
    listed = cli("benchmarks", "list", "--json").json()
    assert [item["id"] for item in listed] == list(default_families().ids())
    shown = cli("benchmarks", "show", "job-shop")
    assert shown.code == 0
    assert "jobs = 10" in shown.out
    assert "E. Taillard (1993)." in shown.out
    assert cli("benchmarks", "show", "no-such-family").code == 1


def test_generate_prints_the_same_problem_document_for_the_same_seed(cli: Cli) -> None:
    arguments = ("benchmarks", "generate", "job-shop", "--seed", "3", "--set", "jobs=3")
    first = cli(*arguments, "--set", "machines=2")
    second = cli(*arguments, "--set", "machines=2")
    assert first.code == 0
    assert first.out == second.out
    problem = loads_problem(first.out)
    assert (problem.id, len(problem.tasks), problem.metadata["seed"]) == ("job-shop-3x2-3", 6, 3)
    expected = default_families().generate("job-shop", 3, jobs=3, machines=2)
    assert problem_to_dict(problem) == problem_to_dict(expected)


def test_generate_writes_a_file_and_reports_it(cli: Cli, tmp_path: Path) -> None:
    target = tmp_path / "nested" / "shop.json"
    result = cli("benchmarks", "generate", "flow-shop", "--set", "jobs=4", "--output", target)
    assert result.code == 0
    assert result.out == f"wrote problem flow-shop-4x5-0 (20 tasks, 5 resources) to {target}\n"
    assert load_problem(target).id == "flow-shop-4x5-0"
    report = cli("benchmarks", "generate", "rcpsp", "--output", target, "--json").json()
    assert report["path"] == str(target)
    assert report["problem"]["family"] == "rcpsp"


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ("colour=red", "family 'job-shop' has no parameters ['colour']"),
        ("jobs=0", "invalid parameters for family 'job-shop'"),
    ],
)
def test_generate_rejects_bad_parameters(cli: Cli, setting: str, message: str) -> None:
    result = cli("benchmarks", "generate", "job-shop", "--set", setting)
    assert result.code == 1
    assert message in result.err
    assert result.out == ""


# ------------------------------------------------------------------- suites


def test_suites_list_and_show(cli: Cli) -> None:
    listed = cli("suites", "list")
    assert listed.code == 0
    assert "taillard-flow-shop" in listed.out
    shown = cli("suites", "show", "taillard-open-shop")
    assert "60: tai4x4-1 to tai20x20-10" in shown.out
    assert "published for 60 instances" in shown.out
    document = cli("suites", "show", "brandimarte", "--json").json()
    assert document["generated"] is False
    assert document["instances"][0] == "mk01"
    assert document["files"]


def test_suites_load_rebuilds_a_generated_instance(cli: Cli, tmp_path: Path) -> None:
    printed = cli("suites", "load", "taillard-flow-shop", "TA001")
    assert printed.code == 0
    problem = loads_problem(printed.out)
    assert (problem.id, len(problem.tasks)) == ("ta001", 100)
    target = tmp_path / "ta001.json"
    assert cli("suites", "load", "taillard-flow-shop", "ta001", "--output", target).code == 0
    assert problem_to_dict(load_problem(target)) == problem_to_dict(problem)


def test_suites_load_reads_a_file_suite_from_its_directory(cli: Cli, tmp_path: Path) -> None:
    missing = cli("suites", "load", "orlib-job-shop", "ft06")
    assert missing.code == 1
    assert "download jobshop1.txt from" in missing.err
    assert "hint: pass --directory D" in missing.err
    (tmp_path / "jobshop1.txt").write_text(
        " instance ft06\n 2 2\n 0 3 1 2\n 1 4 0 1\n", encoding="utf-8"
    )
    loaded = cli("suites", "load", "orlib-job-shop", "ft06", "--directory", tmp_path)
    assert loaded.code == 0
    assert len(loads_problem(loaded.out).tasks) == 4
    unknown = cli("suites", "load", "orlib-job-shop", "ft99", "--directory", tmp_path)
    assert unknown.code == 1
    assert "has no instance 'ft99'" in unknown.err


# ----------------------------------------------------------------- catalogs


def test_catalogs_list_and_show(cli: Cli) -> None:
    listed = cli("catalogs", "list", "--json").json()
    assert {item["name"] for item in listed} >= {"smoke", "continuum", "workflow"}
    shown = cli("catalogs", "show", "smoke")
    assert shown.code == 0
    assert "jobs=4 machines=3" in shown.out
    unknown = cli("catalogs", "show", "nope")
    assert unknown.code == 1
    assert "unknown catalog 'nope'; known: smoke" in unknown.err


def test_catalogs_build_writes_a_verified_directory(cli: Cli, tmp_path: Path) -> None:
    target = tmp_path / "smoke"
    report = cli("catalogs", "build", "smoke", "--output", target, "--seed", "5", "--json")
    assert report.code == 0
    document = report.json()
    problems = load_directory(target)
    assert [problem.id for problem in problems] == document["problems"]
    assert len(problems) == 26
    again = cli("catalogs", "build", "smoke", "--output", target)
    assert again.code == 1
    assert "hint: pass --overwrite" in again.err
    replaced = cli("catalogs", "build", "smoke", "--output", target, "--overwrite")
    assert replaced.code == 0
    assert "wrote 26 problems of catalog smoke (seed 0)" in replaced.out


# --------------------------------------------------------------- references


def test_references_lists_every_entry_with_its_key(cli: Cli) -> None:
    result = cli("references")
    assert result.code == 0
    lines = result.out.splitlines()
    assert len(lines) == len(bibliography())
    assert lines[0].startswith(f"[{next(iter(bibliography()))}] ")


def test_references_selects_keys_and_the_sources_of_solvers_and_suites(cli: Cli) -> None:
    result = cli(
        "references", "wilcoxon1945individual", "--solver", "tabu-search", "--suite",
        "taillard-flow-shop", "--json",
    )  # fmt: skip
    keys = [item["key"] for item in result.json()]
    expected = ["wilcoxon1945individual", *default_registry().info("tabu-search").references]
    assert keys == [*expected, "taillard1993benchmarks", "beasley1990orlibrary"]
    family = cli("references", "--family", "rcpsp", "--json").json()
    assert [item["key"] for item in family] == list(default_families().info("rcpsp").references)


def test_references_exports_bibtex(cli: Cli) -> None:
    result = cli("references", "kolisch1996serial", "mao2017survey", "--format", "bibtex")
    assert result.code == 0
    assert result.out.startswith("@article{kolisch1996serial,\n  author = {Kolisch, R.},\n")
    assert "  pages = {320--333},\n" in result.out
    assert "  journal = {IEEE Communications Surveys \\& Tutorials},\n" in result.out
    assert result.out.count("\n@") == 1
    assert result.out == to_bibtex(["kolisch1996serial", "mao2017survey"])


def test_an_unknown_reference_key_fails_with_a_hint(cli: Cli) -> None:
    result = cli("references", "nobody2099nothing")
    assert result.code == 1
    assert "unknown reference key 'nobody2099nothing'" in result.err
    assert "'dispatchatlas references' lists every key" in result.err
