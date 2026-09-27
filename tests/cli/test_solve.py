"""``solve`` and ``check``: one solver on one problem, and schedule validation."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from dispatchatlas.benchmarks import generate
from dispatchatlas.cli import solve
from dispatchatlas.core import Problem, Schedule, SolveResult, StopCriteria
from dispatchatlas.core.serialization import (
    dumps_problem,
    load_result,
    problem_digest,
    save_problem,
    save_schedule,
    schedule_from_dict,
)
from dispatchatlas.core.validation import check_schedule
from dispatchatlas.solvers import default_registry
from dispatchatlas.solvers.rules import InputOrder
from tests.cli.runner import Cli
from tests.helpers import chain_problem


@pytest.fixture
def problem_file(tmp_path: Path) -> Path:
    return save_problem(generate("job-shop", 1, jobs=3, machines=3), tmp_path / "js.json")


def test_solve_prints_a_checked_summary(cli: Cli, problem_file: Path) -> None:
    result = cli("solve", problem_file, "--solver", "heft")
    assert result.code == 0
    assert "status       feasible" in result.out
    assert "stop reason  completed" in result.out
    assert "check        every constraint holds" in result.out
    assert "makespan" in result.out


def test_solve_writes_the_result_document(cli: Cli, problem_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "out" / "result.json"
    arguments = ("solve", problem_file, "--solver", "tabu-search", "--set", "tenure=3")
    summary = cli(*arguments, "--max-evaluations", "150", "--seed", "4", "--output", output)
    assert summary.code == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["format"] == "dispatchatlas.result"
    assert document["solver"] == "tabu-search"
    assert document["parameters"] == {"tenure": 3, "candidates": 20}
    assert document["stop"] == {"max_evaluations": 150}
    assert (document["seed"], document["evaluations"]) == (4, 150)
    problem = generate("job-shop", 1, jobs=3, machines=3)
    assert (document["problem"], document["problem_digest"]) == (
        problem.id,
        problem_digest(problem),
    )
    result = load_result(output)
    assert result.schedule is not None
    assert check_schedule(problem, result.schedule) == ()
    assert result.objectives["makespan"] == result.schedule.makespan
    assert result.trace
    again = cli(*arguments, "--max-evaluations", "150", "--seed", "4", "--json").json()
    assert again["objectives"] == dict(result.objectives)
    assert again["feasible"] is True
    assert again["output"] is None


def test_solve_reads_a_problem_from_standard_input(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(dumps_problem(chain_problem())))
    result = cli("solve", "-", "--solver", "input-order", "--json")
    assert result.code == 0
    assert (result.json()["problem"], result.json()["objectives"]) == ("chain", {"makespan": 5.0})


def test_solve_reads_instance_files_by_extension_or_format(cli: Cli, tmp_path: Path) -> None:
    fjs = tmp_path / "tiny.fjs"
    fjs.write_text("2 2\n2 1 1 3 2 1 2 2 4\n1 2 1 5 2 6\n", encoding="utf-8")
    by_extension = cli("solve", fjs, "--solver", "heft", "--json")
    assert by_extension.code == 0
    assert by_extension.json()["problem"] == "tiny"
    orlib = tmp_path / "shops.txt"
    orlib.write_text(" instance a\n 2 2\n 0 3 1 2\n 1 4 0 1\n", encoding="utf-8")
    unknown = cli("solve", orlib, "--solver", "heft")
    assert unknown.code == 1
    assert "cannot tell the format of shops.txt" in unknown.err
    assert "hint: pass --format, one of fjs, orlib-flow-shop" in unknown.err
    arguments = ("solve", orlib, "--format", "orlib-job-shop", "--solver", "heft", "--json")
    forced = cli(*arguments, "--instance", "a")
    assert forced.code == 0
    assert forced.json()["problem"] == "a"


@pytest.mark.parametrize(
    ("arguments", "message", "hint"),
    [
        (("--solver", "nope"), "unknown solver 'nope'", "hint: see 'dispatchatlas solve --help'"),
        (
            ("--solver", "tabu-search", "--set", "tenure=0", "--max-evaluations", "5"),
            "invalid parameters for solver 'tabu-search': tenure must be at least 1",
            "hint: see",
        ),
        (
            ("--solver", "tabu-search", "--set", "colour=1", "--max-evaluations", "5"),
            "solver 'tabu-search' has no parameters ['colour']",
            "hint: see",
        ),
    ],
)
def test_solve_errors_name_the_cause_and_a_remedy(
    cli: Cli, problem_file: Path, arguments: tuple[str, ...], message: str, hint: str
) -> None:
    result = cli("solve", problem_file, *arguments)
    assert result.code == 1
    assert result.out == ""
    assert message in result.err
    assert hint in result.err


def test_an_iterative_solver_without_a_limit_is_a_usage_error(cli: Cli, problem_file: Path) -> None:
    result = cli("solve", problem_file, "--solver", "tabu-search", "--target", "10")
    assert result.code == 2
    assert "usage: dispatchatlas solve" in result.err
    assert (
        "tabu-search is iterative: give --time-limit, --max-evaluations, or --max-iterations"
        in result.err
    )


def test_solve_refuses_a_problem_the_solver_does_not_model(cli: Cli, tmp_path: Path) -> None:
    permutation = save_problem(generate("flow-shop", 1, jobs=3, machines=2), tmp_path / "fs.json")
    result = cli("solve", permutation, "--solver", "min-min")
    assert result.code == 1
    assert "does not model features ['permutation']" in result.err
    assert "'dispatchatlas solvers show ID' lists the problem features" in result.err


def test_solve_reports_a_missing_problem_file(cli: Cli, tmp_path: Path) -> None:
    result = cli("solve", tmp_path / "absent.json", "--solver", "heft")
    assert result.code == 1
    assert result.err == (
        f"dispatchatlas: error: {tmp_path / 'absent.json'} does not exist\n"
        "dispatchatlas: hint: check the path\n"
    )


class _Careless:
    """Drops the first task of the input-order schedule but still claims feasibility."""

    info = InputOrder.info

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        result = InputOrder().solve(problem, stop, seed)
        assert result.schedule is not None
        return replace(result, schedule=Schedule(result.schedule.assignments[1:]))


def test_solve_fails_when_the_schedule_breaks_what_its_status_admits(
    cli: Cli, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = default_registry()
    registry.register(_Careless, replace_existing=True)
    monkeypatch.setattr(solve, "default_registry", lambda: registry)
    result = cli("solve", problem_file, "--solver", "input-order")
    assert result.code == 1
    assert "missing-task" in result.out
    assert "breaks constraints its status 'feasible' does not admit: missing-task" in result.err


def test_check_validates_a_problem(cli: Cli, problem_file: Path) -> None:
    result = cli("check", problem_file)
    assert result.code == 0
    assert "job-shop-3x3-1: valid, 9 tasks, 3 resources, 6 dependencies" in result.out
    assert cli("check", problem_file, "--json").json()["valid"] is True


def test_check_reports_an_invalid_problem(cli: Cli, tmp_path: Path) -> None:
    document = json.loads(dumps_problem(chain_problem()))
    document["dependencies"].append({"before": "b", "after": "a"})
    invalid = tmp_path / "cycle.json"
    invalid.write_text(json.dumps(document), encoding="utf-8")
    result = cli("check", invalid)
    assert result.code == 1
    assert "invalid" in result.out
    assert "cycle" in result.out
    report = cli("check", invalid, "--json").json()
    assert report["valid"] is False
    assert "cycle" in report["error"]


def test_check_finds_the_violations_of_a_schedule(cli: Cli, tmp_path: Path) -> None:
    problem = save_problem(chain_problem(), tmp_path / "chain.json")
    solved = cli("solve", problem, "--solver", "input-order", "--output", tmp_path / "r.json")
    assert solved.code == 0
    feasible = cli("check", problem, tmp_path / "r.json", "--json").json()
    assert feasible["schedule"]["feasible"] is True
    assert feasible["schedule"]["kind"] == "result"
    assert feasible["schedule"]["objectives"] == {"makespan": 5.0}
    rows = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["schedule"]
    rows["assignments"] = [
        ["b", 0, 0.0, 2.0] if row[0] == "b" else row for row in rows["assignments"]
    ]
    broken = tmp_path / "broken.json"
    save_schedule(schedule_from_dict(rows), broken, "chain")
    result = cli("check", problem, broken)
    assert result.code == 1
    assert "schedule" in result.out
    assert "precedence" in result.out
    report = cli("check", problem, broken, "--json").json()
    assert [item["code"] for item in report["schedule"]["violations"]] == [
        "precedence",
        "capacity",
    ]


def test_check_rejects_a_document_that_holds_no_schedule(cli: Cli, tmp_path: Path) -> None:
    problem = save_problem(chain_problem(), tmp_path / "chain.json")
    result = cli("check", problem, problem)
    assert result.code == 1
    assert "is not a schedule document, a result document, or a campaign run record" in result.err
