"""Reading a campaign directory into a dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from dispatchatlas.analysis.dataset import load_dataset
from dispatchatlas.campaigns import read_plan, read_runs
from dispatchatlas.core.errors import ConfigError, SerializationError
from tests.analysis.analysis_data import campaign_dir, make_dataset

__all__ = ["campaign_dir"]


def test_every_planned_run_appears_in_plan_order(campaign_dir: Path) -> None:
    dataset = load_dataset(campaign_dir)
    plan = read_plan(campaign_dir)
    assert [run.run_id for run in dataset.runs] == [run.run_id for run in plan.runs]
    assert dataset.counts() == {"succeeded": 56, "infeasible": 0, "failed": 0, "missing": 0}
    assert dataset.campaign == "tiny"
    assert dataset.config_hash == plan.config_hash
    assert [entry.name for entry in dataset.solvers] == [
        "input-order",
        "random-search",
        "hc",
        "shortest-processing-time",
    ]
    assert dataset.objectives == ("makespan",)
    records = {record.run_id: record for record in read_runs(campaign_dir)}
    first = dataset.runs[0]
    assert first.record == records[first.run_id]
    assert first.value("makespan") == records[first.run_id].outcome.value
    assert first.points(("makespan",)) == ((first.value("makespan"),),)


def test_cells_hold_the_replicates_of_one_solver_and_problem(campaign_dir: Path) -> None:
    dataset = load_dataset(campaign_dir)
    problem = dataset.problems[0].id
    cell = dataset.cell("hc", problem, "evals")
    assert [run.replicate for run in cell] == [0, 1, 2]
    assert len(dataset.cell("input-order", problem, "evals")) == 1
    assert dataset.cell("hc", "no-such-problem", "evals") == ()


def test_missing_records_are_missing_runs(campaign_dir: Path) -> None:
    dataset = load_dataset(campaign_dir)
    target = dataset.runs[0]
    path = next((campaign_dir / "runs").rglob("*.json"))
    record = load_dataset(campaign_dir)
    assert record.counts()["missing"] == 0
    moved = path.with_suffix(".bak")
    path.rename(moved)
    try:
        counts = load_dataset(campaign_dir).counts()
    finally:
        moved.rename(path)
    assert counts["missing"] == 1
    assert target.status == "succeeded"


def test_selection_keeps_order_and_refuses_unknown_names() -> None:
    dataset = make_dataset({"a": {"p1": [1.0], "p2": [2.0]}, "b": {"p1": [3.0]}})
    selected = dataset.select(solvers=["b"], problems=["p1"])
    assert [run.run_id for run in selected.runs] == ["run-p-b-p1-0"]
    assert [problem.id for problem in selected.problems] == ["p1"]
    with pytest.raises(ConfigError, match="unknown solver 'c'"):
        dataset.select(solvers=["c"])
    with pytest.raises(ConfigError, match="twice"):
        dataset.select(problems=["p1", "p1"])
    with pytest.raises(ConfigError, match="unknown problem"):
        dataset.problem("p9")


def test_statuses_of_synthetic_runs() -> None:
    dataset = make_dataset({"a": {"p1": [1.0, "infeasible", "failed", None]}})
    assert [run.status for run in dataset.runs] == ["succeeded", "infeasible", "failed", "missing"]
    assert [run.points(("makespan",)) for run in dataset.runs] == [((1.0,),), (), (), ()]
    assert dataset.runs[1].value("makespan") == 1e6
    assert dataset.runs[3].objectives == {}


def test_an_unreadable_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(SerializationError, match=r"plan\.json"):
        load_dataset(tmp_path)
