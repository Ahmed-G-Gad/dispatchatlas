"""The examples: every script runs, every campaign file validates, and the plugin works."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.benchmarks import generate
from dispatchatlas.cli import main
from dispatchatlas.core import ConfigError, StopCriteria, StopReason, check_schedule
from dispatchatlas.solvers import PLUGIN_GROUP, SolverRegistry, default_registry
from dispatchatlas.solvers import registry as solver_registry

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
PLUGIN = EXAMPLES / "plugin"

EXPECTED = {
    "campaign_analysis.py": "replay: 60 of 60 runs reproduced exactly",
    "compare_solvers.py": "optimal makespan 418",
    "multi_objective_placement.py": "hypervolume",
    "quickstart.py": "job-shop-6x4-1: 24 operations on 4 machines",
}
"""A line each script prints when it works."""

SLOW = {"campaign_analysis.py"}
"""Scripts that take several seconds."""


def _scripts() -> list[Any]:
    return [
        pytest.param(path, marks=pytest.mark.slow) if path.name in SLOW else path
        for path in sorted(EXAMPLES.glob("*.py"))
    ]


def test_every_script_has_an_expected_output() -> None:
    assert sorted(path.name for path in EXAMPLES.glob("*.py")) == sorted(EXPECTED)


@pytest.mark.parametrize("script", _scripts(), ids=lambda path: path.name)
def test_the_script_runs(script: Path, tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert EXPECTED[script.name] in completed.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "config", sorted((EXAMPLES / "campaigns").glob("*.toml")), ids=lambda path: path.name
)
def test_the_campaign_file_validates(config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["campaign", "validate", str(config), "--check-compatibility"]) == 0
    assert "fair        yes" in capsys.readouterr().out


# ------------------------------------------------------------------ plugin


def test_the_plugin_declares_its_entry_point() -> None:
    project = tomllib.loads((PLUGIN / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["entry-points"][PLUGIN_GROUP] == {
        "example": "dispatchatlas_example_plugin:register"
    }


@pytest.fixture
def plugin_registry(monkeypatch: pytest.MonkeyPatch) -> SolverRegistry:
    """The default solver registry, with the example plugin found as if installed."""
    monkeypatch.syspath_prepend(str(PLUGIN / "src"))
    entry = EntryPoint("example", "dispatchatlas_example_plugin:register", PLUGIN_GROUP)

    def entry_points(*, group: str) -> list[EntryPoint]:
        return [entry] if group == PLUGIN_GROUP else []

    monkeypatch.setattr(solver_registry, "entry_points", entry_points)
    return default_registry()


def test_the_plugin_registers_through_its_entry_point(plugin_registry: SolverRegistry) -> None:
    info = plugin_registry.info("insertion-descent")
    assert info.family == "local-search"
    assert dict(info.parameters) == {"restart_after": 100}
    assert "insertion-descent" not in default_registry(plugins=False)


@pytest.mark.parametrize(
    ("stop", "reason"),
    [
        (StopCriteria(max_evaluations=300), StopReason.EVALUATION_LIMIT),
        (StopCriteria(max_iterations=200), StopReason.ITERATION_LIMIT),
        (StopCriteria(time_limit=0.2), StopReason.TIME_LIMIT),
    ],
)
def test_the_plugin_solver_honours_every_limit(
    plugin_registry: SolverRegistry, stop: StopCriteria, reason: StopReason
) -> None:
    problem = generate("job-shop", seed=2, jobs=4, machines=3)
    solver = plugin_registry.create("insertion-descent", restart_after=20)
    result = solver.solve(problem, stop, seed=5)
    assert result.stop_reason is reason
    assert result.schedule is not None
    assert check_schedule(problem, result.schedule) == ()
    if stop.max_evaluations is not None:
        assert result.evaluations == stop.max_evaluations
    if stop.max_iterations is not None:
        assert result.iterations == stop.max_iterations


def test_the_plugin_solver_is_deterministic(plugin_registry: SolverRegistry) -> None:
    problem = generate("flow-shop", seed=3, jobs=6, machines=3)
    stop = StopCriteria(max_evaluations=400)
    solver = plugin_registry.create("insertion-descent")
    first, second = (solver.solve(problem, stop, seed=11) for _ in range(2))
    assert first.schedule == second.schedule
    progress = [
        [(p.evaluations, p.iteration, p.value) for p in run.trace] for run in (first, second)
    ]
    assert progress[0] == progress[1]


def test_the_plugin_solver_refuses_invalid_parameters(plugin_registry: SolverRegistry) -> None:
    with pytest.raises(ConfigError, match="restart_after"):
        plugin_registry.create("insertion-descent", restart_after=0)
