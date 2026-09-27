"""The solver registry: registration, creation, plugins, and the ``solve`` shortcut."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field, replace
from importlib.metadata import EntryPoint
from typing import ClassVar

import pytest

from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import Solver, SolveResult, SolverInfo, StopCriteria, StopReason
from dispatchatlas.solvers import registry as solver_module
from dispatchatlas.solvers import solve
from dispatchatlas.solvers.registry import (
    BUILT_IN_MODULES,
    PLUGIN_GROUP,
    SolverRegistry,
    default_registry,
    load_plugins,
)
from dispatchatlas.solvers.search import Search
from tests.helpers import chain_problem


@dataclass(frozen=True)
class Toy:
    """Evaluates the input order ``repeats`` times."""

    repeats: int = 2
    tags: list[str] = field(default_factory=list)
    derived: int = field(default=0, init=False)

    info: ClassVar[SolverInfo] = SolverInfo(
        id="toy-solver",
        name="Toy solver",
        family="random-sampling",
        summary="Evaluates the input order a few times.",
        references=("kolisch1996serial",),
    )

    def __post_init__(self) -> None:
        if self.repeats < 1:
            raise ValueError("repeats must be positive")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        search = Search(problem, stop, seed, self.info)
        with search:
            for _ in range(self.repeats):
                search.next_iteration()
                search.evaluate(list(range(problem.size)))
        return search.result()


PLUGIN_INFO = replace(Toy.info, id="plugin-solver", name="Plugin solver")


def good_plugin(registry: SolverRegistry) -> None:
    registry.register(Toy, PLUGIN_INFO)


def half_plugin(registry: SolverRegistry) -> None:
    registry.register(Toy, replace(PLUGIN_INFO, id="half-solver"))
    raise RuntimeError("second solver is broken")


def _install(monkeypatch: pytest.MonkeyPatch, *entries: tuple[str, str]) -> None:
    points = [
        EntryPoint(name=name, value=f"{__name__}:{target}", group=PLUGIN_GROUP)
        for name, target in entries
    ]

    def fake(*, group: str) -> list[EntryPoint]:
        return [point for point in points if point.group == group]

    monkeypatch.setattr(solver_module, "entry_points", fake)


def _registry() -> SolverRegistry:
    registry = SolverRegistry()
    registry.register(Toy)
    return registry


# -------------------------------------------------------------- registration


def test_registration_reads_parameters_from_dataclass_fields() -> None:
    registry = _registry()
    assert dict(registry.info("toy-solver").parameters) == {"repeats": 2, "tags": []}
    assert Toy.info.parameters == {}
    registry.register(Toy, replace(PLUGIN_INFO, parameters={"repeats": 5}))
    assert dict(registry.info("plugin-solver").parameters) == {"repeats": 5}


def test_registration_needs_metadata_and_a_free_id() -> None:
    registry = _registry()
    with pytest.raises(ConfigError, match="solver 'toy-solver' is already registered"):
        registry.register(Toy)
    registry.register(Toy, replace(Toy.info, name="Renamed"), replace_existing=True)
    assert registry.info("toy-solver").name == "Renamed"

    def bare() -> Toy:
        return Toy()

    with pytest.raises(ConfigError, match="no SolverInfo given"):
        registry.register(bare)
    with pytest.raises(ConfigError, match="unknown solver 'missing'; run 'dispatchatlas solvers"):
        registry.info("missing")


def test_lookups_and_copies() -> None:
    registry = _registry()
    registry.register(Toy, PLUGIN_INFO)
    assert registry.ids() == ("plugin-solver", "toy-solver")
    assert [info.id for info in registry.infos()] == ["plugin-solver", "toy-solver"]
    assert list(registry) == ["plugin-solver", "toy-solver"]
    assert len(registry) == 2
    assert "toy-solver" in registry
    assert "missing" not in registry
    clone = registry.copy()
    clone.register(Toy, replace(PLUGIN_INFO, id="clone-only"))
    assert "clone-only" in clone
    assert "clone-only" not in registry


def test_create_builds_solvers_and_reports_configuration_errors() -> None:
    registry = _registry()
    solver = registry.create("toy-solver", repeats=3)
    assert isinstance(solver, Solver)
    assert solver == Toy(repeats=3)
    result = solver.solve(chain_problem(), StopCriteria(max_iterations=10))
    assert (result.iterations, result.evaluations) == (3, 3)
    assert result.stop_reason is StopReason.COMPLETED
    for parameters, message in (
        (
            {"speed": 2},
            "solver 'toy-solver' has no parameters ['speed']; known: ['repeats', 'tags']",
        ),
        ({"derived": 1}, "has no parameters ['derived']"),
        ({"repeats": 0}, "invalid parameters for solver 'toy-solver': repeats must be positive"),
    ):
        with pytest.raises(ConfigError) as caught:
            registry.create("toy-solver", **parameters)
        assert message in str(caught.value)
    with pytest.raises(ConfigError, match="unknown solver 'missing'"):
        registry.create("missing")


# ------------------------------------------------------------------- plugins


def test_plugins_register_through_entry_points(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install(
        monkeypatch,
        ("b-half", "half_plugin"),
        ("a-missing", "no_such_function"),
        ("c-good", "good_plugin"),
    )
    registry = _registry()
    with caplog.at_level(logging.WARNING, logger="dispatchatlas"):
        assert load_plugins(registry) == ["c-good"]
    assert registry.ids() == ("plugin-solver", "toy-solver")
    warned = [record.getMessage() for record in caplog.records]
    assert len(warned) == 2
    assert "solver plugin 'a-missing' failed to load" in warned[0]
    assert "solver plugin 'b-half' failed to load: second solver is broken" in warned[1]
    assert load_plugins(SolverRegistry(), group="elsewhere") == []


def test_the_default_registry_is_fresh_and_optionally_extended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, ("good", "good_plugin"))
    plain = default_registry(plugins=False)
    assert {"input-order", "random-search", "heft"} <= set(plain.ids())
    assert "plugin-solver" not in plain
    extended = default_registry()
    assert "plugin-solver" in extended
    extended.register(Toy)
    assert "toy-solver" not in default_registry(plugins=False)
    for module in BUILT_IN_MODULES:
        assert callable(importlib.import_module(module).register), module
    for info in plain.infos():
        assert plain.info(info.id) is info


def test_solve_runs_a_registered_solver_by_id() -> None:
    problem = chain_problem()
    result = solve(problem, "input-order")
    assert (result.solver, result.value, result.stop_reason) == (
        "input-order",
        5.0,
        StopReason.COMPLETED,
    )
    sampled = solve(problem, "random-search", max_evaluations=4, seed=3, random_modes=True)
    assert sampled.evaluations == 4
    assert sampled.stop_reason is StopReason.EVALUATION_LIMIT
    again = solve(problem, "random-search", max_evaluations=4, seed=3, random_modes=True)
    assert (again.schedule, again.objectives) == (sampled.schedule, sampled.objectives)
    with pytest.raises(ConfigError, match="has no parameters"):
        solve(problem, "input-order", speed=2)
