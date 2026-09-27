"""Figure data: the outcome policy, the profiles, ranks, surfaces, and the drawing kernel.

The factories at the top build run and failure records without running a
solver, so every test states its data; ``test_figures_render`` reuses them.
"""

from __future__ import annotations

import math
import statistics
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping, Sequence
from itertools import pairwise
from types import SimpleNamespace

import pytest

from dispatchatlas.analysis import AnalysisConfig
from dispatchatlas.analysis.figures import (
    CATALOG,
    GOULD_SCOTT_CAVEAT,
    PALETTE,
    PROFILE_THRESHOLDS,
    Observations,
    Theme,
    attainment_surface,
    cliques,
    convergence,
    data_profile,
    deviation_distribution,
    gantt_chart,
    observe,
    pareto_fronts,
    performance_profile,
    rank_diagram,
    runtime_distribution,
)
from dispatchatlas.analysis.figures.axes import log_axis, numeric_axis, tick_labels
from dispatchatlas.analysis.figures.chart import box_stats
from dispatchatlas.analysis.figures.drawing import Figure, Text, number
from dispatchatlas.analysis.figures.frames import aggregate, lower_median
from dispatchatlas.analysis.figures.svg import escape
from dispatchatlas.analysis.figures.tex import tex_escape
from dispatchatlas.analysis.figures.theme import PALETTE_REFERENCE
from dispatchatlas.campaigns.config import Protocol
from dispatchatlas.campaigns.records import (
    Category,
    ErrorKind,
    FailureRecord,
    Outcome,
    ProblemRef,
    Provenance,
    RunError,
    RunRecord,
    SolverEntry,
    schedule_digest,
)
from dispatchatlas.core import (
    ConfigError,
    Mode,
    Problem,
    Resource,
    Schedule,
    Status,
    StopCriteria,
    StopReason,
    Task,
    TracePoint,
)
from dispatchatlas.core.objectives import Objective, register_objective, unregister_objective
from dispatchatlas.core.schedule import Assignment
from dispatchatlas.references import bibliography

PROTOCOL = Protocol("evals", StopCriteria(max_evaluations=100))
PROVENANCE = Provenance(
    attempt=1,
    session=1,
    started="2026-01-01T00:00:00+00:00",
    finished="2026-01-01T00:00:01+00:00",
    wall_seconds=1.0,
)


def _entry(solver: str) -> SolverEntry:
    return SolverEntry(solver, solver, None, {}, "random-sampling", True, True, False)


def trace(*points: tuple[int, float]) -> tuple[TracePoint, ...]:
    """Trace points ``(evaluations, value)``, one hundredth of a second per evaluation."""
    return tuple(
        TracePoint(evaluations * 0.01, evaluations, evaluations, value)
        for evaluations, value in points
    )


def record(
    solver: str,
    problem: str,
    value: float,
    *,
    replicate: int = 0,
    points: tuple[TracePoint, ...] = (),
    feasible: bool = True,
    objectives: Mapping[str, float] | None = None,
    front: tuple[tuple[float, ...], ...] = (),
    schedule: Schedule | None = None,
    protocol: Protocol = PROTOCOL,
    objective: str = "makespan",
) -> RunRecord:
    """Return the record of a completed run with the given primary value."""
    values = dict(objectives) if objectives is not None else {objective: value}
    evaluations = points[-1].evaluations if points else 1
    outcome = Outcome(
        status=Status.FEASIBLE if feasible else Status.VIOLATED,
        stop_reason=StopReason.EVALUATION_LIMIT,
        objective=objective,
        value=value,
        objectives=values,
        bound=None,
        gap=None,
        feasible=feasible,
        violations=() if feasible else ("deadline",),
        evaluations=evaluations,
        iterations=evaluations,
        elapsed=points[-1].elapsed if points else 0.01,
        trace=points,
        front=front,
        schedule_digest=None if schedule is None else schedule_digest(schedule),
    )
    return RunRecord(
        run_id=f"{solver}.{problem}.{protocol.name}.{replicate}",
        campaign="test",
        config_hash="hash",
        problem=ProblemRef(problem, None, "digest"),
        solver=_entry(solver),
        protocol=protocol,
        replicate=replicate,
        seed=replicate,
        outcome=outcome,
        result_hash=outcome.result_hash(),
        schedule=schedule,
        provenance=PROVENANCE,
    )


def failure(
    solver: str, problem: str, *, replicate: int = 0, protocol: Protocol = PROTOCOL
) -> FailureRecord:
    """Return the record of a failed attempt."""
    return FailureRecord(
        run_id=f"{solver}.{problem}.{protocol.name}.{replicate}",
        campaign="test",
        config_hash="hash",
        problem=ProblemRef(problem, None, "digest"),
        solver=_entry(solver),
        protocol=protocol,
        replicate=replicate,
        seed=replicate,
        category=Category.TERMINAL,
        error=RunError(ErrorKind.EXCEPTION, "the solver refused"),
        provenance=PROVENANCE,
    )


def results(runs: Sequence[RunRecord], failures: Sequence[FailureRecord] = ()) -> SimpleNamespace:
    """Return an object holding runs and failures, as a campaign reader would."""
    return SimpleNamespace(runs=tuple(runs), failures=tuple(failures))


def golden_records() -> list[RunRecord]:
    """The two-problem data of the Moré-Wild golden vector, at tau = 0.5."""
    return [
        record("ga", "p1", 40.0, points=trace((0, 100.0), (5, 60.0), (10, 40.0))),
        record("sa", "p1", 70.0, points=trace((0, 100.0), (5, 80.0), (10, 70.0))),
        record("ga", "p2", 45.0, points=trace((0, 50.0), (10, 45.0))),
        record("sa", "p2", 30.0, points=trace((0, 50.0), (10, 30.0))),
    ]


# ---------------------------------------------------------------- policy


def test_observe_reads_every_run_and_judges_it_by_one_policy() -> None:
    runs = [
        record("a", "p1", 10.0),
        record("a", "p2", 12.0, feasible=False),
        record("b", "p1", 11.0),
    ]
    failures = [failure("b", "p2"), failure("a", "p1")]
    observations = observe(results(runs, failures))
    status = {(run.solver, run.problem): run.status for run in observations.runs}
    assert status == {
        ("a", "p1"): "succeeded",
        ("a", "p2"): "infeasible",
        ("b", "p1"): "succeeded",
        ("b", "p2"): "failed",
    }
    assert observations.solvers == ("a", "b")
    assert observations.problems == ("p1", "p2")
    infeasible = observations.cell("a", "p2")[0]
    assert observations.value(infeasible) is None
    assert observations.score(infeasible) == math.inf
    assert observations.objective("p1") == "makespan"
    assert observe(runs).runs == tuple(run for run in observations.runs if run.status != "failed")


def test_observe_refuses_other_inputs() -> None:
    with pytest.raises(TypeError):
        observe(42)
    with pytest.raises(TypeError):
        observe(["not a record"])
    other = Protocol("time", StopCriteria(time_limit=1.0))
    mixed = [record("a", "p", 1.0), record("a", "p", 1.0, protocol=other)]
    with pytest.raises(ConfigError, match="protocols"):
        performance_profile(mixed)


@pytest.mark.usefixtures("yield_objective")
def test_replicates_aggregate_to_the_median_run_or_the_best_run() -> None:
    def cell(*values: float | None, objective: str = "makespan") -> Observations:
        runs = [
            record("a", "p", value, replicate=index, objective=objective)
            if value is not None
            else record("a", "p", 1.0, replicate=index, feasible=False, objective=objective)
            for index, value in enumerate(values)
        ]
        return observe(runs)

    assert cell(10.0, 14.0, None).cell_score("a", "p", "median") == 14.0
    assert cell(10.0, 14.0, None).cell_score("a", "p", "best") == 10.0
    assert cell(10.0, None).cell_score("a", "p", "median") == 10.0
    assert cell(None, None, 10.0).cell_score("a", "p", "median") == math.inf
    assert cell(3.0, 5.0, objective="figures_test_yield").cell_score("a", "p", "median") == -5.0
    assert aggregate([], "median") == math.inf
    assert lower_median([4.0, 1.0, 3.0, 2.0]) == 2.0


def test_figures_read_the_analysed_objective() -> None:
    def both(makespan: float, completion: float) -> dict[str, float]:
        return {"makespan": makespan, "total_completion_time": completion}

    runs = [
        record("a", "p", 10.0, objectives=both(10.0, 50.0), points=trace((1, 12.0), (5, 10.0))),
        record("b", "p", 12.0, objectives=both(12.0, 40.0), points=trace((2, 12.0))),
    ]
    observations = observe(runs)
    completion = observations.analysing("total_completion_time")
    first = observations.cell("a", "p")[0]
    assert observations.value(first) == 10.0
    assert completion.value(first) == 50.0
    assert completion.objective_label() == "total_completion_time"
    assert observations.progress(first, "evaluations") == ((1.0, 12.0), (5.0, 10.0))
    assert completion.progress(first, "evaluations") == ()
    profile = performance_profile(completion)
    assert {curve.solver: curve.ratios for curve in profile.curves} == {"a": (1.25,), "b": (1.0,)}
    assert profile.objective == "total_completion_time"
    assert data_profile(completion).knots == ()
    assert convergence(completion).curves == ()
    with pytest.raises(ConfigError, match="does not declare"):
        observations.analysing("energy")
    settings = AnalysisConfig(solvers=("b",), objective="total_completion_time")
    narrowed = observations.configured(settings)
    assert narrowed.solvers == ("b",)
    assert {run.solver for run in narrowed.runs} == {"b"}
    assert narrowed.target == "total_completion_time"
    with pytest.raises(ConfigError, match="unknown solver"):
        observations.restrict(solvers=("z",))


# ------------------------------------------------------------- profiles


def test_data_profile_matches_the_hand_computed_vector() -> None:
    profile = data_profile(golden_records(), accuracy=0.5, sizes={"p1": 4, "p2": 9})
    curves = {curve.solver: curve for curve in profile.curves}
    assert (curves["ga"].cells, curves["ga"].solved) == (2, 1)
    assert (curves["sa"].cells, curves["sa"].solved) == (2, 2)
    assert dict(profile.points("ga")) == {1.0: 0.5, 2.0: 0.5}
    assert dict(profile.points("sa")) == {1.0: 0.5, 2.0: 1.0}
    assert profile.unit == "evaluations / (tasks + 1)"


def test_data_profile_budget_limit_clips_the_step_function() -> None:
    profile = data_profile(
        golden_records(), accuracy=0.5, sizes={"p1": 4, "p2": 9}, budget_limit=1.5
    )
    curves = {curve.solver: curve for curve in profile.curves}
    assert curves["sa"].solved == 1
    assert dict(profile.points("sa")) == {1.0: 0.5}


def test_data_profile_validates_its_parameters() -> None:
    with pytest.raises(ConfigError, match="strictly within"):
        data_profile(golden_records(), accuracy=1.5)
    with pytest.raises(ConfigError, match="budget limit"):
        data_profile(golden_records(), budget_limit=0.0)
    with pytest.raises(ConfigError, match="size"):
        data_profile(golden_records(), sizes={"p1": -1})


def test_data_profile_keeps_infeasible_and_failed_cells_in_the_denominator() -> None:
    runs = [
        record("a", "p1", 10.0, points=trace((1, 20.0), (4, 10.0))),
        record("a", "p2", 5.0, points=trace((1, 5.0)), feasible=False),
        record("b", "p1", 20.0, points=trace((1, 20.0))),
        record("b", "p2", 8.0, points=trace((2, 8.0))),
    ]
    profile = data_profile(results(runs, [failure("a", "p3"), failure("b", "p3")]), accuracy=0.1)
    curves = {curve.solver: curve for curve in profile.curves}
    assert (curves["a"].cells, curves["a"].solved) == (3, 1)
    assert curves["a"].share(math.inf) == pytest.approx(1 / 3)
    assert profile.unit == "evaluations"


def test_performance_profile_matches_a_hand_case() -> None:
    runs = [
        record("a", "p1", 10.0),
        record("b", "p1", 11.0),
        record("c", "p1", 20.0),
        record("a", "p2", 12.0),
        record("b", "p2", 10.0),
        record("c", "p2", 9.0, feasible=False),
        record("a", "p3", 30.0),
        record("c", "p3", 30.0),
    ]
    profile = performance_profile(results(runs, [failure("b", "p3")]))
    ratios = {curve.solver: curve.ratios for curve in profile.curves}
    assert ratios["a"] == pytest.approx((1.0, 1.0, 1.2))
    assert ratios["b"][0] == 1.0
    assert ratios["b"][1] == pytest.approx(1.1)
    assert ratios["b"][2] == math.inf
    assert ratios["c"] == (1.0, 2.0, math.inf)
    shares = profile.at()
    assert PROFILE_THRESHOLDS == (1.0, 1.05, 1.1, 1.25, 1.5, 2.0)
    third = 1 / 3
    assert shares["a"] == pytest.approx((2 * third, 2 * third, 2 * third, 1.0, 1.0, 1.0))
    assert shares["b"] == pytest.approx((third, third, 2 * third, 2 * third, 2 * third, 2 * third))
    assert shares["c"] == pytest.approx((third, third, third, third, third, 2 * third))


@pytest.fixture
def yield_objective() -> Iterator[str]:
    """Register a maximised objective for one test and remove it afterwards."""
    register_objective(
        Objective("figures_test_yield", "max", "count", "A test yield.", lambda p, s: 0.0)
    )
    yield "figures_test_yield"
    unregister_objective("figures_test_yield")


@pytest.mark.usefixtures("yield_objective")
def test_performance_profile_of_maximised_objectives_and_undefined_ratios() -> None:
    runs = [
        record("a", "p1", 8.0, objective="figures_test_yield"),
        record("b", "p1", 10.0, objective="figures_test_yield"),
        record("a", "p2", 8.0, objective="energy"),
        record("b", "p2", 10.0, objective="energy"),
        record("a", "p3", 0.0, objective="total_tardiness"),
        record("b", "p3", 3.0, objective="total_tardiness"),
    ]
    profile = performance_profile(runs)
    assert profile.problems == ("p1", "p2")
    assert profile.excluded == ("p3",)
    ratios = {curve.solver: curve.ratios for curve in profile.curves}
    assert ratios == {"a": (1.0, 1.25), "b": (1.0, 1.25)}


def test_performance_profile_aggregation_is_explicit() -> None:
    runs = [
        record("a", "p", 10.0, replicate=0),
        record("a", "p", 30.0, replicate=1),
        record("a", "p", 40.0, replicate=2),
        record("b", "p", 20.0),
    ]
    median = {curve.solver: curve.ratios for curve in performance_profile(runs).curves}
    best = {
        curve.solver: curve.ratios for curve in performance_profile(runs, aggregation="best").curves
    }
    assert median == {"a": (1.5,), "b": (1.0,)}
    assert best == {"a": (1.0,), "b": (2.0,)}
    with pytest.raises(ConfigError, match="aggregation"):
        performance_profile(runs, aggregation="mean")


def test_runtime_distribution_counts_targets_reached_by_each_run() -> None:
    runs = [
        record("a", "p", 40.0, points=trace((100, 100.0), (200, 60.0), (400, 40.0))),
        record("b", "p", 80.0, points=trace((100, 80.0))),
    ]
    result = runtime_distribution(runs, targets=(0.5, 0.0))
    curves = {curve.solver: curve for curve in result.curves}
    assert curves["a"].knots == ((2.0, 0.5), (4.0, 1.0))
    assert curves["b"].knots == ()
    assert curves["a"].share(3.0) == 0.5
    assert result.horizon == pytest.approx(4.0)
    by_evaluations = runtime_distribution(runs, targets=(0.5, 0.0), budget="evaluations")
    assert {c.solver: c.knots for c in by_evaluations.curves}["a"] == ((200.0, 0.5), (400.0, 1.0))
    with pytest.raises(ConfigError, match="targets"):
        runtime_distribution(runs, targets=(0.5, 0.5))


def test_convergence_takes_the_median_gap_and_its_quartiles() -> None:
    runs = [
        record("a", "p", 10.0, replicate=0, points=trace((1, 20.0), (10, 10.0))),
        record("a", "p", 15.0, replicate=1, points=trace((1, 20.0), (10, 15.0))),
        record("a", "p", 20.0, replicate=2, points=trace((1, 20.0))),
    ]
    curves = convergence(runs, budget="evaluations", points=3)
    (curve,) = curves.curves
    assert curve.budgets == pytest.approx((1.0, math.sqrt(10.0), 10.0))
    assert curve.median == (1.0, 1.0, 0.5)
    gaps = [0.0, 0.5, 1.0]
    lower, _, upper = statistics.quantiles(gaps, n=4, method="inclusive")
    assert (curve.lower[-1], curve.upper[-1]) == (lower, upper)
    failed = convergence(results(runs, [failure("b", "p")]), budget="evaluations", points=2)
    assert {c.solver: c.median for c in failed.curves}["b"] == (1.0, 1.0)


def test_box_statistics_use_inclusive_quartiles() -> None:
    stats = box_stats([8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert (stats.minimum, stats.lower, stats.median, stats.upper, stats.maximum) == (
        1.0,
        2.75,
        4.5,
        6.25,
        8.0,
    )
    assert box_stats([3.0]).median == 3.0


def test_distribution_measures_deviation_from_the_best_value_found() -> None:
    runs = [
        record("a", "p1", 100.0),
        record("a", "p1", 110.0, replicate=1),
        record("b", "p1", 120.0),
        record("b", "p2", 0.0, objective="total_tardiness"),
    ]
    boxes = {
        box.solver: box for box in deviation_distribution(results(runs, [failure("a", "p2")])).boxes
    }
    assert boxes["a"].stats is not None
    assert (boxes["a"].stats.minimum, boxes["a"].stats.maximum) == (0.0, 10.0)
    assert boxes["b"].stats is not None
    assert boxes["b"].stats.median == pytest.approx(20.0)
    assert (boxes["a"].succeeded, boxes["a"].planned) == (2, 2)


# --------------------------------------------------------- ranks, fronts


def test_cliques_join_solvers_within_the_critical_difference() -> None:
    ranks = [("a", 1.2), ("b", 1.9), ("c", 2.6), ("d", 3.9)]
    assert cliques(ranks, 1.0) == (("a", "b"), ("b", "c"))
    assert cliques(ranks, 3.0) == (("a", "b", "c", "d"),)
    assert cliques(ranks, 0.5) == ()
    assert cliques([("x", 2.0), ("y", 2.0)], 0.1) == (("x", "y"),)
    diagram = rank_diagram({"c": 2.6, "a": 1.2, "b": 1.9}, 1.0, blocks=5)
    assert diagram.ranks == (("a", 1.2), ("b", 1.9), ("c", 2.6))
    assert diagram.cliques == (("a", "b"), ("b", "c"))
    with pytest.raises(ConfigError):
        rank_diagram({"a": 0.5})


def test_attainment_surfaces_summarise_replicates() -> None:
    runs = [[(1.0, 5.0), (3.0, 2.0)], [(2.0, 4.0)], []]
    assert attainment_surface(runs, 1) == ((1.0, 5.0), (2.0, 4.0), (3.0, 2.0))
    assert attainment_surface(runs, 2) == ((2.0, 5.0), (3.0, 4.0))
    assert attainment_surface(runs, 3) == ()


def test_pareto_fronts_cover_bi_objective_problems_only() -> None:
    both = {"makespan": 10.0, "total_completion_time": 30.0}
    runs = [
        record("a", "bi", 10.0, objectives=both, front=((10.0, 30.0), (12.0, 25.0))),
        record("b", "bi", 11.0, objectives={"makespan": 11.0, "total_completion_time": 20.0}),
        record("a", "mono", 5.0),
    ]
    (fronts,) = pareto_fronts(runs)
    assert fronts.problem == "bi"
    assert fronts.objectives == ("makespan", "total_completion_time")
    surfaces = {surface.solver: surface.points for surface in fronts.surfaces}
    assert surfaces == {"a": ((10.0, 30.0), (12.0, 25.0)), "b": ((11.0, 20.0),)}


# ---------------------------------------------------------------- Gantt


def test_gantt_packs_overlapping_tasks_into_rows() -> None:
    problem = Problem(
        "pool",
        tasks=(
            Task.on("a", "cores", 4, demand=1, group="j1"),
            Task.on("b", "cores", 2, demand=1, group="j2"),
            Task("c", (Mode(3, {"cores": 1, "m": 1}),), group="j1"),
        ),
        resources=(Resource("cores", capacity=2), Resource("m")),
    )
    schedule = Schedule.of(
        [Assignment("a", 0, 0.0, 4.0), Assignment("b", 0, 0.0, 2.0), Assignment("c", 0, 2.0, 5.0)]
    )
    chart = gantt_chart(schedule, problem)
    assert chart.rows == ("cores", "", "m")
    assert chart.keys == ("j1", "j2")
    rows: dict[str, set[int]] = {}
    for bar in chart.bars:
        rows.setdefault(bar.task, set()).add(bar.row)
    assert rows["c"] - {0, 1} == {2}
    assert rows["a"] | rows["b"] == {0, 1}
    for row in range(3):
        spans = sorted((bar.start, bar.end) for bar in chart.bars if bar.row == row)
        assert all(first[1] <= second[0] for first, second in pairwise(spans))
    tasks = gantt_chart(schedule)
    assert tasks.view == "tasks"
    assert tasks.rows == ("a", "b", "c")
    with pytest.raises(ConfigError):
        gantt_chart(Schedule.of([Assignment("z", 0, 0.0, 1.0)]), problem)


# --------------------------------------------------------------- kernel


def test_axes_have_nice_ticks_and_one_label_style() -> None:
    axis = numeric_axis("x", [3.2, 97.0])
    assert (axis.low, axis.high) == (0.0, 100.0)
    assert axis.ticks == (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
    assert axis.labels == ("0", "20", "40", "60", "80", "100")
    shares = numeric_axis("y", [], low=0.0, high=1.0)
    assert shares.labels == ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0")
    wide = log_axis("t", [3e-5, 0.004])
    assert (wide.low, wide.high) == (2e-5, 0.005)
    assert all("e" in label for label in wide.labels)
    assert log_axis("r", [1.0, 1.3], low=1.0).labels[0] == "1.0"
    assert tick_labels([0.0, 50000.0, 100000.0], 50000.0) == ("0", "5e4", "1e5")
    assert number(-0.001) == "0"
    assert number(2.50) == "2.5"


def test_text_is_escaped_for_both_formats() -> None:
    assert escape('a<b & "c" τ') == "a&lt;b &amp; &quot;c&quot; &#964;"
    assert tex_escape("50% of x_1 & {y} ≤ τ Demšar") == (
        r"50\% of x\_1 \& \{y\} $\leq$ $\tau$ Dem\v{s}ar"
    )
    theme = Theme()
    figure = Figure(
        "check",
        "Title <&>",
        "desc",
        400.0,
        100.0,
        (Text(10.0, 50.0, "x < y & z", 12.0, theme.foreground),),
    )
    root = ET.fromstring(figure.svg())
    assert root.findtext("{http://www.w3.org/2000/svg}title") == "Title <&>"
    assert figure.svg().isascii()
    assert figure.tex().isascii()


def test_theme_palette_is_colour_blind_safe_and_cited() -> None:
    assert len(PALETTE) >= 8
    assert len(set(PALETTE)) == len(PALETTE)
    theme = Theme()
    styles = {(s.color, s.marker, s.dash) for s in (theme.series(i) for i in range(56))}
    assert len(styles) == 56
    with pytest.raises(ConfigError):
        Theme(palette=("red",))
    known = bibliography()
    assert PALETTE_REFERENCE in known
    for spec in CATALOG:
        for key in spec.references:
            assert key in known, (spec.id, key)


def test_gould_scott_caveat_names_its_source() -> None:
    assert "Gould & Scott 2016" in GOULD_SCOTT_CAVEAT
    assert bibliography()["gould2016note"].doi == "10.1145/2950048"
