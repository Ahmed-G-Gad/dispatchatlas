"""Rendering: every catalog figure from a real campaign, determinism, layout, and TeX."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

import pytest

from dispatchatlas.analysis import AnalysisConfig, load_dataset, summarize, write_bundle
from dispatchatlas.analysis.figures import (
    CATALOG,
    Convergence,
    DataProfile,
    Distribution,
    Figure,
    FigureContext,
    GanttChart,
    ParetoFronts,
    PerformanceProfile,
    RuntimeDistribution,
    average_ranks,
    figure_spec,
    observe,
    performance_profile,
    plot_convergence,
    plot_critical_difference,
    plot_data_profile,
    plot_distribution,
    plot_gantt,
    plot_pareto_fronts,
    plot_performance_profile,
    plot_runtime_distribution,
    rank_diagram,
    render_figures,
)
from dispatchatlas.analysis.figures.drawing import bounds
from dispatchatlas.campaigns import (
    Campaign,
    Execution,
    FamilySource,
    FileSource,
    Protocol,
    RunRecord,
    SolverSpec,
    read_failures,
    read_runs,
    run,
)
from dispatchatlas.core import (
    ConfigError,
    Problem,
    Resource,
    Schedule,
    SolverError,
    SolveResult,
    SolverInfo,
    StopCriteria,
    Task,
    save_problem,
)
from dispatchatlas.core.schedule import Assignment
from dispatchatlas.references import cite
from dispatchatlas.solvers import default_registry
from tests.analysis.test_figures import failure, record, results, trace

SVG = "{http://www.w3.org/2000/svg}"


@dataclass(frozen=True)
class Refusing:
    """A solver that raises on every problem, so its runs fail."""

    info: ClassVar[SolverInfo] = SolverInfo(
        id="test-refusing",
        name="Refusing",
        family="random-sampling",
        summary="Raises on every problem.",
        stochastic=False,
        iterative=False,
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        raise SolverError("this solver refuses every problem")


@pytest.fixture(scope="module")
def campaign(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run cheap solvers, one of them multi-objective and one failing, on small problems.

    The problem ``late`` has a deadline no schedule meets, so every run on it
    is infeasible.
    """
    root = tmp_path_factory.mktemp("figures")
    late = Problem(
        "late",
        tasks=(Task.on("a", "m1", 3, deadline=2.0), Task.on("b", "m1", 2), Task.on("c", "m2", 4)),
        resources=(Resource("m1"), Resource("m2")),
    )
    save_problem(late, root / "late.json")
    registry = default_registry(plugins=False)
    registry.register(Refusing)
    definition = Campaign(
        id="figures",
        seed=5,
        instances=(
            FamilySource("flow-shop", (1, 2), {"jobs": 4, "machines": 3}),
            FileSource(root / "late.json"),
        ),
        solvers=(
            SolverSpec("input-order"),
            SolverSpec("random-search"),
            SolverSpec("hill-climbing", "hc", {"restart_after": 5}),
            SolverSpec("nsga-ii", parameters={"population_size": 6}),
            SolverSpec("test-refusing"),
        ),
        protocols=(Protocol("evals", StopCriteria(max_evaluations=30)),),
        replicates=2,
        objectives=("makespan", "total_completion_time"),
        execution=Execution(backend="sequential"),
    )
    return run(definition, root / "results", registry=registry).directory


def _stems(paths: list[Path]) -> list[str]:
    return list(dict.fromkeys(path.stem for path in paths))


def _spec_of(stem: str) -> str:
    return "pareto-fronts" if stem.startswith("pareto-front-") else stem


def test_every_catalog_figure_renders_in_both_formats(campaign: Path, tmp_path: Path) -> None:
    paths = render_figures(load_dataset(campaign), tmp_path / "figures")
    assert sorted(paths) == sorted(p for p in (tmp_path / "figures").iterdir())
    stems = _stems(paths)
    order = [spec.id for spec in CATALOG]
    positions = [order.index(_spec_of(stem)) for stem in stems]
    assert positions == sorted(positions)
    assert {_spec_of(stem) for stem in stems} == set(order)
    assert "pareto-front-late" not in stems
    for stem in stems:
        svg, tex = tmp_path / "figures" / f"{stem}.svg", tmp_path / "figures" / f"{stem}.tex"
        assert paths.index(svg) + 1 == paths.index(tex)
        root = ET.parse(svg).getroot()
        assert root.tag == f"{SVG}svg"
        assert root.findtext(f"{SVG}title")
        text = tex.read_text(encoding="ascii")
        assert text.count(r"\begin{tikzpicture}") == 1
        assert text.startswith("% ")
        assert r"\documentclass" in text
    profile = (tmp_path / "figures" / "performance-profile.svg").read_text()
    assert "Gould &amp; Scott 2016" in profile


def test_records_and_failures_render_the_same_figures(campaign: Path, tmp_path: Path) -> None:
    from_dataset = render_figures(load_dataset(campaign), tmp_path / "dataset")
    reader = results(read_runs(campaign), read_failures(campaign))
    from_records = render_figures(reader, tmp_path / "records")
    assert _stems(from_records) == _stems(from_dataset)
    sequence = render_figures(list(read_runs(campaign)), tmp_path / "sequence")
    assert _stems(sequence) == _stems(from_dataset)


def test_rendering_is_byte_identical(campaign: Path, tmp_path: Path) -> None:
    dataset = load_dataset(campaign)
    first = render_figures(dataset, tmp_path / "one")
    second = render_figures(dataset, tmp_path / "two")
    assert [path.name for path in first] == [path.name for path in second]
    for one, two in zip(first, second, strict=True):
        assert one.read_bytes() == two.read_bytes()


def test_the_figure_ranks_agree_with_the_summary(campaign: Path) -> None:
    dataset = load_dataset(campaign)
    summary = summarize(dataset)
    diagram = average_ranks(dataset)
    expected = {row.solver: row.average_rank for row in summary.solver_summaries}
    assert dict(diagram.ranks) == pytest.approx(expected)
    if summary.critical_difference is None:
        assert diagram.critical_difference is None
    else:
        assert diagram.critical_difference == pytest.approx(summary.critical_difference.value)
        assert diagram.cliques == tuple(
            g for g in summary.critical_difference.cliques if len(g) > 1
        )


@pytest.mark.parametrize(
    "settings",
    [
        AnalysisConfig(alpha=0.1),
        AnalysisConfig(statistic="median", omnibus="friedman"),
        AnalysisConfig(objective="total_completion_time", solvers=("input-order", "hc", "nsga-ii")),
    ],
)
def test_the_figures_follow_the_analysis_settings(
    campaign: Path, tmp_path: Path, settings: AnalysisConfig
) -> None:
    dataset = load_dataset(campaign)
    summary = summarize(dataset, settings)
    diagram = average_ranks(dataset, settings)
    expected = {row.solver: row.average_rank for row in summary.solver_summaries}
    assert dict(diagram.ranks) == pytest.approx(expected)
    if summary.critical_difference is None:
        assert diagram.critical_difference is None
    else:
        assert diagram.critical_difference == pytest.approx(summary.critical_difference.value)
    paths = render_figures(dataset, tmp_path, config=settings)
    drawn = (tmp_path / "critical-difference.svg").read_text(encoding="ascii")
    assert drawn == plot_critical_difference(diagram).svg()
    stems = set(_stems(paths))
    if settings.objective == "total_completion_time":
        assert not stems & {"data-profile", "runtime-ecdf", "convergence-evaluations"}
        assert "performance-profile" in stems
        assert "total_completion_time" in (tmp_path / "distribution.svg").read_text()


def _bundle_files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_the_bundle_writes_and_lists_the_figures(campaign: Path, tmp_path: Path) -> None:
    settings = AnalysisConfig(resamples=100, alpha=0.1)
    bundle = write_bundle(campaign, tmp_path / "bundle", settings)
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["figures"]["rendered"] is True
    for name in ("performance-profile.svg", "performance-profile.tex", "gantt.svg"):
        relative = f"figures/evals/{name}"
        assert (bundle / relative).is_file()
        assert relative in manifest["files"]
    focused = replace(settings, protocol="evals")
    diagram = average_ranks(load_dataset(campaign).select(protocols=["evals"]), focused)
    drawn = (bundle / "figures/evals/critical-difference.svg").read_text(encoding="ascii")
    assert drawn == plot_critical_difference(diagram).svg()
    index = (bundle / "index.md").read_text(encoding="utf-8")
    profile = figure_spec("performance-profile")
    links = (
        "[svg](figures/evals/performance-profile.svg) \u00b7 "
        "[tex](figures/evals/performance-profile.tex)"
    )
    assert f"- **{profile.title}** ({links}). {profile.caption}" in index
    for key in profile.references:
        assert f"  - {cite(key)}" in index
    assert "flow-shop-4x3-1: [svg](figures/evals/pareto-front-flow-shop-4x3-1.svg)" in index
    assert index.index("**Performance profile**") < index.index("**Gantt chart**")
    again = write_bundle(campaign, tmp_path / "again", settings)
    assert _bundle_files(again) == _bundle_files(bundle)


def test_empty_input_renders_nothing_and_every_family_has_a_placeholder(tmp_path: Path) -> None:
    assert render_figures([], tmp_path / "none") == []
    assert not (tmp_path / "none").exists()
    empty = observe([])
    for spec in CATALOG:
        assert spec.builder(FigureContext(empty)) == ()
    placeholders = [
        plot_performance_profile(PerformanceProfile((), (), (), "median")),
        plot_data_profile(DataProfile((), (), 0.001, "evaluations", ())),
        plot_runtime_distribution(RuntimeDistribution((), (), (0.0,), "time", 0.0)),
        plot_convergence(Convergence((), "evaluations", ())),
        plot_critical_difference(rank_diagram({})),
        plot_distribution(Distribution((), (), ())),
        plot_pareto_fronts(ParetoFronts("p", ("makespan", "cost"), ("min", "min"), (), "median")),
        plot_gantt(GanttChart("Schedule", "tasks", (), (), (), 0.0)),
    ]
    for figure in placeholders:
        assert ET.fromstring(figure.svg()).findtext(f"{SVG}desc")
        assert figure.tex().count(r"\begin{tikzpicture}") == 1


def _many_solvers(count: int) -> list[RunRecord]:
    """Records of ``count`` solvers with long names on three bi-objective problems."""
    runs: list[RunRecord] = []
    for index in range(count):
        solver = f"a-solver-with-a-rather-long-name-{index:02d}"
        for position, problem in enumerate(("p1", "p2", "p3")):
            value = 100.0 + index + 7.0 * position
            schedule = Schedule.of(
                [
                    Assignment("t1", 0, 0.0, value / 2.0),
                    Assignment("t2", index % 3, value / 2.0, value),
                ]
            )
            runs.append(
                record(
                    solver,
                    problem,
                    value,
                    points=trace((1, 3.0 * value), (5 + index, 1.5 * value), (40 + index, value)),
                    objectives={"makespan": value, "total_completion_time": 900.0 - 5.0 * index},
                    schedule=schedule,
                )
            )
    return runs


def _inside(figure: Figure) -> None:
    for shape in figure.shapes:
        left, top, right, bottom = bounds(shape)
        assert left >= -0.5, (figure.name, shape)
        assert top >= -0.5, (figure.name, shape)
        assert right <= figure.width + 0.5, (figure.name, shape)
        assert bottom <= figure.height + 0.5, (figure.name, shape)
    root = ET.fromstring(figure.svg())
    width, height = figure.width, figure.height
    for element in root.iter():
        for name in ("x", "x1", "x2", "cx"):
            if name in element.attrib:
                assert 0.0 <= float(element.attrib[name]) <= width, (figure.name, element.attrib)
        for name in ("y", "y1", "y2", "cy"):
            if name in element.attrib:
                assert 0.0 <= float(element.attrib[name]) <= height, (figure.name, element.attrib)
        for pair in element.attrib.get("points", "").split():
            x, y = (float(value) for value in pair.split(","))
            assert 0.0 <= x <= width, (figure.name, pair)
            assert 0.0 <= y <= height, (figure.name, pair)


def test_many_series_stay_inside_the_canvas(tmp_path: Path) -> None:
    runs = _many_solvers(40)
    failures = [failure("a-solver-that-always-fails", problem) for problem in ("p1", "p2", "p3")]
    observations = observe(results(runs, failures))
    for spec in CATALOG:
        figures = spec.builder(FigureContext(observations))
        assert figures, spec.id
        for figure in figures:
            _inside(figure)
    profile = figure_spec("performance-profile")
    few = profile.builder(FigureContext(observe(_many_solvers(3))))
    many = profile.builder(FigureContext(observations))
    assert many[0].height > few[0].height


def test_labels_fall_back_to_raw_ids() -> None:
    runs = [record("a", "p", 10.0), record("my-plugin-solver", "p", 12.0)]
    figure = plot_performance_profile(performance_profile(runs), labels={"a": "Algorithm A"})
    svg = figure.svg()
    assert ">Algorithm A<" in svg
    assert ">my-plugin-solver<" in svg
    assert ">a<" not in svg


def test_several_protocols_render_into_subdirectories(tmp_path: Path) -> None:
    other = Protocol("time", StopCriteria(time_limit=1.0))
    runs = [
        record("a", "p", 10.0, points=trace((1, 12.0), (3, 10.0))),
        record("b", "p", 11.0, points=trace((2, 11.0))),
        record("a", "p", 9.0, points=trace((1, 9.0)), protocol=other),
        record("b", "p", 12.0, points=trace((1, 12.0)), protocol=other),
    ]
    paths = render_figures(runs, tmp_path)
    folders = {path.parent.name for path in paths}
    assert folders == {"evals", "time"}
    assert (tmp_path / "evals" / "performance-profile.svg") in paths


def test_formats_are_checked(tmp_path: Path) -> None:
    runs = [record("a", "p", 10.0), record("b", "p", 12.0)]
    with pytest.raises(ConfigError, match="unknown figure format"):
        render_figures(runs, tmp_path, formats=("png",))
    with pytest.raises(ConfigError, match="twice"):
        render_figures(runs, tmp_path, formats=("svg", "svg"))
    only = render_figures(runs, tmp_path, formats=("tex",))
    assert only
    assert all(path.suffix == ".tex" for path in only)
    with pytest.raises(ConfigError, match="unknown figure"):
        figure_spec("pie-chart")


def _latex(directory: Path, name: str, text: str) -> subprocess.CompletedProcess[str]:
    source = directory / f"{name}.tex"
    source.write_text(text, encoding="ascii")
    return subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", source.name],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
def test_every_tex_figure_compiles(campaign: Path, tmp_path: Path) -> None:
    paths = [
        path for path in render_figures(load_dataset(campaign), tmp_path) if path.suffix == ".tex"
    ]
    documents = [path.read_text(encoding="ascii") for path in paths]
    work = tmp_path / "latex"
    work.mkdir()
    standalone = _latex(work, "single", documents[0])
    assert standalone.returncode == 0, standalone.stdout[-2000:]
    pictures = []
    for document in documents:
        preamble, _, body = document.partition(r"\begin{document}")
        assert preamble.strip().endswith(r"\usepackage{tikz}")
        pictures.append(body.replace(r"\end{document}", "").strip())
    combined = (
        "\\documentclass[border=0pt,multi=tikzpicture]{standalone}\n\\usepackage{tikz}\n"
        "\\begin{document}\n" + "\n".join(pictures) + "\n\\end{document}\n"
    )
    together = _latex(work, "combined", combined)
    assert together.returncode == 0, together.stdout[-2000:]
    log = (work / "combined.log").read_text(errors="replace")
    assert not re.search(r"^! ", log, flags=re.MULTILINE)
    assert f"({len(pictures)} pages" in log


def test_families_without_their_inputs_are_skipped(tmp_path: Path) -> None:
    runs = [record("a", "p1", 10.0), record("b", "p1", 12.0), record("a", "p2", 5.0)]
    stems = _stems(render_figures(runs, tmp_path))
    absent = ("pareto-fronts", "gantt")
    assert stems == [spec.id for spec in CATALOG if spec.id not in absent]
    for identifier in absent:
        assert figure_spec(identifier).builder(FigureContext(observe(runs))) == ()
