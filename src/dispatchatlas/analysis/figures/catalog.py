"""The figure catalog, and the function that writes every figure of a campaign.

:data:`CATALOG` is the one list of figure families: each
:class:`FigureSpec` names a family, describes it, cites its method, and
draws it from a :class:`FigureContext`: the runs of one protocol, the
analysis settings, and the theme. :func:`render_figures` writes every figure
whose inputs are present, in both formats, under stable file names:
``<name>.svg`` and ``<name>.tex``, where the name is the family id, or, for
the Pareto fronts, ``pareto-front-<problem>``; :func:`family_of` maps a name
back to its family. A family whose inputs are absent, such as fronts in a
campaign of single-objective problems, writes nothing.

The settings make the figures read what the tables of the same analysis
read: the protocol, solvers, and problems it selects, and the objective and
sense it analyses. The critical-difference diagram also follows its
statistic, omnibus test, and significance level.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.figures.distribution import deviation_distribution, plot_distribution
from dispatchatlas.analysis.figures.drawing import Figure
from dispatchatlas.analysis.figures.frames import Budget, Observations, observe
from dispatchatlas.analysis.figures.gantt import gantt_chart, plot_gantt, representative_schedule
from dispatchatlas.analysis.figures.pareto import pareto_fronts, plot_pareto_fronts
from dispatchatlas.analysis.figures.profiles import (
    data_profile,
    performance_profile,
    plot_data_profile,
    plot_performance_profile,
    plot_runtime_distribution,
    runtime_distribution,
)
from dispatchatlas.analysis.figures.ranking import average_ranks, plot_critical_difference
from dispatchatlas.analysis.figures.theme import DEFAULT_THEME, Theme
from dispatchatlas.analysis.figures.traces import convergence, plot_convergence
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.io import atomic_write_text
from dispatchatlas.core.logs import get_logger

__all__ = [
    "CATALOG",
    "FORMATS",
    "FigureContext",
    "FigureSpec",
    "family_of",
    "figure_spec",
    "render_figures",
]

FORMATS: tuple[str, ...] = ("svg", "tex")
"""The formats figures are written in."""

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class FigureContext:
    """What a figure family draws from.

    Attributes:
        observations: The runs of one protocol, read for the analysed objective.
        config: The analysis settings.
        theme: The appearance.
    """

    observations: Observations
    config: AnalysisConfig = field(default_factory=AnalysisConfig)
    theme: Theme = DEFAULT_THEME


type Builder = Callable[[FigureContext], tuple[Figure, ...]]


@dataclass(frozen=True)
class FigureSpec:
    """One family of figures.

    Attributes:
        id: Stable identifier, also the file stem of the family's figure.
        title: Title of the family.
        caption: What the figures show and how to read them.
        builder: Draws the family's figures, and returns none when their
            inputs are absent.
        references: Bibliography keys of the methods the figures implement.
        prefix: For a family with one figure per problem, the start of
            their file stems; empty when the stem is the id.
    """

    id: str
    title: str
    caption: str
    builder: Builder
    references: tuple[str, ...] = ()
    prefix: str = ""

    def owns(self, name: str) -> bool:
        """Return whether a figure's file stem belongs to this family."""
        return name.startswith(self.prefix) if self.prefix else name == self.id


def _performance(context: FigureContext) -> tuple[Figure, ...]:
    profile = performance_profile(context.observations)
    if not any(curve.solved for curve in profile.curves):
        return ()
    return (plot_performance_profile(profile, theme=context.theme),)


def _data(context: FigureContext) -> tuple[Figure, ...]:
    profile = data_profile(context.observations)
    if not profile.knots:
        return ()
    return (plot_data_profile(profile, theme=context.theme),)


def _runtime(context: FigureContext) -> tuple[Figure, ...]:
    result = runtime_distribution(context.observations)
    if not any(curve.knots for curve in result.curves):
        return ()
    return (plot_runtime_distribution(result, theme=context.theme),)


def _convergence(budget: Budget) -> Builder:
    def build(context: FigureContext) -> tuple[Figure, ...]:
        curves = convergence(context.observations, budget=budget)
        if not curves.curves:
            return ()
        return (plot_convergence(curves, theme=context.theme),)

    return build


def _critical_difference(context: FigureContext) -> tuple[Figure, ...]:
    diagram = average_ranks(context.observations, context.config)
    if len(diagram.ranks) < 2:
        return ()
    return (plot_critical_difference(diagram, theme=context.theme),)


def _distribution(context: FigureContext) -> tuple[Figure, ...]:
    boxes = deviation_distribution(context.observations)
    if not any(box.stats is not None for box in boxes.boxes):
        return ()
    return (plot_distribution(boxes, theme=context.theme),)


def _pareto(context: FigureContext) -> tuple[Figure, ...]:
    return tuple(
        plot_pareto_fronts(fronts, theme=context.theme)
        for fronts in pareto_fronts(context.observations)
        if any(surface.points for surface in fronts.surfaces)
    )


def _gantt(context: FigureContext) -> tuple[Figure, ...]:
    observations = context.observations
    run = representative_schedule(observations)
    if run is None or run.record is None or run.record.schedule is None:
        return ()
    value = observations.value(run)
    title = f"Best recorded schedule on {run.problem}: {run.solver}"
    if value is not None:
        title += f", {observations.objective(run.problem)} {value:g}"
    return (plot_gantt(gantt_chart(run.record.schedule, title=title), theme=context.theme),)


CATALOG: tuple[FigureSpec, ...] = (
    FigureSpec(
        "performance-profile",
        "Performance profile",
        "For each solver, the share of problems on which its median run comes within a factor "
        "tau of the best solver's value; failed runs never do. With more than two solvers the "
        "profile compares each solver with the best one only.",
        _performance,
        ("dolan2002benchmarking", "gould2016note"),
    ),
    FigureSpec(
        "data-profile",
        "Data profile",
        "For each solver, the share of problems its median run solves to accuracy 0.001 "
        "within a budget of evaluations per task plus one.",
        _data,
        ("more2009benchmarking",),
    ),
    FigureSpec(
        "runtime-ecdf",
        "Runtime to reach quality targets",
        "For each solver, the share of (problem, quality target) pairs its runs reach within a "
        "given time, with targets between the best value found and the worst first value.",
        _runtime,
        ("hansen2022anytime",),
    ),
    FigureSpec(
        "convergence-evaluations",
        "Convergence over evaluations",
        "Median gap to the best value found, with its interquartile range, against the "
        "evaluations spent.",
        _convergence("evaluations"),
    ),
    FigureSpec(
        "convergence-time",
        "Convergence over time",
        "Median gap to the best value found, with its interquartile range, against the elapsed "
        "time.",
        _convergence("time"),
    ),
    FigureSpec(
        "critical-difference",
        "Critical-difference diagram",
        "Average ranks over the problems every solver is planned on; solvers the Nemenyi test "
        "does not separate are joined by a bar.",
        _critical_difference,
        ("demsar2006statistical",),
    ),
    FigureSpec(
        "distribution",
        "Relative deviation from the best value found",
        "Box plots of each solver's relative percentage deviation from the best value found, "
        "over its succeeded runs, with the number of succeeded runs.",
        _distribution,
        ("hyndman1996sample",),
    ),
    FigureSpec(
        "pareto-fronts",
        "Pareto fronts",
        "For each problem with two objectives, the median attainment surface of each solver's "
        "runs.",
        _pareto,
        ("fonseca1996performance",),
        prefix="pareto-front-",
    ),
    FigureSpec(
        "gantt",
        "Gantt chart",
        "The best recorded schedule on the first problem, one row per task, coloured by mode.",
        _gantt,
    ),
)
"""Every figure family, in the order figures are written."""


def figure_spec(identifier: str) -> FigureSpec:
    """Return the catalog entry with the given id.

    Raises:
        ConfigError: If no entry has that id.
    """
    for spec in CATALOG:
        if spec.id == identifier:
            return spec
    known = ", ".join(spec.id for spec in CATALOG)
    raise ConfigError(f"unknown figure {identifier!r}; known: {known}")


def family_of(name: str) -> FigureSpec | None:
    """Return the family a figure's file stem belongs to, or ``None`` for a foreign name."""
    return next((spec for spec in CATALOG if spec.owns(name)), None)


def render_figures(
    dataset: object,
    directory: str | Path,
    *,
    formats: Sequence[str] = FORMATS,
    config: AnalysisConfig | None = None,
) -> list[Path]:
    """Write every applicable figure of a campaign's runs.

    Args:
        dataset: An analysis :class:`~dispatchatlas.analysis.dataset.Dataset`,
            a sequence of :class:`~dispatchatlas.campaigns.records.RunRecord`,
            or an object with a ``runs`` attribute holding run records and
            optionally a ``failures`` attribute holding failure records.
        directory: Where to write. When the runs span several protocols,
            each protocol's figures go to a subdirectory named after it.
        formats: Formats to write, from ``"svg"`` and ``"tex"``.
        config: The analysis settings, the defaults when ``None``: its
            protocol, solvers, and problems select the runs, its objective
            and sense are the ones figures read, and the critical-difference
            diagram follows its statistic, omnibus test, and alpha.

    Returns:
        The paths written, in catalog order, then figure order, then format
        order; empty when no figure applies.

    Raises:
        ConfigError: If a format is unknown or given twice, or the settings
            name something the runs lack.
        TypeError: If ``dataset`` is not one of the accepted inputs.
    """
    chosen = tuple(formats)
    for fmt in chosen:
        if fmt not in FORMATS:
            raise ConfigError(f"unknown figure format {fmt!r}; use {', '.join(FORMATS)}")
    if len(set(chosen)) != len(chosen):
        raise ConfigError(f"a figure format is given twice: {list(chosen)}")
    settings = config if config is not None else AnalysisConfig()
    observations = observe(dataset).configured(settings)
    target = Path(directory)
    written: list[Path] = []
    split = len(observations.protocols) > 1
    for protocol in observations.protocols:
        context = FigureContext(observations.select(protocol), settings, DEFAULT_THEME)
        folder = target / protocol if split else target
        for spec in CATALOG:
            figures = spec.builder(context)
            if not figures:
                _LOG.debug("figure %s does not apply to protocol %s", spec.id, protocol)
            for figure in figures:
                for fmt in chosen:
                    path = folder / f"{figure.name}.{fmt}"
                    written.append(atomic_write_text(path, figure.render(fmt)))
    return written
