"""Figures of campaign results, as SVG and as TikZ, without a plotting library.

Write every figure of a campaign at once::

    from dispatchatlas.analysis import load_dataset
    from dispatchatlas.analysis.figures import render_figures

    paths = render_figures(load_dataset("results/my-campaign"), "figures")

or compute one figure's data and draw it::

    from dispatchatlas.analysis.figures import performance_profile, plot_performance_profile

    figure = plot_performance_profile(performance_profile(records))
    svg, tex = figure.svg(), figure.tex()

The families, in :data:`CATALOG` order: performance profiles (Dolan and Moré
2002), data profiles (Moré and Wild 2009), runtime distributions over quality
targets (Hansen et al. 2022), convergence over evaluations and over time,
the critical-difference diagram (Demšar 2006), box plots of relative
deviations, Pareto fronts as attainment surfaces (Fonseca and Fleming 1996),
and the Gantt chart of one schedule.

The modules:

- :mod:`~dispatchatlas.analysis.figures.frames`: the runs, one outcome
  policy, and explicit aggregation over replicates;
- :mod:`~dispatchatlas.analysis.figures.profiles`,
  :mod:`~dispatchatlas.analysis.figures.traces`,
  :mod:`~dispatchatlas.analysis.figures.ranking`,
  :mod:`~dispatchatlas.analysis.figures.distribution`,
  :mod:`~dispatchatlas.analysis.figures.pareto`, and
  :mod:`~dispatchatlas.analysis.figures.gantt`: each family's data and figure;
- :mod:`~dispatchatlas.analysis.figures.catalog`: the list of families and
  :func:`render_figures`;
- :mod:`~dispatchatlas.analysis.figures.chart`,
  :mod:`~dispatchatlas.analysis.figures.axes`,
  :mod:`~dispatchatlas.analysis.figures.drawing`,
  :mod:`~dispatchatlas.analysis.figures.theme`,
  :mod:`~dispatchatlas.analysis.figures.svg`, and
  :mod:`~dispatchatlas.analysis.figures.tex`: the drawing kernel.
"""

from dispatchatlas.analysis.figures.catalog import (
    CATALOG,
    FORMATS,
    FigureContext,
    FigureSpec,
    family_of,
    figure_spec,
    render_figures,
)
from dispatchatlas.analysis.figures.distribution import (
    Box,
    Distribution,
    deviation_distribution,
    plot_distribution,
)
from dispatchatlas.analysis.figures.drawing import Figure
from dispatchatlas.analysis.figures.frames import (
    AGGREGATIONS,
    BUDGETS,
    Aggregation,
    Budget,
    Observation,
    Observations,
    observe,
)
from dispatchatlas.analysis.figures.gantt import (
    GanttChart,
    gantt_chart,
    plot_gantt,
    representative_schedule,
)
from dispatchatlas.analysis.figures.pareto import (
    ParetoFronts,
    Surface,
    attainment_surface,
    pareto_fronts,
    plot_pareto_fronts,
)
from dispatchatlas.analysis.figures.profiles import (
    GOULD_SCOTT_CAVEAT,
    PROFILE_THRESHOLDS,
    QUALITY_TARGETS,
    DataCurve,
    DataProfile,
    PerformanceProfile,
    ProfileCurve,
    RuntimeCurve,
    RuntimeDistribution,
    data_profile,
    performance_profile,
    plot_data_profile,
    plot_performance_profile,
    plot_runtime_distribution,
    runtime_distribution,
)
from dispatchatlas.analysis.figures.ranking import (
    RankDiagram,
    average_ranks,
    cliques,
    plot_critical_difference,
    rank_diagram,
)
from dispatchatlas.analysis.figures.svg import to_svg
from dispatchatlas.analysis.figures.tex import to_tex
from dispatchatlas.analysis.figures.theme import DEFAULT_THEME, PALETTE, Theme
from dispatchatlas.analysis.figures.traces import (
    Convergence,
    ConvergenceCurve,
    convergence,
    plot_convergence,
)

__all__ = [
    "AGGREGATIONS",
    "BUDGETS",
    "CATALOG",
    "DEFAULT_THEME",
    "FORMATS",
    "GOULD_SCOTT_CAVEAT",
    "PALETTE",
    "PROFILE_THRESHOLDS",
    "QUALITY_TARGETS",
    "Aggregation",
    "Box",
    "Budget",
    "Convergence",
    "ConvergenceCurve",
    "DataCurve",
    "DataProfile",
    "Distribution",
    "Figure",
    "FigureContext",
    "FigureSpec",
    "GanttChart",
    "Observation",
    "Observations",
    "ParetoFronts",
    "PerformanceProfile",
    "ProfileCurve",
    "RankDiagram",
    "RuntimeCurve",
    "RuntimeDistribution",
    "Surface",
    "Theme",
    "attainment_surface",
    "average_ranks",
    "cliques",
    "convergence",
    "data_profile",
    "deviation_distribution",
    "family_of",
    "figure_spec",
    "gantt_chart",
    "observe",
    "pareto_fronts",
    "performance_profile",
    "plot_convergence",
    "plot_critical_difference",
    "plot_data_profile",
    "plot_distribution",
    "plot_gantt",
    "plot_pareto_fronts",
    "plot_performance_profile",
    "plot_runtime_distribution",
    "rank_diagram",
    "render_figures",
    "representative_schedule",
    "runtime_distribution",
    "to_svg",
    "to_tex",
]
