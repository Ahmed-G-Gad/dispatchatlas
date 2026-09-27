"""The Gantt chart of one schedule.

Given the problem, the chart has one row per resource, and a task appears on
the row of every resource its mode holds, coloured by its job (its group, or
itself without one). A resource that runs several tasks at once, such as a
pool of cores, gets as many rows as it needs, filled first-fit in order of
start time, so bars never overlap. Without the problem, which the run
records do not hold, the chart has one row per task, in order of start
time, coloured by the index of the mode the task runs in. A dashed line
marks the makespan.

:func:`representative_schedule` picks the schedule the figure catalog draws
from a campaign: the best recorded schedule on the first problem.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from dispatchatlas.analysis.figures.axes import numeric_axis
from dispatchatlas.analysis.figures.chart import (
    Bar,
    Guide,
    LegendEntry,
    Row,
    placeholder,
    row_chart,
)
from dispatchatlas.analysis.figures.drawing import Figure
from dispatchatlas.analysis.figures.frames import Observation, observe
from dispatchatlas.analysis.figures.theme import DEFAULT_THEME, SeriesStyle, Theme
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.model import Problem
from dispatchatlas.core.schedule import Assignment, Schedule

__all__ = ["GanttBar", "GanttChart", "gantt_chart", "plot_gantt", "representative_schedule"]

MAX_LEGEND_KEYS = 16
"""Colour keys beyond which the legend is left out; bars still carry their task ids."""

_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class GanttBar:
    """One task on one row.

    Attributes:
        row: Index of the row.
        task: Task id.
        start: Start time.
        end: Finish time.
        key: Index of the bar's colour key.
    """

    row: int
    task: str
    start: float
    end: float
    key: int


@dataclass(frozen=True)
class GanttChart:
    """What a Gantt chart shows.

    Attributes:
        title: Title of the figure.
        view: ``"resources"`` for one row per resource, ``"tasks"`` for one
            row per task.
        rows: Row labels; the extra rows of a resource are unlabelled.
        bars: The bars.
        keys: What each colour stands for: jobs, or modes.
        makespan: Latest finish time.
    """

    title: str
    view: Literal["resources", "tasks"]
    rows: tuple[str, ...]
    bars: tuple[GanttBar, ...]
    keys: tuple[str, ...]
    makespan: float


def gantt_chart(
    schedule: Schedule, problem: Problem | None = None, *, title: str = "Schedule"
) -> GanttChart:
    """Return the Gantt chart of a schedule.

    Args:
        schedule: The schedule.
        problem: The problem it solves, for rows per resource. Without it,
            the chart has one row per task.
        title: Title of the figure.

    Returns:
        The chart.

    Raises:
        ConfigError: If the schedule names a task or mode the problem lacks.
    """
    if problem is None:
        return _task_view(schedule, title)
    return _resource_view(schedule, problem, title)


def _task_view(schedule: Schedule, title: str) -> GanttChart:
    modes = sorted({item.mode for item in schedule.assignments})
    key = {mode: index for index, mode in enumerate(modes)}
    bars = tuple(
        GanttBar(row, item.task, item.start, item.end, key[item.mode])
        for row, item in enumerate(schedule.assignments)
    )
    return GanttChart(
        title,
        "tasks",
        tuple(item.task for item in schedule.assignments),
        bars,
        tuple(f"mode {mode}" for mode in modes),
        schedule.makespan,
    )


def _resource_view(schedule: Schedule, problem: Problem, title: str) -> GanttChart:
    held: dict[str, list[Assignment]] = {}
    for item in schedule.assignments:
        if item.task not in problem.task_index:
            raise ConfigError(f"the schedule names task {item.task!r}, which the problem lacks")
        task = problem.task(item.task)
        if not 0 <= item.mode < len(task.modes):
            raise ConfigError(f"task {item.task!r} has no mode {item.mode}")
        for resource_id in task.modes[item.mode].demands:
            held.setdefault(resource_id, []).append(item)
    jobs: dict[str, int] = {}
    for task in problem.tasks:
        jobs.setdefault(task.group if task.group is not None else task.id, len(jobs))
    rows: list[str] = []
    bars: list[GanttBar] = []
    for resource in problem.resources:
        items = sorted(
            held.get(resource.id, ()), key=lambda item: (item.start, item.end, item.task)
        )
        if not items:
            continue
        lanes: list[float] = []
        first_row = len(rows)
        for item in items:
            lane = next(
                (index for index, end in enumerate(lanes) if end <= item.start + _EPSILON), None
            )
            if lane is None:
                lane = len(lanes)
                lanes.append(item.end)
            else:
                lanes[lane] = item.end
            task = problem.task(item.task)
            job = task.group if task.group is not None else task.id
            bars.append(GanttBar(first_row + lane, item.task, item.start, item.end, jobs[job]))
        rows.extend([resource.id] + [""] * (len(lanes) - 1))
    return GanttChart(title, "resources", tuple(rows), tuple(bars), tuple(jobs), schedule.makespan)


def representative_schedule(source: object) -> Observation | None:
    """Return the run whose schedule the catalog draws.

    It is the succeeded run with a recorded schedule on the first problem
    that has one, with the best value there; ties go to the solver listed
    first, then to the lowest replicate.

    Args:
        source: Runs of one protocol, anything
            :func:`~dispatchatlas.analysis.figures.frames.observe` accepts.

    Returns:
        The run, or ``None`` when no run recorded a schedule.
    """
    observations = observe(source).comparable()
    for problem in observations.problems:
        candidates: Sequence[Observation] = [
            run
            for solver in observations.solvers
            for run in observations.cell(solver, problem)
            if run.succeeded and run.record is not None and run.record.schedule is not None
        ]
        if candidates:
            order = {solver: index for index, solver in enumerate(observations.solvers)}
            return min(
                candidates,
                key=lambda run: (
                    observations.score(run),
                    order[run.solver],
                    run.replicate,
                    run.run_id,
                ),
            )
    return None


def plot_gantt(chart: GanttChart, *, theme: Theme | None = None) -> Figure:
    """Draw a Gantt chart.

    Args:
        chart: The chart.
        theme: The appearance.

    Returns:
        The figure, or a placeholder for an empty schedule.
    """
    active = theme or DEFAULT_THEME
    name = "gantt"
    if not chart.bars:
        return placeholder(name, chart.title, "The schedule assigns no task.", active)
    palette = active.palette
    row_height = min(24.0, max(4.0, 480.0 / len(chart.rows)))
    bars = [
        Bar(
            bar.row,
            bar.start,
            bar.end,
            palette[bar.key % len(palette)],
            text=bar.task if chart.view == "resources" else "",
            tooltip=f"{bar.task}: {bar.start:g} to {bar.end:g}",
        )
        for bar in chart.bars
    ]
    legend = []
    if 2 <= len(chart.keys) <= MAX_LEGEND_KEYS:
        legend = [
            LegendEntry(key, SeriesStyle(palette[index % len(palette)], "square", ()), "bar")
            for index, key in enumerate(chart.keys)
        ]
    x = numeric_axis("Time", [0.0, chart.makespan], low=0.0)
    colours = "job" if chart.view == "resources" else "mode"
    rows = "resource" if chart.view == "resources" else "task"
    notes = [
        f"One row per {rows}; bars are coloured by {colours}, and the dashed line marks the "
        "makespan.",
    ]
    description = (
        f"{chart.title}. Gantt chart of {len(chart.bars)} bars on {len(chart.rows)} rows, "
        f"makespan {chart.makespan:g}. {notes[0]}"
    )
    return row_chart(
        name,
        chart.title,
        description,
        x=x,
        rows=[Row(label) for label in chart.rows],
        bars=bars,
        row_height=row_height,
        legend=legend,
        guides=[Guide(chart.makespan, f"makespan {chart.makespan:g}")],
        notes=notes,
        theme=active,
    )
