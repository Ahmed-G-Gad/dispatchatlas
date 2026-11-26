"""Measures computed from the raw records of an online run.

**Recovery after an event.** Let ``t`` be the time of an event, and the
**cohort** the workflows that had arrived and not finished just before it.
For a forecast ``p`` of the run (see :mod:`~dispatchatlas.online.runner`), let
``R(p)`` be the cohort's mean planned response: over the cohort, the
workflow's finish in ``p`` minus its arrival, where a workflow finished by the
forecast's time counts with its realised finish. ``R-`` is ``R`` of the
forecast just before the event, which assumed the event would not happen.

The **recovery time** is ``s - t`` for the first forecast ``s`` from the one
just after the event on, at which ``R(s) <= (1 + tolerance) R-``. The
realised outcome at the end of the run counts as a last forecast. The end of
the run is ``summary["end"]``, the time of the last step of the loop (the
latest event, task start or finish, answer end, or pass end), not the horizon
of the trace. An event whose cohort never returns within tolerance is
**censored**: its recovery time runs to that end and ``recovered`` is false.

The **degradation** is the integral of ``max(0, R - R-)`` from ``t`` to the
recovery, with ``R`` held constant from one forecast to the next. It is
``None`` when some forecast in the window cannot predict a cohort workflow,
because a task of it can run on no node that is up.

Events at one instant share their forecasts, and so their recovery; events
within 1e-9 of a decision point are at its instant, as in the runner. An event
with an empty cohort, such as the first arrival, has no recovery.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import TYPE_CHECKING, Any

from dispatchatlas.online._engine import TIME_TOLERANCE

if TYPE_CHECKING:
    from dispatchatlas.online.runner import OnlineResult

__all__ = ["cohort_response", "recovery"]

_EPSILON = TIME_TOLERANCE


def cohort_response(result: OnlineResult, cohort: list[int], row: list[Any] | None) -> float | None:
    """Return the cohort's mean planned response at a forecast row, or at the end.

    Args:
        result: The run.
        cohort: Workflow indices.
        row: A row of :attr:`OnlineResult.forecasts`, or ``None`` for the
            realised outcome at the end of the run.

    Returns:
        The mean, or ``None`` if some workflow of the cohort has no prediction.
    """
    if not cohort:
        return None
    predicted = {} if row is None else dict(row[3])
    at = result.summary["end"] if row is None else row[0]
    total = 0.0
    for workflow in cohort:
        _, arrival, finish, _ = result.workflows[workflow]
        if finish is None or finish > at + _EPSILON:
            finish = predicted.get(workflow) if row is not None else None
        if finish is None:
            return None
        total += finish - arrival
    return total / len(cohort)


def _at(times: list[float], rows: list[int], time: float) -> int | None:
    """Return the last forecast whose time is within the runner's tolerance of ``time``.

    The runner applies the events within :data:`~dispatchatlas.online._engine.TIME_TOLERANCE`
    after a step together at that step's time, so a forecast of the decision point
    can be stamped a hair before its event.
    """
    k = bisect_left(times, time + TIME_TOLERANCE) - 1
    if k < 0 or times[k] < time - TIME_TOLERANCE:
        return None
    return rows[k]


def recovery(result: OnlineResult, tolerance: float) -> list[dict[str, Any]]:
    """Return the recovery of every event of a run, as the module defines it.

    Args:
        result: The run.
        tolerance: Relative tolerance on the pre-event response.

    Returns:
        One mapping per event: ``event`` (its index), ``time``, ``kind``,
        ``subject``, ``cohort`` (its size), ``before`` (``R-``), ``after``
        (``R`` just after the event), ``recovered``, ``recovery_time``, and
        ``degradation``. The last five are ``None`` for an empty cohort or an
        unpredictable ``R-``.
    """
    forecasts = result.forecasts
    pre_times: list[float] = []
    pre_rows: list[int] = []
    post_times: list[float] = []
    post_rows: list[int] = []
    for n, row in enumerate(forecasts):
        if row[1] == "pre":
            pre_times.append(row[0])
            pre_rows.append(n)
        elif row[1] == "post":
            post_times.append(row[0])
            post_rows.append(n)
    end = result.summary["end"]
    rows: list[dict[str, Any]] = []
    for index, (time, kind, subject) in enumerate(result.events):
        entry: dict[str, Any] = {
            "event": index,
            "time": time,
            "kind": kind,
            "subject": subject,
            "cohort": 0,
            "before": None,
            "after": None,
            "recovered": None,
            "recovery_time": None,
            "degradation": None,
        }
        rows.append(entry)
        pre = _at(pre_times, pre_rows, time)
        post = _at(post_times, post_rows, time)
        if pre is None or post is None:
            continue
        cohort = [w for w, _ in forecasts[pre][3]]
        entry["cohort"] = len(cohort)
        before = cohort_response(result, cohort, forecasts[pre])
        if before is None:
            continue
        entry["before"] = before
        series: list[tuple[float, float | None]] = [
            (row[0], cohort_response(result, cohort, row)) for row in forecasts[post:]
        ]
        series.append((end, cohort_response(result, cohort, None)))
        entry["after"] = series[0][1]
        bound = (1.0 + tolerance) * before + _EPSILON
        degradation: float | None = 0.0
        recovered_at: float | None = None
        for k, (at, value) in enumerate(series):
            if value is not None and value <= bound:
                recovered_at = at
                break
            following = series[k + 1][0] if k + 1 < len(series) else at
            if value is None:
                degradation = None
            elif degradation is not None:
                degradation += max(0.0, value - before) * (following - at)
        entry["recovered"] = recovered_at is not None
        entry["recovery_time"] = (end if recovered_at is None else recovered_at) - time
        entry["degradation"] = degradation
    return rows
