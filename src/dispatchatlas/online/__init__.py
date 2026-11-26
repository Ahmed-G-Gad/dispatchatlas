"""Online scheduling: a loop that plans while workflows arrive and nodes fail.

A static solver sees every task at once. An online scheduler sees workflows
as they arrive, cannot move what has started, loses the work running on a
node that fails, and pays for its own computation in the time it schedules.
This package runs that loop, deterministically and with every raw record
kept:

- :mod:`~dispatchatlas.online.events`: workflows, arrivals, failures,
  repairs, and the :class:`Trace` that holds them, with a content hash.
- :mod:`~dispatchatlas.online.generators`: seeded Poisson and submitted
  arrivals, and failures and repairs from an availability model, on common
  random numbers.
- :mod:`~dispatchatlas.online.state`: task statuses, attempts, and the
  :class:`SystemState` a planner sees.
- :mod:`~dispatchatlas.online.planning`: the :class:`Planner` protocol, the
  :class:`Plan` it returns, its :class:`Budget`, and the
  :class:`PlanningClock` that charges planning in simulated time.
- :mod:`~dispatchatlas.online.planners`: the greedy reference
  :class:`OnlineEft` and :class:`WarmRestart` of a static solver.
- :mod:`~dispatchatlas.online.runner`: :func:`run_online`, the loop, and
  the rules it follows.
- :mod:`~dispatchatlas.online.metrics`: response times and recovery after
  each event.
- :mod:`~dispatchatlas.online.records`: one compressed JSON-lines record file
  per job, written atomically, read strictly.
- :mod:`~dispatchatlas.online.validation`: :func:`check_run`, an independent
  check of an executed schedule.

The package sits above :mod:`dispatchatlas.solvers`, whose schedule builder
it uses, and below :mod:`dispatchatlas.campaigns`.
"""

from dispatchatlas.online.events import (
    TRACE_FORMAT,
    Arrival,
    Event,
    EventKind,
    Failure,
    Repair,
    Trace,
    Workflow,
    workflows_from_problem,
)
from dispatchatlas.online.generators import (
    AlternatingRenewal,
    Distribution,
    Exponential,
    LogNormal,
    Weibull,
    availability_events,
    poisson_arrivals,
    rate_for_load,
    submitted_arrivals,
    workflow_work,
)
from dispatchatlas.online.metrics import cohort_response, recovery
from dispatchatlas.online.planners import OnlineEft, WarmRestart
from dispatchatlas.online.planning import (
    Budget,
    BudgetExhausted,
    Estimate,
    PassOutcome,
    Plan,
    Planner,
    PlanningClock,
    PlanningContext,
    Residual,
)
from dispatchatlas.online.records import RECORDS_FORMAT, RecordWriter, read_records, write_records
from dispatchatlas.online.runner import RESULT_FORMAT, OnlineResult, online_problem, run_online
from dispatchatlas.online.state import Attempt, AttemptOutcome, SystemState, TaskStatus
from dispatchatlas.online.validation import check_run

__all__ = [
    "RECORDS_FORMAT",
    "RESULT_FORMAT",
    "TRACE_FORMAT",
    "AlternatingRenewal",
    "Arrival",
    "Attempt",
    "AttemptOutcome",
    "Budget",
    "BudgetExhausted",
    "Distribution",
    "Estimate",
    "Event",
    "EventKind",
    "Exponential",
    "Failure",
    "LogNormal",
    "OnlineEft",
    "OnlineResult",
    "PassOutcome",
    "Plan",
    "Planner",
    "PlanningClock",
    "PlanningContext",
    "RecordWriter",
    "Repair",
    "Residual",
    "SystemState",
    "TaskStatus",
    "Trace",
    "WarmRestart",
    "Weibull",
    "Workflow",
    "availability_events",
    "check_run",
    "cohort_response",
    "online_problem",
    "poisson_arrivals",
    "rate_for_load",
    "read_records",
    "recovery",
    "run_online",
    "submitted_arrivals",
    "workflow_work",
    "workflows_from_problem",
    "write_records",
]
