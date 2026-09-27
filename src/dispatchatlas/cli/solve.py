"""``dispatchatlas solve``: run one solver on one problem and check its schedule."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dispatchatlas.cli._common import (
    PROG,
    CommandError,
    Commands,
    UsageError,
    add_command,
    add_problem_arguments,
    assignment,
    emit_fields,
    emit_json,
    finite_number,
    non_negative_integer,
    parameters,
    positive_integer,
    positive_number,
    read_problem,
    seconds,
    text,
    violation_to_dict,
)
from dispatchatlas.core.model import Problem
from dispatchatlas.core.serialization import save_result
from dispatchatlas.core.solver import SolveResult, Status, StopCriteria
from dispatchatlas.core.validation import Violation, check_schedule
from dispatchatlas.solvers import default_registry

__all__ = ["register"]


def register(commands: Commands) -> None:
    """Add the ``solve`` command."""
    parser = add_command(
        commands,
        "solve",
        _solve,
        "Solve a problem with one solver and check the schedule it returns.",
        description="Solve a problem with one registered solver, check the schedule it "
        "returns against every constraint, and print a summary. The full result, with the "
        "schedule and the trace of improvements, is written with --output.",
        epilog="examples:\n"
        f"  {PROG} solve js.json --solver heft\n"
        f"  {PROG} solve js.json --solver tabu-search --set tenure=10 --max-evaluations 5000 "
        "--seed 3 --output result.json\n"
        f"  {PROG} benchmarks generate flow-shop | {PROG} solve - --solver neh",
    )
    add_problem_arguments(parser)
    parser.add_argument("--solver", metavar="ID", required=True, help="solver id")
    parser.add_argument(
        "--set",
        type=assignment,
        action="append",
        metavar="KEY=VALUE",
        help="set a solver parameter; the value is read as JSON when it parses, else as text",
    )
    limits = parser.add_argument_group(
        "stop criteria", "The first criterion met ends the run. Iterative solvers need a limit."
    )
    limits.add_argument(
        "--time-limit", type=positive_number, metavar="S", help="wall-clock seconds"
    )
    limits.add_argument(
        "--max-evaluations", type=positive_integer, metavar="N", help="schedules evaluated"
    )
    limits.add_argument(
        "--max-iterations", type=positive_integer, metavar="N", help="main-loop iterations"
    )
    limits.add_argument(
        "--target", type=finite_number, metavar="V", help="stop once the objective reaches V"
    )
    limits.add_argument(
        "--patience", type=positive_integer, metavar="N", help="iterations without improvement"
    )
    parser.add_argument(
        "--seed",
        type=non_negative_integer,
        default=0,
        metavar="N",
        help="seed of the run (default: 0)",
    )
    parser.add_argument("--output", metavar="FILE", help="write the full result document here")


def _solve(arguments: argparse.Namespace) -> int:
    registry = default_registry()
    info = registry.info(arguments.solver)
    stop = StopCriteria(
        time_limit=arguments.time_limit,
        max_evaluations=arguments.max_evaluations,
        max_iterations=arguments.max_iterations,
        target=arguments.target,
        patience=arguments.patience,
    )
    if info.iterative and not stop.bounded:
        raise UsageError(
            f"{info.id} is iterative: give --time-limit, --max-evaluations, or --max-iterations"
        )
    given = parameters(arguments.set)
    solver = registry.create(info.id, **given)
    problem = read_problem(arguments.problem, arguments.format_name, arguments.instance)
    result = solver.solve(problem, stop, arguments.seed)
    violations = () if result.schedule is None else check_schedule(problem, result.schedule)
    output: Path | None = None
    if arguments.output is not None:
        output = save_result(
            result,
            arguments.output,
            problem=problem,
            stop=stop,
            seed=arguments.seed,
            parameters={**info.parameters, **given},
        )
    summary = _summary(result, problem, arguments.seed, violations, output)
    if arguments.json:
        emit_json(summary)
    else:
        _print(summary, result)
    _verdict(result, violations)
    return 0


def _summary(
    result: SolveResult,
    problem: Problem,
    seed: int,
    violations: Sequence[Violation],
    output: Path | None,
) -> dict[str, Any]:
    return {
        "solver": result.solver,
        "problem": problem.id,
        "seed": seed,
        "status": result.status.value,
        "stop_reason": result.stop_reason.value,
        "objective": result.objective,
        "value": result.value,
        "objectives": dict(result.objectives),
        "bound": result.bound,
        "gap": result.gap,
        "evaluations": result.evaluations,
        "iterations": result.iterations,
        "elapsed": result.elapsed,
        "front": [list(point.values) for point in result.front],
        "feasible": result.schedule is not None and not violations,
        "violations": [violation_to_dict(item) for item in violations],
        "output": output,
    }


def _print(summary: dict[str, Any], result: SolveResult) -> None:
    objectives = ", ".join(f"{name} {text(value)}" for name, value in result.objectives.items())
    if result.schedule is None:
        check = "no schedule to check"
    elif summary["violations"]:
        check = "\n".join(f"{item['code']}: {item['message']}" for item in summary["violations"])
    else:
        check = "every constraint holds"
    rows: list[tuple[str, object]] = [
        ("solver", f"{result.solver} (seed {summary['seed']})"),
        ("problem", summary["problem"]),
        ("status", result.status.value),
        ("stop reason", result.stop_reason.value),
        ("objectives", objectives or "-"),
    ]
    if result.bound is not None:
        rows.append(("bound", f"{text(result.bound)} (gap {text(result.gap)})"))
    if result.front:
        rows.append(("front", f"{len(result.front)} non-dominated schedules"))
    rows += [
        ("effort", f"{result.evaluations} evaluations, {result.iterations} iterations"),
        ("time", seconds(result.elapsed)),
        ("check", check),
    ]
    if summary["output"] is not None:
        rows.append(("result", summary["output"]))
    emit_fields(rows)


def _verdict(result: SolveResult, violations: Sequence[Violation]) -> None:
    """Fail unless the result is consistent: a schedule its status admits, or a proof.

    Raises:
        CommandError: If the solver returned no schedule without proving
            infeasibility, or a schedule that breaks constraints its status
            does not admit. Only a ``violated`` status admits missed deadlines.
    """
    if result.schedule is None:
        if result.status is Status.INFEASIBLE:
            return
        raise CommandError(
            f"{result.solver} found no schedule before it stopped ({result.stop_reason.value})",
            hint="raise the limits, or try another solver",
        )
    admitted = {"deadline"} if result.status is Status.VIOLATED else set()
    unexpected = sorted({item.code for item in violations} - admitted)
    if unexpected:
        raise CommandError(
            f"the schedule of {result.solver} breaks constraints its status "
            f"{result.status.value!r} does not admit: {', '.join(unexpected)}",
            hint="this is a defect of the solver; please report it with the result document "
            "that --output writes",
        )
