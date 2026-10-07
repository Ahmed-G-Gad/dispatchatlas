"""``dispatchatlas campaign``: validate, estimate, run, check, and replay experiment campaigns."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from dispatchatlas.campaigns import (
    Campaign,
    Caveat,
    CheckReport,
    Compatibility,
    Estimate,
    ReplayEntry,
    ReplayReport,
    RunSummary,
    Verdict,
    check_campaign,
    check_compatibility,
    estimate,
    load_campaign,
    plan,
    read_campaign,
    replay,
    run,
)
from dispatchatlas.campaigns.checking import SEVERITIES
from dispatchatlas.campaigns.config import BACKENDS, CPU_RESERVE
from dispatchatlas.campaigns.store import (
    CAMPAIGN_FILE,
    FAILURES,
    campaign_directory,
    recorded_config_hash,
)
from dispatchatlas.cli._common import (
    PROG,
    CommandError,
    Commands,
    add_command,
    add_group,
    affinity,
    emit_fields,
    emit_json,
    emit_table,
    existing_file,
    non_negative_integer,
    non_negative_number,
    positive_integer,
    seconds,
    text,
    workers,
)
from dispatchatlas.core.errors import ConfigError

__all__ = ["register"]

_RESUME = "records written so far are kept; run the same command again to resume"

_SHOWN = 6
"""Names a summary line lists before it abbreviates."""


def register(commands: Commands) -> None:
    """Add the ``campaign validate``, ``estimate``, ``run``, ``check``, and ``replay`` commands."""
    group = add_group(
        commands,
        "campaign",
        "Validate, estimate, run, check, and replay experiment campaigns.",
        "Run experiment campaigns described by a TOML or JSON file: every solver on every "
        "problem under every stopping protocol, with seeds shared across solvers, one "
        "record per run, and resumption after an interruption.",
    )
    validate = add_command(
        group,
        "validate",
        _validate,
        "Check a campaign file and plan its runs without running them.",
        description="Read a campaign file strictly, load every problem, check every solver "
        "and parameter, and plan the runs, verifying that the comparison is fair. Nothing "
        "runs.",
        epilog=f"example:\n  {PROG} campaign validate campaign.toml --check-compatibility",
    )
    validate.add_argument("config", metavar="CONFIG", help="campaign file (.toml or .json)")
    validate.add_argument(
        "--check-compatibility",
        action="store_true",
        help="list every problem and solver pair that is excluded, with the features the "
        "solver does not model, and every caveat, even when no pair remains",
    )
    validate.add_argument(
        "--runs",
        action="store_true",
        help="list every planned run with its id, problem, solver, protocol, replicate, and "
        "seed, as 'campaign run' would write them to plan.json; with --json, 'runs' is then "
        "that list in place of its length",
    )
    estimated = add_command(
        group,
        "estimate",
        _estimate,
        "Count a campaign's runs and bound how long they can take.",
        description="Plan a campaign, count its runs, and bound their duration from the "
        "protocols' time limits. With --output, runs already completed there are left out.",
    )
    estimated.add_argument("config", metavar="CONFIG", help="campaign file (.toml or .json)")
    estimated.add_argument(
        "--output", metavar="DIR", help="output directory of earlier runs of the campaign"
    )
    _add_execution(estimated)
    running = add_command(
        group,
        "run",
        _run,
        "Run a campaign, resuming from the records it already has.",
        description="Run every planned run that has no record yet. The records go to "
        "DIR/<campaign id>/. Running the command again resumes the campaign. The exit code "
        "is 1 when runs failed.",
        epilog=f"example:\n  {PROG} campaign run campaign.toml --output results --workers auto",
        interrupted=_RESUME,
    )
    running.add_argument("config", metavar="CONFIG", help="campaign file (.toml or .json)")
    running.add_argument("--output", metavar="DIR", required=True, help="output directory")
    _add_execution(running)
    _add_affinity(running)
    running.add_argument(
        "--fresh",
        action="store_true",
        help="discard every record of the campaign directory first",
    )
    running.add_argument(
        "--retry-failed",
        action="store_true",
        help="give every run that failures settled one more attempt",
    )
    running.add_argument(
        "--max-runs",
        type=non_negative_integer,
        metavar="N",
        help="attempt at most N runs in this session",
    )
    checked = add_command(
        group,
        "check",
        _check,
        "Check a campaign directory before its results are used.",
        description="Read a campaign directory and report what qualifies its results: whether "
        "the stored campaign still matches its config hash, whether the plan is the plan that "
        "campaign gives, whether every planned run has a valid record or failures that settle "
        "it, whether the budgets were kept, and whether the sessions agree on code and "
        "settings. Each finding is blocking, a limitation, or a note. Nothing is written and "
        "nothing runs. The exit code is 0 when no finding is blocking, 1 when a finding is "
        "blocking or the directory cannot be checked, and 2 for a command line that is "
        "not valid.",
        epilog=f"example:\n  {PROG} campaign check results/shop-baselines --replicate-floor 10",
    )
    checked.add_argument(
        "directory", metavar="DIR", help="campaign directory written by 'campaign run'"
    )
    checked.add_argument(
        "--replicate-floor",
        type=positive_integer,
        metavar="N",
        help="report the cells of replicated solvers with fewer than N completed replicates "
        "(default: no floor)",
    )
    checked.add_argument(
        "--overrun-tolerance",
        type=non_negative_number,
        metavar="SHARE",
        help="share of a time limit by which a completed run may pass it without counting as "
        "over (default: the margin of the engine's note on a run)",
    )
    checked.add_argument(
        "--no-replan",
        action="store_true",
        help="do not derive the plan again from the stored campaign, which loads every "
        "problem; the plan file is then compared with the last session's digest only",
    )
    replayed = add_command(
        group,
        "replay",
        _replay,
        "Run completed runs again and check that they reproduce.",
        description="Run the completed runs of a campaign directory again from their "
        "recorded seeds and compare the results with the records. Runs stopped by their "
        "wall-clock limit are skipped. Nothing is written. The exit code is 1 when a run "
        "does not reproduce.",
        epilog=f"example:\n  {PROG} campaign replay results/shop-baselines --backend sequential",
    )
    replayed.add_argument(
        "directory", metavar="DIR", help="campaign directory written by 'campaign run'"
    )
    replayed.add_argument(
        "--run",
        action="append",
        metavar="ID",
        help="replay this run only; repeat for several (default: every completed run)",
    )
    _add_execution(replayed)
    _add_affinity(replayed)


def _add_execution(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workers",
        type=workers,
        metavar="N",
        help=f"worker processes, or 'auto' for the available processors minus {CPU_RESERVE} "
        "(default: the campaign's setting)",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        help="run in worker processes, or one at a time in this process "
        "(default: the campaign's setting)",
    )


def _add_affinity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--affinity",
        type=affinity,
        metavar="WHERE",
        help="processors of the workers: 'inherit' for those of this process, 'pin' for one "
        "of them each, or indices such as 0,2,4 (default: the campaign's setting; "
        "--backend sequential implies 'inherit')",
    )


# -------------------------------------------------------------------- validate


def _validate(arguments: argparse.Namespace) -> int:
    campaign = load_campaign(existing_file(arguments.config))
    compatibility = check_compatibility(campaign) if arguments.check_compatibility else None
    if compatibility is not None and not arguments.json:
        _print_compatibility(compatibility)
    try:
        planned = plan(campaign)
    except ConfigError:
        if compatibility is not None and arguments.json:
            emit_json({"compatibility": compatibility.to_dict()})
        raise
    document: dict[str, Any] = planned.to_dict()
    if not arguments.runs:
        document["runs"] = len(planned.runs)
    if compatibility is not None:
        document["compatibility"] = compatibility.to_dict()
    if arguments.json:
        emit_json(document)
        return 0
    fairness = document["fairness"]
    emit_fields(
        [
            ("campaign", f"{campaign.id} (config hash {campaign.config_hash[:12]})"),
            ("problems", _counted([item.id for item in planned.problems], "problem")),
            ("solvers", _counted([item.name for item in planned.solvers], "solver")),
            ("protocols", "\n".join(_protocol(item) for item in document["protocols"])),
            ("replicates", _replicates(campaign)),
            ("seeds", _seeds(campaign)),
            ("runs", len(planned.runs)),
            ("excluded", _count(len(planned.exclusions), "problem and solver pair")),
            ("caveats", "\n".join(_caveat(item) for item in planned.caveats) or "none"),
            ("fair", "yes" if fairness["fair"] else "no: " + "; ".join(fairness["issues"])),
            (
                "notes",
                "\n".join([*fairness["notes"], *fairness["conditional_notes"]]) or "none",
            ),
        ]
    )
    if arguments.runs:
        print()
        emit_table(
            ("run", "problem", "solver", "protocol", "replicate", "seed"),
            [
                (run.run_id, run.problem, run.solver, run.protocol, run.replicate, run.seed)
                for run in planned.runs
            ],
        )
    return 0


def _replicates(campaign: Campaign) -> str:
    """Describe how many replicates each kind of solver of a campaign runs."""
    timed = (
        ", and per solver declared deterministic under a time limit"
        if campaign.replicate_timed
        else ""
    )
    own = "".join(
        f"\n{solver.name}: {solver.replicates}"
        for solver in campaign.solvers
        if solver.replicates is not None
    )
    return f"{campaign.replicates} per stochastic solver{timed}{own}"


def _seeds(campaign: Campaign) -> str:
    """Describe what a campaign derives its run seeds from."""
    if campaign.seed_namespace is None:
        return f"derived from seed {campaign.seed}"
    return f"derived from seed {campaign.seed} in namespace {campaign.seed_namespace}"


def _print_compatibility(compatibility: Compatibility) -> None:
    rows: list[tuple[str, object]] = [
        ("excluded", "\n".join(item.reason for item in compatibility.exclusions) or "none"),
        ("caveats", "\n".join(_caveat(item) for item in compatibility.caveats) or "none"),
    ]
    emit_fields(rows)
    print()


def _protocol(document: dict[str, Any]) -> str:
    stop = ", ".join(f"{key}={text(value)}" for key, value in document["stop"].items())
    parities = ", ".join(document["parities"]) or "no parity"
    return f"{document['name']}: {stop} ({parities})"


def _caveat(item: Caveat) -> str:
    return f"{item.solver} on {_count(len(item.problems), 'problem')}: {item.message}"


def _counted(names: Sequence[str], what: str) -> str:
    """Return ``"<n> <what>s: a, b, c, ..."``, naming the first few."""
    shown = ", ".join(names[:_SHOWN]) + (", ..." if len(names) > _SHOWN else "")
    return f"{_count(len(names), what)}: {shown}"


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


# -------------------------------------------------------------------- estimate


def _estimate(arguments: argparse.Namespace) -> int:
    campaign = load_campaign(existing_file(arguments.config))
    result = estimate(
        campaign, root=arguments.output, backend=arguments.backend, workers=arguments.workers
    )
    document = _estimate_document(campaign, result)
    if arguments.json:
        emit_json(document)
        return 0
    emit_fields(
        [
            ("campaign", f"{campaign.id} (config hash {campaign.config_hash[:12]})"),
            (
                "runs",
                f"{result.planned} planned: {result.completed} completed, {result.failed} "
                f"failed, {result.remaining} remaining",
            ),
            (
                "grid",
                f"{_count(result.problems, 'problem')}, {_count(result.solvers, 'solver')}, "
                f"{_count(result.protocols, 'protocol')}, "
                f"{_count(result.exclusions, 'excluded pair')}",
            ),
            ("by protocol", _tally(result.by_protocol)),
            ("by solver", _tally(result.by_solver)),
            ("workers", result.workers),
            ("duration", _duration(result)),
        ]
    )
    return 0


def _estimate_document(campaign: Campaign, result: Estimate) -> dict[str, Any]:
    return {
        "campaign": campaign.id,
        "config_hash": campaign.config_hash,
        "planned": result.planned,
        "completed": result.completed,
        "failed": result.failed,
        "remaining": result.remaining,
        "problems": result.problems,
        "solvers": result.solvers,
        "protocols": result.protocols,
        "exclusions": result.exclusions,
        "by_protocol": dict(result.by_protocol),
        "by_solver": dict(result.by_solver),
        "workers": result.workers,
        "timed": result.timed,
        "untimed": result.untimed,
        "budget_seconds": result.budget_seconds,
        "worst_case_seconds": result.worst_case_seconds,
        "budget_wall_seconds": result.budget_wall_seconds,
        "worst_case_wall_seconds": result.worst_case_wall_seconds,
    }


def _tally(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name} {count}" for name, count in counts.items()) or "-"


def _duration(result: Estimate) -> str:
    lines = []
    if result.timed:
        lines.append(
            f"{result.timed} timed runs: {seconds(result.budget_wall_seconds)} of time limits "
            f"on {result.workers} workers, at most {seconds(result.worst_case_wall_seconds)} "
            "with hard limits and retries"
        )
    if result.untimed:
        lines.append(f"{result.untimed} runs without a time limit, whose duration has no bound")
    return "\n".join(lines) or "nothing remains to run"


# ------------------------------------------------------------------------- run


def _run(arguments: argparse.Namespace) -> int:
    campaign = load_campaign(existing_file(arguments.config))
    directory = campaign_directory(arguments.output, campaign.id)
    found = recorded_config_hash(directory)
    if found is not None and found != campaign.config_hash and not arguments.fresh:
        raise CommandError(
            f"{directory} holds results of another configuration of campaign {campaign.id} "
            f"(config hash {found[:12]}, now {campaign.config_hash[:12]})",
            hint="pass --fresh to discard them, or give the campaign another id",
        )
    summary = run(
        campaign,
        arguments.output,
        fresh=arguments.fresh,
        retry_failed=arguments.retry_failed,
        max_runs=arguments.max_runs,
        backend=arguments.backend,
        workers=arguments.workers,
        affinity=arguments.affinity,
    )
    if arguments.json:
        emit_json(_run_document(summary))
    else:
        emit_fields(
            [
                ("campaign", f"{summary.campaign}, session {summary.session}"),
                ("directory", summary.directory),
                (
                    "runs",
                    f"{summary.planned} planned: {summary.completed} completed, "
                    f"{summary.failed} failed, {summary.pending} pending",
                ),
                (
                    "session",
                    f"{summary.attempted} attempts: {summary.succeeded} succeeded, "
                    f"{summary.errors} failed, in {seconds(summary.elapsed)}",
                ),
            ]
        )
    if summary.failed:
        raise CommandError(
            f"{summary.failed} of {summary.planned} runs failed; their records are under "
            f"{summary.directory / FAILURES}",
            hint="fix the cause, then run again with --retry-failed",
        )
    return 0


def _run_document(summary: RunSummary) -> dict[str, Any]:
    return {
        "campaign": summary.campaign,
        "config_hash": summary.config_hash,
        "directory": summary.directory,
        "session": summary.session,
        "planned": summary.planned,
        "completed": summary.completed,
        "failed": summary.failed,
        "pending": summary.pending,
        "attempted": summary.attempted,
        "succeeded": summary.succeeded,
        "errors": summary.errors,
        "elapsed": summary.elapsed,
        "done": summary.done,
    }


# ---------------------------------------------------------------------- replay


def _check(arguments: argparse.Namespace) -> int:
    directory = Path(arguments.directory).resolve()
    if not directory.is_dir():
        raise CommandError(
            f"{directory} is not a directory",
            hint=f"pass the directory that '{PROG} campaign run' reported",
        )
    options: dict[str, Any] = {}
    if arguments.overrun_tolerance is not None:
        options["overrun_tolerance"] = arguments.overrun_tolerance
    report = check_campaign(
        directory,
        replicate_floor=arguments.replicate_floor,
        replan=not arguments.no_replan,
        **options,
    )
    if arguments.json:
        emit_json(report.to_dict())
    else:
        emit_fields(_check_rows(report))
    blocking = len(report.of("blocking"))
    if blocking:
        raise CommandError(
            f"{_count(blocking, 'blocking finding')} in {directory}",
            hint="the results are not ready to be used; a run without a record gets one when "
            f"'{PROG} campaign run' is run again with the same file and output",
        )
    return 0


def _check_rows(report: CheckReport) -> list[tuple[str, object]]:
    """Lay a check report out as labelled values, each finding under its severity."""
    hash_text = "unknown" if report.config_hash is None else report.config_hash[:12]
    summary = ", ".join(
        f"{len(report.of(severity))} blocking"
        if severity == "blocking"
        else _count(len(report.of(severity)), severity)
        for severity in SEVERITIES
    )
    rows: list[tuple[str, object]] = [
        ("campaign", f"{report.campaign or 'unknown'} (config hash {hash_text})"),
        ("runs", _tally(report.counts) if report.counts else "not counted"),
        ("findings", summary),
    ]
    for finding in report.findings:
        under = "" if finding.protocol is None else f" (protocol {finding.protocol})"
        lines = [f"{finding.code}{under}: {finding.message}"]
        lines += [f"- {item}" for item in finding.items[:_SHOWN]]
        if len(finding.items) > _SHOWN:
            lines.append(f"- and {len(finding.items) - _SHOWN} more; --json lists every item")
        rows.append((finding.severity, "\n".join(lines)))
    return rows


def _replay(arguments: argparse.Namespace) -> int:
    directory = Path(arguments.directory).resolve()
    if not (directory / CAMPAIGN_FILE).is_file():
        raise CommandError(
            f"{directory} is not a campaign directory: it has no {CAMPAIGN_FILE}",
            hint=f"pass the directory that '{PROG} campaign run' reported",
        )
    campaign = read_campaign(directory)
    if campaign.id != directory.name:
        # The id only names the directory, so a moved campaign keeps its config hash.
        campaign = replace(campaign, id=directory.name)
    report = replay(
        campaign,
        directory.parent,
        runs=arguments.run,
        backend=arguments.backend,
        workers=arguments.workers,
        affinity=arguments.affinity,
    )
    if arguments.json:
        emit_json(_replay_document(report))
    else:
        counts = ", ".join(f"{report.count(verdict)} {verdict.value}" for verdict in Verdict)
        emit_fields(
            [
                ("campaign", f"{report.campaign} (config hash {report.config_hash[:12]})"),
                ("verdicts", counts),
                *(
                    (entry.run_id, _verdict(entry))
                    for entry in report.entries
                    if entry.verdict is not Verdict.MATCH
                ),
            ]
        )
    if not report.ok:
        failing = report.count(Verdict.MISMATCH) + report.count(Verdict.FAILED)
        raise CommandError(
            f"{failing} of {len(report.entries)} runs did not reproduce",
            hint="the entries above list the fields that differ",
        )
    return 0


def _verdict(entry: ReplayEntry) -> str:
    """Describe a verdict other than a match, one line per differing field."""
    lines = [
        f"{entry.verdict.value}: {entry.solver} on {entry.problem}, protocol {entry.protocol}, "
        f"replicate {entry.replicate}: {entry.detail}"
    ]
    lines += [
        f"{item.field}: recorded {item.recorded}, replayed {item.replayed}"
        for item in entry.differences
    ]
    return "\n".join(lines)


def _replay_document(report: ReplayReport) -> dict[str, Any]:
    return {
        "campaign": report.campaign,
        "config_hash": report.config_hash,
        "ok": report.ok,
        "counts": {verdict.value: report.count(verdict) for verdict in Verdict},
        "entries": [
            {
                "run_id": entry.run_id,
                "solver": entry.solver,
                "problem": entry.problem,
                "protocol": entry.protocol,
                "replicate": entry.replicate,
                "verdict": entry.verdict.value,
                "detail": entry.detail,
                "differences": [
                    {"field": item.field, "recorded": item.recorded, "replayed": item.replayed}
                    for item in entry.differences
                ],
            }
            for entry in report.entries
        ],
    }
