"""``campaign`` and ``analyze``, and the whole flow from a generated problem to a bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from dispatchatlas.campaigns import read_runs
from dispatchatlas.cli._common import affinity
from tests.cli.runner import Cli, campaign_file

CHEAP_SOLVERS = (
    'id = "input-order"\n\n[[solvers]]\nid = "random-search"\n\n'
    '[[solvers]]\nid = "hill-climbing"\nparameters = { restart_after = 5 }'
)


def _files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_from_a_generated_problem_to_a_verified_analysis_bundle(cli: Cli, tmp_path: Path) -> None:
    problem = tmp_path / "js.json"
    generated = cli(
        "benchmarks", "generate", "job-shop", "--seed", "1", "--set", "jobs=3", "--set",
        "machines=3", "--output", problem,
    )  # fmt: skip
    assert generated.code == 0
    result = tmp_path / "result.json"
    arguments = ("solve", problem, "--solver", "hill-climbing", "--max-evaluations", "100")
    solved = cli(*arguments, "--seed", "2", "--output", result, "--json")
    assert solved.code == 0
    assert solved.json()["feasible"] is True
    checked = cli("check", problem, result, "--json")
    assert checked.code == 0
    assert checked.json()["schedule"]["objectives"] == solved.json()["objectives"]

    config = campaign_file(tmp_path, problem=problem, solvers=CHEAP_SOLVERS)
    validated = cli("campaign", "validate", config, "--json")
    assert validated.code == 0
    plan = validated.json()
    assert (plan["runs"], len(plan["problems"]), plan["fairness"]["fair"]) == (15, 3, True)
    estimated = cli("campaign", "estimate", config, "--json").json()
    assert (estimated["planned"], estimated["remaining"], estimated["untimed"]) == (15, 15, 15)

    output = tmp_path / "runs"
    ran = cli("campaign", "run", config, "--output", output, "--json")
    assert ran.code == 0
    summary = ran.json()
    assert (summary["planned"], summary["completed"], summary["failed"]) == (15, 15, 0)
    assert "dispatchatlas: info: campaign tiny, session 1" in ran.err
    directory = output / "tiny"
    assert summary["directory"] == str(directory)
    assert len(read_runs(directory)) == 15
    resumed = cli("campaign", "run", config, "--output", output, "--json").json()
    assert (resumed["session"], resumed["attempted"], resumed["completed"]) == (2, 0, 15)
    left = cli("campaign", "estimate", config, "--output", output, "--json").json()
    assert (left["completed"], left["remaining"]) == (15, 0)

    replayed = cli("campaign", "replay", directory, "--json")
    assert replayed.code == 0
    assert replayed.json()["counts"]["match"] == 15

    record = directory / "runs" / "hill-climbing" / "job-shop-3x3-1" / "r000-evals.json"
    checked_record = cli("check", problem, record, "--json")
    assert checked_record.code == 0
    assert checked_record.json()["schedule"]["kind"] == "run record"

    bundle = tmp_path / "bundle"
    analyzed = cli(
        "analyze", directory, "--output", bundle, "--bootstrap-samples", "50",
        "--bootstrap-seed", "3", "--correction", "bonferroni", "--confidence", "0.9",
        "--replicate-floor", "2", "--json",
    )  # fmt: skip
    assert analyzed.code == 0
    report = analyzed.json()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    config_block = manifest["config"]
    assert (config_block["resamples"], config_block["seed"]) == (50, 3)
    assert (config_block["correction"], config_block["confidence"]) == ("bonferroni", 0.9)
    assert config_block["replicate_floor"] == 2
    assert manifest["campaign"]["id"] == "tiny"
    assert manifest["campaign"]["runs"]["succeeded"] == 15
    files = _files(bundle)
    assert set(manifest["files"]) == set(files) - {"manifest.json"}
    for name, digest in manifest["files"].items():
        assert hashlib.sha256(files[name]).hexdigest() == digest
    assert report["files"] == sorted(manifest["files"])
    assert report["protocols"] == ["evals"]
    assert "tables/evals/pairwise.md" in files


def test_validate_reports_the_plan_and_the_compatibility(cli: Cli, tmp_path: Path) -> None:
    config = campaign_file(tmp_path, solvers='id = "min-min"\n\n[[solvers]]\nid = "neh"')
    result = cli("campaign", "validate", config, "--check-compatibility")
    assert result.code == 0
    assert "min-min does not model permutation (problem flow-shop-4x3-1)" in result.out
    assert "runs        2" in result.out
    assert "fair        yes" in result.out
    document = cli("campaign", "validate", config, "--check-compatibility", "--json").json()
    assert len(document["compatibility"]["exclusions"]) == 2
    assert len(document["exclusions"]) == 2


def test_validate_explains_an_invalid_campaign_file(cli: Cli, tmp_path: Path) -> None:
    config = campaign_file(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace("replicates = 2", "replicate = 2"),
        encoding="utf-8",
    )
    result = cli("campaign", "validate", config)
    assert result.code == 1
    assert "the campaign has unknown key(s) 'replicate'" in result.err
    assert "hint: see 'dispatchatlas campaign validate --help'" in result.err
    missing = cli("campaign", "validate", tmp_path / "absent.toml")
    assert missing.code == 1
    assert "absent.toml does not exist" in missing.err
    excluded = campaign_file(tmp_path, campaign_id="none", solvers='id = "min-min"')
    nothing = cli("campaign", "validate", excluded, "--check-compatibility")
    assert nothing.code == 1
    assert "min-min does not model permutation" in nothing.out
    assert "plans no runs" in nothing.err
    as_json = cli("campaign", "validate", excluded, "--check-compatibility", "--json")
    assert as_json.code == 1
    assert len(as_json.json()["compatibility"]["exclusions"]) == 2


def test_run_refuses_results_of_another_configuration_unless_fresh(
    cli: Cli, tmp_path: Path
) -> None:
    output = tmp_path / "runs"
    first = campaign_file(tmp_path, execution='backend = "process"')
    assert cli("campaign", "run", first, "--output", output, "--backend", "sequential").code == 0
    changed = campaign_file(tmp_path, seeds="[1, 2, 3]")
    refused = cli("campaign", "run", changed, "--output", output)
    assert refused.code == 1
    assert "holds results of another configuration of campaign tiny" in refused.err
    assert "hint: pass --fresh to discard them" in refused.err
    fresh = cli("campaign", "run", changed, "--output", output, "--fresh", "--json")
    assert fresh.code == 0
    assert (fresh.json()["session"], fresh.json()["completed"]) == (1, 9)


def test_run_attempts_at_most_max_runs(cli: Cli, tmp_path: Path) -> None:
    config = campaign_file(tmp_path)
    output = tmp_path / "runs"
    partial = cli("campaign", "run", config, "--output", output, "--max-runs", "2", "--json")
    assert partial.code == 0
    assert (partial.json()["attempted"], partial.json()["pending"]) == (2, 4)
    text = cli("campaign", "run", config, "--output", output)
    assert text.code == 0
    assert "6 planned: 6 completed, 0 failed, 0 pending" in text.out


def test_run_exits_with_1_when_runs_fail(cli: Cli, tmp_path: Path) -> None:
    solvers = (
        'id = "input-order"\n\n[[solvers]]\nid = "enumeration"\nparameters = { max_tasks = 2 }'
    )
    config = campaign_file(tmp_path, solvers=solvers)
    output = tmp_path / "runs"
    result = cli("campaign", "run", config, "--output", output)
    assert result.code == 1
    assert "4 planned: 2 completed, 2 failed, 0 pending" in result.out
    assert "2 of 4 runs failed; their records are under" in result.err
    assert "hint: fix the cause, then run again with --retry-failed" in result.err
    retried = cli("campaign", "run", config, "--output", output, "--retry-failed", "--json")
    assert retried.code == 1
    assert retried.json()["attempted"] == 2


def test_replay_needs_a_campaign_directory_and_follows_a_moved_one(
    cli: Cli, tmp_path: Path
) -> None:
    missing = cli("campaign", "replay", tmp_path)
    assert missing.code == 1
    assert "is not a campaign directory: it has no campaign.json" in missing.err
    config = campaign_file(tmp_path)
    assert cli("campaign", "run", config, "--output", tmp_path / "runs").code == 0
    moved = tmp_path / "elsewhere" / "renamed"
    moved.parent.mkdir()
    (tmp_path / "runs" / "tiny").rename(moved)
    records = read_runs(moved)
    chosen = records[0].run_id
    result = cli("campaign", "replay", moved, "--run", chosen)
    assert result.code == 0
    assert "1 match, 0 mismatch, 0 skipped, 0 failed" in result.out
    unknown = cli("campaign", "replay", moved, "--run", "run-0000000000000000")
    assert unknown.code == 1
    assert "the plan has no runs run-0000000000000000" in unknown.err


def test_replay_detects_a_record_edited_after_the_run(cli: Cli, tmp_path: Path) -> None:
    config = campaign_file(tmp_path)
    assert cli("campaign", "run", config, "--output", tmp_path / "runs").code == 0
    directory = tmp_path / "runs" / "tiny"
    path = next((directory / "runs" / "random-search").rglob("*.json"))
    document = json.loads(path.read_text(encoding="utf-8"))
    document["outcome"]["evaluations"] += 1
    path.write_text(json.dumps(document), encoding="utf-8")
    result = cli("campaign", "replay", directory)
    assert result.code == 1
    assert "mismatch" in result.out
    assert "evaluations: recorded" in result.out
    assert "1 of 6 runs did not reproduce" in result.err


def test_analyze_compares_a_focal_solver_and_repeats_a_manifest(cli: Cli, tmp_path: Path) -> None:
    config = campaign_file(tmp_path, solvers=CHEAP_SOLVERS, seeds="[1, 2, 3]")
    assert cli("campaign", "run", config, "--output", tmp_path / "runs").code == 0
    directory = tmp_path / "runs" / "tiny"
    first = tmp_path / "first"
    arguments = ("analyze", directory, "--bootstrap-samples", "40", "--no-figures")
    compared = cli(*arguments, "--output", first, "--focal", "hill-climbing", "--baseline",
                   "random-search")  # fmt: skip
    assert compared.code == 0
    assert "**hill-climbing against baselines (makespan, protocol evals)**" in compared.out
    assert "| hill-climbing | random-search |" in compared.out
    document = cli(*arguments, "--output", first, "--focal", "hill-climbing", "--json").json()
    assert (document["config"]["focal"], document["config"]["baselines"]) == ("hill-climbing", None)
    assert "tables/evals/baselines.csv" in document["files"]
    rows = (first / "tables" / "evals" / "baselines.csv").read_text(encoding="utf-8").splitlines()
    assert [row.split(",")[1] for row in rows[1:]] == ["input-order", "random-search"]
    again = tmp_path / "again"
    repeated = cli("analyze", directory, "--output", again, "--no-figures", "--config",
                   first / "manifest.json")  # fmt: skip
    assert repeated.code == 0
    assert _files(again) == _files(first)


def test_analyze_errors_point_to_the_fix(cli: Cli, tmp_path: Path) -> None:
    not_campaign = cli("analyze", tmp_path, "--output", tmp_path / "bundle")
    assert not_campaign.code == 1
    assert "is not a campaign directory: it has no plan.json" in not_campaign.err
    assert "hint: pass the directory that 'dispatchatlas campaign run' reported" in not_campaign.err
    config = campaign_file(tmp_path)
    assert cli("campaign", "run", config, "--output", tmp_path / "runs").code == 0
    directory = tmp_path / "runs" / "tiny"
    bundle = tmp_path / "bundle"
    alone = cli("analyze", directory, "--output", bundle, "--baseline", "random-search")
    assert alone.code == 1
    assert "baselines need a focal solver to compare with them" in alone.err
    usage = cli("analyze", directory, "--output", bundle, "--formats", "svg,png")
    assert usage.code == 2
    assert "must list distinct formats of svg, tex, got 'svg,png'" in usage.err
    invalid = cli("analyze", directory, "--output", bundle, "--confidence", "1.5")
    assert invalid.code == 1
    assert "confidence must lie strictly between 0 and 1" in invalid.err
    settings = tmp_path / "settings.json"
    settings.write_text('{"resamples": 10, "colour": "red"}', encoding="utf-8")
    unknown = cli("analyze", directory, "--output", bundle, "--config", settings)
    assert unknown.code == 1
    assert "unknown analysis settings colour" in unknown.err
    focal = cli("analyze", directory, "--output", bundle, "--focal", "nobody")
    assert focal.code == 1
    assert "nobody" in focal.err


def test_analyze_writes_the_figure_formats_asked_for(cli: Cli, tmp_path: Path) -> None:
    config = campaign_file(tmp_path, solvers=CHEAP_SOLVERS)
    assert cli("campaign", "run", config, "--output", tmp_path / "runs").code == 0
    bundle = tmp_path / "bundle"
    arguments = ("analyze", tmp_path / "runs" / "tiny", "--output", bundle, "--formats", " tex")
    result = cli(*arguments, "--bootstrap-samples", "20", "--json")
    assert result.code == 0
    assert result.json()["figures"] == {"rendered": True, "formats": ["tex"]}
    figures = [name for name in result.json()["files"] if name.startswith("figures/")]
    assert figures
    assert all(name.endswith(".tex") for name in figures)


def test_the_affinity_option_reads_names_and_processor_indices() -> None:
    assert affinity("inherit") == "inherit"
    assert affinity("pin") == "pin"
    assert affinity("0,2,4") == (0, 2, 4)
    assert affinity("7") == (7,)
    for text in ("", "a,b", "1,,2", "-1", "pinned"):
        with pytest.raises(argparse.ArgumentTypeError, match="processor indices such as 0,2,4"):
            affinity(text)


def test_the_affinity_option_frees_a_campaign_from_its_processors(cli: Cli, tmp_path: Path) -> None:
    config = campaign_file(
        tmp_path,
        solvers=CHEAP_SOLVERS,
        execution='backend = "process"\naffinity = [4096]',
        campaign_id="pinned",
    )
    output = tmp_path / "runs"
    refused = cli("campaign", "run", config, "--output", output)
    assert refused.code != 0
    assert "affinity" in refused.err
    assert not (output / "pinned").exists()
    ran = cli("campaign", "run", config, "--output", output, "--affinity", "inherit", "--json")
    assert ran.code == 0
    summary = ran.json()
    assert summary["completed"] == summary["planned"] > 0
    directory = output / "pinned"
    assert cli("campaign", "run", config, "--output", output).code == 0
    refused = cli("campaign", "replay", directory)
    assert refused.code != 0
    assert "affinity" in refused.err
    assert cli("campaign", "replay", directory, "--backend", "sequential").code == 0
    assert cli("campaign", "replay", directory, "--affinity", "inherit").code == 0
