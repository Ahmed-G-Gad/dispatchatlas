"""``dispatchatlas analyze``: the analysis bundle of a campaign."""

from __future__ import annotations

import argparse
import dataclasses
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dispatchatlas.analysis import AnalysisConfig, write_bundle
from dispatchatlas.analysis.bundle import FIGURE_FORMATS, MANIFEST_FILE
from dispatchatlas.analysis.manifest import MANIFEST_ARTIFACT
from dispatchatlas.analysis.stats.corrections import CORRECTIONS
from dispatchatlas.campaigns.store import PLAN_FILE
from dispatchatlas.cli._common import (
    PROG,
    CommandError,
    Commands,
    add_command,
    emit_fields,
    emit_json,
    existing_file,
    finite_number,
    non_negative_integer,
    positive_integer,
)
from dispatchatlas.core.errors import ConfigError

__all__ = ["register"]

_OPTIONS: Mapping[str, str] = {
    "protocol": "protocol",
    "objective": "objective",
    "sense": "sense",
    "solver": "solvers",
    "problem": "problems",
    "alpha": "alpha",
    "confidence": "confidence",
    "correction": "correction",
    "bootstrap_samples": "resamples",
    "bootstrap_seed": "seed",
    "replicate_floor": "replicate_floor",
    "focal": "focal",
    "baseline": "baselines",
}
"""Command-line option (as an attribute) to the :class:`AnalysisConfig` field it sets."""

_DEFAULTS = AnalysisConfig()
"""The default settings, quoted in the help."""


def register(commands: Commands) -> None:
    """Add the ``analyze`` command."""
    parser = add_command(
        commands,
        "analyze",
        _analyze,
        "Write the analysis bundle of a campaign: tables, figures, and a manifest.",
        description="Analyse the runs of a campaign directory and write the analysis bundle: "
        "an index, per-protocol tables in Markdown, CSV, and LaTeX, figures, the caveats of "
        "the analysis, and a manifest with the settings and every file's SHA-256 digest. "
        "Settings come from --config, then from the options. With --focal, the bundle also "
        "holds the table of the focal solver against each baseline, which is printed too.",
        epilog="examples:\n"
        f"  {PROG} analyze results/shop-baselines --output analysis/shop-baselines\n"
        f"  {PROG} analyze results/shop-baselines --output again "
        "--config analysis/shop-baselines/manifest.json\n"
        f"  {PROG} analyze results/shop-baselines --output analysis --focal tabu-search "
        "--baseline random-search",
    )
    parser.add_argument("campaign_dir", metavar="CAMPAIGN_DIR", help="campaign directory")
    parser.add_argument("--output", metavar="DIR", required=True, help="bundle directory")
    parser.add_argument(
        "--config",
        metavar="FILE",
        help="JSON or TOML file of analysis settings, such as the manifest of an earlier "
        "bundle, whose settings it repeats",
    )
    parser.add_argument(
        "--formats",
        type=_formats,
        default=FIGURE_FORMATS,
        metavar="LIST",
        help=f"figure formats, comma-separated (default: {','.join(FIGURE_FORMATS)})",
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="write the tables and the manifest only"
    )
    chosen = parser.add_argument_group("what to analyse")
    chosen.add_argument("--protocol", metavar="NAME", help="one protocol (default: every one)")
    chosen.add_argument(
        "--objective", metavar="NAME", help="objective to compare (default: the primary one)"
    )
    chosen.add_argument(
        "--sense", choices=("min", "max"), help="default: the sense the objective declares"
    )
    chosen.add_argument(
        "--solver", action="append", metavar="NAME", help="analyse this solver; repeat for more"
    )
    chosen.add_argument(
        "--problem", action="append", metavar="ID", help="analyse this problem; repeat for more"
    )
    statistics = parser.add_argument_group("statistics")
    statistics.add_argument(
        "--alpha",
        type=finite_number,
        metavar="A",
        help=f"significance level (default: {_DEFAULTS.alpha:g})",
    )
    statistics.add_argument(
        "--confidence",
        type=finite_number,
        metavar="C",
        help=f"interval coverage (default: {_DEFAULTS.confidence:g})",
    )
    statistics.add_argument(
        "--correction",
        choices=CORRECTIONS,
        help=f"multiple-comparison correction of the pairwise tests "
        f"(default: {_DEFAULTS.correction})",
    )
    statistics.add_argument(
        "--bootstrap-samples",
        type=non_negative_integer,
        metavar="N",
        help=f"bootstrap resamples per interval (default: {_DEFAULTS.resamples})",
    )
    statistics.add_argument(
        "--bootstrap-seed",
        type=non_negative_integer,
        metavar="N",
        help=f"seed of the bootstrap and of the power simulation (default: {_DEFAULTS.seed})",
    )
    statistics.add_argument(
        "--replicate-floor",
        type=positive_integer,
        metavar="N",
        help="report cells of stochastic solvers with fewer replicates as a limitation",
    )
    focal = parser.add_argument_group("comparison with baselines")
    focal.add_argument(
        "--focal", metavar="NAME", help="solver under study, compared with each baseline"
    )
    focal.add_argument(
        "--baseline",
        action="append",
        metavar="NAME",
        help="a baseline; repeat for more (default: every other solver)",
    )


def _analysis_config(arguments: argparse.Namespace) -> AnalysisConfig:
    """Build the analysis settings: the ``--config`` file, overridden by the options.

    Raises:
        ConfigError: If the file holds unknown settings or a value is invalid.
    """
    values: dict[str, Any] = {}
    if arguments.config is not None:
        values.update(_settings_file(existing_file(arguments.config)))
    for option, name in _OPTIONS.items():
        value = getattr(arguments, option)
        if value is not None:
            values[name] = tuple(value) if isinstance(value, list) else value
    known = {item.name for item in dataclasses.fields(AnalysisConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ConfigError(
            f"unknown analysis settings {', '.join(unknown)}; known: {', '.join(sorted(known))}"
        )
    return AnalysisConfig(**values)


def _analyze(arguments: argparse.Namespace) -> int:
    directory = Path(arguments.campaign_dir)
    if not (directory / PLAN_FILE).is_file():
        raise CommandError(
            f"{directory} is not a campaign directory: it has no {PLAN_FILE}",
            hint=f"pass the directory that '{PROG} campaign run' reported",
        )
    config = _analysis_config(arguments)
    target = write_bundle(
        directory,
        arguments.output,
        config,
        figures=not arguments.no_figures,
        formats=arguments.formats,
    )
    manifest = json.loads((target / MANIFEST_FILE).read_text(encoding="utf-8"))
    protocols = [item["protocol"] for item in manifest["analyses"]]
    comparisons = [
        name
        for name in (f"tables/{protocol}/baselines.md" for protocol in protocols)
        if name in manifest["files"]
    ]
    report = {
        "directory": target,
        "manifest": target / MANIFEST_FILE,
        "campaign": manifest["campaign"]["id"],
        "config_hash": manifest["campaign"]["config_hash"],
        "protocols": protocols,
        "files": sorted(manifest["files"]),
        "figures": manifest["figures"],
        "config": manifest["config"],
    }
    if arguments.json:
        emit_json(report)
        return 0
    figures = manifest["figures"]
    emit_fields(
        [
            ("bundle", target),
            ("campaign", f"{report['campaign']} (config hash {report['config_hash'][:12]})"),
            ("protocols", report["protocols"]),
            ("files", f"{len(report['files'])}, listed with their digests in {MANIFEST_FILE}"),
            (
                "figures",
                ", ".join(figures["formats"]) if figures["rendered"] else figures["reason"],
            ),
            ("start with", target / "index.md"),
        ]
    )
    for name in comparisons:
        print()
        print((target / name).read_text(encoding="utf-8"), end="")
    return 0


def _formats(argument: str) -> tuple[str, ...]:
    """Parse a comma-separated list of figure formats, such as ``svg,tex``."""
    names = tuple(name.strip() for name in argument.split(","))
    if set(names) - set(FIGURE_FORMATS) or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(
            f"must list distinct formats of {', '.join(FIGURE_FORMATS)}, got {argument!r}"
        )
    return names


def _settings_file(path: Path) -> dict[str, Any]:
    """Read analysis settings from JSON or TOML; a bundle manifest gives its ``config``."""
    raw = path.read_text(encoding="utf-8")
    toml = path.suffix.lower() == ".toml"
    try:
        document = tomllib.loads(raw) if toml else json.loads(raw)
    except (tomllib.TOMLDecodeError, json.JSONDecodeError) as error:
        raise ConfigError(f"{path} is not valid {'TOML' if toml else 'JSON'}: {error}") from error
    if isinstance(document, Mapping) and document.get("artifact") == MANIFEST_ARTIFACT:
        document = document.get("config")
    if not isinstance(document, Mapping):
        raise ConfigError(f"{path} must hold a table of analysis settings")
    return dict(document)
