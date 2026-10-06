r"""The analysis bundle: every table, figure, and caveat of a campaign in one directory.

:func:`write_bundle` analyses a campaign directory and writes::

    <output>/
        index.md                  what the bundle holds: tables, figures, methods, and sources
        limitations.md            the caveats of every analysis
        runs.csv                  one row per analysed run, at full precision
        tables/<protocol>/<table>.md, .csv, .tex
        figures/<protocol>/...    SVG and TikZ figures, when the figure package is available
        manifest.json             configuration, versions, environment, and file digests

Each protocol is analysed separately (see
:func:`~dispatchatlas.analysis.summary.summarize`). Tables with a p-value
always carry an effect size and an interval beside it. The multi-objective
indicator table appears when a problem has several objectives, the
deviation table when the configuration gives reference values, and the
baselines table, the focal solver against each baseline (see
:func:`~dispatchatlas.analysis.comparison.compare_to_baselines`), when it
names a focal solver. Figures are drawn with the same settings as the
tables, in the formats asked for, and the index lists each with its caption
and the sources of its method.

Output is byte-reproducible: equal inputs, configuration, and software give
identical files. Content files hold no timestamps or paths, JSON is written
with sorted keys, and text with ``\n`` line endings; the environment appears
in the manifest only.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from dispatchatlas.analysis.comparison import compare_to_baselines
from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.dataset import Dataset, load_dataset
from dispatchatlas.analysis.deviation import arpd
from dispatchatlas.analysis.indicators import indicator_report
from dispatchatlas.analysis.manifest import (
    MANIFEST_ARTIFACT,
    analysis_record,
    build_manifest,
    sha256,
)
from dispatchatlas.analysis.methods import Method, method
from dispatchatlas.analysis.results import Limitation, Summary
from dispatchatlas.analysis.summary import summarize
from dispatchatlas.analysis.tables import (
    Table,
    baselines_table,
    cells_table,
    deviation_table,
    indicators_table,
    omnibus_table,
    pairwise_table,
    runs_table,
    solvers_table,
    to_csv,
    to_latex,
    to_markdown,
)
from dispatchatlas.core.errors import ConfigError, SerializationError
from dispatchatlas.core.io import atomic_write_bytes
from dispatchatlas.core.logs import get_logger
from dispatchatlas.references import cite

__all__ = ["FIGURE_FORMATS", "MANIFEST_FILE", "write_bundle"]

MANIFEST_FILE = "manifest.json"
"""Name of the manifest in a bundle directory."""

FIGURE_FORMATS: tuple[str, ...] = ("svg", "tex")
"""The formats figures can be written in: SVG, and TikZ for LaTeX."""

_LOG = get_logger(__name__)

_RENDERERS: tuple[tuple[str, Callable[[Table], str]], ...] = (
    ("md", to_markdown),
    ("csv", to_csv),
    ("tex", to_latex),
)
_FIGURE_MODULE = "dispatchatlas.analysis.figures"

type _RenderFigures = Callable[..., list[Path]]


def write_bundle(
    campaign_directory: str | Path,
    output_directory: str | Path,
    config: AnalysisConfig | None = None,
    *,
    figures: bool = True,
    formats: Sequence[str] = FIGURE_FORMATS,
) -> Path:
    """Analyse a campaign and write the bundle.

    Args:
        campaign_directory: A campaign directory written by
            :func:`dispatchatlas.campaigns.run`.
        output_directory: Where to write the bundle. It must be absent,
            empty, or hold an earlier bundle, whose files are replaced once
            the analysis has succeeded.
        config: The analysis settings. Without a protocol, every protocol of
            the campaign is analysed. With a focal solver, the bundle holds
            the baselines table of every protocol.
        figures: Render figures through
            ``dispatchatlas.analysis.figures.render_figures``, with the
            settings of each protocol's analysis. When that package is not
            installed, the bundle is written without figures and the
            manifest says why.
        formats: Formats of the figures, from :data:`FIGURE_FORMATS`. Their
            order does not matter.

    Returns:
        The bundle directory.

    Raises:
        ConfigError: If the output directory holds other files, a format is
            unknown, or the configuration does not fit the campaign.
        SerializationError: If the campaign directory cannot be read.
    """
    settings = config if config is not None else AnalysisConfig()
    chosen = _figure_formats(formats)
    dataset = load_dataset(campaign_directory)
    protocols = (
        [settings.protocol]
        if settings.protocol is not None
        else [protocol.name for protocol in dataset.protocols]
    )
    analysed = dataset.select(
        protocols=protocols, solvers=settings.solvers, problems=settings.problems
    )
    target = Path(output_directory)
    runs = runs_table(analysed, settings.timeout_score)
    contents: dict[str, bytes] = {"runs.csv": to_csv(runs).encode("utf-8")}
    records: list[dict[str, Any]] = []
    limitations: list[tuple[str, Limitation]] = []
    sections: list[str] = []
    used: set[str] = {"effect-sizes-beside-p-values"}
    for protocol in protocols:
        focused = replace(settings, protocol=protocol)
        summary = summarize(dataset, focused)
        tables = [
            solvers_table(summary),
            cells_table(summary),
            pairwise_table(summary),
            omnibus_table(summary),
        ]
        used.update(_methods_of(summary))
        report = indicator_report(dataset, focused)
        if report.rows:
            tables.append(indicators_table(report))
            used.update(("hypervolume", "igd", "igd-plus"))
        limitations.extend((protocol, item) for item in (*summary.limitations, *report.limitations))
        if settings.references is not None:
            tables.append(
                deviation_table(
                    arpd(dataset, focused), summary.objective, settings.reference_source
                )
            )
            used.add("relative-percentage-deviation")
        if settings.focal is not None:
            comparison = compare_to_baselines(dataset, settings.focal, settings.baselines, focused)
            tables.append(baselines_table(comparison))
            if comparison.correction != "none":
                used.add(comparison.correction)
            limitations.extend(
                (protocol, item)
                for item in comparison.limitations
                if item not in summary.limitations
            )
        for table in tables:
            for extension, render in _RENDERERS:
                path = f"tables/{protocol}/{table.name}.{extension}"
                contents[path] = render(table).encode("utf-8")
        records.append(analysis_record(dataset, summary))
        sections.append(_section(summary, tables))
    _clear(target)
    for relative, data in contents.items():
        atomic_write_bytes(target / relative, data)
    digests = {relative: sha256(data) for relative, data in contents.items()}
    rendered, figure_files, figure_lines = _figures(
        analysed, protocols, target, settings, formats=chosen, enabled=figures
    )
    digests.update(figure_files)
    limitation_text = _limitations_markdown(limitations, rendered)
    index_text = _index_markdown(dataset, settings, sections, figure_lines, used)
    for relative, text in (("limitations.md", limitation_text), ("index.md", index_text)):
        data = text.encode("utf-8")
        atomic_write_bytes(target / relative, data)
        digests[relative] = sha256(data)
    manifest = build_manifest(dataset, settings, records, digests, rendered)
    document = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    atomic_write_bytes(target / MANIFEST_FILE, (document + "\n").encode("ascii"))
    _LOG.info("wrote the analysis bundle of campaign %s to %s", dataset.campaign, target)
    return target


def _clear(target: Path) -> None:
    """Make ``target`` ready for a bundle, removing the files of an earlier one."""
    if target.exists() and not target.is_dir():
        raise ConfigError(f"{target} is a file, not a directory")
    if not target.exists() or not any(target.iterdir()):
        target.mkdir(parents=True, exist_ok=True)
        return
    manifest = target / MANIFEST_FILE
    previous: Any = None
    if manifest.is_file():
        try:
            previous = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
    if not isinstance(previous, dict) or previous.get("artifact") != MANIFEST_ARTIFACT:
        raise ConfigError(
            f"{target} is not empty and holds no analysis bundle; choose an empty directory"
        )
    root = target.resolve()
    for relative in previous.get("files", {}):
        path = (target / relative).resolve()
        if path.is_relative_to(root) and path.is_file():
            path.unlink()
    manifest.unlink()
    for folder in sorted((p for p in target.rglob("*") if p.is_dir()), reverse=True):
        if not any(folder.iterdir()):
            folder.rmdir()


def _figure_formats(formats: Sequence[str]) -> tuple[str, ...]:
    """Return the requested figure formats in the order of :data:`FIGURE_FORMATS`.

    A canonical order keeps the bundle identical however the formats are
    listed.

    Raises:
        ConfigError: If no format is given, a format is unknown, or one is
            repeated.
    """
    requested = [formats] if isinstance(formats, str) else list(formats)
    unknown = sorted(set(requested) - set(FIGURE_FORMATS))
    if not requested or unknown or len(set(requested)) != len(requested):
        raise ConfigError(
            f"figure formats must be distinct entries of {list(FIGURE_FORMATS)}, got {requested}"
        )
    return tuple(name for name in FIGURE_FORMATS if name in requested)


def _figures(
    dataset: Dataset,
    protocols: list[str],
    target: Path,
    settings: AnalysisConfig,
    *,
    formats: tuple[str, ...],
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Render the figures of every protocol.

    Returns:
        The manifest entry, the digest of every figure file in the order the
        figures were written, and the lines of the index that list them.
    """
    if not enabled:
        return {"rendered": False, "reason": "figures were not requested"}, {}, []
    try:
        module = importlib.import_module(_FIGURE_MODULE)
        render: _RenderFigures = module.render_figures
    except (ImportError, AttributeError) as error:
        _LOG.warning("figures are skipped: %s", error)
        reason = f"the figure package is not available: {error}"
        return {"rendered": False, "reason": reason}, {}, []
    root = target.resolve()
    digests: dict[str, str] = {}
    try:
        for protocol in protocols:
            directory = target / "figures" / protocol
            for path in render(
                dataset.select(protocols=[protocol]),
                directory,
                formats=formats,
                config=replace(settings, protocol=protocol),
            ):
                resolved = Path(path).resolve()
                if not resolved.is_relative_to(root):
                    raise SerializationError(f"a figure was written outside the bundle: {path}")
                digests[resolved.relative_to(root).as_posix()] = sha256(resolved)
    except ImportError as error:
        _LOG.warning("figures are skipped: %s", error)
        return {"rendered": False, "reason": f"a figure dependency is missing: {error}"}, {}, []
    lines = _figure_lines(list(digests), getattr(module, "family_of", None))
    return {"rendered": True, "formats": list(formats)}, digests, lines


def _figure_lines(files: Sequence[str], family_of: Callable[[str], Any] | None) -> list[str]:
    """Return the index lines listing each figure with its caption and sources.

    Figures are grouped by protocol, then by family in the order they were
    written. A file whose family the figure package does not know is listed
    as a plain link.
    """
    grouped: dict[str, dict[str, list[str]]] = {}
    for relative in files:
        parts = Path(relative).parts
        protocol = parts[1] if len(parts) > 2 else ""
        grouped.setdefault(protocol, {}).setdefault(Path(relative).stem, []).append(relative)
    lines: list[str] = []
    for protocol, stems in grouped.items():
        lines.extend(["", f"### Protocol `{protocol}`", ""])
        families: dict[str, tuple[Any, list[tuple[str, list[str]]]]] = {}
        for stem, paths in stems.items():
            spec = family_of(stem) if family_of is not None else None
            if spec is None:
                lines.extend(f"- [{path}]({path})" for path in paths)
                continue
            families.setdefault(spec.id, (spec, []))[1].append((stem, paths))
        for spec, figures in families.values():
            links = [
                " · ".join(f"[{Path(path).suffix[1:]}]({path})" for path in paths)
                for _, paths in figures
            ]
            if spec.prefix:
                links = [
                    f"{stem[len(spec.prefix) :]}: {link}"
                    for (stem, _), link in zip(figures, links, strict=True)
                ]
            lines.append(f"- **{spec.title}** ({'; '.join(links)}). {spec.caption}")
            lines.extend(f"  - {cite(key)}" for key in spec.references)
    return lines


def _methods_of(summary: Summary) -> set[str]:
    """Return the catalog ids of the methods a summary applied."""
    interval = "bootstrap-bca" if summary.config.interval == "bca" else "bootstrap-percentile"
    used = {"wilcoxon-signed-rank", "paired-win-share", "cohens-dz", interval}
    if summary.config.correction != "none":
        used.add(summary.config.correction)
    if summary.omnibus is not None:
        used.update(("friedman", "iman-davenport", "kendalls-w"))
    if summary.critical_difference is not None:
        used.add("nemenyi")
    return used


def _section(summary: Summary, tables: list[Table]) -> str:
    counts = {row.solver: row for row in summary.solver_summaries}
    planned = sum(row.planned for row in counts.values())
    succeeded = sum(row.succeeded for row in counts.values())
    sense = "minimised" if summary.sense == "min" else "maximised"
    lines = [
        f"## Protocol `{summary.protocol}`",
        "",
        f"Objective `{summary.objective}` ({sense}): {len(summary.solvers)} solvers on "
        f"{len(summary.problems)} problems, {succeeded} of {planned} planned runs succeeded.",
        "",
        "| Table | Files |",
        "| --- | --- |",
    ]
    for table in tables:
        links = " · ".join(
            f"[{extension}](tables/{summary.protocol}/{table.name}.{extension})"
            for extension, _ in _RENDERERS
        )
        lines.append(f"| {table.title.replace('|', '/')} | {links} |")
    return "\n".join(lines)


def _index_markdown(
    dataset: Dataset,
    config: AnalysisConfig,
    sections: list[str],
    figure_lines: list[str],
    used: set[str],
) -> str:
    lines = [
        f"# Analysis of campaign `{dataset.campaign}`",
        "",
        f"Config hash `{dataset.config_hash}`. Runs are listed in [runs.csv](runs.csv), the "
        "caveats in [limitations.md](limitations.md), and the settings, versions, and file "
        "digests in [manifest.json](manifest.json).",
        "",
        f"Tests use alpha = {config.alpha:g}; intervals are {config.confidence * 100:g}% "
        f"{'BCa' if config.interval == 'bca' else 'percentile'} bootstrap intervals with "
        f"{config.resamples} resamples and seed {config.seed}. A run without a feasible result "
        "counts as a failure: success rates divide by the planned runs, and scores count a "
        "failure as the worst value.",
    ]
    for section in sections:
        lines.extend(["", section])
    if figure_lines:
        lines.extend(["", "## Figures"])
        lines.extend(figure_lines)
    catalog: list[Method] = sorted((method(identifier) for identifier in used), key=_name)
    lines.extend(["", "## Methods", ""])
    lines.extend(f"- {item.name}: {item.summary}" for item in catalog)
    keys = sorted({key for item in catalog for key in item.references})
    lines.extend(["", "## References", ""])
    lines.extend(f"- {cite(key)}" for key in keys)
    return "\n".join(lines) + "\n"


def _name(item: Method) -> str:
    return item.name


def _limitations_markdown(
    limitations: list[tuple[str, Limitation]], figures: dict[str, Any]
) -> str:
    lines = ["# Limitations"]
    protocols = list(dict.fromkeys(protocol for protocol, _ in limitations))
    for protocol in protocols:
        lines.extend(["", f"## Protocol `{protocol}`", ""])
        lines.extend(
            f"- **{item.topic}**: {item.message}" for name, item in limitations if name == protocol
        )
    if not limitations:
        lines.extend(["", "No limitation was found."])
    if not figures.get("rendered", False):
        lines.extend(["", "## Figures", "", f"- Figures were not rendered: {figures['reason']}."])
    return "\n".join(lines) + "\n"
