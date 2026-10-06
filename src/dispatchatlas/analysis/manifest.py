"""The reproducibility manifest of an analysis bundle.

``manifest.json`` records what a reader needs to repeat or to judge an
analysis:

- the campaign: id, config hash, seed, replicates, the digest of the analysed
  plan's runs, and run counts by status, and, when the campaign sets them,
  its seed namespace, its replication under a time limit, and the solvers'
  own replicate counts;
- the full :class:`~dispatchatlas.analysis.config.AnalysisConfig`;
- the DispatchAtlas version and the environment, as
  :func:`~dispatchatlas.campaigns.environment.capture_environment` records it;
- for each analysed protocol, the two sample sizes of the design: the
  replicates behind each per-problem value and the problems behind each
  paired test, with the smallest standardized effect the paired tests detect
  at the configured power, or the reason none is detectable;
- whether figures were rendered;
- the SHA-256 digest of every other file of the bundle.

The manifest holds no timestamp, so equal inputs give an identical file on
the same machine and software.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dispatchatlas import __version__
from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.dataset import Dataset
from dispatchatlas.analysis.results import Summary
from dispatchatlas.analysis.stats.power import attainable_floor, detectable_effect_size
from dispatchatlas.campaigns.environment import capture_environment

__all__ = ["MANIFEST_ARTIFACT", "MANIFEST_VERSION", "analysis_record", "build_manifest", "sha256"]

MANIFEST_ARTIFACT = "dispatchatlas.analysis-bundle"
"""The ``artifact`` field that identifies an analysis bundle manifest."""

MANIFEST_VERSION = 1
"""Version of the manifest format."""


def sha256(data: bytes | Path) -> str:
    """Return the SHA-256 hex digest of bytes, or of a file's content."""
    content = data.read_bytes() if isinstance(data, Path) else data
    return hashlib.sha256(content).hexdigest()


def analysis_record(dataset: Dataset, summary: Summary) -> dict[str, Any]:
    """Describe the design behind one protocol's summary.

    Args:
        dataset: The dataset the summary was computed from.
        summary: The summary.

    Returns:
        A JSON-ready dictionary: protocol, objective and sense, the analysed
        solvers and problems, the complete blocks, the replicates per cell,
        the problems per paired test, and the power statement.
    """
    config = summary.config
    planned = [cell.planned for cell in summary.cells]
    stochastic = {entry.name for entry in dataset.solvers if entry.stochastic}
    replicates = [cell.planned for cell in summary.cells if cell.solver in stochastic]
    pairs = [row.pairs for row in summary.pairwise]
    record: dict[str, Any] = {
        "protocol": summary.protocol,
        "objective": summary.objective,
        "sense": summary.sense,
        "solvers": list(summary.solvers),
        "problems": len(summary.problems),
        "complete_blocks": len(summary.blocks),
        "runs_per_cell": _span(planned),
        "stochastic_replicates": _span(replicates),
        "problems_per_pair": _span(pairs),
    }
    if not pairs:
        record["power"] = None
        return record
    fewest = min(pairs)
    floor = attainable_floor(config.alpha)
    power: dict[str, Any] = {
        "test": "wilcoxon-signed-rank",
        "pairs": fewest,
        "alpha": config.alpha,
        "power": config.power,
        "attainable_floor": floor,
    }
    if fewest < floor:
        power["detectable_dz"] = None
        power["note"] = (
            f"with {fewest} problems no two-sided Wilcoxon test reaches p <= {config.alpha:g}"
        )
    else:
        assessment = detectable_effect_size(
            fewest, alpha=config.alpha, power=config.power, seed=config.seed
        )
        power["detectable_dz"] = assessment.effect
        power["replicates"] = assessment.replicates
        power["seed"] = assessment.seed
    record["power"] = power
    return record


def build_manifest(
    dataset: Dataset,
    config: AnalysisConfig,
    analyses: list[dict[str, Any]],
    files: Mapping[str, str],
    figures: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the manifest document of a bundle.

    Args:
        dataset: The analysed dataset, before narrowing to a protocol.
        config: The configuration of the analysis.
        analyses: One :func:`analysis_record` per analysed protocol.
        files: SHA-256 digest of every other file, keyed by path relative to
            the bundle, with ``/`` separators.
        figures: Whether figures were rendered, and why not.
    """
    return {
        "artifact": MANIFEST_ARTIFACT,
        "version": MANIFEST_VERSION,
        "package": {"name": "dispatchatlas", "version": __version__},
        "campaign": _campaign(dataset),
        "config": config.to_dict(),
        "analyses": analyses,
        "figures": dict(figures),
        "environment": capture_environment(),
        "files": dict(sorted(files.items())),
    }


def _campaign(dataset: Dataset) -> dict[str, Any]:
    """Return what a manifest states about the campaign of a dataset.

    The seed and the replicates alone do not say which seeds the runs
    received or how many ran when the campaign sets a seed namespace,
    replicates solvers declared deterministic under a time limit, or gives a
    solver its own count. Each of the three is stated when it is set, the
    rule its place in the config hash follows: ``seed_namespace``,
    ``replicate_timed``, and ``solver_replicates``, the count of each solver
    that has its own, by solver name. ``runs_digest`` is the digest of the
    run ids and seeds of the analysed plan.
    """
    plan = dataset.plan
    record: dict[str, Any] = {
        "id": dataset.campaign,
        "config_hash": dataset.config_hash,
        "seed": plan.seed,
    }
    if plan.seed_namespace is not None:
        record["seed_namespace"] = plan.seed_namespace
    record["replicates"] = plan.replicates
    if plan.replicate_timed:
        record["replicate_timed"] = True
    counts = {
        solver.name: solver.replicates for solver in plan.solvers if solver.replicates is not None
    }
    if counts:
        record["solver_replicates"] = counts
    record["runs_digest"] = plan.runs_digest
    record["runs"] = dict(dataset.counts())
    return record


def _span(values: list[int]) -> dict[str, int] | None:
    return {"min": min(values), "max": max(values)} if values else None
