"""The analysis bundle: contents, determinism, digests, and figure fallback."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from dispatchatlas import __version__
from dispatchatlas.analysis.bundle import write_bundle
from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.dataset import Dataset
from dispatchatlas.analysis.manifest import build_manifest, sha256
from dispatchatlas.campaigns import Protocol
from dispatchatlas.core import StopCriteria
from dispatchatlas.core.errors import ConfigError, MissingDependencyError, SerializationError
from tests.analysis.analysis_data import campaign_dir, make_dataset, run_tiny_campaign

__all__ = ["campaign_dir"]

TABLES = ("solvers", "cells", "pairwise", "omnibus")


def _files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_two_bundles_of_one_campaign_are_byte_identical(campaign_dir: Path, tmp_path: Path) -> None:
    config = AnalysisConfig(resamples=300)
    first = _files(write_bundle(campaign_dir, tmp_path / "one", config, figures=False))
    second = _files(write_bundle(campaign_dir, tmp_path / "two", config, figures=False))
    assert first == second
    expected = {"index.md", "limitations.md", "manifest.json", "runs.csv"} | {
        f"tables/evals/{name}.{extension}" for name in TABLES for extension in ("md", "csv", "tex")
    }
    assert set(first) == expected
    assert all(b"\r\n" not in data for data in first.values())


def test_the_manifest_states_the_optional_campaign_settings_only_when_set() -> None:
    dataset = make_dataset({"a": {"p1": [1.0, 2.0]}, "b": {"p1": [2.0, 1.0]}})
    plain = build_manifest(dataset, AnalysisConfig(), [], {}, {"rendered": False})["campaign"]
    assert plain == {
        "id": dataset.campaign,
        "config_hash": dataset.config_hash,
        "seed": dataset.plan.seed,
        "replicates": dataset.plan.replicates,
        "runs_digest": dataset.plan.runs_digest,
        "runs": dict(dataset.counts()),
    }

    solvers = tuple(
        replace(entry, replicates=4) if entry.name == "a" else entry
        for entry in dataset.plan.solvers
    )
    plan = replace(dataset.plan, seed_namespace="comparison", replicate_timed=True, solvers=solvers)
    stated = build_manifest(
        replace(dataset, plan=plan), AnalysisConfig(), [], {}, {"rendered": False}
    )["campaign"]
    assert stated == {
        **plain,
        "seed_namespace": "comparison",
        "replicate_timed": True,
        "solver_replicates": {"a": 4},
    }
    assert list(stated)[:4] == ["id", "config_hash", "seed", "seed_namespace"]


def test_the_manifest_records_digests_config_and_design(campaign_dir: Path, tmp_path: Path) -> None:
    config = AnalysisConfig(resamples=300, seed=4)
    target = write_bundle(campaign_dir, tmp_path / "bundle", config, figures=False)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    files = _files(target)
    assert set(manifest["files"]) == set(files) - {"manifest.json"}
    for name, digest in manifest["files"].items():
        assert digest == sha256(files[name])
        assert sha256(target / name) == digest
    assert manifest["artifact"] == "dispatchatlas.analysis-bundle"
    assert manifest["package"] == {"name": "dispatchatlas", "version": __version__}
    assert manifest["config"] == json.loads(json.dumps(config.to_dict()))
    assert manifest["campaign"]["runs"]["succeeded"] == 56
    assert manifest["figures"]["rendered"] is False
    (analysis,) = manifest["analyses"]
    assert analysis["problems_per_pair"] == {"min": 7, "max": 7}
    assert analysis["stochastic_replicates"] == {"min": 3, "max": 3}
    assert analysis["power"]["pairs"] == 7
    assert analysis["power"]["detectable_dz"] > 1.0
    assert "python" in manifest["environment"]
    index = (target / "index.md").read_text(encoding="utf-8")
    assert "Wilcoxon, F. " not in index
    assert "F. Wilcoxon (1945). Individual comparisons by ranking methods." in index
    assert "## Protocol `evals`" in index


def test_figures_are_skipped_and_recorded_when_unavailable(
    campaign_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "dispatchatlas.analysis.figures", None)
    target = write_bundle(campaign_dir, tmp_path / "bundle", AnalysisConfig(resamples=100))
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["figures"]["rendered"] is False
    assert "not available" in manifest["figures"]["reason"]
    assert "Figures were not rendered" in (target / "limitations.md").read_text(encoding="utf-8")
    assert not (target / "figures").exists()


def test_an_earlier_bundle_is_replaced_and_other_directories_are_refused(
    campaign_dir: Path, tmp_path: Path
) -> None:
    target = tmp_path / "bundle"
    write_bundle(campaign_dir, target, AnalysisConfig(resamples=100), figures=False)
    stale = target / "tables" / "evals" / "stale.md"
    stale.write_text("old", encoding="utf-8")
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["tables/evals/stale.md"] = sha256(stale)
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_bundle(campaign_dir, target, AnalysisConfig(resamples=100), figures=False)
    assert not stale.exists()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(ConfigError, match="holds no analysis bundle"):
        write_bundle(campaign_dir, foreign, figures=False)


def test_every_protocol_and_the_deviation_table_are_written(tmp_path: Path) -> None:
    protocols = (
        Protocol("short", StopCriteria(max_evaluations=5)),
        Protocol("long", StopCriteria(max_evaluations=15)),
    )
    directory = run_tiny_campaign(
        tmp_path / "runs", protocols=protocols, seeds=(1, 2), replicates=2
    )
    references = {f"flow-shop-4x3-{seed}": 300.0 for seed in (1, 2)}
    config = AnalysisConfig(resamples=100, references=references, reference_source="test values")
    target = write_bundle(directory, tmp_path / "bundle", config, figures=False)
    names = set(_files(target))
    for protocol in ("short", "long"):
        for table in (*TABLES, "deviation"):
            assert f"tables/{protocol}/{table}.md" in names
    limitations = (target / "limitations.md").read_text(encoding="utf-8")
    assert "## Protocol `short`" in limitations
    assert "fewer than 6 shared problems" in limitations
    deviation = (target / "tables" / "long" / "deviation.md").read_text(encoding="utf-8")
    assert "Reference values: test values." in deviation


def _stub_figures(render: Callable[..., list[Path]]) -> ModuleType:
    module = ModuleType("dispatchatlas.analysis.figures")
    module.render_figures = render  # type: ignore[attr-defined]
    return module


def test_rendered_figures_are_hashed_and_listed(
    campaign_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[set[str], Path, tuple[str, ...], str | None]] = []

    def render(
        dataset: Dataset,
        directory: Path,
        formats: tuple[str, ...],
        config: AnalysisConfig | None = None,
    ) -> list[Path]:
        protocol = None if config is None else config.protocol
        calls.append(({run.protocol for run in dataset.runs}, directory, formats, protocol))
        directory.mkdir(parents=True, exist_ok=True)
        figure = directory / "ranks.svg"
        figure.write_text(f"<svg>{len(dataset.runs)}</svg>\n", encoding="utf-8")
        return [figure]

    monkeypatch.setitem(sys.modules, "dispatchatlas.analysis.figures", _stub_figures(render))
    target = write_bundle(campaign_dir, tmp_path / "bundle", AnalysisConfig(resamples=100))
    assert calls == [({"evals"}, target / "figures" / "evals", ("svg", "tex"), "evals")]
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["figures"] == {"rendered": True, "formats": ["svg", "tex"]}
    assert manifest["files"]["figures/evals/ranks.svg"] == sha256(
        target / "figures/evals/ranks.svg"
    )
    assert "[figures/evals/ranks.svg](figures/evals/ranks.svg)" in (target / "index.md").read_text(
        encoding="utf-8"
    )


def test_figures_outside_the_bundle_or_without_their_dependency(
    campaign_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "elsewhere.svg"

    def escape(
        dataset: Dataset,
        directory: Path,
        formats: tuple[str, ...],
        config: AnalysisConfig | None = None,
    ) -> list[Path]:
        outside.write_text("<svg/>", encoding="utf-8")
        return [outside]

    monkeypatch.setitem(sys.modules, "dispatchatlas.analysis.figures", _stub_figures(escape))
    with pytest.raises(SerializationError, match="outside the bundle"):
        write_bundle(campaign_dir, tmp_path / "one", AnalysisConfig(resamples=100))

    def missing(
        dataset: Dataset,
        directory: Path,
        formats: tuple[str, ...],
        config: AnalysisConfig | None = None,
    ) -> list[Path]:
        raise MissingDependencyError("matplotlib", "plot", "Figures")

    monkeypatch.setitem(sys.modules, "dispatchatlas.analysis.figures", _stub_figures(missing))
    target = write_bundle(campaign_dir, tmp_path / "two", AnalysisConfig(resamples=100))
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["figures"]["rendered"] is False
    assert "matplotlib" in manifest["figures"]["reason"]


def test_formats_choose_the_figure_files_in_one_order(
    campaign_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []

    def render(
        dataset: Dataset,
        directory: Path,
        formats: tuple[str, ...],
        config: AnalysisConfig | None = None,
    ) -> list[Path]:
        calls.append(formats)
        directory.mkdir(parents=True, exist_ok=True)
        paths = [directory / f"ranks.{name}" for name in formats]
        for path in paths:
            path.write_text(f"{path.suffix}\n", encoding="utf-8")
        return paths

    monkeypatch.setitem(sys.modules, "dispatchatlas.analysis.figures", _stub_figures(render))
    config = AnalysisConfig(resamples=100)
    only_tex = write_bundle(campaign_dir, tmp_path / "tex", config, formats=["tex"])
    manifest = json.loads((only_tex / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["figures"] == {"rendered": True, "formats": ["tex"]}
    assert "figures/evals/ranks.tex" in manifest["files"]
    assert "figures/evals/ranks.svg" not in manifest["files"]
    first = _files(write_bundle(campaign_dir, tmp_path / "a", config, formats=("tex", "svg")))
    second = _files(write_bundle(campaign_dir, tmp_path / "b", config, formats=("svg", "tex")))
    assert calls == [("tex",), ("svg", "tex"), ("svg", "tex")]
    assert first == second
    for formats in (("png",), (), ("svg", "svg")):
        with pytest.raises(ConfigError, match="figure formats must be distinct entries"):
            write_bundle(campaign_dir, tmp_path / "refused", config, formats=formats)
    assert not (tmp_path / "refused").exists()


def test_a_focal_solver_adds_the_baselines_table(campaign_dir: Path, tmp_path: Path) -> None:
    config = AnalysisConfig(resamples=100, focal="hc", baselines=("random-search", "input-order"))
    target = write_bundle(campaign_dir, tmp_path / "one", config, figures=False)
    files = _files(target)
    manifest = json.loads(files["manifest.json"])
    for extension in ("md", "csv", "tex"):
        name = f"tables/evals/baselines.{extension}"
        assert manifest["files"][name] == sha256(files[name])
    assert (manifest["config"]["focal"], manifest["config"]["baselines"]) == (
        "hc",
        ["random-search", "input-order"],
    )
    rows = files["tables/evals/baselines.csv"].decode("utf-8").splitlines()
    assert [row.split(",")[:2] for row in rows[1:]] == [
        ["hc", "input-order"],
        ["hc", "random-search"],
    ]
    index = files["index.md"].decode("utf-8")
    assert "hc against baselines (makespan, protocol evals)" in index
    assert "[csv](tables/evals/baselines.csv)" in index
    again = _files(write_bundle(campaign_dir, tmp_path / "two", config, figures=False))
    assert again == files


def test_a_failed_analysis_leaves_the_earlier_bundle_in_place(
    campaign_dir: Path, tmp_path: Path
) -> None:
    target = write_bundle(campaign_dir, tmp_path / "bundle", AnalysisConfig(resamples=100))
    before = _files(target)
    with pytest.raises(ConfigError, match="unknown focal solver 'nobody'"):
        write_bundle(campaign_dir, target, AnalysisConfig(resamples=100, focal="nobody"))
    shaffer = AnalysisConfig(resamples=100, focal="hc", correction="shaffer")
    with pytest.raises(ConfigError, match="Shaffer's correction applies to all pairwise"):
        write_bundle(campaign_dir, target, shaffer, figures=False)
    assert _files(target) == before
