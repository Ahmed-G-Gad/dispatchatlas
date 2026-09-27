"""Tests of the instance catalogs and of writing and reading instance sets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import ClassVar

import pytest

from dispatchatlas.benchmarks import FamilyInfo, FamilyRegistry, default_registry
from dispatchatlas.benchmarks.catalog import (
    MANIFEST_FORMAT,
    Catalog,
    CatalogEntry,
    build_catalog,
    catalogs,
    load_directory,
    materialize,
)
from dispatchatlas.core import ConfigError, Problem, SerializationError, derive_seed
from dispatchatlas.core.model import Mode, Resource, Scalar, Task

REGISTRY = default_registry(plugins=False)


def _require(catalog: Catalog) -> None:
    missing = sorted(set(catalog.families) - set(REGISTRY.ids()))
    if missing:
        pytest.skip(f"families not registered yet: {missing}")


def _continuum_sample() -> Catalog:
    """The small entries of the continuum catalog, one instance each."""
    entries = tuple(
        CatalogEntry(entry.family, entry.parameters, instances=1, name=entry.label)
        for entry in catalogs()["continuum"].entries
        if entry.label.endswith("-small")
    )
    return Catalog("continuum", "The small continuum entries.", entries)


def _parameters(problem: Problem) -> dict[str, Scalar]:
    return {key: value for key, value in problem.metadata.items() if key != "seed"}


@dataclass(frozen=True)
class _Unseeded:
    """A family whose problems do not record their seed."""

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="unseeded", name="Unseeded", problem_class="test", summary="One task."
    )

    def generate(self, seed: int = 0) -> Problem:
        task = Task("t", (Mode(float(seed % 7 + 1), {"m": 1.0}),))
        return Problem(id=f"unseeded-{seed}", tasks=(task,), resources=(Resource("m"),))


# ----------------------------------------------------------------- catalogs


def test_built_in_catalogs_are_declared():
    assert set(catalogs()) >= {"smoke", "continuum", "workflow"}
    for name, catalog in catalogs().items():
        assert catalog.name == name
        assert catalog.description.endswith(".")
        assert catalog.size == sum(entry.instances for entry in catalog.entries)


def test_smoke_catalog_covers_every_built_in_family():
    smoke = catalogs()["smoke"]
    assert set(REGISTRY.ids()) <= set(smoke.families)
    assert all(entry.instances == 2 for entry in smoke.entries)
    _require(smoke)


def test_smoke_catalog_builds_tiny_distinct_instances():
    smoke = catalogs()["smoke"]
    _require(smoke)
    problems = build_catalog("smoke", seed=3)
    assert len(problems) == smoke.size
    assert len({problem.id for problem in problems}) == len(problems)
    assert max(problem.size for problem in problems) <= 30
    assert problems == build_catalog("smoke", seed=3)
    assert problems != build_catalog("smoke", seed=4)


def test_instances_regenerate_from_their_family_parameters_and_seed():
    catalog = _continuum_sample()
    for entry, problem in zip(catalog.entries, build_catalog(catalog, seed=7), strict=True):
        seed = derive_seed(7, f"continuum/{entry.label}", 0)
        assert problem.id == f"{entry.label}-0"
        assert problem.metadata["seed"] == seed
        original = REGISTRY.generate(entry.family, seed, **entry.parameters)
        assert problem.renamed(original.id) == original


def test_instances_are_validated_once(monkeypatch):
    catalog = _continuum_sample()
    validated: list[str] = []
    check = Problem.__post_init__

    def counting(problem: Problem) -> None:
        validated.append(problem.id)
        check(problem)

    monkeypatch.setattr(Problem, "__post_init__", counting)
    for entry in catalog.entries:
        REGISTRY.generate(
            entry.family, derive_seed(2, f"continuum/{entry.label}"), **entry.parameters
        )
    generating = len(validated)
    validated.clear()
    build_catalog(catalog, seed=2)
    assert len(validated) == generating


def test_a_seed_the_family_does_not_record_is_added():
    registry = FamilyRegistry()
    registry.register(_Unseeded)
    catalog = Catalog("plain", "A family without seed metadata.", (CatalogEntry("unseeded"),))
    (problem,) = build_catalog(catalog, seed=4, registry=registry)
    seed = derive_seed(4, "plain/unseeded", 0)
    assert problem.id == "unseeded-0"
    assert problem.metadata == {"seed": seed}
    assert problem.tasks == _Unseeded().generate(seed).tasks


def test_instances_do_not_depend_on_the_order_of_entries():
    catalog = _continuum_sample()
    reversed_catalog = Catalog(catalog.name, catalog.description, catalog.entries[::-1])
    forward = {problem.id: problem for problem in build_catalog(catalog, seed=1)}
    backward = {problem.id: problem for problem in build_catalog(reversed_catalog, seed=1)}
    assert forward == backward


def test_continuum_catalog_is_a_size_ladder():
    continuum = catalogs()["continuum"]
    assert set(continuum.families) == {
        "continuum-offloading",
        "continuum-workflow",
        "continuum-pipeline",
    }
    by_label = {entry.label: entry for entry in continuum.entries}
    for stem in (
        "continuum-offloading",
        "continuum-workflow-layered",
        "continuum-workflow-fork-join",
        "continuum-pipeline",
    ):
        sizes = [by_label[f"{stem}-{size}"].parameters for size in ("small", "medium", "large")]
        nodes = [
            sum(p[k] for k in ("devices", "edge_nodes", "fog_nodes", "cloud_nodes")) for p in sizes
        ]
        assert nodes == sorted(set(nodes))
    problems = build_catalog(_continuum_sample(), seed=0)
    assert [problem.family for problem in problems] == [
        entry.family for entry in _continuum_sample().entries
    ]
    offloading = tuple(
        CatalogEntry(entry.family, entry.parameters, name=entry.label)
        for entry in continuum.entries
        if entry.family == "continuum-offloading"
    )
    task_counts = [
        problem.size for problem in build_catalog(Catalog("ladder", "Sizes.", offloading))
    ]
    assert task_counts == [60, 150, 300]


@pytest.mark.slow
def test_full_continuum_catalog_builds():
    problems = build_catalog("continuum", seed=0)
    assert len(problems) == catalogs()["continuum"].size
    assert len({problem.id for problem in problems}) == len(problems)


def test_workflow_catalog_spans_the_published_design():
    workflow = catalogs()["workflow"]
    grid = {
        (entry.parameters["tasks"], entry.parameters["ccr"], entry.parameters["shape"])
        for entry in workflow.entries
    }
    assert grid == {
        (tasks, ccr, shape)
        for tasks in (20, 60, 100)
        for ccr in (0.1, 1.0, 10.0)
        for shape in (0.5, 1.0, 2.0)
    }
    problems = build_catalog(Catalog("workflow", "Two entries.", workflow.entries[:2]), seed=0)
    assert [problem.size for problem in problems] == [20, 20, 20, 20]


def test_unknown_catalogs_families_and_parameters_are_refused():
    with pytest.raises(ConfigError, match="unknown catalog"):
        build_catalog("no-such-catalog")
    with pytest.raises(ConfigError):
        build_catalog(Catalog("x", "Unknown family.", (CatalogEntry("no-such-family"),)))
    with pytest.raises(ConfigError):
        build_catalog(Catalog("x", "Bad parameter.", (CatalogEntry("workflow", {"moons": 3}),)))


@pytest.mark.parametrize("seed", [-1, True, 1.0, "1"])
def test_invalid_root_seeds_are_refused(seed):
    with pytest.raises(ConfigError, match="seed"):
        build_catalog("smoke", seed=seed)


@pytest.mark.parametrize(
    "arguments",
    [
        {"family": ""},
        {"family": "workflow", "instances": 0},
        {"family": "workflow", "instances": 1.5},
        {"family": "workflow", "name": "../escape"},
    ],
)
def test_catalog_entries_are_validated(arguments):
    with pytest.raises(ConfigError):
        CatalogEntry(**arguments)


def test_catalogs_are_validated():
    entry = CatalogEntry("workflow")
    with pytest.raises(ConfigError):
        Catalog("x", "Twice the same entry.", (entry, entry))
    with pytest.raises(ConfigError):
        Catalog("bad name", "Unsafe name.", (entry,))
    with pytest.raises(ConfigError):
        Catalog("x", "No entries.", ())


# ---------------------------------------------------- writing and reading


@pytest.fixture(scope="module")
def sample_problems() -> list[Problem]:
    return build_catalog(_continuum_sample(), seed=5) + build_catalog(
        Catalog("workflow", "One entry.", catalogs()["workflow"].entries[:1]), seed=5
    )


def test_materialize_and_load_round_trip(tmp_path, sample_problems):
    directory = materialize(sample_problems, tmp_path / "set")
    assert directory == tmp_path / "set"
    assert load_directory(directory) == sample_problems


def test_manifest_records_digests_families_parameters_and_seeds(tmp_path, sample_problems):
    directory = materialize(sample_problems, tmp_path / "set")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"format", "version", "problems"}
    assert manifest["format"] == MANIFEST_FORMAT
    assert manifest["version"] == 1
    assert len(manifest["problems"]) == len(sample_problems)
    for entry, problem in zip(manifest["problems"], sample_problems, strict=True):
        data = (directory / entry["file"]).read_bytes()
        assert entry["file"] == f"problems/{problem.id}.json"
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["family"] == problem.family
        assert entry["seed"] == problem.metadata["seed"]
        assert entry["parameters"] == _parameters(problem)


def test_materialized_files_are_byte_reproducible(tmp_path, sample_problems):
    first = materialize(sample_problems, tmp_path / "first")
    again = build_catalog(_continuum_sample(), seed=5) + build_catalog(
        Catalog("workflow", "One entry.", catalogs()["workflow"].entries[:1]), seed=5
    )
    second = materialize(again, tmp_path / "second")
    files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    assert files == sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    for relative in files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_loading_detects_modified_files(tmp_path, sample_problems):
    directory = materialize(sample_problems, tmp_path / "set")
    target = directory / "problems" / f"{sample_problems[0].id}.json"
    text = target.read_text(encoding="utf-8")
    target.write_text(text.replace('"release": ', '"release": 1', 1), encoding="utf-8")
    with pytest.raises(SerializationError, match="digest"):
        load_directory(directory)
    changed = load_directory(directory, verify=False)
    assert changed[0] != sample_problems[0]
    assert changed[1:] == sample_problems[1:]


def test_loading_refuses_broken_sets(tmp_path, sample_problems):
    with pytest.raises(SerializationError, match="manifest"):
        load_directory(tmp_path)
    directory = materialize(sample_problems[:1], tmp_path / "set")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["problems"][0]["file"] = "../outside.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SerializationError, match="unexpected file"):
        load_directory(directory)

    manifest["problems"][0]["file"] = f"problems/{sample_problems[0].id}.json"
    manifest["version"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SerializationError, match="version"):
        load_directory(directory)

    manifest["version"] = 1
    manifest["format"] = "something-else"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SerializationError, match="manifest"):
        load_directory(directory)

    manifest["format"] = MANIFEST_FORMAT
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "problems" / f"{sample_problems[0].id}.json").unlink()
    with pytest.raises(SerializationError, match="missing"):
        load_directory(directory)


def test_materialize_refuses_to_mix_with_existing_files(tmp_path, sample_problems):
    directory = materialize(sample_problems, tmp_path / "set")
    with pytest.raises(SerializationError, match="not empty"):
        materialize(sample_problems[:1], directory)
    (directory / "notes.txt").write_text("kept", encoding="utf-8")
    materialize(sample_problems[:1], directory, overwrite=True)
    assert sorted(path.name for path in (directory / "problems").iterdir()) == [
        f"{sample_problems[0].id}.json"
    ]
    assert (directory / "notes.txt").read_text(encoding="utf-8") == "kept"
    assert load_directory(directory) == sample_problems[:1]


def test_materialize_refuses_unsafe_or_repeated_ids(tmp_path, sample_problems):
    problem = sample_problems[0]
    with pytest.raises(SerializationError, match="safe"):
        materialize([replace(problem, id="../escape")], tmp_path / "a")
    with pytest.raises(SerializationError, match="distinct"):
        materialize([problem, problem], tmp_path / "b")
    assert not (tmp_path / "a").exists()
    assert not (tmp_path / "b").exists()
