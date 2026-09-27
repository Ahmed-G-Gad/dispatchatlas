"""The benchmark-family registry: metadata, registration, creation, and plugins."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from importlib.metadata import EntryPoint
from typing import Any, ClassVar

import pytest

from dispatchatlas.benchmarks import registry as family_module
from dispatchatlas.benchmarks.registry import (
    PLUGIN_GROUP,
    FamilyInfo,
    FamilyRegistry,
    Generator,
    default_registry,
    generate,
    load_plugins,
)
from dispatchatlas.core._frozen import FrozenMap
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.model import Problem, Resource, Task


@dataclass(frozen=True)
class Chain:
    """Unit tasks on one machine; the seed sets their duration."""

    size: int = 3
    label: str = "chain"
    tags: list[str] = field(default_factory=list)
    derived: int = field(default=0, init=False)

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="toy-chain",
        name="Toy chain",
        problem_class="single-machine",
        summary="Unit tasks on one machine.",
        references=("kolisch1996serial",),
    )

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("size must be positive")

    def generate(self, seed: int = 0) -> Problem:
        return Problem(
            f"{self.label}-{self.size}-{seed}",
            tasks=tuple(Task.on(f"t{i}", "m", 1 + seed) for i in range(self.size)),
            resources=(Resource("m"),),
        )


PLUGIN_INFO = FamilyInfo("plugin-chain", "Plugin chain", "single-machine", "From a plugin.")


def good_plugin(registry: FamilyRegistry) -> None:
    registry.register(Chain, PLUGIN_INFO)


def half_plugin(registry: FamilyRegistry) -> None:
    registry.register(Chain, replace(PLUGIN_INFO, id="half-chain"))
    raise RuntimeError("second family is broken")


def clashing_plugin(registry: FamilyRegistry) -> None:
    registry.register(Chain, replace(PLUGIN_INFO, id="another-chain"))
    registry.register(Chain)


def _entry(name: str, target: str, group: str = PLUGIN_GROUP) -> EntryPoint:
    return EntryPoint(name=name, value=f"{__name__}:{target}", group=group)


def _install(monkeypatch: pytest.MonkeyPatch, *entries: EntryPoint) -> None:
    def fake(*, group: str) -> list[EntryPoint]:
        return [entry for entry in entries if entry.group == group]

    monkeypatch.setattr(family_module, "entry_points", fake)


def _registry() -> FamilyRegistry:
    registry = FamilyRegistry()
    registry.register(Chain)
    return registry


# ------------------------------------------------------------------ metadata


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"id": "Toy"}, "family id 'Toy' must be lowercase kebab case"),
        ({"id": "toy_chain"}, "must be lowercase kebab case"),
        ({"id": "toy.chain"}, "must be lowercase kebab case"),
        ({"id": "toy--chain"}, "must be lowercase kebab case"),
        ({"name": ""}, "family 'toy-chain' needs a name and a summary"),
        ({"summary": ""}, "family 'toy-chain' needs a name and a summary"),
    ],
)
def test_family_info_rejects_invalid_metadata(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        replace(Chain.info, **changes)


def test_family_info_normalises_its_collections() -> None:
    parameters = {"size": 3}
    info = FamilyInfo(
        "x-y",
        "X",
        "shop",
        "S.",
        ["a2000b"],  # type: ignore[arg-type]
        ["energy"],  # type: ignore[arg-type]
        parameters,
    )
    parameters["size"] = 4
    assert info.references == ("a2000b",)
    assert info.objectives == ("energy",)
    assert info.parameters == {"size": 3}
    assert isinstance(info.parameters, FrozenMap)
    assert Chain.info.objectives == ("makespan",)


# -------------------------------------------------------------- registration


def test_registration_reads_parameters_from_dataclass_fields() -> None:
    registry = _registry()
    info = registry.info("toy-chain")
    assert dict(info.parameters) == {"size": 3, "label": "chain", "tags": []}
    assert Chain.info.parameters == {}
    explicit = replace(Chain.info, id="fixed-chain", parameters={"size": 9})
    registry.register(Chain, explicit)
    assert dict(registry.info("fixed-chain").parameters) == {"size": 9}


def test_registration_needs_metadata_and_a_free_id() -> None:
    registry = _registry()
    with pytest.raises(ConfigError, match="benchmark family 'toy-chain' is already registered"):
        registry.register(Chain)
    renamed = replace(Chain.info, name="Renamed chain")
    registry.register(Chain, renamed, replace_existing=True)
    assert registry.info("toy-chain").name == "Renamed chain"

    def bare(seed: int = 0) -> Problem:
        return Chain().generate(seed)

    with pytest.raises(ConfigError, match="no FamilyInfo given"):
        registry.register(bare)
    with pytest.raises(ConfigError, match="no FamilyInfo given"):
        registry.register(Chain, info="toy-chain")  # type: ignore[arg-type]


def test_lookups_and_copies() -> None:
    registry = _registry()
    registry.register(Chain, PLUGIN_INFO)
    assert registry.ids() == ("plugin-chain", "toy-chain")
    assert [info.id for info in registry.infos()] == ["plugin-chain", "toy-chain"]
    assert list(registry) == ["plugin-chain", "toy-chain"]
    assert len(registry) == 2
    assert "toy-chain" in registry
    assert "missing" not in registry
    clone = registry.copy()
    clone.register(Chain, replace(PLUGIN_INFO, id="clone-only"))
    assert "clone-only" in clone
    assert "clone-only" not in registry
    with pytest.raises(ConfigError, match="unknown benchmark family 'missing'"):
        registry.info("missing")


# ------------------------------------------------------------------ creation


def test_create_and_generate() -> None:
    registry = _registry()
    generator = registry.create("toy-chain", size=2, label="pair")
    assert isinstance(generator, Generator)
    assert generator == Chain(size=2, label="pair")
    problem = registry.generate("toy-chain", 4, size=2, label="pair")
    assert problem.id == "pair-2-4"
    assert problem == generator.generate(4)
    assert problem.tasks[0].modes[0].duration == 5.0
    assert registry.generate("toy-chain").id == "chain-3-0"


@pytest.mark.parametrize(
    ("family", "parameters", "message"),
    [
        ("missing", {}, "unknown benchmark family 'missing'"),
        ("toy-chain", {"colour": "red"}, "has no parameters ['colour']; known: ['label', 'size'"),
        ("toy-chain", {"derived": 1}, "has no parameters ['derived']"),
        ("toy-chain", {"size": 0}, "invalid parameters for family 'toy-chain': size must be"),
    ],
)
def test_create_reports_configuration_errors(
    family: str, parameters: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        _registry().create(family, **parameters)
    assert message in str(caught.value)


def test_non_dataclass_factories_take_no_parameters() -> None:
    registry = FamilyRegistry()

    def factory(**parameters: Any) -> Chain:
        return Chain(**parameters)

    registry.register(factory, PLUGIN_INFO)
    assert registry.info("plugin-chain").parameters == {}
    assert registry.generate("plugin-chain").id == "chain-3-0"
    with pytest.raises(ConfigError, match="has no parameters"):
        registry.create("plugin-chain", size=2)


# ------------------------------------------------------------------- plugins


def test_plugins_register_through_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        _entry("zeta", "good_plugin"),
        _entry("other-group", "half_plugin", group="elsewhere"),
    )
    registry = _registry()
    assert load_plugins(registry) == ["zeta"]
    assert registry.ids() == ("plugin-chain", "toy-chain")
    assert load_plugins(FamilyRegistry(), group="elsewhere") == []


def test_a_failing_plugin_is_skipped_whole(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install(
        monkeypatch,
        _entry("b-half", "half_plugin"),
        _entry("a-missing", "no_such_function"),
        _entry("d-good", "good_plugin"),
        _entry("c-clash", "clashing_plugin"),
    )
    registry = _registry()
    with caplog.at_level(logging.WARNING, logger="dispatchatlas"):
        assert load_plugins(registry) == ["d-good"]
    assert registry.ids() == ("plugin-chain", "toy-chain")
    assert registry.info("toy-chain") == _registry().info("toy-chain")
    warned = [record.getMessage() for record in caplog.records]
    assert len(warned) == 3
    assert "benchmark plugin 'a-missing' failed to load" in warned[0]
    assert "second family is broken" in warned[1]
    assert "'toy-chain' is already registered" in warned[2]


def test_the_default_registry_is_fresh_and_optionally_extended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _entry("zeta", "good_plugin"))
    plain = default_registry(plugins=False)
    assert {"flow-shop", "job-shop", "rcpsp"} <= set(plain.ids())
    assert "plugin-chain" not in plain
    extended = default_registry()
    assert "plugin-chain" in extended
    extended.register(Chain)
    assert "toy-chain" not in default_registry(plugins=False)
    first = generate("flow-shop", seed=2, jobs=3, machines=2)
    assert first == plain.generate("flow-shop", 2, jobs=3, machines=2)
    assert first.id != generate("flow-shop", seed=3, jobs=3, machines=2).id
