"""Campaign files: the strict reader, round trips, and the config hash."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.campaigns import (
    Campaign,
    CatalogSource,
    DirectorySource,
    Execution,
    FamilySource,
    FileSource,
    Protocol,
    SolverSpec,
    SuiteSource,
    campaign_from_dict,
    campaign_to_dict,
    load_campaign,
    loads_campaign,
)
from dispatchatlas.core import ConfigError, StopCriteria

TOML = """
schema_version = 1
id = "demo"
seed = 3
replicates = 2
objectives = ["makespan"]

[[instances]]
family = "flow-shop"
parameters = { jobs = 3, machines = 2 }
seeds = [1, 2]

[[instances]]
suite = "taillard-flow-shop"
instances = ["ta001"]

[[instances]]
catalog = "smoke"
seed = 4

[[instances]]
directory = "materialized"

[[instances]]
file = "problems/one.json"

[[solvers]]
id = "random-search"

[[solvers]]
id = "hill-climbing"
label = "hc"
parameters = { restart_after = 10 }

[[protocols]]
name = "evals"
stop = { max_evaluations = 20 }

[[protocols]]
name = "clock"
stop = { time_limit = 1, max_evaluations = 1000 }

[execution]
backend = "process"
workers = "auto"
timeout_factor = 1.5
grace = 2.0
retries = 1
record_schedules = false

[metadata]
purpose = "test"
tags = ["a", "b"]
"""


def _write(tmp_path: Path, text: str, name: str = "campaign.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_every_section_of_a_toml_file(tmp_path: Path) -> None:
    campaign = load_campaign(_write(tmp_path, TOML))
    assert campaign.id == "demo"
    assert campaign.seed == 3
    assert campaign.replicates == 2
    assert campaign.objectives == ("makespan",)
    assert campaign.instances == (
        FamilySource("flow-shop", (1, 2), {"jobs": 3, "machines": 2}),
        SuiteSource("taillard-flow-shop", ("ta001",)),
        CatalogSource("smoke", 4),
        DirectorySource(Path("materialized")),
        FileSource(Path("problems/one.json")),
    )
    assert campaign.solvers == (
        SolverSpec("random-search"),
        SolverSpec("hill-climbing", "hc", {"restart_after": 10}),
    )
    assert campaign.protocols[1] == Protocol(
        "clock", StopCriteria(time_limit=1.0, max_evaluations=1000)
    )
    assert campaign.execution == Execution("process", "auto", 1.5, 2.0, 1, False)
    assert campaign.metadata["tags"] == ("a", "b")
    assert campaign.base == tmp_path.absolute()
    assert campaign.resolve(Path("materialized")) == tmp_path.absolute() / "materialized"


def test_json_and_toml_forms_are_the_same_campaign(tmp_path: Path) -> None:
    from_toml = load_campaign(_write(tmp_path, TOML))
    document = campaign_to_dict(from_toml)
    from_json = load_campaign(_write(tmp_path, json.dumps(document), "campaign.json"))
    assert from_json == from_toml
    assert campaign_from_dict(document, base=from_toml.base) == from_toml
    assert from_json.config_hash == from_toml.config_hash


def test_optional_keys_take_their_defaults() -> None:
    campaign = loads_campaign(
        """
        schema_version = 1
        id = "small"
        seed = 0
        instances = [{ family = "job-shop", seeds = [1] }]
        solvers = [{ id = "input-order" }]
        protocols = [{ name = "p", stop = { max_iterations = 5 } }]
        """
    )
    assert campaign.replicates == 30
    assert campaign.objectives is None
    assert campaign.execution == Execution()
    assert campaign.metadata == {}
    assert campaign.base is None


VALID = {
    "schema_version": 1,
    "id": "strict",
    "seed": 1,
    "instances": [{"family": "flow-shop", "seeds": [1]}],
    "solvers": [{"id": "random-search"}],
    "protocols": [{"name": "p", "stop": {"max_evaluations": 5}}],
}


_DELETE = object()


def _with(path: str, value: object) -> Any:
    """Return a copy of the valid document with the value at a dotted path replaced."""
    document = json.loads(json.dumps(VALID))
    keys: list[int | str] = [int(key) if key.isdigit() else key for key in path.split(".")]
    target = document
    for key in keys[:-1]:
        target = target[key]
    if value is _DELETE:
        del target[keys[-1]]
    else:
        target[keys[-1]] = value
    return document


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("schema_version", _DELETE, "declares no schema_version"),
        ("schema_version", 2, "schema_version 2 is not supported"),
        ("schema_version", True, "not supported"),
        ("solver", [], "unknown key(s) 'solver'"),
        ("instances.0.seed", 3, "instances[0] has unknown key(s) 'seed'"),
        ("protocols.0.stop.max_evaluation", 3, "unknown key(s) 'max_evaluation'"),
        ("seed", "1", "seed must be an integer, got the string '1'"),
        ("seed", True, "seed must be an integer, got true"),
        ("seed", 1.0, "seed must be an integer"),
        ("seed", -1, "seed must be an integer of at least 0"),
        ("replicates", 1.5, "replicates must be an integer"),
        ("replicates", 0, "replicates must be an integer of at least 1"),
        ("instances.0.seeds", [1, "2"], "instances[0].seeds[1] must be an integer"),
        ("instances.0.seeds", [], "at least one seed"),
        ("instances.0.suite", "x", "mixes the source kinds"),
        ("instances.0", {"parameters": {}}, "names no instance source"),
        ("solvers.0.label", "a/b", "must start with a letter or digit"),
        ("solvers", [{"id": "random-search"}, {"id": "random-search"}], "unique name"),
        ("protocols.0.stop", {"target": 3}, "must set time_limit, max_evaluations"),
        ("protocols.0.stop", {"max_evaluations": 10.0}, "must be an integer, got the number"),
        ("protocols.0.stop", {"time_limit": -1}, "time_limit must be a positive number"),
        ("protocols.0.name", "a b", "protocol name 'a b'"),
        ("execution", {"record_schedules": "false"}, "must be true or false"),
        ("execution", {"workers": True}, "workers must be an integer"),
        ("execution", {"workers": "many"}, "positive integer or 'auto'"),
        (
            "execution",
            {"timeout_factor": 0.5},
            "timeout_factor must be a finite number of at least 1",
        ),
        ("execution", {"backend": "sequential", "workers": 2}, "one run at a time"),
        ("execution", {"backend": "threads"}, "backend must be one of"),
        ("objectives", [], "objectives must not be empty"),
        ("id", "../escape", "campaign id '../escape'"),
    ],
)
def test_strict_reader_rejects_invalid_documents(path: str, value: object, message: str) -> None:
    with pytest.raises(ConfigError) as caught:
        campaign_from_dict(_with(path, value))
    assert message in str(caught.value)


def test_json_rejects_duplicate_keys_and_non_finite_numbers() -> None:
    text = json.dumps(VALID)
    with pytest.raises(ConfigError, match="appears twice"):
        loads_campaign(text.replace('"seed": 1', '"seed": 1, "seed": 2'), format="json")
    with pytest.raises(ConfigError, match="NaN is not allowed"):
        loads_campaign(text.replace('"seed": 1', '"seed": NaN'), format="json")


def test_file_errors_name_the_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"campaign\.toml.*needs the key 'seed'"):
        load_campaign(_write(tmp_path, TOML.replace("seed = 3\n", "")))
    with pytest.raises(ConfigError, match=r"must end in \.toml or \.json"):
        load_campaign(tmp_path / "campaign.yaml")
    with pytest.raises(ConfigError, match="cannot read"):
        load_campaign(tmp_path / "missing.toml")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_campaign(_write(tmp_path, "id = ", "broken.toml"))


def test_config_hash_covers_exactly_the_identity_fields(tmp_path: Path) -> None:
    campaign = load_campaign(_write(tmp_path, TOML))
    unchanged = [
        replace(campaign, id="renamed"),
        replace(campaign, metadata={"purpose": "other"}),
        replace(campaign, base=None),
        campaign.with_execution(workers=3, retries=0, record_schedules=True),
    ]
    for other in unchanged:
        assert other.config_hash == campaign.config_hash
    changed = [
        replace(campaign, seed=4),
        replace(campaign, replicates=3),
        replace(campaign, objectives=("total_completion_time",)),
        replace(campaign, instances=campaign.instances[:1]),
        replace(campaign, solvers=(SolverSpec("random-search", "rs"), campaign.solvers[1])),
        replace(campaign, solvers=campaign.solvers[::-1]),
        replace(
            campaign,
            protocols=(Protocol("evals", StopCriteria(max_evaluations=21)), campaign.protocols[1]),
        ),
    ]
    hashes = {other.config_hash for other in changed}
    assert campaign.config_hash not in hashes
    assert len(hashes) == len(changed)
    assert set(campaign.identity()) == {
        "seed",
        "instances",
        "solvers",
        "protocols",
        "replicates",
        "objectives",
    }


def test_file_locations_are_not_part_of_the_identity(tmp_path: Path) -> None:
    campaign = load_campaign(_write(tmp_path, TOML))
    moved = replace(
        campaign,
        instances=(
            campaign.instances[0],
            SuiteSource("taillard-flow-shop", ("ta001",), Path("/data/downloads")),
            campaign.instances[2],
            DirectorySource(Path("elsewhere/materialized")),
            FileSource(Path("/other/one.json")),
        ),
    )
    assert campaign_to_dict(moved) != campaign_to_dict(campaign)
    assert moved.config_hash == campaign.config_hash
    read_otherwise = replace(
        campaign, instances=(*campaign.instances[:4], FileSource(Path("one.sm"), "psplib"))
    )
    assert read_otherwise.config_hash != campaign.config_hash


def test_stop_criteria_are_normalised_before_hashing() -> None:
    first = Protocol("p", StopCriteria(time_limit=1))
    second = Protocol("p", StopCriteria(time_limit=1.0))
    assert first.to_dict() == second.to_dict() == {"name": "p", "stop": {"time_limit": 1.0}}


def test_names_must_be_unique_ignoring_case() -> None:
    with pytest.raises(ConfigError, match="unique name"):
        Campaign(
            "c",
            0,
            (FamilySource("flow-shop", (1,)),),
            (SolverSpec("random-search", "RS"), SolverSpec("hill-climbing", "rs")),
            (Protocol("p", StopCriteria(max_evaluations=5)),),
        )


def test_solver_arguments_are_plain_and_picklable() -> None:
    spec = SolverSpec("x", parameters={"weights": {"a": 1.0}, "order": [3, 1], "flag": True})
    arguments = spec.arguments()
    assert arguments == {"weights": {"a": 1.0}, "order": (3, 1), "flag": True}
    assert type(arguments["weights"]) is dict
    assert pickle.loads(pickle.dumps(arguments)) == arguments
    source = FamilySource("flow-shop", (1,), {"jobs": 3, "shape": {"a": [1, 2]}})
    assert source.arguments() == {"jobs": 3, "shape": {"a": (1, 2)}}


def test_worker_count_resolves_auto() -> None:
    assert Execution(workers=3).worker_count() == 3
    assert Execution(workers="auto").worker_count() >= 1
    assert Execution(retries=2).max_attempts == 3
