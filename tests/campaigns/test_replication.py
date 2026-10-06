"""Replicates under a time limit, per-solver counts, seed namespaces, and the config hash."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.campaigns import (
    Campaign,
    Plan,
    Protocol,
    SolverSpec,
    campaign_from_dict,
    campaign_to_dict,
    load_campaign,
    loads_campaign,
    plan,
)
from dispatchatlas.campaigns.fairness import assess, conditional_notes, notes
from dispatchatlas.campaigns.planning import PlannedRun, run_seed
from dispatchatlas.campaigns.records import SolverEntry
from dispatchatlas.core import ConfigError, StopCriteria
from tests.campaigns.solvers import registry, tiny_campaign

REGISTRY = registry()
EXAMPLES = Path(__file__).parents[2] / "examples" / "campaigns"
TIMED = Protocol("timed", StopCriteria(time_limit=0.05))
COUNTED = Protocol("evals", StopCriteria(max_evaluations=15))

# The seeds, run ids, and config hashes below were computed with the code of
# the main branch, before any of the optional keys existed.
TINY_HASH = "ed30da70c23546a280d608d885fa3f4d3026fe693c1e4916e7b42d598f1d6524"
EXAMPLE_HASHES = {
    "continuum-placement.toml": "02cd59aee41854069963bbd38618788aeb9dcfa4795d10efeedf19dd2fde2d22",
    "quickstart.toml": "06cb9a987c5c164ff47a30265c003abe3aa5aabb8facd9efbadb07c0e51fd4fc",
    "reference.toml": "3279ef6b1af75f9e5b7f07c092795e760c11c21298d2d623c0d7117673d043c9",
    "taillard-flow-shop.toml": "17f27dfb2b2e0bd41a53333a46137034d5cf107371180401fb7d530e0075f8c3",
}


def _per_cell(result: Plan) -> dict[tuple[str, str], int]:
    """Count the runs of each solver and protocol on the first problem."""
    first = result.problems[0].id
    return dict(Counter((run.solver, run.protocol) for run in result.runs if run.problem == first))


# ------------------------------------------------------------- replicate_timed


def test_a_deterministic_solver_runs_once_under_a_time_limit_and_the_plan_notes_it() -> None:
    campaign = tiny_campaign(protocols=(TIMED, COUNTED), replicates=3)
    result = plan(campaign, registry=REGISTRY)
    assert _per_cell(result) == {
        ("input-order", "timed"): 1,
        ("input-order", "evals"): 1,
        ("random-search", "timed"): 3,
        ("random-search", "evals"): 3,
        ("hc", "timed"): 3,
        ("hc", "evals"): 3,
    }
    assert not campaign.replicate_timed
    assert not result.replicate_timed
    verdict = assess(result)
    assert verdict.fair
    # A constructive rule ends when its algorithm does: the note is conditional.
    assert verdict.notes == ()
    (note,) = verdict.conditional_notes
    assert "'timed'" in note
    assert "declared deterministic that are not iterative (input-order)" in note
    assert "only if the time limit ends it" in note
    assert "replicate_timed" in note
    assert "random-search" not in note
    document = result.to_dict()["fairness"]
    assert document["notes"] == []
    assert document["conditional_notes"] == [note]


def test_an_iterative_deterministic_solver_under_a_time_limit_is_noted_whatever_its_runs_do(
    caplog: pytest.LogCaptureFixture,
) -> None:
    solvers = (SolverSpec("input-order"), SolverSpec("test-sleeper"), SolverSpec("random-search"))
    campaign = tiny_campaign(solvers=solvers, protocols=(TIMED, COUNTED), replicates=3)
    with caplog.at_level(logging.INFO):
        result = plan(campaign, registry=REGISTRY)
    verdict = assess(result)
    (note,) = verdict.notes
    assert "the iterative solvers declared deterministic (test-sleeper)" in note
    assert "input-order" not in note
    (conditional,) = verdict.conditional_notes
    assert "(input-order)" in conditional
    assert "test-sleeper" not in conditional
    assert notes(result, result.protocol("timed")) == (note,)
    assert conditional_notes(result, result.protocol("timed")) == (conditional,)
    assert conditional_notes(result, result.protocol("timed"), ["test-sleeper"]) == ()
    assert conditional_notes(result, result.protocol("evals")) == ()
    # The planner warns about the first and only informs about the second.
    levels = {record.getMessage(): record.levelno for record in caplog.records}
    assert levels[f"campaign {campaign.id}: {note}"] == logging.WARNING
    assert levels[f"campaign {campaign.id}: {conditional}"] == logging.INFO


@pytest.mark.parametrize("name", sorted(EXAMPLE_HASHES))
def test_the_example_campaigns_plan_without_a_warning_about_replication(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        result = plan(load_campaign(EXAMPLES / name))
    warned = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and "declared deterministic" in record.getMessage()
    ]
    assert warned == []
    assert not any("declared deterministic" in note for note in assess(result).notes)


def test_replicate_timed_gives_a_deterministic_solver_every_replicate_under_a_time_limit() -> None:
    base = tiny_campaign(protocols=(TIMED, COUNTED), replicates=3)
    campaign = replace(base, replicate_timed=True)
    result = plan(campaign, registry=REGISTRY)
    assert _per_cell(result) == {
        ("input-order", "timed"): 3,
        ("input-order", "evals"): 1,
        ("random-search", "timed"): 3,
        ("random-search", "evals"): 3,
        ("hc", "timed"): 3,
        ("hc", "evals"): 3,
    }
    assert result.replicate_timed
    entry = result.solver("input-order")
    assert result.replicates_of(entry, result.protocol("timed")) == 3
    assert result.replicates_of(entry, result.protocol("evals")) == 1
    assert result.replicates_of(result.solver("hc"), result.protocol("evals")) == 3
    verdict = assess(result)
    assert verdict.fair
    assert verdict.paired
    assert verdict.notes == verdict.conditional_notes == ()
    for run in result.runs:
        assert run.seed == run_seed(campaign.seed, run.problem, run.replicate)
    assert campaign.config_hash != base.config_hash
    assert campaign.identity()["replicate_timed"] is True


def test_no_note_when_every_solver_runs_once_or_no_protocol_sets_a_time_limit() -> None:
    single = plan(tiny_campaign(protocols=(TIMED,), replicates=1), registry=REGISTRY)
    assert assess(single).notes == assess(single).conditional_notes == ()
    counted = plan(tiny_campaign(protocols=(COUNTED,), replicates=3), registry=REGISTRY)
    assert assess(counted).notes == assess(counted).conditional_notes == ()


def test_a_stored_plan_keeps_its_replication_rule() -> None:
    campaign = replace(
        tiny_campaign(protocols=(TIMED, COUNTED), replicates=3), replicate_timed=True
    )
    result = plan(campaign, registry=REGISTRY)
    again = Plan.from_dict(result.to_dict())
    assert again == result
    assert assess(again).fair


# ------------------------------------------------------- per-solver replicates


def test_a_solver_entry_sets_its_own_replicate_count() -> None:
    solvers = (
        SolverSpec("input-order", replicates=4),
        SolverSpec("random-search"),
        SolverSpec("hill-climbing", "hc", {"restart_after": 5}, replicates=1),
    )
    campaign = tiny_campaign(solvers=solvers, protocols=(TIMED, COUNTED), replicates=2)
    result = plan(campaign, registry=REGISTRY)
    assert _per_cell(result) == {
        ("input-order", "timed"): 4,
        ("input-order", "evals"): 4,
        ("random-search", "timed"): 2,
        ("random-search", "evals"): 2,
        ("hc", "timed"): 1,
        ("hc", "evals"): 1,
    }
    assert result.solver("input-order").replicates == 4
    assert result.solver("random-search").replicates is None
    assert {run.replicate for run in result.runs if run.solver == "input-order"} == {0, 1, 2, 3}
    for run in result.runs:
        assert run.seed == run_seed(campaign.seed, run.problem, run.replicate)
    verdict = assess(result)
    assert verdict.fair
    # A count the entry states is a choice, so the plan has nothing to note.
    assert verdict.notes == verdict.conditional_notes == ()
    assert Plan.from_dict(result.to_dict()) == result


def test_a_solver_count_enters_the_config_hash_only_when_set() -> None:
    base = tiny_campaign()
    assert all("replicates" not in solver.to_dict() for solver in base.solvers)
    counted = replace(base, solvers=(SolverSpec("input-order", replicates=2), *base.solvers[1:]))
    assert counted.solvers[0].to_dict()["replicates"] == 2
    assert counted.config_hash != base.config_hash


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_a_solver_replicate_count_must_be_a_positive_integer(value: Any) -> None:
    with pytest.raises(ConfigError, match="replicates"):
        SolverSpec("input-order", replicates=value)


# -------------------------------------------------------------- seed namespace


def test_seeds_without_a_namespace_are_those_of_the_main_branch() -> None:
    assert run_seed(11, "flow-shop-3x2-1", 0) == 222429825019766731
    assert run_seed(11, "flow-shop-3x2-2", 1) == 6826608615701984612
    assert run_seed(2024, "ta001", 3) == 518631194580767496
    campaign = tiny_campaign()
    assert campaign.seed_namespace is None
    assert campaign.config_hash == TINY_HASH
    result = plan(campaign, registry=REGISTRY)
    assert result.runs[0] == PlannedRun(
        "run-b67a5b83dfd0695f", "flow-shop-3x2-1", "input-order", "evals", 0, 222429825019766731
    )
    assert result.runs[-1] == PlannedRun(
        "run-8629df2c06387aa9", "flow-shop-3x2-2", "hc", "evals", 1, 6826608615701984612
    )


def test_two_campaigns_that_differ_only_in_namespace_share_no_seed() -> None:
    base = tiny_campaign(replicates=3)
    plans = {
        name: plan(replace(base, seed_namespace=name), registry=REGISTRY)
        for name in (None, "study-a", "study-b")
    }
    seeds = {name: {run.seed for run in result.runs} for name, result in plans.items()}
    assert all(len(values) == 2 * 3 for values in seeds.values())
    assert not seeds[None] & seeds["study-a"]
    assert not seeds[None] & seeds["study-b"]
    assert not seeds["study-a"] & seeds["study-b"]
    hashes = {result.config_hash for result in plans.values()}
    assert len(hashes) == 3
    assert not {run.run_id for run in plans["study-a"].runs} & {
        run.run_id for run in plans["study-b"].runs
    }
    for name, result in plans.items():
        assert result.seed_namespace == name
        assert assess(result).paired
        for run in result.runs:
            assert run.seed == run_seed(base.seed, run.problem, run.replicate, name)
        assert Plan.from_dict(result.to_dict()) == result


def test_the_same_namespace_gives_the_same_seeds() -> None:
    campaign = replace(tiny_campaign(), seed_namespace="study-a")
    first = plan(campaign, registry=REGISTRY)
    again = plan(replace(campaign, id="renamed"), registry=REGISTRY)
    assert [run.seed for run in first.runs] == [run.seed for run in again.runs]
    assert [run.run_id for run in first.runs] == [run.run_id for run in again.runs]


@pytest.mark.parametrize("value", ["", "   ", 3, True])
def test_a_seed_namespace_must_be_a_non_empty_string(value: Any) -> None:
    with pytest.raises(ConfigError, match="seed_namespace"):
        replace(tiny_campaign(), seed_namespace=value)


# ----------------------------------------------------------------- config hash


@pytest.mark.parametrize("name", sorted(EXAMPLE_HASHES))
def test_the_example_campaigns_keep_their_config_hash(name: str) -> None:
    assert load_campaign(EXAMPLES / name).config_hash == EXAMPLE_HASHES[name]


def test_every_example_campaign_is_pinned() -> None:
    assert sorted(path.name for path in EXAMPLES.glob("*.toml")) == sorted(EXAMPLE_HASHES)


def test_the_identity_docstrings_state_the_rule_and_compare_no_versions() -> None:
    written, identity = StopCriteria.to_dict.__doc__, Campaign.identity.__doc__
    assert written is not None
    assert identity is not None
    assert "written only when it is not the default" in written
    assert "left at its default is absent from the identity" in identity
    for text in (written, identity):
        assert "without the field" not in text
        assert "without them" not in text


def test_the_identity_holds_the_optional_keys_only_when_they_are_set() -> None:
    base = tiny_campaign()
    assert sorted(base.identity()) == [
        "instances",
        "objectives",
        "protocols",
        "replicates",
        "seed",
        "solvers",
    ]
    assert base.identity()["protocols"] == [{"name": "evals", "stop": {"max_evaluations": 15}}]
    changed = [
        replace(base, replicate_timed=True),
        replace(base, seed_namespace="study-a"),
        replace(base, solvers=(SolverSpec("input-order", replicates=2), *base.solvers[1:])),
        replace(
            base,
            protocols=(Protocol("evals", StopCriteria(max_evaluations=15, schedule="binding")),),
        ),
    ]
    hashes = {campaign.config_hash for campaign in changed}
    assert len(hashes) == len(changed)
    assert base.config_hash not in hashes
    explicit = replace(
        base,
        replicate_timed=False,
        seed_namespace=None,
        protocols=(Protocol("evals", StopCriteria(max_evaluations=15, schedule="deterministic")),),
    )
    assert explicit.config_hash == base.config_hash == TINY_HASH


# ------------------------------------------------------------------ the reader

DOCUMENT = """
schema_version = 1
id = "keys"
seed = 5
replicates = 4
replicate_timed = true
seed_namespace = "study-a"

[[instances]]
family = "flow-shop"
parameters = { jobs = 3, machines = 2 }
seeds = [1]

[[solvers]]
id = "input-order"
replicates = 2

[[solvers]]
id = "random-search"

[[protocols]]
name = "timed"
stop = { time_limit = 0.5, max_evaluations = 50, schedule = "binding" }
"""


def test_a_campaign_file_sets_the_optional_keys() -> None:
    campaign = loads_campaign(DOCUMENT)
    assert campaign.replicate_timed is True
    assert campaign.seed_namespace == "study-a"
    assert [solver.replicates for solver in campaign.solvers] == [2, None]
    assert campaign.protocols[0].stop.schedule == "binding"
    again = campaign_from_dict(campaign_to_dict(campaign))
    assert again.config_hash == campaign.config_hash
    assert again.replicate_timed is True
    assert again.seed_namespace == "study-a"
    assert again.solvers == campaign.solvers
    assert again.protocols == campaign.protocols


def test_a_campaign_file_without_the_optional_keys_reads_as_before() -> None:
    document = campaign_to_dict(tiny_campaign())
    for key in ("replicate_timed", "seed_namespace"):
        document.pop(key, None)
    for solver in document["solvers"]:
        solver.pop("replicates", None)
    campaign = campaign_from_dict(document)
    assert campaign.config_hash == TINY_HASH
    assert not campaign.replicate_timed
    assert campaign.seed_namespace is None


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("replicate_timed = true", "replicate_timed = 1", "replicate_timed must be true or false"),
        ('seed_namespace = "study-a"', "seed_namespace = 3", "seed_namespace must be a string"),
        ("replicates = 2", 'replicates = "2"', r"solvers\[0\].replicates must be an integer"),
        ("replicates = 2", "replicates = 0", r"solvers\[0\]: .*replicates"),
        ('schedule = "binding"', 'schedule = "clock"', r"protocols\[0\]: .*schedule"),
        ('schedule = "binding"', "schedule = 2", r"protocols\[0\].stop.schedule must be a string"),
    ],
)
def test_the_reader_rejects_wrong_values_of_the_optional_keys(
    old: str, new: str, message: str
) -> None:
    assert old in DOCUMENT
    with pytest.raises(ConfigError, match=message):
        loads_campaign(DOCUMENT.replace(old, new))


# ------------------------------------------------------------ stored documents


def test_a_plan_stored_without_the_new_fields_still_loads() -> None:
    result = plan(tiny_campaign(), registry=REGISTRY)
    document = result.to_dict()
    del document["replicate_timed"], document["seed_namespace"]
    for solver in document["solvers"]:
        del solver["replicates"]
    assert Plan.from_dict(document) == result


def test_a_solver_entry_stored_without_a_count_still_loads() -> None:
    entry = plan(tiny_campaign(), registry=REGISTRY).solvers[0]
    document = entry.to_dict()
    del document["replicates"]
    assert SolverEntry.from_dict(document) == entry
    assert SolverEntry.from_dict(document).replicates is None


def test_a_campaign_object_takes_the_keys_directly() -> None:
    base = tiny_campaign()
    campaign = Campaign(
        id="direct",
        seed=1,
        instances=base.instances,
        solvers=base.solvers,
        protocols=base.protocols,
        replicate_timed=True,
        seed_namespace="x",
    )
    assert campaign.replicate_timed
    wrong: Any = "yes"
    with pytest.raises(ConfigError, match="replicate_timed"):
        replace(base, replicate_timed=wrong)
