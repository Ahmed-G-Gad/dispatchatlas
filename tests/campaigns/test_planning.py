"""Planning: problems, replicates, paired seeds, run ids, fairness, and compatibility."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dispatchatlas.benchmarks import build_catalog, materialize
from dispatchatlas.campaigns import (
    CatalogSource,
    DirectorySource,
    FamilySource,
    FileSource,
    Plan,
    Protocol,
    SolverSpec,
    SuiteSource,
    check_compatibility,
    compatibility,
    plan,
)
from dispatchatlas.campaigns.compatibility import MANY_OBJECTIVES
from dispatchatlas.campaigns.fairness import PARITIES, assess, parities
from dispatchatlas.campaigns.planning import PlannedRun, run_id, run_seed
from dispatchatlas.core import (
    ConfigError,
    Dependency,
    Problem,
    Resource,
    SolverInfo,
    StopCriteria,
    Task,
    problem_digest,
    save_problem,
)
from dispatchatlas.references import reference
from tests.campaigns.solvers import registry, tiny_campaign
from tests.helpers import chain_problem, random_problem, small_job_shop

REGISTRY = registry()
TWO_PROTOCOLS = (
    Protocol("evals", StopCriteria(max_evaluations=15)),
    Protocol("iterations", StopCriteria(max_iterations=4)),
)


def test_plans_replicates_for_stochastic_solvers_and_one_run_for_deterministic_ones() -> None:
    result = plan(tiny_campaign(protocols=TWO_PROTOCOLS, replicates=3), registry=REGISTRY)
    counts: dict[str, int] = {}
    for item in result.runs:
        counts[item.solver] = counts.get(item.solver, 0) + 1
    # Two problems and two protocols: 1 replicate for input order, 3 for the others.
    assert counts == {"input-order": 4, "random-search": 12, "hc": 12}
    assert {item.replicate for item in result.runs if item.solver == "input-order"} == {0}
    assert [problem.id for problem in result.problems] == ["flow-shop-3x2-1", "flow-shop-3x2-2"]
    assert result.replicates_of(result.solver("hc")) == 3


def test_seeds_are_common_random_numbers() -> None:
    campaign = tiny_campaign(protocols=TWO_PROTOCOLS, replicates=3)
    result = plan(campaign, registry=REGISTRY)
    seeds: dict[tuple[str, int], set[int]] = {}
    for item in result.runs:
        seeds.setdefault((item.problem, item.replicate), set()).add(item.seed)
        assert item.seed == run_seed(campaign.seed, item.problem, item.replicate)
    assert all(len(values) == 1 for values in seeds.values())
    assert len({next(iter(values)) for values in seeds.values()}) == len(seeds)
    assert assess(result).paired


def test_adding_a_solver_changes_no_seed() -> None:
    base = tiny_campaign()
    extended = replace(base, solvers=(*base.solvers, SolverSpec("random-search", "rs2")))
    first = {(i.problem, i.solver, i.replicate): i.seed for i in plan(base, registry=REGISTRY).runs}
    second = {
        (i.problem, i.solver, i.replicate): i.seed for i in plan(extended, registry=REGISTRY).runs
    }
    assert all(second[key] == seed for key, seed in first.items())


def test_run_ids_are_stable_and_follow_the_identity() -> None:
    campaign = tiny_campaign()
    first = plan(campaign, registry=REGISTRY)
    again = plan(campaign.with_execution(retries=2), registry=REGISTRY)
    renamed = plan(replace(campaign, id="other-name", metadata={"note": "x"}), registry=REGISTRY)
    reseeded = plan(replace(campaign, seed=campaign.seed + 1), registry=REGISTRY)
    ids = [item.run_id for item in first.runs]
    assert ids == [item.run_id for item in again.runs] == [item.run_id for item in renamed.runs]
    assert not set(ids) & {item.run_id for item in reseeded.runs}
    assert len(set(ids)) == len(ids)
    assert all(identifier.startswith("run-") and len(identifier) == 20 for identifier in ids)
    item = first.runs[-1]
    problem = first.problem(item.problem)
    spec = campaign.solver(item.solver)
    assert item.run_id == run_id(
        first.config_hash, problem.id, problem.digest, spec, item.protocol, item.replicate
    )


def test_run_ids_follow_the_problem_content(tmp_path: Path) -> None:
    save_problem(chain_problem(), tmp_path / "chain.json")
    campaign = replace(
        tiny_campaign(
            instances=(
                FileSource(Path("chain.json")),
                FamilySource("flow-shop", (1,), {"jobs": 3, "machines": 2}),
            )
        ),
        base=tmp_path,
    )
    before = {item.run_id: item.problem for item in plan(campaign, registry=REGISTRY).runs}
    changed = replace(chain_problem(), objectives=("total_completion_time",))
    save_problem(changed, tmp_path / "chain.json")
    after = {item.run_id: item.problem for item in plan(campaign, registry=REGISTRY).runs}
    kept = set(before) & set(after)
    assert {before[identifier] for identifier in kept} == {"flow-shop-3x2-1"}
    assert {after[identifier] for identifier in set(after) - kept} == {"chain"}


def test_every_source_kind_loads(tmp_path: Path) -> None:
    materialize(build_catalog("smoke")[:2], tmp_path / "set")
    save_problem(chain_problem(), tmp_path / "chain.json")
    campaign = replace(
        tiny_campaign(
            solvers=(SolverSpec("input-order"),),
            instances=(
                FamilySource("job-shop", (5,), {"jobs": 2, "machines": 2}),
                SuiteSource("taillard-flow-shop", ("ta001",)),
                CatalogSource("smoke", 0),
                DirectorySource(Path("set")),
                FileSource(Path("chain.json")),
            ),
            replicates=1,
        ),
        base=tmp_path,
    )
    with pytest.raises(ConfigError, match=r"problem id 'flow-shop-0' occurs more than once"):
        plan(campaign, registry=REGISTRY)
    campaign = replace(
        campaign, instances=tuple(s for s in campaign.instances if s != CatalogSource("smoke", 0))
    )
    result = plan(campaign, registry=REGISTRY)
    sources = [problem.source for problem in result.problems]
    assert sources == [0, 1, 2, 2, 3]
    assert [problem.id for problem in result.problems][-3:] == [
        "flow-shop-0",
        "flow-shop-1",
        "chain",
    ]


def test_objective_override_applies_to_every_problem() -> None:
    campaign = replace(tiny_campaign(), objectives=("total_completion_time", "makespan"))
    result = plan(campaign, registry=REGISTRY)
    assert {problem.objectives for problem in result.problems} == {
        ("total_completion_time", "makespan")
    }
    original = plan(tiny_campaign(), registry=REGISTRY)
    assert result.problems[0].digest != original.problems[0].digest


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"solvers": (SolverSpec("no-such-solver"),)}, "unknown solver 'no-such-solver'"),
        (
            {"solvers": (SolverSpec("hill-climbing", parameters={"cooling": 2}),)},
            "solver 'hill-climbing'.*has no parameters",
        ),
        (
            {"instances": (FamilySource("no-such-family", (1,)),)},
            r"instances\[0\].*unknown benchmark",
        ),
        (
            {"instances": (FileSource(Path("missing.json")),)},
            r"instances\[0\] \(file 'missing.json'\)",
        ),
        ({"objectives": ("no-such-objective",)}, "unknown objective"),
    ],
)
def test_planning_errors_name_their_cause(change: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        plan(replace(tiny_campaign(), **change), registry=REGISTRY)  # type: ignore[arg-type]


def test_plan_round_trips_through_its_document() -> None:
    result = plan(tiny_campaign(protocols=TWO_PROTOCOLS), registry=REGISTRY)
    document = result.to_dict()
    assert Plan.from_dict(document) == result
    assert document["fairness"]["fair"] is True
    assert document["protocols"][1]["parities"] == ["equal-iterations"]


# ------------------------------------------------------------------- fairness


def test_parities_are_read_off_the_stop_criteria() -> None:
    assert parities(StopCriteria(max_evaluations=5)) == ("equal-evaluations",)
    assert (
        parities(StopCriteria(time_limit=1.0, max_evaluations=5, max_iterations=3, target=1.0))
        == PARITIES
    )
    assert parities(StopCriteria(time_limit=2.0, patience=3)) == ("equal-wall-clock",)


def test_fairness_verifies_the_plan() -> None:
    result = plan(tiny_campaign(protocols=TWO_PROTOCOLS), registry=REGISTRY)
    verdict = assess(result)
    assert verdict.fair
    assert verdict.parities == ("equal-evaluations", "equal-iterations")
    assert verdict.protocols == {
        "evals": ("equal-evaluations",),
        "iterations": ("equal-iterations",),
    }

    missing = replace(result, runs=result.runs[1:])
    assert any("1 missing runs" in issue for issue in assess(missing).issues)
    doubled = replace(result, runs=(*result.runs, result.runs[0]))
    assert any("planned twice" in issue for issue in assess(doubled).issues)
    first = result.runs[0]
    reseeded = replace(result, runs=(replace(first, seed=first.seed + 1), *result.runs[1:]))
    verdict = assess(reseeded)
    assert not verdict.paired
    assert any("several seeds" in issue for issue in verdict.issues)
    stranger = PlannedRun(
        "run-0000000000000000", first.problem, first.solver, "other", 0, first.seed
    )
    assert any(
        "outside the grid" in issue
        for issue in assess(replace(result, runs=(*result.runs, stranger))).issues
    )


# -------------------------------------------------------------- compatibility


def test_solvers_are_excluded_from_problems_they_do_not_model(tmp_path: Path) -> None:
    deadlines = random_problem(10, tasks=6, max_modes=1, pools=0, deadlines=True)
    assert "deadline" in deadlines.features
    save_problem(deadlines, tmp_path / "deadlines.json")
    campaign = replace(
        tiny_campaign(
            solvers=(SolverSpec("input-order"), SolverSpec("test-no-deadlines")),
            instances=(
                FamilySource("flow-shop", (1,), {"jobs": 3, "machines": 2}),
                FileSource(Path("deadlines.json")),
            ),
        ),
        base=tmp_path,
    )
    result = plan(campaign, registry=REGISTRY)
    assert [(e.problem, e.solver, e.features) for e in result.exclusions] == [
        (deadlines.id, "test-no-deadlines", ("deadline",))
    ]
    assert "does not model deadline" in result.exclusions[0].reason
    assert not any(
        item.problem == deadlines.id and item.solver == "test-no-deadlines" for item in result.runs
    )
    assert assess(result).fair
    report = check_compatibility(campaign, registry=REGISTRY)
    assert report.exclusions == result.exclusions
    assert report.excludes(deadlines.id, "test-no-deadlines")


def test_a_campaign_whose_every_pair_is_excluded_is_refused(tmp_path: Path) -> None:
    problem = random_problem(10, tasks=6, max_modes=1, pools=0, deadlines=True)
    save_problem(problem, tmp_path / "p.json")
    campaign = replace(
        tiny_campaign(
            solvers=(SolverSpec("test-no-deadlines"),), instances=(FileSource(Path("p.json")),)
        ),
        base=tmp_path,
    )
    with pytest.raises(ConfigError, match="plans no runs"):
        plan(campaign, registry=REGISTRY)
    assert len(check_compatibility(campaign, registry=REGISTRY).exclusions) == 1


def test_a_solver_that_needs_a_shape_is_excluded_from_problems_without_it() -> None:
    fork = Problem(
        "fork",
        tasks=(Task.on("a", "m1", 1), Task.on("b", "m2", 1), Task.on("c", "m3", 1)),
        resources=(Resource("m1"), Resource("m2"), Resource("m3")),
        dependencies=(Dependency("a", "b"), Dependency("a", "c")),
    )
    needs = SolverInfo(
        id="needs-shape",
        name="Needs a shape",
        family="local-search",
        summary="Works on job shops only.",
        shape="job-shop",
    )
    report = compatibility.assess([small_job_shop(), fork], [("needs-shape", needs)])
    assert [(e.problem, e.solver, e.features) for e in report.exclusions] == [
        ("fork", "needs-shape", ("job-shop",))
    ]
    assert "does not model job-shop" in report.exclusions[0].reason
    assert report.excludes("fork", "needs-shape")
    assert not report.excludes("small-job-shop", "needs-shape")


def test_objective_caveats_are_recorded_with_their_source() -> None:
    many = ("makespan", "total_completion_time", "total_flow_time", "energy")
    assert len(many) == MANY_OBJECTIVES
    campaign = replace(
        tiny_campaign(solvers=(SolverSpec("random-search"), SolverSpec("test-pareto"))),
        objectives=many,
    )
    caveats = {caveat.solver: caveat for caveat in plan(campaign, registry=REGISTRY).caveats}
    assert "primary objective" in caveats["random-search"].message
    assert caveats["random-search"].problems == ("flow-shop-3x2-1", "flow-shop-3x2-2")
    assert caveats["test-pareto"].reference == "deb2014evolutionary"
    assert reference("deb2014evolutionary").doi == "10.1109/TEVC.2013.2281535"
    assert plan(tiny_campaign(), registry=REGISTRY).caveats == ()


def test_problem_digests_are_those_of_the_problems_as_solved() -> None:
    campaign = replace(tiny_campaign(), objectives=("total_completion_time",))
    result = plan(campaign, registry=REGISTRY)
    from dispatchatlas.benchmarks import generate  # noqa: PLC0415

    problem = generate("flow-shop", 1, jobs=3, machines=2).with_objectives("total_completion_time")
    assert result.problems[0].digest == problem_digest(problem)
