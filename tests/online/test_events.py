"""Events, traces, and their seeded generators."""

from __future__ import annotations

import math
import random
from itertools import pairwise
from pathlib import Path

import pytest

from dispatchatlas.core import Dependency, Problem, Resource, Task
from dispatchatlas.core.errors import ModelError, SerializationError
from dispatchatlas.online import (
    AlternatingRenewal,
    Arrival,
    Exponential,
    Failure,
    LogNormal,
    Repair,
    Trace,
    Weibull,
    Workflow,
    availability_events,
    poisson_arrivals,
    rate_for_load,
    submitted_arrivals,
    workflow_work,
    workflows_from_problem,
)
from tests.online.cases import continuum_case, fork, fork_trace


def test_a_trace_sorts_its_events_and_applies_repairs_first_at_an_instant():
    trace = Trace.of(
        [
            Arrival(5.0, fork()),
            Failure(5.0, "n1"),
            Repair(1.0, "n2"),
            Failure(0.5, "n2"),
            Repair(5.0, "n3"),
            Failure(2.0, "n3"),
        ]
    )
    assert [(e.time, e.kind.value, e.subject) for e in trace.events] == [
        (0.5, "failure", "n2"),
        (1.0, "repair", "n2"),
        (2.0, "failure", "n3"),
        (5.0, "repair", "n3"),
        (5.0, "failure", "n1"),
        (5.0, "arrival", "w"),
    ]
    assert trace.nodes == {"n1", "n2", "n3"}
    assert [a.workflow.id for a in trace.arrivals] == ["w"]


@pytest.mark.parametrize(
    "events",
    [
        [Failure(1.0, "n1"), Failure(2.0, "n1")],
        [Repair(1.0, "n1")],
        [Arrival(0.0, fork()), Arrival(1.0, fork())],
        [Failure(-1.0, "n1")],
        [Failure(math.inf, "n1")],
    ],
)
def test_a_trace_refuses_inconsistent_events(events):
    with pytest.raises(ModelError):
        Trace.of(events)


def test_a_trace_refuses_a_task_id_in_two_workflows():
    other = Workflow("v", fork().tasks, fork().dependencies)
    with pytest.raises(ModelError, match="two workflows"):
        Trace.of([Arrival(0.0, fork()), Arrival(1.0, other)])


def test_a_workflow_refuses_a_dependency_outside_its_tasks():
    with pytest.raises(ModelError, match="outside"):
        Workflow("w", fork().tasks, (Dependency("a", "z"),))


def test_the_digest_is_the_hash_of_the_content_whatever_the_input_order(tmp_path: Path):
    trace = fork_trace()
    shuffled = list(trace.events)
    random.Random(4).shuffle(shuffled)
    assert Trace.of(shuffled).digest == trace.digest
    assert len(trace.digest) == 64
    moved = Trace.of([Arrival(0.0, fork()), Failure(3.5, "n1"), Repair(10.0, "n1")])
    assert moved.digest != trace.digest
    assert Trace.loads(trace.dumps()) == trace
    assert Trace.from_dict(trace.to_dict()).digest == trace.digest
    path = trace.save(tmp_path / "trace.json")
    assert Trace.load(path) == trace


def test_a_trace_document_of_another_format_is_refused():
    document = fork_trace().to_dict()
    document["format"] = "something-else"
    with pytest.raises(SerializationError):
        Trace.from_dict(document)
    with pytest.raises(SerializationError):
        Trace.loads("{not json")


def test_generators_give_the_same_trace_for_the_same_seed():
    first = continuum_case(3)[1]
    again = continuum_case(3)[1]
    other = continuum_case(4)[1]
    assert first.digest == again.digest
    assert first.digest != other.digest
    assert all(e.time == round(e.time, 6) for e in first.events)


def test_arrivals_do_not_depend_on_the_availability_model_or_the_other_nodes():
    problem, trace = continuum_case(5, failures=False)
    workflows = [a.workflow for a in trace.arrivals]
    rate = rate_for_load(workflows, problem.resources, 0.7)
    assert poisson_arrivals(workflows, rate, seed=5).digest == trace.digest
    model = AlternatingRenewal(Exponential(2.0), Weibull(0.7, 0.5))
    alone = availability_events(["edge-0"], model, seed=9, horizon=50.0)
    with_others = availability_events(["edge-0", "fog-0", "fog-1"], model, seed=9, horizon=50.0)
    assert [e for e in with_others.events if e.subject == "edge-0"] == list(alone.events)
    assert len(with_others) > len(alone)


def test_poisson_gaps_have_the_stated_mean():
    workflows = [Workflow(f"w{k}", (Task.on(f"t{k}", "n1", 1.0),)) for k in range(4000)]
    trace = poisson_arrivals(workflows, rate=2.0, seed=1, start=10.0)
    times = [e.time for e in trace.events]
    assert times[0] > 10.0
    assert (times[-1] - 10.0) / len(times) == pytest.approx(0.5, rel=0.05)


def test_outages_alternate_and_keep_the_last_repair_past_the_horizon():
    model = AlternatingRenewal(Exponential(3.0), Exponential(1.0))
    outages = model.outages(random.Random(2), horizon=100.0)
    assert outages
    assert all(f < r for f, r in outages)
    assert all(r < f for (_, r), (f, _) in pairwise(outages))
    assert all(f < 100.0 for f, _ in outages)
    assert model.describe() == {
        "model": "alternating-renewal",
        "ttf_distribution": "exponential",
        "ttf_mean": 3.0,
        "ttr_distribution": "exponential",
        "ttr_mean": 1.0,
    }


def test_weibull_durations_have_their_mean():
    weibull = Weibull(shape=0.7, scale=2.0)
    rng = random.Random(3)
    draws = [weibull.sample(rng) for _ in range(20000)]
    assert sum(draws) / len(draws) == pytest.approx(weibull.mean, rel=0.05)
    with pytest.raises(ModelError):
        Weibull(shape=0.0, scale=1.0)
    with pytest.raises(ModelError):
        Exponential(-1.0)


def test_rate_for_load_divides_capacity_by_mean_work():
    # fork(): a 2, b 3, c 1 on their cheapest modes, 6 units of work; two
    # machines of capacity 1, so a load of 0.5 offers 0.5 * 2 / 6 per second.
    assert workflow_work(fork()) == 6.0
    resources = (Resource("n1"), Resource("n2"), Resource("unused", capacity=8))
    assert rate_for_load([fork()], resources, 0.5) == pytest.approx(1.0 / 6.0)
    with pytest.raises(ModelError):
        rate_for_load([], resources, 0.5)


def test_submitted_arrivals_pair_workflows_with_times():
    trace = submitted_arrivals([fork()], [7.5])
    assert [(e.time, e.subject) for e in trace.events] == [(7.5, "w")]
    with pytest.raises(ModelError):
        submitted_arrivals([fork()], [1.0, 2.0])


def test_a_static_problem_splits_into_workflows_with_offsets():
    problem = Problem(
        "two-groups",
        tasks=(
            Task.on("a", "m", 1.0, release=4.0, due=9.0, group="g"),
            Task.on("b", "m", 1.0, release=5.0, group="g"),
            Task.on("c", "m", 2.0, release=1.0),
        ),
        resources=(Resource("m"),),
        dependencies=(Dependency("a", "b"),),
    )
    (g, at_g), (c, at_c) = workflows_from_problem(problem)
    assert (g.id, at_g, c.id, at_c) == ("g", 4.0, "c", 1.0)
    assert [(t.release, t.due) for t in g.tasks] == [(0.0, 5.0), (1.0, None)]
    assert g.dependencies == (Dependency("a", "b"),)
    crossing = Problem(
        "crossing",
        tasks=(Task.on("a", "m", 1.0, group="g"), Task.on("b", "m", 1.0, group="h")),
        resources=(Resource("m"),),
        dependencies=(Dependency("a", "b"),),
    )
    with pytest.raises(ModelError, match="joins two workflows"):
        workflows_from_problem(crossing)


def test_lognormal_validates_its_parameters():
    assert LogNormal(mu=-1.02, sigma=1.27).mean == pytest.approx(math.exp(-1.02 + 1.27**2 / 2))
    for mu, sigma in [(math.nan, 1.0), (math.inf, 1.0), (0.0, 0.0), (0.0, -1.0), (0.0, math.inf)]:
        with pytest.raises(ModelError):
            LogNormal(mu, sigma)
    with pytest.raises(ModelError):
        LogNormal(True, 1.0)
    with pytest.raises(ModelError):
        LogNormal("0", 1.0)  # type: ignore[arg-type]


def test_lognormal_durations_have_their_mean_and_are_deterministic():
    lognormal = LogNormal(mu=0.5, sigma=0.6)
    draws = [lognormal.sample(r) for r in [random.Random(4)] for _ in range(40000)]
    assert min(draws) > 0.0
    assert sum(draws) / len(draws) == pytest.approx(lognormal.mean, rel=0.03)
    again = [lognormal.sample(r) for r in [random.Random(4)] for _ in range(40000)]
    assert again == draws


def test_lognormal_draws_are_those_of_the_standard_library_formula():
    # X = exp(mu + sigma * Z) with Z from the stream's normalvariate
    lognormal = LogNormal(mu=0.91, sigma=1.64)
    ours, theirs = random.Random(11), random.Random(11)
    for _ in range(100):
        assert lognormal.sample(ours) == math.exp(0.91 + 1.64 * theirs.normalvariate(0.0, 1.0))


def test_lognormal_equality_repr_and_description():
    assert LogNormal(0.1, 2.0) == LogNormal(0.1, 2.0)
    assert LogNormal(0.1, 2.0) != LogNormal(0.1, 2.5)
    assert repr(LogNormal(0.1, 2.0)) == "LogNormal(mu=0.1, sigma=2.0)"
    assert LogNormal(0.1, 2.0).describe() == {"distribution": "lognormal", "mu": 0.1, "sigma": 2.0}


def test_lognormal_in_an_alternating_renewal():
    model = AlternatingRenewal(Weibull(0.5, 10.0), LogNormal(0.05, 1.42))
    first = model.outages(random.Random(6), horizon=500.0)
    assert first
    assert first == model.outages(random.Random(6), horizon=500.0)
    assert all(f < r for f, r in first)
    assert model.describe()["ttr_distribution"] == "lognormal"
    assert model.describe()["ttr_mu"] == 0.05
    trace = availability_events(["a", "b"], model, seed=3, horizon=500.0)
    assert trace.digest == availability_events(["a", "b"], model, seed=3, horizon=500.0).digest
