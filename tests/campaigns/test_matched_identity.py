"""The matched job-shop families through campaign records: the arms of a pair are one run.

A campaign gives every solver of one problem and replicate the same seed, so the two arms of a
matched family, random keys and job sequences, draw the same control and operator numbers. The
tests write a tiny campaign file, run it with the library's runner on the sequential backend, and
compare the records that come back, as an analysis reads them. The arms of random sampling decode
the same schedules in the same order, so their records agree in everything that is not the solver,
a time, or a measure of the representation; the arms of simulated annealing start from the same
first schedule.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from dispatchatlas.campaigns import RunRecord, load_campaign, read_runs, run
from dispatchatlas.campaigns.records import schedule_rows
from dispatchatlas.core import StopReason

EVALUATIONS = 300
REPLICATES = 2
REPRESENTATIONS = ("keys", "sequence")
BUILDERS = ("semi-active", "active")

CAMPAIGN = """schema_version = 1
id = "matched"
seed = 7
replicates = {replicates}

[[instances]]
family = "job-shop"
parameters = {{ jobs = 3, machines = 3 }}
seeds = [1]

[[instances]]
suite = "taillard-job-shop"
instances = ["ta01"]

{solvers}[[protocols]]
name = "evaluations"
stop = {{ max_evaluations = {evaluations} }}

[execution]
backend = "sequential"
"""

ARM = """[[solvers]]
id = "{solver}"
label = "{representation}-{builder}"
parameters = {{ representation = "{representation}", builder = "{builder}" }}

"""

type Pairs = dict[tuple[str, str, int], dict[str, RunRecord]]
"""The records of a campaign by problem, builder and replicate, then by representation."""


def _pairs(tmp_path: Path, solver: str, builders: tuple[str, ...]) -> Pairs:
    """Run the campaign of ``solver``'s arms from a file and return its records in pairs."""
    arms = "".join(
        ARM.format(solver=solver, representation=representation, builder=builder)
        for builder in builders
        for representation in REPRESENTATIONS
    )
    file = tmp_path / "matched.toml"
    text = CAMPAIGN.format(replicates=REPLICATES, solvers=arms, evaluations=EVALUATIONS)
    file.write_text(text, encoding="utf-8")
    summary = run(load_campaign(file), tmp_path / "results")
    planned = 2 * len(builders) * len(REPRESENTATIONS) * REPLICATES
    assert (summary.planned, summary.completed, summary.failed) == (planned, planned, 0)
    pairs: Pairs = defaultdict(dict)
    for record in read_runs(summary.directory):
        parameters = record.solver.parameters
        pairs[record.problem.id, parameters["builder"], record.replicate][
            parameters["representation"]
        ] = record
    assert {problem for problem, _, _ in pairs} == {"job-shop-3x3-1", "ta01"}
    assert all(sorted(arms) == sorted(REPRESENTATIONS) for arms in pairs.values())
    assert len(pairs) == planned // len(REPRESENTATIONS)
    return dict(pairs)


def _first_digest(record: RunRecord) -> str:
    digest = record.outcome.extras["first_schedule_digest"]
    assert isinstance(digest, str)
    assert len(digest) == 32
    return digest


def _ends_at_its_limit(record: RunRecord) -> None:
    """Check the stop rule: exactly the evaluation limit, and no repaired order."""
    assert record.outcome.evaluations == EVALUATIONS
    assert record.outcome.stop_reason is StopReason.EVALUATION_LIMIT
    assert record.outcome.extras["repairs"] == 0


def test_random_sampling_arms_are_equal_through_campaign_records(tmp_path: Path) -> None:
    pairs = _pairs(tmp_path, "shop-random-sampling", BUILDERS)
    for arms in pairs.values():
        keys, sequence = arms["keys"], arms["sequence"]
        assert keys.seed == sequence.seed
        for record in (keys, sequence):
            _ends_at_its_limit(record)
        assert keys.schedule is not None
        assert sequence.schedule is not None
        assert schedule_rows(keys.schedule) == schedule_rows(sequence.schedule)
        assert keys.outcome.schedule_digest == sequence.outcome.schedule_digest
        assert keys.outcome.value == sequence.outcome.value
        assert keys.outcome.objectives == sequence.outcome.objectives
        assert keys.outcome.evaluations == sequence.outcome.evaluations
        assert keys.outcome.iterations == sequence.outcome.iterations
        assert [(p.evaluations, p.iteration, p.value, p.feasible) for p in keys.outcome.trace] == [
            (p.evaluations, p.iteration, p.value, p.feasible) for p in sequence.outcome.trace
        ]
        assert _first_digest(keys) == _first_digest(sequence)
        # Every field the result hash covers, the schedule's digest among them, is equal.
        assert keys.result_hash == sequence.result_hash
        assert keys.identity_hash != sequence.identity_hash
    # The check is not trivial: the replicates start from schedules of their own.
    assert len({_first_digest(arms["keys"]) for arms in pairs.values()}) >= 4


def test_annealing_arms_start_from_one_first_schedule_through_campaign_records(
    tmp_path: Path,
) -> None:
    pairs = _pairs(tmp_path, "shop-simulated-annealing", ("active",))
    for arms in pairs.values():
        keys, sequence = arms["keys"], arms["sequence"]
        assert keys.seed == sequence.seed
        assert _first_digest(keys) == _first_digest(sequence)
        for record in (keys, sequence):
            _ends_at_its_limit(record)
        # The calibration costs both arms the same evaluations, which count toward the limit.
        keys_calibration = keys.outcome.extras["calibration_evaluations"]
        assert keys_calibration == sequence.outcome.extras["calibration_evaluations"]
        assert isinstance(keys_calibration, int)
        assert keys_calibration > 0
    assert len({_first_digest(arms["keys"]) for arms in pairs.values()}) == len(pairs)
