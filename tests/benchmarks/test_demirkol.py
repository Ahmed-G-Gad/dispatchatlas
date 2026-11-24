"""The job shops of Demirkol, Mehta and Uzsoy: the stored files, their shapes, and their digests.

The oracle below parses the packaged files with the library's OR-Library reader and with plain
token splitting, so the loader of :mod:`dispatchatlas.benchmarks.demirkol` is checked against
code that shares nothing with it. The SHA-256 digests of the files are those that the
download of the ScheduleOpt collection gave.
"""

from __future__ import annotations

import hashlib
import re
from importlib.resources import files
from typing import Any

import pytest

from dispatchatlas.benchmarks import demirkol, suites
from dispatchatlas.benchmarks.demirkol import (
    JOB_SHOP_ROUTE_DIGESTS,
    check_demirkol_job_shop,
    demirkol_instances,
    demirkol_job_shop,
)
from dispatchatlas.benchmarks.formats import read_orlib_job_shop
from dispatchatlas.benchmarks.shop import job_shop_problem
from dispatchatlas.benchmarks.taillard import check_taillard_job_shop, job_shop_route_digest
from dispatchatlas.core import ConfigError, ModelError, SerializationError

FOLDER = files("dispatchatlas.data").joinpath("instances", "demirkol-job-shop")
SIZES = [(20, 15), (20, 20), (30, 15), (30, 20), (40, 15), (40, 20), (50, 15), (50, 20)]
NAMES = [f"dmu{number:02d}" for number in range(1, 81)]


def _text(name: str) -> str:
    return FOLDER.joinpath(f"{name}.txt").read_bytes().decode("ascii")


def _numbers(name: str) -> list[int]:
    return [int(token) for token in _text(name).split()]


def _oracle(name: str) -> str:
    """Digest of the routes, written from the raw numbers of the file alone."""
    numbers = _numbers(name)
    jobs, machines = numbers[0], numbers[1]
    width = len(str(machines - 1))
    digest = hashlib.sha256()
    for j in range(jobs):
        row = numbers[2 + 2 * j * machines : 2 + 2 * (j + 1) * machines]
        route = " ".join(f"m{row[2 * p]:0{width}d}:{row[2 * p + 1]}" for p in range(machines))
        digest.update(f"{name} {route}\n".encode())
    return digest.hexdigest()


def test_the_instances_are_dmu01_to_dmu80_in_blocks_of_five() -> None:
    assert list(demirkol_instances()) == NAMES
    for index, name in enumerate(NAMES):
        jobs, machines = SIZES[(index // 5) % 8]
        assert _numbers(name)[:2] == [jobs, machines], name
    assert sorted(path.name for path in FOLDER.iterdir()) == sorted(
        [*(f"{name}.txt" for name in NAMES), "NOTICE.txt"]
    )


def test_every_file_is_a_job_shop_with_positive_integer_times_each_machine_once() -> None:
    for name in NAMES:
        numbers = _numbers(name)
        jobs, machines = numbers[:2]
        assert len(numbers) == 2 + 2 * jobs * machines, name
        for j in range(jobs):
            row = numbers[2 + 2 * j * machines : 2 + 2 * (j + 1) * machines]
            assert sorted(row[0::2]) == list(range(machines)), (name, j)
            assert all(1 <= time <= 200 for time in row[1::2]), (name, j)


def test_the_first_set_has_random_routes_and_the_second_two_machine_sets() -> None:
    for index, name in enumerate(NAMES):
        numbers = _numbers(name)
        jobs, machines = numbers[:2]
        first = 7 if machines == 15 else 10
        two_sets = True
        for j in range(jobs):
            machine_order = numbers[2 + 2 * j * machines : 2 + 2 * (j + 1) * machines : 2]
            two_sets &= set(machine_order[:first]) == set(range(first))
        assert two_sets is (index >= 40), name


def test_the_problems_have_the_routes_of_the_files() -> None:
    for name in NAMES:
        problem = demirkol_job_shop(name)
        read = read_orlib_job_shop(_text(name), problem_id=name)
        assert problem.tasks == read.tasks, name
        assert problem.groups == read.groups, name
        assert job_shop_route_digest(problem) == job_shop_route_digest(read), name
        assert problem.metadata["suite"] == "demirkol-job-shop"
        assert (problem.metadata["jobs"], problem.metadata["machines"]) == tuple(_numbers(name)[:2])
        assert problem.size == _numbers(name)[0] * _numbers(name)[1]
        assert problem.metadata["routing"] == ("random" if int(name[3:]) <= 40 else "two-sets")


def test_the_stored_route_digests_match_an_independent_oracle() -> None:
    assert list(JOB_SHOP_ROUTE_DIGESTS) == NAMES
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in JOB_SHOP_ROUTE_DIGESTS.values())
    assert len(set(JOB_SHOP_ROUTE_DIGESTS.values())) == 80
    for name, stored in JOB_SHOP_ROUTE_DIGESTS.items():
        assert _oracle(name) == stored, name
    with pytest.raises(TypeError):
        JOB_SHOP_ROUTE_DIGESTS["dmu01"] = "0" * 64  # type: ignore[index]


def test_the_files_are_those_that_were_downloaded() -> None:
    notice = FOLDER.joinpath("NOTICE.txt").read_text(encoding="utf-8")
    recorded = {
        name: digest
        for digest, name in re.findall(r"^([0-9a-f]{64})  (dmu\d\d\.txt)$", notice, re.M)
    }
    assert len(recorded) == 80
    for name in NAMES:
        raw = FOLDER.joinpath(f"{name}.txt").read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(raw).hexdigest() == recorded[f"{name}.txt"], name
    assert "af68a647c1d62a342b596d23dcb57011715f3ffd" in notice
    assert "CC-BY-SA-4.0" in notice


def test_a_known_instance_has_the_first_job_of_its_file() -> None:
    problem = demirkol_job_shop("dmu01")
    first = [problem.tasks[i].modes[0] for i in problem.groups["j00"]]
    assert [next(iter(mode.demands)) for mode in first][:3] == ["m00", "m13", "m06"]
    assert [mode.duration for mode in first][:3] == [160.0, 5.0, 139.0]
    assert (problem.size, problem.id) == (300, "dmu01")


def test_names_numbers_and_case_are_accepted_and_others_refused() -> None:
    assert demirkol_job_shop(" DMU7 ").id == "dmu07" == demirkol_job_shop(7).id
    assert demirkol_job_shop("dmu80") is demirkol_job_shop(80)
    for bad in ("dmu00", "dmu81", "ta01", "", 0, 81, True, -1):
        with pytest.raises(ConfigError, match="unknown Demirkol instance"):
            demirkol_job_shop(bad)


def test_the_suite_lists_the_instances_and_cites_the_paper() -> None:
    item = suites.suite("demirkol-job-shop")
    assert item.instances == tuple(NAMES)
    assert item.references == ("demirkol1998benchmarks",)
    assert item.bundled
    assert item.offline
    assert not item.generated
    assert item.files is None
    assert item.load("dmu12") is demirkol_job_shop("dmu12")
    assert suites.suites()[3].id == "demirkol-job-shop"


# ------------------------------------------------------------- damaged files and the check


def _routes(text: str, jobs: int = 2, machines: int = 2) -> Any:
    return demirkol._routes("dmu99", text, jobs, machines)


def test_a_well_formed_text_parses() -> None:
    assert _routes("2 2\n0 5 1 6\n1 7 0 8\n") == [[(0, 5.0), (1, 6.0)], [(1, 7.0), (0, 8.0)]]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("2 2\n0 5 1 6\n1 7 0 x\n", "not a non-negative integer"),
        ("2 2\n0 5 1 6\n1 7 0 -8\n", "not a non-negative integer"),
        ("2 3\n0 5 1 6\n1 7 0 8\n", "not a 2 x 2 job shop"),
        ("2 2\n0 5 1 6\n1 7 0\n", "not a 2 x 2 job shop"),
        ("2 2\n0 5 1 6\n1 7 1 8\n", "job 1 does not visit every machine once"),
        ("2 2\n0 5 1 6\n1 7 2 8\n", "job 1 does not visit every machine once"),
        ("2 2\n0 5 1 0\n1 7 0 8\n", "job 0 has a processing time below 1"),
    ],
)
def test_a_damaged_text_is_refused(text: str, message: str) -> None:
    with pytest.raises(SerializationError, match=message):
        _routes(text)


def test_the_check_accepts_the_stored_routes_and_refuses_others() -> None:
    problem = demirkol_job_shop("dmu01")
    check_demirkol_job_shop(problem)
    routes = [
        [
            (
                int(next(iter(problem.tasks[i].modes[0].demands))[1:]),
                problem.tasks[i].modes[0].duration,
            )
            for i in indices
        ]
        for indices in problem.groups.values()
    ]
    rebuilt = job_shop_problem(routes, machines=15, problem_id="dmu01")
    check_demirkol_job_shop(rebuilt)
    routes[0][0] = (routes[0][0][0], routes[0][0][1] + 1)
    changed = job_shop_problem(routes, machines=15, problem_id="dmu01")
    with pytest.raises(ModelError, match="does not have the routes") as raised:
        check_demirkol_job_shop(changed)
    assert raised.value.issues[0] == f"stored route digest: {JOB_SHOP_ROUTE_DIGESTS['dmu01']}"
    with pytest.raises(ModelError, match="not one of the job shops dmu01 to dmu80"):
        check_demirkol_job_shop(job_shop_problem([[(0, 1.0)]], problem_id="ta01"))
    with pytest.raises(ModelError, match="not one of the job shops ta01 to ta80"):
        check_taillard_job_shop(problem)


def test_a_file_that_differs_from_its_digest_is_refused_on_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _text("dmu01")
    tokens = text.split()
    tokens[3] = str(int(tokens[3]) + 1)  # the first processing time
    damaged = " ".join(tokens)

    class Folder:
        def joinpath(self, *parts: str) -> Folder:
            return self

        def read_bytes(self) -> bytes:
            return damaged.encode("ascii")

    monkeypatch.setattr(demirkol, "files", lambda package: Folder())
    demirkol._build.cache_clear()
    try:
        with pytest.raises(ModelError, match="does not have the routes"):
            demirkol_job_shop("dmu01")
    finally:
        monkeypatch.undo()
        demirkol._build.cache_clear()
    assert demirkol_job_shop("dmu01").id == "dmu01"
