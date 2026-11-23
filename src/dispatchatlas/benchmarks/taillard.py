"""Taillard's benchmark instances, regenerated exactly from the published seeds.

Taillard (1993) defined 120 permutation flow-shop instances (``ta001`` to
``ta120``), 80 job-shop instances (``ta01`` to ``ta80``), and 60 open-shop
instances by a portable random number generator and one or two seeds per
instance. This module implements that generator and carries the seed tables
distributed with OR-Library (Beasley 1990) in ``flowshop2.txt``,
``jobshop2.txt``, and ``openshop.txt``, so every instance is available offline
and identical to the published data.

The generator ``unif`` is the Lehmer generator
``seed <- 16807 * seed mod (2^31 - 1)``, computed with Schrage's factorisation
(``b = 127773``, ``c = 2836``) so that no intermediate value leaves the 32-bit
range, followed by ``low + floor(seed / (2^31 - 1) * (high - low + 1))``. A
processing time is ``unif(seed, 1, 99)``:

- Flow shop: ``d[i][j]`` for machine ``i``, then job ``j``, all from the time
  seed.
- Job shop and open shop: ``d[i][j]`` for job ``j``, then operation ``i``, all
  from the time seed; then each job's machines ``M[i][j] = i`` are shuffled by
  swapping ``M[i][j]`` with ``M[unif(machine seed, i, m)][j]`` for
  ``i = 1..m``. Operation ``i`` of job ``j`` runs on machine ``M[i][j]`` for
  ``d[i][j]``; in an open shop the operations may run in any order.

The open-shop table also records the bounds printed in ``openshop.txt``: an
upper bound, marked there with ``*`` when it is optimal, and a lower bound.

Instances are named ``ta001`` to ``ta120`` (flow shop), ``ta01`` to ``ta80``
(job shop), and ``tai<n>x<n>-<k>`` for the ``k``-th open shop with ``n`` jobs
and machines, for example ``tai4x4-1``; open shops are also numbered 1 to 60 in
file order. The problems follow the conventions of
:mod:`dispatchatlas.benchmarks.shop`, so machine ``M = 1`` is resource ``m0``.

:func:`taillard_job_shop_seeded` builds job shops of any size by the same
generator from seeds that Taillard did not publish: it derives the time seed and
the machine seed from one seed with
:func:`~dispatchatlas.core.seeds.derive_seed` and refuses a seed pair of the
published job-shop and open-shop tables. The family ``taillard-job-shop-seeded``
generates them, and :func:`register` adds it to a registry.

:data:`JOB_SHOP_ROUTE_DIGESTS` stores a digest of the routes of each job shop
``ta01`` to ``ta80`` as the generator rebuilds it, and
:func:`check_taillard_job_shop` compares a rebuilt instance with its entry, so
that a study can check from the installed package that it rebuilds the instances
the library was tested on.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import ClassVar

from dispatchatlas.benchmarks._common import check_int, check_seed
from dispatchatlas.benchmarks.registry import FamilyInfo, FamilyRegistry
from dispatchatlas.benchmarks.shop import flow_shop_problem, job_shop_problem, open_shop_problem
from dispatchatlas.core.errors import ConfigError, ModelError
from dispatchatlas.core.model import Problem, Scalar
from dispatchatlas.core.seeds import derive_seed

__all__ = [
    "JOB_SHOP_ROUTE_DIGESTS",
    "TaillardInstance",
    "TaillardJobShopSeeded",
    "check_taillard_job_shop",
    "flow_shop_instances",
    "flow_shop_matrix",
    "job_shop_instances",
    "job_shop_matrices",
    "job_shop_route_digest",
    "open_shop_instances",
    "open_shop_matrices",
    "register",
    "taillard_flow_shop",
    "taillard_instance",
    "taillard_job_shop",
    "taillard_job_shop_seeded",
    "taillard_open_shop",
    "unif",
]

MODULUS = 2147483647
"""The modulus ``2^31 - 1`` of Taillard's generator."""

MULTIPLIER = 16807
"""The multiplier of Taillard's generator."""

_QUOTIENT = 127773  # MODULUS // MULTIPLIER
_REMAINDER = 2836  # MODULUS % MULTIPLIER

_ORLIB = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"

_SEEDED_FAMILY = "taillard-job-shop-seeded"


@dataclass(frozen=True, slots=True)
class TaillardInstance:
    """The published definition of one Taillard instance.

    Attributes:
        name: Instance name, for example ``"ta001"``.
        problem_class: ``"flow-shop"``, ``"job-shop"``, or ``"open-shop"``.
        jobs: Number of jobs.
        machines: Number of machines.
        time_seed: Seed of the processing times.
        machine_seed: Seed of the machine orders; ``None`` for flow shops.
        upper_bound: Makespan of the best schedule listed with the seeds, if any.
        lower_bound: Makespan lower bound listed with the seeds, if any.
        optimal: Whether the listed upper bound is marked optimal.
        source: URL of the OR-Library file that lists the seeds.
    """

    name: str
    problem_class: str
    jobs: int
    machines: int
    time_seed: int
    machine_seed: int | None = None
    upper_bound: int | None = None
    lower_bound: int | None = None
    optimal: bool = False
    source: str = ""


def unif(seed: int, low: int, high: int) -> tuple[int, int]:
    """Advance Taillard's generator once and draw an integer in ``[low, high]``.

    Args:
        seed: Current state, in ``[1, 2^31 - 2]``.
        low: Smallest value to draw.
        high: Largest value to draw.

    Returns:
        The next state and the drawn value.
    """
    k = seed // _QUOTIENT
    seed = MULTIPLIER * (seed % _QUOTIENT) - k * _REMAINDER
    if seed < 0:
        seed += MODULUS
    return seed, low + int(seed / MODULUS * (high - low + 1))


def flow_shop_matrix(time_seed: int, jobs: int, machines: int) -> list[list[int]]:
    """Return the processing times ``d[i][j]`` of machine ``i`` and job ``j``.

    This is Taillard's ``generate_flow_shop``, with rows and columns as printed
    in the paper and in ``flowshop2.txt``.
    """
    seed = time_seed
    matrix = []
    for _ in range(machines):
        row = []
        for _ in range(jobs):
            seed, value = unif(seed, 1, 99)
            row.append(value)
        matrix.append(row)
    return matrix


def job_shop_matrices(
    time_seed: int, machine_seed: int, jobs: int, machines: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Return ``(d, M)``: operation ``i`` of job ``j`` runs on machine ``M[i][j]`` for ``d[i][j]``.

    This is Taillard's ``generate_job_shop``, with rows (operations) and
    columns (jobs) as printed in ``jobshop2.txt``, and machines numbered from 1.
    """
    durations = [[0] * jobs for _ in range(machines)]
    seed = time_seed
    for j in range(jobs):
        for i in range(machines):
            seed, durations[i][j] = unif(seed, 1, 99)
    order = [[i + 1] * jobs for i in range(machines)]
    seed = machine_seed
    for j in range(jobs):
        for i in range(machines):
            seed, k = unif(seed, i, machines - 1)
            order[i][j], order[k][j] = order[k][j], order[i][j]
    return durations, order


def open_shop_matrices(
    time_seed: int, machine_seed: int, jobs: int, machines: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Return ``(d, M)`` of Taillard's ``generate_open_shop``.

    The procedure is the same as :func:`job_shop_matrices`: job ``j`` needs
    machine ``M[i][j]`` for ``d[i][j]``, in any order.
    """
    return job_shop_matrices(time_seed, machine_seed, jobs, machines)


def flow_shop_instances() -> tuple[TaillardInstance, ...]:
    """Return the 120 flow-shop instances ``ta001`` to ``ta120`` in order."""
    return _tables()[0]


def job_shop_instances() -> tuple[TaillardInstance, ...]:
    """Return the 80 job-shop instances ``ta01`` to ``ta80`` in order."""
    return _tables()[1]


def open_shop_instances() -> tuple[TaillardInstance, ...]:
    """Return the 60 open-shop instances ``tai4x4-1`` to ``tai20x20-10`` in order."""
    return _tables()[2]


def taillard_instance(problem_class: str, instance: str | int) -> TaillardInstance:
    """Return the definition of one instance.

    Args:
        problem_class: ``"flow-shop"``, ``"job-shop"``, or ``"open-shop"``.
        instance: Name, such as ``"ta001"``, ``"ta01"``, or ``"tai4x4-1"``, or
            the instance's number in its table, starting at 1.

    Raises:
        ConfigError: If the class or the instance is unknown.
    """
    tables = dict(zip(("flow-shop", "job-shop", "open-shop"), _tables(), strict=True))
    if problem_class not in tables:
        raise ConfigError(
            f"unknown Taillard problem class {problem_class!r}; known: {sorted(tables)}"
        )
    table = tables[problem_class]
    number = _number(problem_class, instance, table)
    if number is None or not 1 <= number <= len(table):
        first, last = table[0].name, table[-1].name
        raise ConfigError(
            f"unknown Taillard {problem_class} instance {instance!r}; "
            f"use {first} to {last} or a number from 1 to {len(table)}"
        )
    return table[number - 1]


def taillard_flow_shop(instance: str | int) -> Problem:
    """Return a Taillard permutation flow shop, for example ``taillard_flow_shop("ta001")``.

    Raises:
        ConfigError: If the instance is unknown.
    """
    item = taillard_instance("flow-shop", instance)
    matrix = flow_shop_matrix(item.time_seed, item.jobs, item.machines)
    times = [[float(matrix[i][j]) for i in range(item.machines)] for j in range(item.jobs)]
    return flow_shop_problem(
        times, problem_id=item.name, permutation=True, metadata=_metadata(item)
    )


def taillard_job_shop(instance: str | int) -> Problem:
    """Return a Taillard job shop, for example ``taillard_job_shop("ta01")``.

    Raises:
        ConfigError: If the instance is unknown.
    """
    item = taillard_instance("job-shop", instance)
    return _job_shop(
        item.time_seed,
        _machine_seed(item),
        item.jobs,
        item.machines,
        problem_id=item.name,
        metadata=_metadata(item),
    )


def taillard_open_shop(instance: str | int) -> Problem:
    """Return a Taillard open shop, for example ``taillard_open_shop("tai4x4-1")``.

    The published bounds are recorded in the problem's metadata as
    ``upper_bound``, ``lower_bound``, and ``optimal``.

    Raises:
        ConfigError: If the instance is unknown.
    """
    item = taillard_instance("open-shop", instance)
    durations, order = open_shop_matrices(
        item.time_seed, _machine_seed(item), item.jobs, item.machines
    )
    times = [[0.0] * item.machines for _ in range(item.jobs)]
    for j in range(item.jobs):
        for i in range(item.machines):
            times[j][order[i][j] - 1] = float(durations[i][j])
    return open_shop_problem(times, problem_id=item.name, metadata=_metadata(item))


def job_shop_route_digest(problem: Problem) -> str:
    """Return the SHA-256 digest, in hexadecimal, of the routes of a job shop.

    The digest hashes one line for each job, the jobs in the order of the problem's groups:
    the problem's id, then the operations of the job in order, each written
    ``<machine>:<time>`` with the machine the operation demands and its processing time in
    Python's ``g`` format, separated by spaces, and a line feed. It covers the machine and
    the processing time of every operation and the order of the operations and of the jobs.
    Release times, due dates, metadata, and the rest of the problem are not in it;
    :func:`~dispatchatlas.core.serialization.problem_digest` digests the whole problem.

    Args:
        problem: A job shop whose every operation belongs to a job group, as the problems
            of this module are.

    Returns:
        The digest, 64 lowercase hexadecimal digits.

    Raises:
        ModelError: If the problem is not a job shop, or an operation belongs to no group.
    """
    if "job-shop" not in problem.shapes:
        raise ModelError(f"problem {problem.id!r} is not a job shop")
    if sum(len(indices) for indices in problem.groups.values()) != len(problem.tasks):
        raise ModelError(f"problem {problem.id!r} has operations that belong to no job group")
    digest = hashlib.sha256()
    for indices in problem.groups.values():
        route = []
        for index in indices:
            mode = problem.tasks[index].modes[0]
            route.append(f"{next(iter(mode.demands))}:{mode.duration:g}")
        digest.update(f"{problem.id} {' '.join(route)}\n".encode())
    return digest.hexdigest()


def check_taillard_job_shop(instance: Problem) -> None:
    """Raise :class:`~dispatchatlas.core.errors.ModelError` unless a job shop has its stored routes.

    ``instance`` is a job shop named ``ta01`` to ``ta80``, such as :func:`taillard_job_shop`
    builds. Its route digest (:func:`job_shop_route_digest`) must equal the entry of
    :data:`JOB_SHOP_ROUTE_DIGESTS` for its id. The tests of the library rebuild ta01 to ta80
    and compare each with its entry, so a study that calls this function on the machine where
    it runs, with the package it installed, checks that it rebuilds the instances the library
    was tested on.

    This checks the reproducibility of the generator: that it gives here the instance it gave
    the tests. It does not compare the instance with the processing-time files that Taillard
    published; that is a separate check, which needs those files.

    Args:
        instance: The rebuilt job shop. Its id names the instance.

    Raises:
        ModelError: If the id is not one of ``ta01`` to ``ta80``, the problem is not a job
            shop, or its routes differ from the stored ones. The ``issues`` of a difference
            hold the stored digest and the digest found.
    """
    name = instance.id
    stored = JOB_SHOP_ROUTE_DIGESTS.get(name)
    if stored is None:
        raise ModelError(
            f"{name!r} is not one of the job shops ta01 to ta80, so no route digest is stored "
            "for it"
        )
    found = job_shop_route_digest(instance)
    if found != stored:
        raise ModelError(
            f"job shop {name!r} does not have the routes that the library stored for it",
            issues=(f"stored route digest: {stored}", f"digest of the routes found: {found}"),
        )


def taillard_job_shop_seeded(jobs: int, machines: int, seed: int) -> Problem:
    """Return a job shop of Taillard's generator built from seeds that Taillard did not publish.

    The time seed and the machine seed derive from ``seed`` and the size with
    :func:`~dispatchatlas.core.seeds.derive_seed`, and are mapped onto the states ``1`` to
    ``2^31 - 2`` of the generator. Equal arguments give the same problem on every platform,
    and another seed or another size gives an unrelated one. The problem is built as
    :func:`taillard_job_shop` builds a published instance: the processing times from the
    time seed, then each job's machine order from the machine seed.

    Args:
        jobs: Number of jobs.
        machines: Number of machines. Every job visits each machine once.
        seed: Non-negative seed of the instance. The derived time and machine seeds are
            recorded in the problem's metadata as ``time_seed`` and ``machine_seed``.

    Returns:
        The problem, named ``taillard-job-shop-seeded-<jobs>x<machines>-<seed>``.

    Raises:
        ValueError: If ``jobs`` or ``machines`` is not a positive integer, or ``seed`` is not
            a non-negative integer.
        ConfigError: If the derived seeds are a pair that Taillard published, for a job shop
            or an open shop. Another ``seed`` gives other seeds.
    """
    jobs = check_int("jobs", jobs)
    machines = check_int("machines", machines)
    seed = check_seed(seed)
    time_seed, machine_seed = _derived_seeds(jobs, machines, seed)
    metadata: dict[str, Scalar] = {
        "jobs": jobs,
        "machines": machines,
        "seed": seed,
        "time_seed": time_seed,
        "machine_seed": machine_seed,
    }
    return _job_shop(
        time_seed,
        machine_seed,
        jobs,
        machines,
        problem_id=f"{_SEEDED_FAMILY}-{jobs}x{machines}-{seed}",
        metadata=metadata,
        family=_SEEDED_FAMILY,
    )


@dataclass(frozen=True)
class TaillardJobShopSeeded:
    """Job shops of Taillard's generator with seeds that Taillard did not publish.

    The time seed and the machine seed of a problem derive from the seed given to
    :meth:`generate`, as :func:`taillard_job_shop_seeded` describes. The family suits instance
    sets that must not contain a published instance, such as the instances a method is tuned
    on.

    Attributes:
        jobs: Number of jobs.
        machines: Number of machines. Every job visits each machine once.
    """

    jobs: int = 15
    machines: int = 15

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id=_SEEDED_FAMILY,
        name="Taillard job shop with derived seeds",
        problem_class="job-shop",
        summary="Job shops built by Taillard's (1993) generator from a time seed and a machine "
        "seed derived from one seed, never a seed pair that Taillard published.",
        references=("taillard1993benchmarks",),
    )

    def __post_init__(self) -> None:
        check_int("jobs", self.jobs)
        check_int("machines", self.machines)

    def generate(self, seed: int = 0) -> Problem:
        """Return the job shop for ``seed``."""
        return taillard_job_shop_seeded(self.jobs, self.machines, seed)


def register(registry: FamilyRegistry) -> None:
    """Add the family of job shops with derived seeds to ``registry``."""
    registry.register(TaillardJobShopSeeded)


# ------------------------------------------------------------------ helpers


def _machine_seed(item: TaillardInstance) -> int:
    if item.machine_seed is None:  # pragma: no cover - every job and open shop has one
        raise ConfigError(f"instance {item.name!r} has no machine seed")
    return item.machine_seed


def _job_shop(
    time_seed: int,
    machine_seed: int,
    jobs: int,
    machines: int,
    *,
    problem_id: str,
    metadata: Mapping[str, Scalar],
    family: str | None = None,
) -> Problem:
    """Return the job shop that Taillard's generator makes of a time seed and a machine seed."""
    durations, order = job_shop_matrices(time_seed, machine_seed, jobs, machines)
    routes = [
        [(order[i][j] - 1, float(durations[i][j])) for i in range(machines)] for j in range(jobs)
    ]
    return job_shop_problem(
        routes, machines=machines, problem_id=problem_id, family=family, metadata=metadata
    )


def _derived_seeds(jobs: int, machines: int, seed: int) -> tuple[int, int]:
    """Return the time seed and the machine seed that ``seed`` derives for a size.

    Raises:
        ConfigError: If the pair is one that Taillard published.
    """
    stream = f"{_SEEDED_FAMILY}:{jobs}x{machines}"
    time_seed = _generator_state(derive_seed(seed, stream, 0))
    machine_seed = _generator_state(derive_seed(seed, stream, 1))
    if (time_seed, machine_seed) in _published_pairs():
        raise ConfigError(
            f"seed {seed} derives the seeds {time_seed} and {machine_seed} for {jobs} jobs and "
            f"{machines} machines, which are a pair that Taillard published; use another seed"
        )
    return time_seed, machine_seed


def _generator_state(derived: int) -> int:
    """Map a derived seed onto the states ``1`` to ``MODULUS - 1`` of Taillard's generator."""
    return derived % (MODULUS - 1) + 1


@cache
def _published_pairs() -> frozenset[tuple[int, int]]:
    """Return the ``(time seed, machine seed)`` pairs of the job-shop and open-shop tables."""
    _, job, open_ = _tables()
    return frozenset((item.time_seed, _machine_seed(item)) for item in (*job, *open_))


def _metadata(item: TaillardInstance) -> dict[str, Scalar]:
    values: dict[str, Scalar] = {
        "suite": f"taillard-{item.problem_class}",
        "instance": item.name,
        "problem_class": item.problem_class,
        "jobs": item.jobs,
        "machines": item.machines,
        "time_seed": item.time_seed,
        "source": item.source,
    }
    if item.machine_seed is not None:
        values["machine_seed"] = item.machine_seed
    if item.upper_bound is not None:
        values["upper_bound"] = item.upper_bound
        values["optimal"] = item.optimal
    if item.lower_bound is not None:
        values["lower_bound"] = item.lower_bound
    return values


def _number(
    problem_class: str, instance: str | int, table: Sequence[TaillardInstance]
) -> int | None:
    """Return the 1-based table position an instance name or number denotes."""
    if isinstance(instance, bool):
        return None
    if isinstance(instance, int):
        return instance
    text = str(instance).strip().lower()
    if text.isdigit():
        return int(text)
    if problem_class == "open-shop":
        match = re.fullmatch(r"tai_?(\d+)[x_](\d+)[-_](\d+)", text)
        if match is None:
            return None
        size, other, index = (int(group) for group in match.groups())
        for position, item in enumerate(table, start=1):
            if (item.jobs, item.machines) == (size, other):
                return position + index - 1 if 1 <= index <= 10 else None
        return None
    match = re.fullmatch(r"ta0*(\d+)", text)
    return int(match.group(1)) if match else None


@cache
def _tables() -> tuple[
    tuple[TaillardInstance, ...], tuple[TaillardInstance, ...], tuple[TaillardInstance, ...]
]:
    flow = tuple(
        TaillardInstance(
            f"ta{10 * block + k + 1:03d}",
            "flow-shop",
            jobs,
            machines,
            seed,
            source=_ORLIB + "flowshop2.txt",
        )
        for block, ((jobs, machines), seeds) in enumerate(_FLOW_SHOP_SEEDS)
        for k, seed in enumerate(seeds)
    )
    job = tuple(
        TaillardInstance(
            f"ta{10 * block + k + 1:02d}",
            "job-shop",
            jobs,
            machines,
            time_seed,
            machine_seed,
            source=_ORLIB + "jobshop2.txt",
        )
        for block, ((jobs, machines), seeds) in enumerate(_JOB_SHOP_SEEDS)
        for k, (time_seed, machine_seed) in enumerate(seeds)
    )
    open_ = tuple(
        TaillardInstance(
            f"tai{size}x{size}-{k + 1}",
            "open-shop",
            size,
            size,
            time_seed,
            machine_seed,
            upper_bound=upper,
            lower_bound=lower,
            optimal=optimal,
            source=_ORLIB + "openshop.txt",
        )
        for size, rows in _OPEN_SHOP_SEEDS
        for k, (time_seed, machine_seed, upper, optimal, lower) in enumerate(rows)
    )
    return flow, job, open_


# ------------------------------------------------------------------ published data
#
# Seeds as listed in OR-Library's flowshop2.txt, jobshop2.txt, and openshop.txt.
# Each block of ten instances shares one size. jobshop2.txt misprints the name
# of ta42 as "ta4@"; its seeds are the second pair of the 30 x 20 block.

_FLOW_SHOP_SEEDS: tuple[tuple[tuple[int, int], tuple[int, ...]], ...] = (
    ((20, 5), (873654221, 379008056, 1866992158, 216771124, 495070989,
               402959317, 1369363414, 2021925980, 573109518, 88325120)),
    ((20, 10), (587595453, 1401007982, 873136276, 268827376, 1634173168,
                691823909, 73807235, 1273398721, 2065119309, 1672900551)),
    ((20, 20), (479340445, 268827376, 1958948863, 918272953, 555010963,
                2010851491, 1519833303, 1748670931, 1923497586, 1829909967)),
    ((50, 5), (1328042058, 200382020, 496319842, 1203030903, 1730708564,
               450926852, 1303135678, 1273398721, 587288402, 248421594)),
    ((50, 10), (1958948863, 575633267, 655816003, 1977864101, 93805469,
                1803345551, 49612559, 1899802599, 2013025619, 578962478)),
    ((50, 20), (1539989115, 691823909, 655816003, 1315102446, 1949668355,
                1923497586, 1805594913, 1861070898, 715643788, 464843328)),
    ((100, 5), (896678084, 1179439976, 1122278347, 416756875, 267829958,
                1835213917, 1328833962, 1418570761, 161033112, 304212574)),
    ((100, 10), (1539989115, 655816003, 960914243, 1915696806, 2013025619,
                 1168140026, 1923497586, 167698528, 1528387973, 993794175)),
    ((100, 20), (450926852, 1462772409, 1021685265, 83696007, 508154254,
                 1861070898, 26482542, 444956424, 2115448041, 118254244)),
    ((200, 10), (471503978, 1215892992, 135346136, 1602504050, 160037322,
                 551454346, 519485142, 383947510, 1968171878, 540872513)),
    ((200, 20), (2013025619, 475051709, 914834335, 810642687, 1019331795,
                 2056065863, 1342855162, 1325809384, 1988803007, 765656702)),
    ((500, 20), (1368624604, 450181436, 1927888393, 1759567256, 606425239,
                 19268348, 1298201670, 2041736264, 379756761, 28837162)),
)  # fmt: skip

_JOB_SHOP_SEEDS: tuple[tuple[tuple[int, int], tuple[tuple[int, int], ...]], ...] = (
    ((15, 15), ((840612802, 398197754), (1314640371, 386720536), (1227221349, 316176388),
                (342269428, 1806358582), (1603221416, 1501949241), (1357584978, 1734077082),
                (44531661, 1374316395), (302545136, 2092186050), (1153780144, 1393392374),
                (73896786, 1544979948))),
    ((20, 15), ((533484900, 317419073), (1894307698, 1474268163), (874340513, 509669280),
                (1124986343, 1209573668), (1463788335, 529048107), (1056908795, 25321885),
                (195672285, 1717580117), (961965583, 1353003786), (1610169733, 1734469503),
                (532794656, 998486810))),
    ((20, 20), ((1035939303, 773961798), (5997802, 1872541150), (1357503601, 722225039),
                (806159563, 1166962073), (1902815253, 1879990068), (1503184031, 1850351876),
                (1032645967, 99711329), (229894219, 1158117804), (823349822, 108033225),
                (1297900341, 489486403))),
    ((30, 15), ((98640593, 1981283465), (1839268120, 248890888), (573875290, 2081512253),
                (1670898570, 788294565), (1118914567, 1074349202), (178750207, 294279708),
                (1549372605, 596993084), (798174738, 151685779), (553410952, 1329272528),
                (1661531649, 1173386294))),
    ((30, 20), ((1841414609, 1357882888), (2116959593, 1546338557), (796392706, 1230864158),
                (532496463, 254174057), (2020525633, 978943053), (524444252, 185526083),
                (1569394691, 487269855), (1460267840, 1631446539), (198324822, 1937476577),
                (38071822, 1541985579))),
    ((50, 15), ((17271, 718939), (660481279, 449650254), (352229765, 949737911),
                (1197518780, 166840558), (1376020303, 483922052), (2106639239, 955932362),
                (1765352082, 1209982549), (1105092880, 1349003108), (907248070, 919544535),
                (2011630757, 1845447001))),
    ((50, 20), ((8493988, 2738939), (1991925010, 709517751), (342093237, 786960785),
                (1634043183, 973178279), (341706507, 286513148), (320167954, 1411193018),
                (1089696753, 298068750), (433032965, 1589656152), (615974477, 331205412),
                (236150141, 592292984))),
    ((100, 20), ((302034063, 1203569070), (1437643198, 1692025209), (1792475497, 1039908559),
                 (1647273132, 1012841433), (696480901, 1689682358), (1785569423, 1092647459),
                 (117806902, 739059626), (1639154709, 1319962509), (2007423389, 749368241),
                 (682761130, 262763021))),
)  # fmt: skip

# Per row: time seed, machine seed, upper bound, whether it is marked optimal,
# lower bound.
_OPEN_SHOP_SEEDS: tuple[tuple[int, tuple[tuple[int, int, int, bool, int], ...]], ...] = (
    (4, ((1166510396, 164000672, 193, True, 186), (1624514147, 1076870026, 236, True, 229),
         (1116611914, 1729673136, 271, True, 262), (410579806, 1453014524, 250, True, 245),
         (1036100146, 375655500, 295, True, 287), (597897640, 322140729, 189, True, 185),
         (1268670769, 556009645, 201, True, 197), (307928077, 421384574, 217, True, 212),
         (667545295, 485515899, 261, True, 258), (35780816, 492238933, 217, True, 213))),
    (5, ((527556884, 1343124817, 300, True, 295), (1046824493, 1973406531, 262, True, 255),
         (1165033492, 86711717, 323, True, 321), (476292817, 24463110, 310, True, 306),
         (1181363416, 606981348, 326, True, 321), (897739730, 513119113, 312, True, 307),
         (577107303, 2046387124, 303, True, 298), (1714191910, 1928475945, 300, True, 292),
         (1813128617, 2091141708, 353, True, 349), (808919936, 183753764, 326, True, 321))),
    (7, ((1840686215, 1827454623, 438, False, 435), (1026771938, 1312166461, 449, False, 443),
         (609471574, 670843185, 479, False, 468), (1022295947, 398226875, 467, False, 463),
         (1513073047, 1250759651, 419, False, 416), (1612211197, 95606345, 460, False, 451),
         (435024109, 1118234860, 435, False, 422), (1760865440, 1099909092, 426, False, 424),
         (122574075, 10979313, 460, False, 458), (248031774, 1685251301, 400, False, 398))),
    (10, ((1344106948, 1868311537, 645, False, 637), (425990073, 1111853152, 588, True, 588),
          (666128954, 1750328066, 611, False, 598), (442723456, 1369177184, 577, True, 577),
          (2033800800, 1344077538, 641, False, 640), (964467313, 1735817385, 538, True, 538),
          (1004528509, 967002400, 623, False, 616), (1667495107, 818777384, 596, False, 595),
          (1806968543, 1561913259, 595, True, 595), (938376228, 344628625, 602, False, 596))),
    (15, ((1561423441, 1787167667, 937, True, 937), (204120997, 213027331, 918, True, 918),
          (801158374, 1812110433, 871, True, 871), (1502847623, 1527847153, 934, True, 934),
          (282791231, 1855451778, 950, False, 946), (1130361878, 849417380, 933, True, 933),
          (379464508, 944419714, 891, True, 891), (1760142791, 1955448160, 893, True, 893),
          (1993140927, 179408412, 908, False, 899), (1678386613, 1567160817, 902, True, 902))),
    (20, ((957638, 9237185, 1155, True, 1155), (162587311, 1489531109, 1244, False, 1241),
          (965299017, 1054695706, 1257, True, 1257), (1158457671, 1499999517, 1248, True, 1248),
          (1191143707, 1530757746, 1256, True, 1256), (1826671743, 901609771, 1209, False, 1204),
          (1591533998, 1146547719, 1294, True, 1294), (937297777, 92726463, 1173, False, 1169),
          (687896268, 1731298717, 1289, True, 1289), (687034842, 684013066, 1241, True, 1241))),
)  # fmt: skip


# ------------------------------------------------------------------ route digests
#
# The route digest of each job shop, as job_shop_route_digest computes it from the
# instance that the seeds above rebuild.

JOB_SHOP_ROUTE_DIGESTS: Mapping[str, str] = MappingProxyType(
    {
        "ta01": "8367dc56d16ded75eae478d50e66a37260c1d6232cc63a6d933147779e9dcd0f",
        "ta02": "bdb5cded0710db8cc30d989fcb30be5fcd7bdf59da27dcd453a3ec7402a63fe9",
        "ta03": "e82892d31579fb7025d63e66db9cd14b891dd8a9e55125cc6b988a5cc3a6caed",
        "ta04": "06bc5fc9795fa6e2d59eea07ce0c9d921e2b33a4670accb6a3b3e5b3bed00893",
        "ta05": "36bca665d09418e511b18a3344d96a026a4b3008335d1af1a322ac954821b032",
        "ta06": "e325bc2e4d818492cba7f43964f5ba9e3dc305367918ea4342e18238eb835699",
        "ta07": "f3c84f13e8025e65face31ab061928d706cb69507950da2207875ed44f0cb177",
        "ta08": "5bc754fe9e305ef57112aba2ebf351cdb072f9ca5a8ba64987fe15cd37289211",
        "ta09": "ab4c13aeaea4fcb37158acff04572f06646059bec03aecdf474d152f7ac40cc0",
        "ta10": "60f9f3bb5c0e0732df3def8de3d67073ab46750ac35b640b9565c91753fcbfda",
        "ta11": "7504100355a84e97a971b677fb77f0bc13c7ba3653ba58ec499559fcded63358",
        "ta12": "c014aa8cd6bcc91bebd04ad1286adb1866b3da9416c8800c85abe2ef81ab9133",
        "ta13": "897b26dc658392702695b0746b89f39d900b1d90fe4e52741f706f2c66828ee4",
        "ta14": "6371759fba36ccd178c165e23f571053a6930d85f2c19f0dddca22bf84e7d670",
        "ta15": "bd946f5e165789abd1165035bbcefbf4cea3bdaa8765e4ae1c4f65e083a97a72",
        "ta16": "3b20b726112953237d7cf55ccd1c16e9c5880f3acbba27ef68d62541e9758020",
        "ta17": "579164327ceac0263c4213dc04ad13fbe31edd2699ba1ea52af89cd6753f0cb3",
        "ta18": "47b75514b5461dc7e6b61d53ac8c4a75584eb568a7ff5d852f69f0ee9e019513",
        "ta19": "51f35c81de53e3db05d6b5dfff415a7bcb7206e71ed8afdcd56b18ae00907f57",
        "ta20": "c6b6aa6e4fe3ac3e74a21ed49fcddd006ee2daf26e7a71ef795d3968ac72e2aa",
        "ta21": "57b1ca5836589b4c23f3c76793138210a40cc216b93aa54e3f485afc9646c755",
        "ta22": "d6882da3d26a95d967b6528256642b28573a5c1c5ba734db95ecd7786cc14feb",
        "ta23": "779de048a0b2fd90aea665d6d07fcf045156811625e87cfad202bea9dd745784",
        "ta24": "011f5bfa06464f61244116e7c37f09ef3aee6c56c07efd3e9e85fe6f867b7d26",
        "ta25": "b34cc1515e30489287245fa511fcce93d8cf969144bbc6ad71b82edf6122d7a2",
        "ta26": "7f4c15d3022aa874292dd604977ee4115f1df2416c8639aa2f8ccb875d52b78e",
        "ta27": "803911794060f13063a034ea13ba225c480977c82c825511480d1f0e75936a63",
        "ta28": "37c8a04cf9c8800419f7e377be8ed756a8ce9c5bb0ac2bc59b2a33e3a271e31f",
        "ta29": "d443922852cd5d6eadfa65d9f6cdb6f777df4ce8f5c90ea56f323a2f83c761b9",
        "ta30": "ad3a2e5280ef053ea1a203a9e395d4b9ea9180e15a1cc46d02607e5a2f5cde46",
        "ta31": "62887f7d74096eef5aaa169ae6c65448d27db3821e6fa860c25c40c543e9d3c4",
        "ta32": "3bfadcc4b05b364a0380292b5a46d349647ce10f1707b78819c99c08796984d3",
        "ta33": "ed1be7587ffc1362dc5f5747cd8204c0b30ca5f89ecfc49ba7ce2eae27ca1573",
        "ta34": "cd5ac810b3e060036f80ce5e763f90afb511979738137c241888559dd9fc4458",
        "ta35": "29012048dabef1abb91fb317d20929f7a4a23bc2016a124b83bf2eb5bb1fcdf3",
        "ta36": "37c5b3177f9a11c5797c245ea6d651386089962b9d840cd2e545a516059a732b",
        "ta37": "a0cdbad48b0f166c47b83313a334140797b02eca671d59a24b4e2b93d98ae704",
        "ta38": "7e7faf6ecf902bd9bc84cfdbc5eebe2e70049af60d7967bd3d698afa816dd666",
        "ta39": "1c92bf7d870e25ae60050f08bffc062d31225fd7d1f5010649033f38e9ad0bf0",
        "ta40": "354487951c79cfd279fcea3d8d21f503790d0a9d4fa06225f1f6b2e67a35ec33",
        "ta41": "6a5190615b3d29fffe883c5c54c9beada85739105d93d5612ffc2992d6deed38",
        "ta42": "89e95a539936e8d43d88e3eaa8e1c5bff7030734267603c078cd07da76789fdb",
        "ta43": "672b160e8e0351312a1d28506a25ff4947209805037111741ca0957d5fe91da9",
        "ta44": "c0cd7ec791a68dd6b4826feb4f985913681480d368cd8fc5d7801fb489e78acc",
        "ta45": "7f66632182b52069cdd19b18ce84e9ca6bbbcc086d51120244e547b7c9cc792e",
        "ta46": "636a617326225880e7ccdbf96101133af1c7bd983fd3a121d5f8dd129d939d32",
        "ta47": "48d45045bba5d899f7cdcc36a055ed053936ddb94c7fb28a2899b2226704ac56",
        "ta48": "bfd19feca4bfd34fe053590df75b5dd228720650bfe34e7099179742161ad6ca",
        "ta49": "8b0aec0ae8d3da94a77a2cb7946dfc7fd2601db75fa4d427a69d12c2bd718df5",
        "ta50": "a75c3cf66b44112e95d3221ceae8b785e63462f8204fd8f18087798dfc0cff08",
        "ta51": "f0a59f59f96bb9e61cb5483a1687f4a6a27896bd4c5072b2485dc35fef742d2a",
        "ta52": "8601a7798b1f7d9ea0548aa76c795ed9ff04aa518b65dc67d987093d94ad6a75",
        "ta53": "9ec3c72d4a42e26215328adadd461935fa8a5c598e7621d560582f1584c46985",
        "ta54": "08b5e6e89d34c6aa5ac8fda26902b79432b21b395097935ade61949b010041bf",
        "ta55": "d634d64207453dea56869c7b8935eb835d2582f1f65eb96911e3bda004c2b928",
        "ta56": "9f8b4b00396b686bc5ad1edaa78bebb72200e19174e3d267443a7c9701a88507",
        "ta57": "74cf3f37ed553d62b2d006479af748199ca73f11e75747f5e237a718a7cd9738",
        "ta58": "7f2b7cf42f52501e4844ca3924f441d74484b5e2fd537ca77888cc3a4d03c128",
        "ta59": "4aa8c9c8d7c334e113a328ac7c9eb20e5a01e9b9c544d94b1476737988683c9c",
        "ta60": "b8178d0a868634832827579daf214b093fade031fa4719519f31d494de0521de",
        "ta61": "7ba5020b13c689a07a092e150530782e1eb29c7abc6d31945e6c3948b6872b6f",
        "ta62": "2fdebb6ea96e814e61e03dc754fa70dbb51a4f647b27380264e87f1a8393447f",
        "ta63": "9759e01550dbca5f6a7d678724dbc162f8b9787d3cfde1060ecdfb7dcc72e1e6",
        "ta64": "fd5d0ca6759f0cbb46bd4826e3d6ca4a956d6831c1072d17987e51c92c1484d9",
        "ta65": "c3fb6d2a9d2046a11d8a1a98688e2af554aa0933258fe628081d08a485914a22",
        "ta66": "0c9af5dccdb55eff7f7b240a5d8adefca532f1ded8fec4acde6d6522575107aa",
        "ta67": "07bbee8a85e5817723f4e9edf9594d64032200053285a403e8fe00b03d20115b",
        "ta68": "b924dfbf87ad3997416dd16ec3e9f933af737bd39921090923d9e966dcb7e00d",
        "ta69": "284b979689300720f1040ec1a83acb5c281ec2d2967cc563fde14fe1f5305d53",
        "ta70": "ff0a1557d9cd0b4fab93f14ea8a749dc1333e047040d04ae25a0b9cc2e68bf23",
        "ta71": "cf6d9d98ef5fab3676bf300f9924294950e04a1ec2dd03490de05ff546e8854c",
        "ta72": "ec2baeadfe928ccdf494f4a7bc1c71ff6a8e02f92be00eb91b76c607c38a3fa2",
        "ta73": "21fe7f9ff0ef5269af3128a9b8f7cfbd3fbf8530fc66137761ddd2d8e9c3e7d3",
        "ta74": "f462f2a7a0d836585b17f744a260de7ae3bd885623749d29ae48bf391cbced26",
        "ta75": "af19a794e61d9326183b4aaa3bc2fcb061f020ab77082bc326c66b5e604247f5",
        "ta76": "98199fd387215bfd5a65326db028d7132affa721558ddc0b49d0df5d0d05a7da",
        "ta77": "1ce20b77d56d099c0af8d3e82f4703685619271b80cf6952d43041d7487dbeaf",
        "ta78": "d066d3f1dbbbd5adcf9a64ae299c598808f4e35b9afe2ba13b95486c565b9199",
        "ta79": "c91192e0df9cd2ac1eb62be1e9a9f6c24541ae8c63eba213df369bc7b39d1d2b",
        "ta80": "ee33fbf4118627fe43f3a32de945659dfd27af38e4792a400592310b54408f4b",
    }
)
"""The route digest of each job shop ta01 to ta80, by name, read-only.

Each value is what :func:`job_shop_route_digest` returns for the instance that
:func:`taillard_job_shop` builds. :func:`check_taillard_job_shop` compares a rebuilt
instance with its entry.
"""
