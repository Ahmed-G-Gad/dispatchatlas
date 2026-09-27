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
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

from dispatchatlas.benchmarks.shop import flow_shop_problem, job_shop_problem, open_shop_problem
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.model import Problem, Scalar

__all__ = [
    "TaillardInstance",
    "flow_shop_instances",
    "flow_shop_matrix",
    "job_shop_instances",
    "job_shop_matrices",
    "open_shop_instances",
    "open_shop_matrices",
    "taillard_flow_shop",
    "taillard_instance",
    "taillard_job_shop",
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
    durations, order = job_shop_matrices(
        item.time_seed, _machine_seed(item), item.jobs, item.machines
    )
    routes = [
        [(order[i][j] - 1, float(durations[i][j])) for i in range(item.machines)]
        for j in range(item.jobs)
    ]
    return job_shop_problem(
        routes, machines=item.machines, problem_id=item.name, metadata=_metadata(item)
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


# ------------------------------------------------------------------ helpers


def _machine_seed(item: TaillardInstance) -> int:
    if item.machine_seed is None:  # pragma: no cover - every job and open shop has one
        raise ConfigError(f"instance {item.name!r} has no machine seed")
    return item.machine_seed


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
