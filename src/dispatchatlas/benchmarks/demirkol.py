"""The job shops ``dmu01`` to ``dmu80`` of Demirkol, Mehta and Uzsoy (1998), from package data.

Demirkol, Mehta and Uzsoy (1998) generated job shops of 20, 30, 40, and 50 jobs on 15 and
20 machines, with processing times drawn uniformly from 1 to 200, and published the five
instances of each size and routing structure whose heuristic bounds were furthest apart.
The instances are data, not generator output, so the library ships the 80 files:

- ``dmu01`` to ``dmu40``: every job visits all machines in its own random order.
- ``dmu41`` to ``dmu80``: the machines form two sets, machines 0 to 6 (15 machines) or 0 to 9
  (20 machines) and the rest, and every job visits all machines of the first set, in a
  random order, before any machine of the second set.

Within each set the sizes run in blocks of five instances: 20 x 15, 20 x 20, 30 x 15,
30 x 20, 40 x 15, 40 x 20, 50 x 15, and 50 x 20 jobs by machines.

The files are the ones that the ScheduleOpt benchmark collection distributes, in the text
format that :func:`~dispatchatlas.benchmarks.formats.orlib.read_orlib_job_shop` reads: a size
line, then one line per job of ``machine time`` pairs, with machines numbered from 0. They
are stored under ``dispatchatlas/data/instances/demirkol-job-shop`` with a notice that
records their source, version, licence, and SHA-256 digests. Every instance read is checked
for its shape (each job visits each machine once, with a positive integer processing time)
and against :data:`JOB_SHOP_ROUTE_DIGESTS`, so a damaged or edited file is refused.

The problems follow the conventions of :mod:`dispatchatlas.benchmarks.shop`, so machine ``0``
is resource ``m00``. :func:`demirkol_job_shop` builds one; the suite ``"demirkol-job-shop"``
(:mod:`dispatchatlas.benchmarks.suites`) lists all of them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from types import MappingProxyType

from dispatchatlas.benchmarks.formats._text import parse_error
from dispatchatlas.benchmarks.shop import job_shop_problem
from dispatchatlas.benchmarks.taillard import job_shop_route_digest
from dispatchatlas.core.errors import ConfigError, ModelError
from dispatchatlas.core.model import Problem, Scalar

__all__ = [
    "DEMIRKOL_SOURCE",
    "JOB_SHOP_ROUTE_DIGESTS",
    "check_demirkol_job_shop",
    "demirkol_instances",
    "demirkol_job_shop",
]

DEMIRKOL_SOURCE = (
    "https://github.com/ScheduleOpt/benchmarks/tree/main/jobshop/instances/text/"
    "DemirkolMehtaUzsoy1998"
)
"""Where the instance files come from."""

_SIZES = ((20, 15), (20, 20), (30, 15), (30, 20), (40, 15), (40, 20), (50, 15), (50, 20))
_NAME = re.compile(r"dmu0*(\d+)")


def demirkol_instances() -> tuple[str, ...]:
    """Return the names ``dmu01`` to ``dmu80`` in order."""
    return tuple(f"dmu{number:02d}" for number in range(1, 81))


def demirkol_job_shop(instance: str | int) -> Problem:
    """Return one instance of Demirkol, Mehta and Uzsoy, for example ``demirkol_job_shop("dmu01")``.

    The problem's metadata hold ``suite``, ``instance``, ``problem_class``, ``jobs``,
    ``machines``, ``routing`` (``"random"`` for ``dmu01`` to ``dmu40``, ``"two-sets"`` for
    ``dmu41`` to ``dmu80``), and ``source``.

    Args:
        instance: Name such as ``"dmu01"`` (case does not matter), or its number from 1 to 80.

    Returns:
        The job shop, named after the instance.

    Raises:
        ConfigError: If the instance is unknown.
        SerializationError: If the stored file is malformed.
        ModelError: If the routes differ from those that :data:`JOB_SHOP_ROUTE_DIGESTS` stores.
    """
    name = _name(instance)
    return _build(name)


def check_demirkol_job_shop(instance: Problem) -> None:
    """Raise :class:`~dispatchatlas.core.errors.ModelError` unless a job shop has its stored routes.

    ``instance`` is a job shop named ``dmu01`` to ``dmu80``. Its route digest
    (:func:`~dispatchatlas.benchmarks.taillard.job_shop_route_digest`) must equal the entry of
    :data:`JOB_SHOP_ROUTE_DIGESTS` for its id.

    Args:
        instance: The job shop to check. Its id names the instance.

    Raises:
        ModelError: If the id is not one of ``dmu01`` to ``dmu80``, the problem is not a job
            shop, or its routes differ from the stored ones. The ``issues`` of a difference
            hold the stored digest and the digest found.
    """
    stored = JOB_SHOP_ROUTE_DIGESTS.get(instance.id)
    if stored is None:
        raise ModelError(
            f"{instance.id!r} is not one of the job shops dmu01 to dmu80, so no route digest "
            "is stored for it"
        )
    found = job_shop_route_digest(instance)
    if found != stored:
        raise ModelError(
            f"job shop {instance.id!r} does not have the routes that the library stored for it",
            issues=(f"stored route digest: {stored}", f"digest of the routes found: {found}"),
        )


def _name(instance: str | int) -> str:
    if isinstance(instance, bool):
        number = None
    elif isinstance(instance, int):
        number = instance
    else:
        match = _NAME.fullmatch(str(instance).strip().lower())
        number = int(match.group(1)) if match else None
    if number is None or not 1 <= number <= 80:
        raise ConfigError(
            f"unknown Demirkol instance {instance!r}; use dmu01 to dmu80 or a number from 1 to 80"
        )
    return f"dmu{number:02d}"


@cache
def _build(name: str) -> Problem:
    number = int(name[3:])
    jobs, machines = _SIZES[((number - 1) // 5) % 8]
    text = (
        files("dispatchatlas.data")
        .joinpath("instances", "demirkol-job-shop", f"{name}.txt")
        .read_bytes()
        .decode("ascii")
    )
    routes = _routes(name, text, jobs, machines)
    metadata: dict[str, Scalar] = {
        "suite": "demirkol-job-shop",
        "instance": name,
        "problem_class": "job-shop",
        "jobs": jobs,
        "machines": machines,
        "routing": "random" if number <= 40 else "two-sets",
        "source": DEMIRKOL_SOURCE,
    }
    problem = job_shop_problem(routes, machines=machines, problem_id=name, metadata=metadata)
    check_demirkol_job_shop(problem)
    return problem


def _routes(name: str, text: str, jobs: int, machines: int) -> list[list[tuple[int, float]]]:
    """Parse a stored file and check its shape."""
    tokens = text.split()
    if not all(token.isdigit() for token in tokens):
        raise parse_error(name, "the file holds a token that is not a non-negative integer")
    if [int(tokens[0]), int(tokens[1])] != [jobs, machines] or len(
        tokens
    ) != 2 + 2 * jobs * machines:
        raise parse_error(name, f"the file is not a {jobs} x {machines} job shop")
    values = [int(token) for token in tokens[2:]]
    routes: list[list[tuple[int, float]]] = []
    for job in range(jobs):
        route = [
            (values[2 * (job * machines + p)], float(values[2 * (job * machines + p) + 1]))
            for p in range(machines)
        ]
        if sorted(machine for machine, _ in route) != list(range(machines)):
            raise parse_error(name, f"job {job} does not visit every machine once")
        if any(time < 1 for _, time in route):
            raise parse_error(name, f"job {job} has a processing time below 1")
        routes.append(route)
    return routes


# ------------------------------------------------------------------ route digests
#
# The route digest of each job shop, as job_shop_route_digest computes it from the stored file.

JOB_SHOP_ROUTE_DIGESTS: Mapping[str, str] = MappingProxyType(
    {
        "dmu01": "c3c5f24aff359bd2e7b9c78cd28b6b78bb7b9040537f5397c90277e5f3802844",
        "dmu02": "19ff081039b41fc4832327ad60232df9b6481a194177f51c64fa78f8a0950778",
        "dmu03": "ac6d2a24d04f67d449995b57a146956bb5e2ea06191ff4877038bea069af0447",
        "dmu04": "39576eb0ca3ec377495affab9c55c0b1fbf4d11b65e05206c2700194e678a8c5",
        "dmu05": "7e12e17d1654425711f7cd960b000f25370f95400c75c5b72cd422d16388219d",
        "dmu06": "bef62be891acd76f83d10f9e044c25298e3e1de0ec6ca421d18392a8180e82db",
        "dmu07": "120252dd3511118bcd274a96fa4a133371ea81eb3d444530271e42b138c5a04b",
        "dmu08": "10550cf236d7c33a958bef374aa1166034930eb689c1df116f08df8725ec9dd9",
        "dmu09": "d3860bd1c0f3f6b5902089a800fbbf30817da553a25b4dab0e5be0e2f12e5b05",
        "dmu10": "c77c834b61cfec61662e1aa8302703b2fd0e73ff6c6221c73744634c933c26a4",
        "dmu11": "c346e971d2b1a8eda79940eedaab5d8c5cfdd654ead9fc8559c0bfa1ff345d7c",
        "dmu12": "b3a4070e1b6b014548e36e0ee77a0f80e2a220d7c55c7c5b7ab3eace6af805d1",
        "dmu13": "6b546ed7230dbd59647a4dcb127445efb6f0d712cbf2cf1581ef3a1460ac5a86",
        "dmu14": "5db1fa05a309e59b1e1c0b08549b2458b3f173c5c7cfdf274e446d69e65a1efd",
        "dmu15": "df31a044458cbbe7f17170f5ee553e083ce1a51b5ca76770d5a36c2509a6855a",
        "dmu16": "d67d7af1088089224342d8c9dd9e850d7f063c883fc8aeb5390b0191bf982234",
        "dmu17": "16a0083a5565770f6846894eb045d01d8b41ba082fd103997b77e115236af48b",
        "dmu18": "70bc7cc692d8bb78b4cae58ec856c64c6369e42262722194891f33be3fe99d62",
        "dmu19": "1a6fe50343924f319761f1807aae98ef96a812776917960e92df105763383cdd",
        "dmu20": "ee0ab34fb3ae844bb9066353a91aa2001794df463a2e151ea385e28066f8d884",
        "dmu21": "2e1be727272117bacde1f5e972910d005488942b9ab6a3ac1a2d2bdbb9ae57be",
        "dmu22": "a373e0cdd3b81d7486d548ec19aebf67d4d8f875f8ac2beb4aca9171ac3d1db0",
        "dmu23": "e72e1d153dff18e025b9a3bcb0d74879c292cb5f4f86d6f566df58fff2ab522c",
        "dmu24": "f05d7179c18edec6191b8cc40a182657b79415b015e2bc791b3d795342dfb32b",
        "dmu25": "5e296d224053499f4c6808a2bdc8be3bd6b6604a46bc59a036c66cb1f87eb99d",
        "dmu26": "a8f025d8395c0838261dccc3ec4ee4dab68c7f18700dd9cdbd328c8aa0b52827",
        "dmu27": "e08a6ab0e3e31096f989c0e7e02f64557944ccb203dfc06aa9126df269d396bb",
        "dmu28": "74b7bfd98e72ff165ba9a97fcc31950bb3b1d0aea57fb1e4f7f4216009584116",
        "dmu29": "c5015cdbc15594f1c7f1931f1a11ad9ab09e675308a74b8b239e0654cf1da2cb",
        "dmu30": "93d8a6cc1d11af0325c4a9f71c7f03f7ba051d787d8fb27673aed705b2db6f85",
        "dmu31": "b60ed6aa4ccee45640b0065ba3fb69437a2a365559b6ab8ebdb1f966d18bf064",
        "dmu32": "89b6d6e85d2d9f09150552f5db2f68e40de08a5dcfa3d9cbd78b4d04ec98d3d2",
        "dmu33": "e3be88d64906aa744ad68aba0db6f7b086860cc40531547139b8147956bbf155",
        "dmu34": "3bc16ab0864e8de9ded4bfdc3d0f218e70e6ec8a8ea218f120658c9f69f7abd6",
        "dmu35": "060f060b79197bcaf16e540dbaf886c198d1292891f0ee0bc0dbbbe86ae2c93d",
        "dmu36": "309fba95b9adb2af81f404da2046484f13b7953d10aae2f2485a5d034f8d75b7",
        "dmu37": "f0386de26d867b9dbc4115fc5fb95b1ee4cb33c5cdf3a0eec2875a3636ac7d3d",
        "dmu38": "b57efbdcccde7dee87aa6a2d9a4453a62f78a8b43d3c8b22a94f21a5031ee07f",
        "dmu39": "e6f197cf8baddab37549d9acb5d4641bc7e2691dd5864e72ba662eb4c5b8fb48",
        "dmu40": "41f808960132e24b4d037ae5e3a23e447725815a48266ed171b4715ef3850ef2",
        "dmu41": "3ce20038957710a95741cdf5cd6e7bd1e6125f79084a7065dcf10e07f3d76bf4",
        "dmu42": "826ec4c7eda78869b9c43b804bc543310b483925b0648fb311ed78f6927a8eff",
        "dmu43": "7bc37e7b5cd9807fbd4730e3569909cd14e8f69f8da16be162598b300e7f0ed4",
        "dmu44": "77e5455405917e45b41422a1de30f6ee8cae843cda781ba6e024c6eaee5f1d36",
        "dmu45": "f7f2d06b5afc0a0e0bc60d2f60081833654bd0d836aa2142b8e2c29ebf6ea530",
        "dmu46": "1242097a7b15b55d547c5b0a070fac8538774334e6f6d3de763e4f83579adec9",
        "dmu47": "8d9193fd596fdeaf133754f8591e2bbc7447aa13531118519858b9418667c026",
        "dmu48": "59beff973e0a1b7967282331789880b747caa4031a8c71b6ad19e7a8c291177e",
        "dmu49": "32ddfd39ddb095c00e1af32010a0fdda6663fbfbc47d083e91358635694b007c",
        "dmu50": "edd761da3d3c6a29c0ae1a47f4454a169dc9f3f7f24f6ad9ca7c181e42906f25",
        "dmu51": "73c7cc700d3c43fe51cfe756416aed3693ac53f6f725f6aef5c74fe3783beed5",
        "dmu52": "9227ea4e27d9473095565ef58edee17539c95a6ccae16ecfb6ca8612fbad68c3",
        "dmu53": "2e08e144d4379a09d4ebd9255ace0014ec6a8a7fc63607452d0a683c61f466b6",
        "dmu54": "e7a126df17b204d9d0841fc78ca6555db3fb927e87a8ba6c90950ba1341d7fc9",
        "dmu55": "f799980291fe48e903a254e9e2932ad83b5c04cf61988861caa22f864bb8f7c5",
        "dmu56": "8bed62a24ad304e5c6accbb71b45cd16fd0014975a0c934f8da9b65d3a21a442",
        "dmu57": "51cdc2ab65c712657616b01da68092059fc1c44cca646b31194c06631984ac30",
        "dmu58": "205c776ba4cbc86a38f12dc27d0a1a61363ef5fc16fc36396f55e0cba36063cf",
        "dmu59": "67539d8104088756c663ef38b0a450eb6e399cb8ec2a0c16d4a4d712388a5fc1",
        "dmu60": "b94e6dd6a5bd1414610c5d902709ca69d396b624ab515b68578f1afb447ae1af",
        "dmu61": "d0c9b4b10adfe851eb484aa7b3f37a4d33f2dad68c524007a53a2d247570656f",
        "dmu62": "8854256522eeca57a410d4e342d458816486f9fc3aa2cc7a81ea70e196c0ed3c",
        "dmu63": "140bb99e568115c56e6c3f5c66eb934e16a3b2233dca0abe341006ecf6ad319f",
        "dmu64": "1e5ecfd3df39188bce5781cdc730f3f81f91b9998fe7a2b3822a3f50a0cc2ed0",
        "dmu65": "74dbb30d4855e6a52e9d5bc649eeb673df3cc27713932c1e00c0edbb44a7a84a",
        "dmu66": "1c6457efd875c1797df007b8cb1bb4db64e994a8bf17fbb97ccf902e7d143b38",
        "dmu67": "d6f21811c4697be2cc43fd0714beb3392543fd85248981bdce901cf3e92d58ad",
        "dmu68": "55a72acc1d1537f978b6d763366f94e8d04d9ac77c8f0621eecc399ac35812a1",
        "dmu69": "3fbc5c9269636767e20a082a150c9b3a611c2c464113e4c39ef9203bdd0ddaaf",
        "dmu70": "23c682427560fd7b990d1c2548381c316464d63096f5ac62175c00a228d192f4",
        "dmu71": "9b8a4910f7e1b17bff340de09b2dde93a0b5c66af57db2053b3dab315fbbdd19",
        "dmu72": "52a464eed367a9980c70ac8e99bd2a8a7d99b5d838e6b877e570afa05a9e6f0f",
        "dmu73": "a5d95fb846935332494e77a7e4c79b09d80de26276e53240d5d2c25d23aa083e",
        "dmu74": "48e414dd22f4a1e653df1ef77bd18e25f423951a9c6e39bbeab7d966405e72d5",
        "dmu75": "7e812bca1fd4a88416725ebda85f0bc8e84bfddfa3eca59df695bbc30b74bb09",
        "dmu76": "a91b5bd386a804ff2924a59ddaa82d435dde7d307fd1b8f7417440dbe0528fef",
        "dmu77": "b72dec0fb555109455a929059e8780901d15451bc0cda27ffbb962a7fdea54e6",
        "dmu78": "ec7871053aa0d88b17efd49ff47c5b3408a86f752bcb21c3dec5239f1e695214",
        "dmu79": "d903d5a952dfc5869b10feaa3b42299c352f7f1c2bf7d9b8febad91c018a2b42",
        "dmu80": "bff2edceb4a46e1687cc064d45441043393e6d70d66511eaeb2b689e3e0e9fcd",
    }
)
"""The route digest of each job shop dmu01 to dmu80, by name, read-only."""
