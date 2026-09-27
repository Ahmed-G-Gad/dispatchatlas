"""Published benchmark suites: their instances, sources, references, and loaders.

A :class:`Suite` describes one published instance set. Two kinds exist:

- **Generated** suites, Taillard's (1993) flow, job, and open shops, are
  rebuilt exactly from their published seeds by
  :mod:`dispatchatlas.benchmarks.taillard`, so they load offline.
- **File** suites, the OR-Library job and flow shops, Brandimarte's flexible
  job shops, and the PSPLIB projects, are read from files that the user
  downloads from :attr:`Suite.source` into a directory. The library does not
  bundle these files.

Example::

    from dispatchatlas.benchmarks import suites
    ta001 = suites.load("taillard-flow-shop", "ta001")
    ft06 = suites.load("orlib-job-shop", "ft06", directory="downloads/orlib")
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, partial
from pathlib import Path
from types import MappingProxyType

from dispatchatlas.benchmarks.formats import (
    read_fjs,
    read_orlib_flow_shop,
    read_orlib_job_shop,
    read_psplib,
)
from dispatchatlas.benchmarks.taillard import (
    open_shop_instances,
    taillard_flow_shop,
    taillard_job_shop,
    taillard_open_shop,
)
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.model import Problem

__all__ = ["Bound", "Suite", "load", "suite", "suites"]

type Loader = Callable[[str, Path | None], Problem]
"""Builds one instance from its name and, for file suites, a directory."""

_ORLIB = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"


@dataclass(frozen=True, slots=True)
class Bound:
    """Published bounds on the optimal value of one instance.

    Attributes:
        lower: Lower bound, if published.
        upper: Value of the best published schedule, if any.
        optimal: Whether ``upper`` is proven optimal.
        objective: Objective the bounds refer to.
    """

    lower: float | None = None
    upper: float | None = None
    optimal: bool = False
    objective: str = "makespan"


@dataclass(frozen=True)
class Suite:
    """One published benchmark suite.

    Attributes:
        id: Registry key, for example ``"taillard-flow-shop"``.
        name: Human-readable name.
        problem_class: Problem class of every instance, as in
            :attr:`~dispatchatlas.benchmarks.registry.FamilyInfo.problem_class`.
        instances: Instance names, in the order of the source.
        source: URL of the published instances or of the seeds that generate
            them.
        references: Bibliography keys of the works that define the instances.
        loader: Builds an instance from its name and, for file suites, the
            directory that holds the downloaded files.
        files: What to download from :attr:`source` into the directory passed
            to :meth:`load`. ``None`` for suites generated offline.
        bounds: Bounds published with the instances, keyed by instance name.
        notes: Further facts about the suite.
    """

    id: str
    name: str
    problem_class: str
    instances: tuple[str, ...]
    source: str
    references: tuple[str, ...]
    loader: Loader = field(repr=False, compare=False)
    files: str | None = None
    bounds: Mapping[str, Bound] = field(default_factory=dict, hash=False, compare=False)
    notes: str = ""

    @property
    def generated(self) -> bool:
        """Whether the instances are generated offline rather than read from files."""
        return self.files is None

    def load(self, instance: str, directory: str | Path | None = None) -> Problem:
        """Return one instance of the suite.

        Args:
            instance: Instance name; case does not matter.
            directory: Directory holding the downloaded files, searched
                recursively. Required for file suites, ignored otherwise.

        Raises:
            ConfigError: If the instance is unknown, or a file suite gets no
                directory or cannot find the instance's file in it.
        """
        names = {name.lower(): name for name in self.instances}
        name = names.get(str(instance).strip().lower())
        if name is None:
            first, last = self.instances[0], self.instances[-1]
            raise ConfigError(
                f"suite {self.id!r} has no instance {instance!r}; "
                f"its {len(self.instances)} instances run from {first} to {last}"
            )
        if self.generated:
            return self.loader(name, None)
        if directory is None:
            raise ConfigError(
                f"suite {self.id!r} is read from files: download {self.files} from "
                f"{self.source} and pass directory="
            )
        folder = Path(directory)
        if not folder.is_dir():
            raise ConfigError(f"{folder} is not a directory")
        return self.loader(name, folder)


def suites() -> tuple[Suite, ...]:
    """Return every suite: the generated ones first, then the file suites."""
    return tuple(_registry().values())


def suite(suite_id: str) -> Suite:
    """Return the suite with the given id.

    Raises:
        ConfigError: If no suite has that id.
    """
    registry = _registry()
    try:
        return registry[suite_id]
    except KeyError:
        raise ConfigError(f"unknown suite {suite_id!r}; known: {sorted(registry)}") from None


def load(suite_id: str, instance: str, directory: str | Path | None = None) -> Problem:
    """Return one instance of a suite. See :meth:`Suite.load`."""
    return suite(suite_id).load(instance, directory)


# ------------------------------------------------------------------ loaders


def _generated(
    instance: str, directory: Path | None, *, build: Callable[[str], Problem]
) -> Problem:
    del directory
    return build(instance)


def _find(directory: Path | None, file_name: str) -> str:
    """Return the text of the one file called ``file_name`` (any case) under ``directory``."""
    if directory is None:  # pragma: no cover - Suite.load passes a directory
        raise ConfigError("a file suite needs a directory")
    wanted = file_name.lower()
    matches = sorted(
        path for path in directory.rglob("*") if path.name.lower() == wanted and path.is_file()
    )
    if not matches:
        raise ConfigError(f"no file {file_name!r} under {directory}")
    if len(matches) > 1:
        listed = ", ".join(str(path) for path in matches)
        raise ConfigError(f"several files {file_name!r} under {directory}: {listed}")
    return matches[0].read_bytes().decode("utf-8", errors="replace")


def _orlib_file(
    instance: str, directory: Path | None, *, file_name: str, read: Callable[..., Problem]
) -> Problem:
    return read(_find(directory, file_name), instance=instance)


def _fjs_file(instance: str, directory: Path | None) -> Problem:
    return read_fjs(_find(directory, f"{instance}.fjs"), problem_id=instance)


def _psplib_file(instance: str, directory: Path | None) -> Problem:
    return read_psplib(_find(directory, f"{instance}.sm"), problem_id=instance)


# ------------------------------------------------------------------ registry


def _numbered(prefix: str, numbers: Sequence[int], width: int = 2) -> tuple[str, ...]:
    return tuple(f"{prefix}{number:0{width}d}" for number in numbers)


def _psplib_suite(size: int, parameters: int) -> Suite:
    names = tuple(
        f"j{size}{parameter}_{instance}"
        for parameter in range(1, parameters + 1)
        for instance in range(1, 11)
    )
    return Suite(
        id=f"psplib-j{size}",
        name=f"PSPLIB J{size} single-mode projects",
        problem_class="rcpsp",
        instances=names,
        source="https://www.om-db.wi.tum.de/psplib/getdata.php?mode=sm",
        references=("kolisch1997psplib", "kolisch1995characterization"),
        loader=_psplib_file,
        files=f"j{size}.sm.zip, extracted (files {names[0]}.sm to {names[-1]}.sm)",
        notes=f"{parameters} ProGen parameter combinations with 10 instances each, "
        f"{size} activities and 4 renewable resources. The source also lists optimal "
        "values, lower bounds, and best-known solutions.",
    )


@cache
def _registry() -> Mapping[str, Suite]:
    open_shops = open_shop_instances()
    entries = (
        Suite(
            id="taillard-flow-shop",
            name="Taillard permutation flow shops",
            problem_class="flow-shop",
            instances=_numbered("ta", range(1, 121), width=3),
            source=_ORLIB + "flowshop2.txt",
            references=("taillard1993benchmarks", "beasley1990orlibrary"),
            loader=partial(_generated, build=taillard_flow_shop),
            notes="120 instances from 20 jobs x 5 machines to 500 x 20, generated "
            "exactly from the published seeds.",
        ),
        Suite(
            id="taillard-job-shop",
            name="Taillard job shops",
            problem_class="job-shop",
            instances=_numbered("ta", range(1, 81)),
            source=_ORLIB + "jobshop2.txt",
            references=("taillard1993benchmarks", "beasley1990orlibrary"),
            loader=partial(_generated, build=taillard_job_shop),
            notes="80 instances from 15 jobs x 15 machines to 100 x 20, generated "
            "exactly from the published seeds.",
        ),
        Suite(
            id="taillard-open-shop",
            name="Taillard open shops",
            problem_class="open-shop",
            instances=tuple(item.name for item in open_shops),
            source=_ORLIB + "openshop.txt",
            references=("taillard1993benchmarks", "beasley1990orlibrary"),
            loader=partial(_generated, build=taillard_open_shop),
            bounds=MappingProxyType(
                {
                    item.name: Bound(item.lower_bound, item.upper_bound, item.optimal)
                    for item in open_shops
                }
            ),
            notes="60 instances from 4 jobs x 4 machines to 20 x 20, generated exactly "
            "from the published seeds. The bounds are those listed with the seeds.",
        ),
        Suite(
            id="orlib-job-shop",
            name="OR-Library job shops",
            problem_class="job-shop",
            instances=(
                *(f"abz{number}" for number in range(5, 10)),
                "ft06",
                "ft10",
                "ft20",
                *_numbered("la", range(1, 41)),
                *_numbered("orb", range(1, 11)),
                *_numbered("swv", range(1, 21)),
                *(f"yn{number}" for number in range(1, 5)),
            ),
            source=_ORLIB + "jobshop1.txt",
            references=(
                "beasley1990orlibrary",
                "adams1988shifting",
                "fisher1963probabilistic",
                "lawrence1984resource",
                "applegate1991computational",
                "storer1992new",
                "yamada1992genetic",
            ),
            loader=partial(_orlib_file, file_name="jobshop1.txt", read=read_orlib_job_shop),
            files="jobshop1.txt",
            notes="82 classic job shops in one file: abz5-9, ft06, ft10, ft20, la01-40, "
            "orb01-10, swv01-20, and yn1-4.",
        ),
        Suite(
            id="orlib-flow-shop",
            name="OR-Library flow shops",
            problem_class="flow-shop",
            instances=(
                *(f"car{number}" for number in range(1, 9)),
                "hel1",
                "hel2",
                *_numbered("reC", range(1, 42, 2)),
            ),
            source=_ORLIB + "flowshop1.txt",
            references=(
                "beasley1990orlibrary",
                "carlier1978ordonnancements",
                "heller1960numerical",
                "reeves1995genetic",
            ),
            loader=partial(_orlib_file, file_name="flowshop1.txt", read=read_orlib_flow_shop),
            files="flowshop1.txt",
            notes="31 permutation flow shops in one file: car1-8, hel1-2, and the "
            "odd-numbered reC01-41; each even-numbered reC instance reverses the "
            "machine order of the preceding one.",
        ),
        Suite(
            id="brandimarte",
            name="Brandimarte flexible job shops",
            problem_class="flexible-job-shop",
            instances=_numbered("mk", range(1, 11)),
            source="https://people.idsia.ch/~monaldo/fjsp.html",
            references=("brandimarte1993routing", "mastrolilli2000effective"),
            loader=_fjs_file,
            files="TextData.zip from the page, extracted (Mk01.fjs to Mk10.fjs)",
            notes="The Mk01-Mk10 instances in the FJSPLib format of Mastrolilli and "
            "Gambardella, with machines numbered from 1.",
        ),
        _psplib_suite(30, 48),
        _psplib_suite(60, 48),
        _psplib_suite(90, 48),
        _psplib_suite(120, 60),
    )
    return MappingProxyType({entry.id: entry for entry in entries})
