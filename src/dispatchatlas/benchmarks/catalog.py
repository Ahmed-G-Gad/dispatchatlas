"""Named instance catalogs, and writing instance sets to disk reproducibly.

A :class:`Catalog` is a declarative list of registered families, each with
parameter values and an instance count. :func:`build_catalog` generates every
instance, deriving each seed from one root seed with
:func:`~dispatchatlas.core.seeds.derive_seed` under the namespace
``<catalog>/<entry>`` and the instance index, so an instance never depends on
the entries listed before it. The instance ids are ``<entry>-<index>``.

Built-in catalogs:

- ``smoke``: two tiny instances of every built-in family, for fast checks.
- ``continuum``: the continuum families on a small, medium, and large
  platform, five instances each.
- ``workflow``: random task graphs over part of the design of Topcuoglu,
  Hariri and Wu (2002): 20, 60, and 100 tasks, communication-to-computation
  ratios 0.1, 1, and 10, and shapes 0.5, 1, and 2.

:func:`materialize` writes a set of problems to a directory as
``problems/<id>.json`` plus a ``manifest.json`` that records, for every
file, its SHA-256 digest, its family, its parameters, and its seed. Nothing
in either depends on the clock or the machine, so writing the same problems
twice gives identical bytes. :func:`load_directory` reads the set back and,
by default, checks every file against its digest.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dispatchatlas.benchmarks._common import check_seed
from dispatchatlas.benchmarks.registry import FamilyRegistry, default_registry
from dispatchatlas.core.errors import ConfigError, SerializationError
from dispatchatlas.core.io import atomic_write_bytes, atomic_write_json, read_json
from dispatchatlas.core.logs import get_logger
from dispatchatlas.core.model import Problem
from dispatchatlas.core.seeds import derive_seed
from dispatchatlas.core.serialization import dumps_problem, loads_problem

__all__ = [
    "MANIFEST_FORMAT",
    "Catalog",
    "CatalogEntry",
    "build_catalog",
    "catalogs",
    "load_directory",
    "materialize",
]

MANIFEST_FORMAT = "dispatchatlas.catalog"
"""Value of the ``format`` field of a manifest."""

_MANIFEST_VERSION = 1
_MANIFEST = "manifest.json"
_PROBLEMS = "problems"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class CatalogEntry:
    """One family of a catalog, with its parameters and instance count.

    Attributes:
        family: Id of a registered benchmark family.
        parameters: Parameter values passed to the family; the others keep
            their defaults.
        instances: Number of instances to generate.
        name: Name of the entry, unique in its catalog and used in instance
            ids and seed namespaces. Defaults to the family id.
    """

    family: str
    parameters: Mapping[str, Any] = field(default_factory=dict, hash=False)
    instances: int = 1
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise ConfigError("a catalog entry needs a family id")
        if isinstance(self.instances, bool) or not isinstance(self.instances, int):
            raise ConfigError("the instance count of a catalog entry must be an integer")
        if self.instances < 1:
            raise ConfigError(f"catalog entry {self.label!r} needs at least one instance")
        if not _SAFE_ID.match(self.label):
            raise ConfigError(f"catalog entry name {self.label!r} is not a safe identifier")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @property
    def label(self) -> str:
        """The entry's name, or its family id when it has none."""
        return self.name if self.name is not None else self.family


@dataclass(frozen=True)
class Catalog:
    """A named, declarative set of benchmark instances.

    Attributes:
        name: Name of the catalog, used in seed namespaces.
        description: One sentence on what the catalog is for.
        entries: The families to generate, with parameters and counts.
    """

    name: str
    description: str
    entries: tuple[CatalogEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _SAFE_ID.match(self.name):
            raise ConfigError(f"catalog name {self.name!r} is not a safe identifier")
        entries = tuple(self.entries)
        labels = [entry.label for entry in entries]
        if not entries or len(set(labels)) != len(labels):
            raise ConfigError(f"catalog {self.name!r} needs entries with distinct names")
        object.__setattr__(self, "entries", entries)

    @property
    def families(self) -> tuple[str, ...]:
        """Ids of the families the catalog uses, sorted."""
        return tuple(sorted({entry.family for entry in self.entries}))

    @property
    def size(self) -> int:
        """Number of instances the catalog generates."""
        return sum(entry.instances for entry in self.entries)


_SMOKE = Catalog(
    name="smoke",
    description="Two tiny instances of every built-in family, for fast checks.",
    entries=tuple(
        CatalogEntry(family, parameters, instances=2)
        for family, parameters in (
            ("flow-shop", {"jobs": 4, "machines": 3}),
            ("job-shop", {"jobs": 3, "machines": 3}),
            ("open-shop", {"jobs": 3, "machines": 3}),
            (
                "flexible-job-shop",
                {"jobs": 3, "machines": 3, "min_operations": 2, "max_operations": 3},
            ),
            ("hybrid-flow-shop", {"jobs": 4, "stages": 2, "max_machines": 2}),
            ("parallel-machines", {"jobs": 6, "machines": 2}),
            ("rcpsp", {"activities": 8, "resources": 2}),
            ("multi-mode-rcpsp", {"activities": 6, "resources": 2}),
            ("workflow", {"tasks": 8, "processors": 3}),
            ("continuum-offloading", {"applications": 6, "devices": 2}),
            ("continuum-workflow", {"applications": 2, "tasks": 5, "devices": 2}),
            ("continuum-pipeline", {"streams": 2, "frames": 2, "stages": 2, "devices": 2}),
            ("taillard-job-shop-seeded", {"jobs": 3, "machines": 3}),
        )
    ),
)


def _continuum_entries() -> tuple[CatalogEntry, ...]:
    """Return every continuum family on a small, a medium, and a large platform."""
    ladder = {
        # size: node counts, then (requests, arrival rate), (workflows, tasks), (streams, stages)
        "small": ((4, 2, 1, 1), (20, 4.0), (4, 10), (4, 3)),
        "medium": ((8, 4, 2, 1), (50, 8.0), (8, 20), (8, 4)),
        "large": ((16, 8, 4, 2), (100, 16.0), (16, 30), (16, 5)),
    }
    entries: list[CatalogEntry] = []
    for size, (nodes, (requests, rate), (workflows, tasks), (streams, stages)) in ladder.items():
        counts = ("devices", "edge_nodes", "fog_nodes", "cloud_nodes")
        platform = dict(zip(counts, nodes, strict=True))
        entries.append(
            CatalogEntry(
                "continuum-offloading",
                {**platform, "applications": requests, "arrival_rate": rate},
                instances=5,
                name=f"continuum-offloading-{size}",
            )
        )
        for structure in ("layered", "fork-join"):
            entries.append(
                CatalogEntry(
                    "continuum-workflow",
                    {**platform, "applications": workflows, "tasks": tasks, "structure": structure},
                    instances=5,
                    name=f"continuum-workflow-{structure}-{size}",
                )
            )
        entries.append(
            CatalogEntry(
                "continuum-pipeline",
                {**platform, "streams": streams, "frames": 10, "stages": stages},
                instances=5,
                name=f"continuum-pipeline-{size}",
            )
        )
    return tuple(entries)


_CONTINUUM = Catalog(
    name="continuum",
    description="The continuum families on small, medium, and large platforms.",
    entries=_continuum_entries(),
)

_WORKFLOW = Catalog(
    name="workflow",
    description="Random task graphs over part of the design of Topcuoglu, Hariri and Wu (2002).",
    entries=tuple(
        CatalogEntry(
            "workflow",
            {"tasks": tasks, "ccr": ccr, "shape": shape, "out_degree": 3, "processors": 4},
            instances=2,
            name=f"workflow-{tasks}-ccr{ccr:g}-shape{shape:g}",
        )
        for tasks in (20, 60, 100)
        for ccr in (0.1, 1.0, 10.0)
        for shape in (0.5, 1.0, 2.0)
    ),
)

_CATALOGS: Mapping[str, Catalog] = MappingProxyType(
    {catalog.name: catalog for catalog in (_SMOKE, _CONTINUUM, _WORKFLOW)}
)


def catalogs() -> Mapping[str, Catalog]:
    """Return the built-in catalogs keyed by name."""
    return _CATALOGS


def build_catalog(
    catalog: str | Catalog, *, seed: int = 0, registry: FamilyRegistry | None = None
) -> list[Problem]:
    """Generate every instance of a catalog.

    Instance ``i`` of an entry uses the seed
    ``derive_seed(seed, "<catalog>/<entry>", i)`` and gets the id
    ``<entry>-<i>``. Its metadata records that seed under ``seed``, as the
    built-in families do; for a family that records none, it is added, at
    the price of validating the problem a second time.

    Args:
        catalog: Name of a built-in catalog, or a catalog.
        seed: Root seed of the catalog.
        registry: Registry to look families up in. Defaults to the default
            registry with plugins.

    Returns:
        The problems, entry by entry.

    Raises:
        ConfigError: If the catalog, a family, or a parameter is unknown, or
            ``seed`` is not a non-negative integer.
    """
    try:
        root = check_seed(seed)
    except ValueError as error:
        raise ConfigError(f"invalid catalog seed: {error}") from error
    if isinstance(catalog, str):
        if catalog not in _CATALOGS:
            raise ConfigError(f"unknown catalog {catalog!r}; known: {', '.join(_CATALOGS)}")
        catalog = _CATALOGS[catalog]
    registry = registry if registry is not None else default_registry()
    problems: list[Problem] = []
    for entry in catalog.entries:
        generator = registry.create(entry.family, **entry.parameters)
        for index in range(entry.instances):
            instance_seed = derive_seed(root, f"{catalog.name}/{entry.label}", index)
            problem = generator.generate(instance_seed)
            problem_id = f"{entry.label}-{index}"
            if problem.metadata.get("seed") == instance_seed:
                problems.append(problem.renamed(problem_id))
            else:
                metadata = {**problem.metadata, "seed": instance_seed}
                problems.append(replace(problem, id=problem_id, metadata=metadata))
    _LOG.debug("built catalog %r: %d problems", catalog.name, len(problems))
    return problems


def materialize(
    problems: Iterable[Problem], directory: str | Path, *, overwrite: bool = False
) -> Path:
    """Write problems and their manifest to a directory.

    Each problem goes to ``problems/<id>.json`` in the format of
    :func:`~dispatchatlas.core.serialization.dumps_problem`. The manifest,
    ``manifest.json``, lists every file with its SHA-256 digest, the problem's
    family, its seed (the metadata ``seed`` entry, ``null`` when absent), and
    its parameters: the rest of the problem metadata, which is where the
    built-in families record their parameters. All files are written
    atomically, with sorted keys where the format allows.

    Args:
        problems: The problems. Their ids must be distinct and consist of
            letters, digits, ``.``, ``_``, and ``-``, starting with a letter
            or digit.
        directory: Target directory. Created when missing.
        overwrite: Allow a non-empty directory. The manifest is replaced and
            JSON files in ``problems/`` that the new set does not contain are
            removed; other files are left alone.

    Returns:
        The directory.

    Raises:
        SerializationError: If an id is unsafe or repeated, or the directory
            is not empty and ``overwrite`` is false.
    """
    items = list(problems)
    ids = [problem.id for problem in items]
    for problem_id in ids:
        if not _SAFE_ID.match(problem_id):
            raise SerializationError(f"problem id {problem_id!r} is not a safe file name")
    if len(set(ids)) != len(ids):
        raise SerializationError("problem ids must be distinct")
    root = Path(directory)
    folder = root / _PROBLEMS
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise SerializationError(f"{root} is not empty; pass overwrite=True to replace it")
        if folder.is_dir():
            keep = {f"{problem_id}.json" for problem_id in ids}
            for stale in sorted(folder.glob("*.json")):
                if stale.name not in keep:
                    stale.unlink()
    entries: list[dict[str, Any]] = []
    for problem in items:
        data = dumps_problem(problem).encode("utf-8")
        name = f"{_PROBLEMS}/{problem.id}.json"
        atomic_write_bytes(root / name, data)
        metadata = problem.metadata
        entries.append(
            {
                "id": problem.id,
                "file": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "family": problem.family,
                "seed": metadata.get("seed"),
                "parameters": {key: value for key, value in metadata.items() if key != "seed"},
            }
        )
    manifest = {"format": MANIFEST_FORMAT, "version": _MANIFEST_VERSION, "problems": entries}
    atomic_write_json(root / _MANIFEST, manifest)
    _LOG.debug("materialized %d problems in %s", len(items), root)
    return root


def load_directory(directory: str | Path, *, verify: bool = True) -> list[Problem]:
    """Read the problems a manifest lists, in manifest order.

    Args:
        directory: A directory written by :func:`materialize`.
        verify: Check every file against the SHA-256 digest in the manifest.

    Returns:
        The problems.

    Raises:
        SerializationError: If the manifest is missing or malformed, a file
            is missing, a digest does not match, or a file holds a problem
            with another id.
    """
    root = Path(directory)
    path = root / _MANIFEST
    if not path.is_file():
        raise SerializationError(f"{root} has no {_MANIFEST}")
    manifest = read_json(path)
    if not isinstance(manifest, Mapping) or manifest.get("format") != MANIFEST_FORMAT:
        raise SerializationError(f"{path} is not a {MANIFEST_FORMAT} manifest")
    if manifest.get("version") != _MANIFEST_VERSION:
        raise SerializationError(
            f"unsupported manifest version {manifest.get('version')!r}; "
            f"this release reads version {_MANIFEST_VERSION}"
        )
    entries = manifest.get("problems")
    if not isinstance(entries, list):
        raise SerializationError(f"{path} has no list of problems")
    problems: list[Problem] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
            raise SerializationError(f"{path} has a malformed problem entry")
        problem_id = entry["id"]
        name = f"{_PROBLEMS}/{problem_id}.json"
        if not _SAFE_ID.match(problem_id) or entry.get("file") != name:
            raise SerializationError(f"manifest entry {problem_id!r} names an unexpected file")
        try:
            data = (root / name).read_bytes()
        except FileNotFoundError:
            raise SerializationError(f"{root / name} is missing") from None
        if verify and hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            raise SerializationError(f"{root / name} does not match its manifest digest")
        problem = loads_problem(data.decode("utf-8"))
        if problem.id != problem_id:
            raise SerializationError(f"{root / name} holds problem {problem.id!r}")
        problems.append(problem)
    return problems
