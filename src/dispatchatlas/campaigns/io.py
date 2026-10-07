"""Reading and writing campaign files.

Campaign files are TOML, read with the standard library's :mod:`tomllib`, or
JSON. One strict reader serves both: every table rejects keys it does not
know, every value must have its documented type (``true`` is not a number and
``"false"`` is not a boolean), a JSON object may not repeat a key, and the
document must declare ``schema_version = 1``. Relative paths are resolved
against the directory of the file.

Example::

    schema_version = 1
    id = "shop-baselines"
    seed = 2024
    replicates = 10

    [[instances]]
    family = "flow-shop"
    parameters = { jobs = 10, machines = 5 }
    seeds = [1, 2, 3]

    [[instances]]
    suite = "taillard-flow-shop"
    instances = ["ta001", "ta002"]

    [[solvers]]
    id = "random-search"

    [[solvers]]
    id = "hill-climbing"
    label = "hill-climbing-r50"
    parameters = { restart_after = 50 }

    [[protocols]]
    name = "evaluations"
    stop = { max_evaluations = 5000 }

    [execution]
    workers = "auto"
    retries = 1

An instance source is one of ``{family, seeds, parameters?}``,
``{suite, instances, directory?}``, ``{catalog, seed?}``, ``{directory}``,
or ``{file, format?}``. A solver is ``{id, label?, parameters?, replicates?}``.
A protocol's ``stop`` table takes the fields of
:class:`~dispatchatlas.core.solver.StopCriteria`. The optional top-level keys
are ``replicates``, ``replicate_timed``, ``seed_namespace``, ``objectives``,
``execution`` (the fields of
:class:`~dispatchatlas.campaigns.config.Execution`), and ``metadata``. In
JSON, ``null`` stands for an absent optional value.
"""

from __future__ import annotations

import json
import math
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from dispatchatlas.campaigns.config import (
    SCHEMA_VERSION,
    Campaign,
    CatalogSource,
    DirectorySource,
    Execution,
    FamilySource,
    FileSource,
    InstanceSource,
    Protocol,
    SolverSpec,
    SuiteSource,
    jsonable,
)
from dispatchatlas.core.errors import ConfigError, ModelError
from dispatchatlas.core.solver import StopCriteria

__all__ = ["campaign_from_dict", "campaign_to_dict", "load_campaign", "loads_campaign"]

_ABSENT: Any = object()

_TOP_LEVEL = (
    "schema_version",
    "id",
    "seed",
    "instances",
    "solvers",
    "protocols",
    "replicates",
    "replicate_timed",
    "seed_namespace",
    "objectives",
    "execution",
    "metadata",
)
_SOURCE_KEYS = {
    "family": ("family", "seeds", "parameters"),
    "suite": ("suite", "instances", "directory"),
    "catalog": ("catalog", "seed"),
    "file": ("file", "format"),
    "directory": ("directory",),
}
_STOP_KEYS = {
    "time_limit": "number",
    "max_evaluations": "integer",
    "max_iterations": "integer",
    "target": "number",
    "patience": "integer",
    "schedule": "string",
}
_EXECUTION_KEYS = (
    "backend",
    "workers",
    "timeout_factor",
    "grace",
    "startup_limit",
    "retries",
    "record_schedules",
    "affinity",
    "threads_per_worker",
    "on_changed_code",
    "on_changed_problem",
    "log_level",
)


def load_campaign(path: str | Path) -> Campaign:
    """Read a campaign file.

    Args:
        path: A ``.toml`` or ``.json`` file.

    Returns:
        The campaign. Relative paths in its instance sources refer to the
        file's directory.

    Raises:
        ConfigError: If the file cannot be read or is not a valid campaign.
    """
    file = Path(path)
    kind = file.suffix.lower().lstrip(".")
    if kind not in ("toml", "json"):
        raise ConfigError(f"{file.name}: campaign files must end in .toml or .json")
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read the campaign file {file}: {error}") from error
    try:
        return loads_campaign(text, format=kind, base=file.absolute().parent)
    except ConfigError as error:
        raise ConfigError(f"{file}: {error}") from error


def loads_campaign(text: str, *, format: str = "toml", base: Path | None = None) -> Campaign:  # noqa: A002
    """Parse a campaign from TOML or JSON text.

    Args:
        text: The document.
        format: ``"toml"`` or ``"json"``.
        base: Directory that relative paths refer to.

    Raises:
        ConfigError: If the text is not a valid campaign document.
    """
    if format == "toml":
        try:
            document: object = tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"invalid TOML: {error}") from error
    elif format == "json":
        try:
            document = json.loads(text, object_pairs_hook=_unique_keys, parse_constant=_constant)
        except json.JSONDecodeError as error:
            raise ConfigError(f"invalid JSON: {error}") from error
    else:
        raise ConfigError(f"unknown campaign format {format!r}; use 'toml' or 'json'")
    if not isinstance(document, Mapping):
        raise ConfigError("a campaign document must be a table")
    return campaign_from_dict(document, base=base)


def campaign_to_dict(campaign: Campaign) -> dict[str, Any]:
    """Return a campaign as a JSON-ready document that :func:`campaign_from_dict` reads back.

    The directory that relative paths refer to, :attr:`Campaign.base`, is not
    part of the document.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "id": campaign.id,
        "seed": campaign.seed,
        "replicates": campaign.replicates,
        "replicate_timed": campaign.replicate_timed,
        "seed_namespace": campaign.seed_namespace,
        "objectives": None if campaign.objectives is None else list(campaign.objectives),
        "instances": [source.to_dict() for source in campaign.instances],
        "solvers": [solver.to_dict() for solver in campaign.solvers],
        "protocols": [protocol.to_dict() for protocol in campaign.protocols],
        "execution": campaign.execution.to_dict(),
        "metadata": jsonable(campaign.metadata),
    }


def campaign_from_dict(document: Mapping[str, Any], *, base: Path | None = None) -> Campaign:
    """Build a campaign from a parsed TOML or JSON document.

    Args:
        document: The document, as :func:`campaign_to_dict` returns or a
            campaign file holds.
        base: Directory that relative paths refer to.

    Raises:
        ConfigError: If a key is unknown, a value has the wrong type, a
            required key is missing, or the values are inconsistent. The
            message names the offending entry, for example
            ``instances[1].seeds[0]``.
    """
    table = _Table(document, "the campaign", _TOP_LEVEL)
    version = table.get("schema_version")
    if version is _ABSENT:
        raise ConfigError(
            f"the campaign declares no schema_version; add schema_version = {SCHEMA_VERSION}"
        )
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version {version!r} is not supported; this release reads "
            f"schema_version = {SCHEMA_VERSION}"
        )
    campaign_id = _string(table.required("id"), "id")
    seed = _integer(table.required("seed"), "seed")
    instances = tuple(
        _source(item, f"instances[{i}]")
        for i, item in enumerate(_array(table.required("instances"), "instances"))
    )
    solvers = tuple(
        _solver(item, f"solvers[{i}]")
        for i, item in enumerate(_array(table.required("solvers"), "solvers"))
    )
    protocols = tuple(
        _protocol(item, f"protocols[{i}]")
        for i, item in enumerate(_array(table.required("protocols"), "protocols"))
    )
    options: dict[str, Any] = {}
    if (replicates := table.get("replicates")) is not _ABSENT:
        options["replicates"] = _integer(replicates, "replicates")
    if (replicate_timed := table.get("replicate_timed")) is not _ABSENT:
        options["replicate_timed"] = _boolean(replicate_timed, "replicate_timed")
    if (namespace := table.get("seed_namespace")) is not _ABSENT:
        options["seed_namespace"] = _string(namespace, "seed_namespace")
    if (objectives := table.get("objectives")) is not _ABSENT:
        options["objectives"] = tuple(
            _string(item, f"objectives[{i}]")
            for i, item in enumerate(_array(objectives, "objectives"))
        )
    if (execution := table.get("execution")) is not _ABSENT:
        options["execution"] = _execution(execution, "execution")
    if (metadata := table.get("metadata")) is not _ABSENT:
        options["metadata"] = _json_table(metadata, "metadata")
    return _build(
        "the campaign",
        lambda: Campaign(
            id=campaign_id,
            seed=seed,
            instances=instances,
            solvers=solvers,
            protocols=protocols,
            base=base,
            **options,
        ),
    )


# ------------------------------------------------------------------ sections


def _source(value: object, where: str) -> InstanceSource:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a table, got {_describe(value)}")
    kinds = [kind for kind in ("family", "suite", "catalog", "file") if value.get(kind) is not None]
    if len(kinds) > 1:
        raise ConfigError(f"{where} mixes the source kinds {', '.join(kinds)}; give exactly one")
    if kinds:
        kind = kinds[0]
    elif value.get("directory") is not None:
        kind = "directory"
    else:
        raise ConfigError(
            f"{where} names no instance source; give one of family, suite, catalog, file, "
            "or directory"
        )
    table = _Table(value, where, _SOURCE_KEYS[kind])
    if kind == "family":
        family = _string(table.required("family"), f"{where}.family")
        seeds = tuple(
            _integer(item, f"{where}.seeds[{i}]")
            for i, item in enumerate(_array(table.required("seeds"), f"{where}.seeds"))
        )
        parameters = table.get("parameters")
        options = {} if parameters is _ABSENT else _json_table(parameters, f"{where}.parameters")
        return _build(where, lambda: FamilySource(family, seeds, options))
    if kind == "suite":
        suite = _string(table.required("suite"), f"{where}.suite")
        names = tuple(
            _string(item, f"{where}.instances[{i}]")
            for i, item in enumerate(_array(table.required("instances"), f"{where}.instances"))
        )
        directory = table.get("directory")
        folder = None if directory is _ABSENT else Path(_string(directory, f"{where}.directory"))
        return _build(where, lambda: SuiteSource(suite, names, folder))
    if kind == "catalog":
        catalog = _string(table.required("catalog"), f"{where}.catalog")
        seed = table.get("seed")
        root = 0 if seed is _ABSENT else _integer(seed, f"{where}.seed")
        return _build(where, lambda: CatalogSource(catalog, root))
    if kind == "file":
        file = Path(_string(table.required("file"), f"{where}.file"))
        format_value = table.get("format")
        name = None if format_value is _ABSENT else _string(format_value, f"{where}.format")
        return _build(where, lambda: FileSource(file, name))
    path = Path(_string(table.required("directory"), f"{where}.directory"))
    return _build(where, lambda: DirectorySource(path))


def _solver(value: object, where: str) -> SolverSpec:
    table = _Table(value, where, ("id", "label", "parameters", "replicates"))
    solver_id = _string(table.required("id"), f"{where}.id")
    label = table.get("label")
    name = None if label is _ABSENT else _string(label, f"{where}.label")
    parameters = table.get("parameters")
    options = {} if parameters is _ABSENT else _json_table(parameters, f"{where}.parameters")
    replicates = table.get("replicates")
    count = None if replicates is _ABSENT else _integer(replicates, f"{where}.replicates")
    return _build(where, lambda: SolverSpec(solver_id, name, options, count))


def _protocol(value: object, where: str) -> Protocol:
    table = _Table(value, where, ("name", "stop"))
    name = _string(table.required("name"), f"{where}.name")
    stop = _Table(table.required("stop"), f"{where}.stop", tuple(_STOP_KEYS))
    criteria: dict[str, Any] = {}
    for key, kind in _STOP_KEYS.items():
        item = stop.get(key)
        if item is not _ABSENT:
            criteria[key] = _STOP_READERS[kind](item, f"{where}.stop.{key}")

    def build() -> Protocol:
        try:
            criterion = StopCriteria(**criteria)
        except ModelError as error:
            raise ConfigError(str(error)) from error
        return Protocol(name, criterion)

    return _build(where, build)


def _execution(value: object, where: str) -> Execution:
    table = _Table(value, where, _EXECUTION_KEYS)
    settings: dict[str, Any] = {}
    readers: dict[str, Callable[[object, str], Any]] = {
        "backend": _string,
        "timeout_factor": _number,
        "grace": _number,
        "startup_limit": _number,
        "retries": _integer,
        "record_schedules": _boolean,
        "threads_per_worker": _integer,
        "on_changed_code": _string,
        "on_changed_problem": _string,
        "log_level": _string,
    }
    for key, read in readers.items():
        item = table.get(key)
        if item is not _ABSENT:
            settings[key] = read(item, f"{where}.{key}")
    workers = table.get("workers")
    if isinstance(workers, str):
        settings["workers"] = workers
    elif workers is not _ABSENT:
        settings["workers"] = _integer(workers, f"{where}.workers")
    affinity = table.get("affinity")
    if isinstance(affinity, str):
        settings["affinity"] = affinity
    elif affinity is not _ABSENT:
        settings["affinity"] = tuple(
            _integer(item, f"{where}.affinity[{index}]")
            for index, item in enumerate(_array(affinity, f"{where}.affinity"))
        )
    return _build(where, lambda: Execution(**settings))


# ------------------------------------------------------------------- helpers


class _Table:
    """A table of a campaign document with a fixed set of allowed keys.

    Unknown keys are rejected on construction, so a misspelt key is reported
    as such rather than as a missing one. A JSON ``null`` reads as absent.
    """

    def __init__(self, value: object, where: str, allowed: tuple[str, ...]) -> None:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where} must be a table, got {_describe(value)}")
        unknown = sorted(str(key) for key in value if key not in allowed)
        if unknown:
            listed = ", ".join(repr(key) for key in unknown)
            raise ConfigError(
                f"{where} has unknown key(s) {listed}; the allowed keys are {', '.join(allowed)}"
            )
        self._value: Mapping[str, Any] = value
        self._where = where

    def get(self, key: str) -> Any:
        item = self._value.get(key)
        return _ABSENT if item is None else item

    def required(self, key: str) -> Any:
        item = self.get(key)
        if item is _ABSENT:
            raise ConfigError(f"{self._where} needs the key {key!r}")
        return item


def _build[T](where: str, factory: Callable[[], T]) -> T:
    """Call ``factory`` and prefix a configuration error with ``where``."""
    try:
        return factory()
    except ConfigError as error:
        raise ConfigError(f"{where}: {error}") from error


def _describe(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f"the string {value!r}"
    if isinstance(value, (int, float)):
        return f"the number {value!r}"
    if isinstance(value, Mapping):
        return "a table"
    if isinstance(value, list):
        return "a list"
    return f"a value of type {type(value).__name__}"


def _integer(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an integer, got {_describe(value)}")
    return value


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number, got {_describe(value)}")
    if not math.isfinite(value):
        raise ConfigError(f"{where} must be finite, got {value!r}")
    return float(value)


def _string(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a string, got {_describe(value)}")
    return value


def _boolean(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be true or false, got {_describe(value)}")
    return value


_STOP_READERS: Mapping[str, Callable[[object, str], Any]] = {
    "integer": _integer,
    "number": _number,
    "string": _string,
}
"""The reader of each kind of value in a protocol's ``stop`` table."""


def _array(value: object, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{where} must be a list, got {_describe(value)}")
    return value


def _json_table(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a table, got {_describe(value)}")
    return {str(key): _json_value(item, f"{where}.{key}") for key, item in value.items()}


def _json_value(value: object, where: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _number(value, where)
    if isinstance(value, list):
        return [_json_value(item, f"{where}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, Mapping):
        return _json_table(value, where)
    raise ConfigError(
        f"{where} must be a string, number, boolean, list, or table, got {_describe(value)}"
    )


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ConfigError(f"the key {key!r} appears twice in one JSON object")
        document[key] = value
    return document


def _constant(name: str) -> Any:
    raise ConfigError(f"the JSON constant {name} is not allowed; numbers must be finite")
