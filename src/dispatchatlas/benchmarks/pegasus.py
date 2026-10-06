"""Reading Pegasus DAX workflow files onto a continuum platform.

The Pegasus workflow management system (Deelman et al. 2015) describes an
abstract workflow in an XML document, the DAX. The synthetic scientific
workflows of Bharathi et al. (2008) and Juve et al. (2013), which model
Montage, CyberShake, Epigenomics, LIGO Inspiral, and SIPHT, are distributed in
schema 2.1 of that format with two additions: a ``runtime`` on every job and
a ``size`` on every file a job uses. :func:`read_pegasus_dax` maps one such
file onto a :class:`~dispatchatlas.benchmarks.continuum.platform.Platform`:

- Every job becomes a task with one mode per eligible node. Its ``runtime``
  is taken as its work, so it runs ``runtime / speed`` seconds on one core of
  the chosen node.
- Every ``parent`` listed in a ``child`` element becomes a dependency that
  carries the bytes of the files the parent writes and the child reads,
  divided by ``bytes_per_unit`` to express them in the platform's data unit
  (megabytes by default). The network then prices the transfer whenever the
  two tasks run on different nodes.
- All tasks form one group, the workflow, and are released at time 0.

The elements and attributes read:

- The root ``adag``, in the namespace :data:`DAX_NAMESPACE` or in none. Its
  ``name`` and ``version``, when present, are kept in the problem's metadata.
- Every ``job`` under the root: ``id`` (the task's id), ``runtime`` in
  seconds, and ``name`` and ``namespace``, kept in the task's metadata as
  ``transformation`` and ``namespace``.
- Every ``uses`` under a job: ``file``, ``link``, and ``size`` in bytes. The
  link ``input`` marks a file the job reads, ``output`` one it writes,
  ``inout`` one it reads and writes, and ``none`` one it does neither with.
- Every ``child`` under the root and every ``parent`` under a child: ``ref``,
  the id of a job.

Everything else is ignored: other elements, such as ``filename``,
``argument``, and ``profile``; other attributes, such as ``version``,
``level``, and ``cores`` of a job, ``register``, ``transfer``, ``optional``,
and ``type`` of a file, and the counts of the root, which are not compared
with the content. Every task holds one core, whatever ``cores`` says.

Sizes belong to one job's listing of a file, not to the file name: one name
can be listed with different sizes by different jobs, and written by several
jobs. The data of a dependency is therefore read on that dependency alone. For
every file name the parent lists as written and the child lists as read, it
takes the size the parent gives, or the size the child gives when the parent
gives none; the sizes other jobs give for the same name play no part. A
``uses`` without a ``size`` is accepted, and only a file passed along a
dependency with a size on neither end is refused. A job that lists one file
twice with the same link must give it one size.

Only ``child`` elements create dependencies. A file that one job writes and
another reads adds none by itself, and a dependency between two jobs that
share no file carries no data. Each parent-child pair is counted once.

Runtimes and sizes are never invented or corrected. They are finite decimal
numbers in ASCII digits, with an optional sign, fraction, and exponent, and
may have white space around them. A job without a runtime, or with a negative
one, is refused, as is a negative size. Zero runtimes are accepted. A runtime
too large to give a finite duration on a node, and files too large to give a
dependency a finite amount of data, are refused as well.

The file must be UTF-8 text, with or without a byte-order mark. A file that
contains a NUL character is refused: XML allows none, a file in UTF-16 or
UTF-32 has them, and the XML parser takes a NUL among the first two bytes of
a document as the sign of UTF-16, whatever encoding it is told to use. The
decoded text is then searched for a document type declaration (``<!DOCTYPE``)
and for an entity declaration (``<!ENTITY``) anywhere, a comment included, and
a document that has one is refused before the XML parser is given it, so the
parser meets no declared entity to expand or fetch. The parser reads the same
decoded text as UTF-8, whatever encoding the document declares, and its tree
builder raises when the parser reports a document type declaration to it, so
no problem is built from a declaration the search missed. Anything missing or
malformed raises :class:`~dispatchatlas.core.errors.SerializationError` naming
the file and the element.
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dispatchatlas.benchmarks._common import check_number, check_objectives
from dispatchatlas.benchmarks.continuum.platform import Platform
from dispatchatlas.core.errors import ConfigError, ModelError, SerializationError
from dispatchatlas.core.model import Dependency, Problem, Resource, Task

__all__ = ["DAX_NAMESPACE", "read_pegasus_dax"]

DAX_NAMESPACE = "http://pegasus.isi.edu/schema/DAX"
"""XML namespace of the DAX elements. Documents without a namespace are read too."""

_DECLARATION = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)
"""A document type or entity declaration, in any letter case."""

_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)
"""A decimal number in ASCII digits, with an optional exponent.

Each digit can be matched by one part of the pattern only, so refusing a text
takes time proportional to its length.
"""

_SPACES = " \t\r\n"
"""The white space of XML, which may surround a number."""

_SHOWN = 40
"""How many characters of a malformed number a message repeats."""

_LINKS: Mapping[str, tuple[bool, bool]] = {
    "input": (True, False),
    "output": (False, True),
    "inout": (True, True),
    "none": (False, False),
}
"""Whether a job reads and writes a file, by the ``link`` of its ``uses`` element."""


@dataclass(frozen=True, slots=True)
class _Job:
    """The content of one ``job`` element that the reader uses."""

    key: str
    runtime: float
    reads: Mapping[str, float | None]
    writes: Mapping[str, float | None]
    metadata: Mapping[str, str]


class _Builder(ET.TreeBuilder):
    """A tree builder that refuses a document type declaration.

    :func:`_decode` refuses a document with a declaration before the parser
    runs. This builder is a second check: it raises when the parser reports a
    declaration to it, so that no tree is returned from a document the first
    check missed. The parser has been given the whole text by then, so the
    builder turns the reading into an error without keeping the parser from
    processing the declaration. The accelerated parser of the standard library
    reports every declaration; the one written in Python reports only a
    declaration with a public or system identifier.
    """

    def doctype(self, *_identifiers: str | None) -> None:
        """Refuse the declaration, whatever its name and its public and system identifiers.

        Raises:
            SerializationError: Always.
        """
        raise SerializationError(_refusal("DOCTYPE"))


def read_pegasus_dax(
    source: str | os.PathLike[str],
    platform: Platform,
    *,
    tiers: Sequence[str] | None = None,
    problem_id: str | None = None,
    objectives: str | Sequence[str] = ("makespan",),
    bytes_per_unit: float = 1e6,
) -> Problem:
    """Read a Pegasus DAX workflow file as a problem on ``platform``.

    Args:
        source: Path of the DAX file.
        platform: Nodes and network the workflow runs on.
        tiers: Tiers whose nodes may run tasks. Defaults to every tier of the
            platform except ``"device"``.
        problem_id: Id of the problem. Defaults to the file name without its
            extension, with characters outside letters, digits, ``.``, ``_``,
            and ``-`` replaced, or else the ``name`` of the ``adag`` element
            treated the same way. The file name comes first because many
            files can share one ``adag`` name.
        objectives: One objective name or several, primary first.
        bytes_per_unit: Bytes per data unit of the platform's bandwidths.

    Returns:
        The problem.

    Raises:
        SerializationError: If the file is not a DAX document the reader
            accepts, its content is inconsistent, or a runtime or the files
            of a dependency are too large for a finite duration or amount of
            data.
        ConfigError: If an objective is unknown or repeated, no node of
            ``tiers`` exists, ``bytes_per_unit`` is not a positive number, or
            ``problem_id`` is not a non-empty string without surrounding
            spaces.
        OSError: If the file cannot be read.
    """
    try:
        names = check_objectives(objectives)
        check_number("bytes_per_unit", bytes_per_unit, positive=True)
    except ValueError as error:
        raise ConfigError(f"invalid arguments for read_pegasus_dax: {error}") from error
    if problem_id is not None and (
        not isinstance(problem_id, str) or not problem_id or problem_id != problem_id.strip()
    ):
        raise ConfigError(
            "invalid arguments for read_pegasus_dax: problem_id must be a non-empty string "
            f"without surrounding spaces, got {problem_id!r}"
        )
    path = Path(source)
    nodes = _nodes(platform, tiers)
    try:
        root = _parse(_decode(path.read_bytes()))
        jobs = _jobs(root)
        dependencies = tuple(
            Dependency(before, after, data=_shared_data(jobs[before], jobs[after], bytes_per_unit))
            for before, after in _edges(root, jobs)
        )
        name = root.get("name") or ""
        identifier = problem_id or _slug(path.stem) or _slug(name) or "pegasus-dax"
        tasks = tuple(_task(job, platform, nodes, identifier) for job in jobs.values())
    except SerializationError as error:
        raise SerializationError(f"{path}: {error}") from error
    version = root.get("version") or ""
    metadata: dict[str, str] = {"source": "pegasus-dax"}
    if name:
        metadata["name"] = name
    if version:
        metadata["schema_version"] = version
    try:
        return Problem(
            id=identifier,
            tasks=tasks,
            resources=platform.resources,
            dependencies=dependencies,
            objectives=names,
            network=platform.network,
            metadata=metadata,
        )
    except ModelError as error:
        raise SerializationError(
            f"{path}: Pegasus DAX workflow {identifier!r} is invalid: {error}"
        ) from error


def _decode(data: bytes) -> str:
    """Return the text of the document held in ``data``, checked before any parsing.

    The bytes must be UTF-8 text without a NUL character and without a
    document type or entity declaration. The search for declarations and the
    parser are given this same text. A NUL is refused first: the parser takes
    a NUL among the first two bytes of its input as the sign of UTF-16,
    whatever encoding it is told to use, and would then read other characters
    than the ones searched.

    Raises:
        SerializationError: If the bytes are refused.
    """
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise SerializationError(f"the document is not UTF-8 text: {error}") from error
    if "\x00" in text:
        raise SerializationError("the document is not UTF-8 text: it contains a NUL character")
    declaration = _DECLARATION.search(text)
    if declaration:
        raise SerializationError(_refusal(declaration.group(1).upper()))
    return text


def _parse(text: str) -> ET.Element:
    """Return the root ``adag`` element of the DAX document written in ``text``.

    The parser reads ``text`` as UTF-8, whatever encoding the document
    declares, and builds the tree with a :class:`_Builder`.
    """
    parser = ET.XMLParser(target=_Builder(), encoding="utf-8")
    try:
        root = ET.fromstring(text, parser=parser)
    except ET.ParseError as error:
        raise SerializationError(f"invalid XML: {error}") from error
    if _local(root) != "adag":
        raise SerializationError(
            f"the root element must be adag, in the namespace {DAX_NAMESPACE} or in none, "
            f"found {root.tag!r}"
        )
    return root


def _refusal(keyword: str) -> str:
    """Return the message that refuses a ``DOCTYPE`` or an ``ENTITY`` declaration."""
    declared = "a document type" if keyword == "DOCTYPE" else "an entity"
    return (
        f"the document declares {declared} (<!{keyword}); "
        "document type and entity declarations are refused"
    )


def _jobs(root: ET.Element) -> dict[str, _Job]:
    """Read the ``job`` elements under ``root``, keyed by id in document order."""
    jobs: dict[str, _Job] = {}
    for number, element in enumerate(_children(root, "job")):
        key = _attribute(element, "id", f"job element {number}")
        if key != key.strip():
            raise SerializationError(f"id {key!r} of job element {number} has surrounding spaces")
        if key in jobs:
            raise SerializationError(f"id {key!r} is used by two job elements")
        where = f"job {key!r}"
        runtime = _number(_attribute(element, "runtime", where), f"runtime of {where}")
        reads: dict[str, float | None] = {}
        writes: dict[str, float | None] = {}
        for index, uses in enumerate(_children(element, "uses")):
            place = f"uses element {index} of {where}"
            file_name = _attribute(uses, "file", place)
            link = _attribute(uses, "link", place)
            if link not in _LINKS:
                raise SerializationError(
                    f"file {file_name!r} of {where} has link {link!r}, "
                    f"expected one of {', '.join(_LINKS)}"
                )
            text = uses.get("size")
            size = None if text is None else _number(text, f"size of file {file_name!r} of {where}")
            read, written = _LINKS[link]
            if read:
                _record(reads, file_name, size, f"{where} lists input file {file_name!r}")
            if written:
                _record(writes, file_name, size, f"{where} lists output file {file_name!r}")
        metadata = {
            label: value
            for label, value in (
                ("transformation", element.get("name")),
                ("namespace", element.get("namespace")),
            )
            if value
        }
        jobs[key] = _Job(key, runtime, reads, writes, metadata)
    if not jobs:
        raise SerializationError("the adag element has no job element")
    return jobs


def _edges(root: ET.Element, jobs: Mapping[str, _Job]) -> list[tuple[str, str]]:
    """Return the parent-child pairs, ordered by parent and then child position."""
    position = {key: number for number, key in enumerate(jobs)}
    pairs: set[tuple[str, str]] = set()
    for number, element in enumerate(_children(root, "child")):
        after = _reference(element, position, f"child element {number}")
        for index, parent in enumerate(_children(element, "parent")):
            before = _reference(parent, position, f"parent element {index} of child {after!r}")
            if before == after:
                raise SerializationError(f"job {after!r} depends on itself")
            pairs.add((before, after))
    return sorted(pairs, key=lambda pair: (position[pair[0]], position[pair[1]]))


def _shared_data(parent: _Job, child: _Job, bytes_per_unit: float) -> float:
    """Return the data ``parent`` passes to ``child``, in units of ``bytes_per_unit`` bytes.

    The data is the total size of the files ``parent`` writes and ``child``
    reads. Each file has the size the parent gives it, or else the size the
    child gives it.
    """
    sizes: list[float] = []
    for name in sorted(parent.writes.keys() & child.reads.keys()):
        size = parent.writes[name]
        if size is None:
            size = child.reads[name]
        if size is None:
            raise SerializationError(
                f"file {name!r} passed from job {parent.key!r} to job {child.key!r} has no size"
            )
        sizes.append(size)
    try:
        data = math.fsum(sizes) / bytes_per_unit
    except OverflowError:
        data = math.inf
    if not math.isfinite(data):
        raise SerializationError(
            f"the files passed from job {parent.key!r} to job {child.key!r} are too large: "
            "their total size is not a finite number of data units"
        )
    return data


def _task(job: _Job, platform: Platform, nodes: Sequence[Resource], group: str) -> Task:
    """Return the task of ``job``, in ``group``, with one mode per node of ``nodes``."""
    try:
        modes = platform.modes(job.runtime, nodes)
        return Task(job.key, modes, group=group, metadata=job.metadata)
    except ModelError as error:
        raise SerializationError(
            f"job {job.key!r} with runtime {job.runtime!r} cannot become a task "
            f"on the platform: {error}"
        ) from error


def _nodes(platform: Platform, tiers: Sequence[str] | None) -> tuple[Resource, ...]:
    chosen = [tier for tier in platform.tiers if tier != "device"] if tiers is None else tiers
    nodes = tuple(node for tier in chosen for node in platform.nodes(tier))
    if not nodes:
        raise ConfigError(f"the platform has no node in tiers {list(chosen)}")
    return nodes


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")


def _local(element: ET.Element) -> str:
    """Return the name of a DAX element, or an empty string for any other element."""
    tag = element.tag
    if not tag.startswith("{"):
        return tag
    namespace, _, name = tag[1:].partition("}")
    return name if namespace == DAX_NAMESPACE else ""


def _children(parent: ET.Element, name: str) -> Iterator[ET.Element]:
    return (child for child in parent if _local(child) == name)


def _attribute(element: ET.Element, name: str, where: str) -> str:
    value = element.get(name)
    if value is None or not value.strip():
        raise SerializationError(f"{where} needs a non-empty {name} attribute")
    return value


def _reference(element: ET.Element, position: Mapping[str, int], where: str) -> str:
    key = _attribute(element, "ref", where)
    if key not in position:
        raise SerializationError(f"{where} refers to unknown job {key!r}")
    return key


def _number(text: str, what: str) -> float:
    """Return the finite, non-negative decimal number written in ``text``.

    The number is written in ASCII digits. XML white space around it is
    allowed, and a negative zero is returned as zero.
    """
    stripped = text.strip(_SPACES)
    if not _NUMBER.fullmatch(stripped):
        raise SerializationError(f"{what} must be a number, found {_shown(text)}")
    value = float(stripped)
    if not math.isfinite(value) or value < 0.0:
        raise SerializationError(f"{what} must be finite and non-negative, found {_shown(text)}")
    return value + 0.0


def _shown(text: str) -> str:
    """Return ``text`` quoted for a message, cut after its first characters when it is long."""
    if len(text) <= _SHOWN:
        return repr(text)
    return f"{text[:_SHOWN]!r} and {len(text) - _SHOWN} more characters"


def _record(sizes: dict[str, float | None], name: str, size: float | None, what: str) -> None:
    """Record one listing of a file, keeping the size of an earlier listing that gave one."""
    known = sizes.get(name)
    if known is not None and size is not None and known != size:
        raise SerializationError(f"{what} with two different sizes")
    sizes[name] = known if size is None else size
