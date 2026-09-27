"""Readers for published instance-file formats.

Each reader turns the text of one instance into a
:class:`~dispatchatlas.core.model.Problem` built with the same builders as the
generated families, so a read instance and a generated one have the same shape.
The formats, by name:

- ``orlib-job-shop``: OR-Library job shops, read by :func:`read_orlib_job_shop`.
- ``orlib-flow-shop``: OR-Library flow shops, read by :func:`read_orlib_flow_shop`.
- ``fjs``: FJSPLib flexible job shops (``.fjs``), read by :func:`read_fjs`.
- ``psplib``: PSPLIB projects (``.sm``, ``.mm``), read by :func:`read_psplib`.

:func:`load_instance` reads a file with the reader its extension or the
``format`` argument selects; OR-Library files, usually ``.txt``, need
``format``::

    from dispatchatlas.benchmarks.formats import load_instance
    problem = load_instance("jobshop1.txt", format="orlib-job-shop", instance="ft06")
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dispatchatlas.benchmarks.formats.fjs import read_fjs
from dispatchatlas.benchmarks.formats.orlib import (
    read_orlib_flow_shop,
    read_orlib_job_shop,
    split_orlib,
)
from dispatchatlas.benchmarks.formats.psplib import read_psplib
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.model import Problem

__all__ = [
    "EXTENSIONS",
    "FORMATS",
    "load_instance",
    "read_fjs",
    "read_orlib_flow_shop",
    "read_orlib_job_shop",
    "read_psplib",
    "split_orlib",
]

FORMATS: Mapping[str, Callable[..., Problem]] = MappingProxyType(
    {
        "orlib-job-shop": read_orlib_job_shop,
        "orlib-flow-shop": read_orlib_flow_shop,
        "fjs": read_fjs,
        "psplib": read_psplib,
    }
)
"""Readers by format name."""

EXTENSIONS: Mapping[str, str] = MappingProxyType({".fjs": "fjs", ".sm": "psplib", ".mm": "psplib"})
"""Format name implied by a file extension. OR-Library files need an explicit format."""


def load_instance(path: str | Path, format: str | None = None, **options: Any) -> Problem:  # noqa: A002
    """Read one instance file.

    Args:
        path: The file.
        format: A key of :data:`FORMATS`. Defaults to the format its extension
            implies, see :data:`EXTENSIONS`.
        **options: Keyword arguments of the reader, such as ``instance`` for
            OR-Library files or ``ignore_nonrenewable`` for PSPLIB files. The
            problem id defaults to ``instance`` when given, else to the file
            name without its extension.

    Returns:
        The problem.

    Raises:
        ConfigError: If the format is unknown or cannot be inferred, or an
            option is not accepted by the reader.
        SerializationError: If the file is malformed.
    """
    file = Path(path)
    name = format if format is not None else EXTENSIONS.get(file.suffix.lower())
    if name is None:
        raise ConfigError(
            f"cannot infer the format of {file.name!r}; pass format= one of {sorted(FORMATS)}"
        )
    reader = FORMATS.get(name)
    if reader is None:
        raise ConfigError(f"unknown instance format {name!r}; known: {sorted(FORMATS)}")
    accepted = set(inspect.signature(reader).parameters) - {"text"}
    unknown = set(options) - accepted
    if unknown:
        raise ConfigError(
            f"format {name!r} does not accept {sorted(unknown)}; it accepts {sorted(accepted)}"
        )
    if "problem_id" not in options or options["problem_id"] is None:
        options["problem_id"] = options.get("instance") or file.stem
    text = file.read_bytes().decode("utf-8", errors="replace")
    return reader(text, **options)
