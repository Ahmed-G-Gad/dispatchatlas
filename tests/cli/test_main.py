"""The command line as a whole: the command tree, exit codes, errors, logs, and values."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections.abc import Iterator

import pytest

from dispatchatlas import __version__
from dispatchatlas.cli import build_parser, solvers, version
from dispatchatlas.cli._common import parse_value
from tests.cli.runner import Cli

COMMANDS = (
    ("version",),
    ("solvers", "list"),
    ("solvers", "show"),
    ("benchmarks", "list"),
    ("benchmarks", "show"),
    ("benchmarks", "generate"),
    ("suites", "list"),
    ("suites", "show"),
    ("suites", "load"),
    ("catalogs", "list"),
    ("catalogs", "show"),
    ("catalogs", "build"),
    ("solve",),
    ("check",),
    ("campaign", "validate"),
    ("campaign", "estimate"),
    ("campaign", "run"),
    ("campaign", "replay"),
    ("analyze",),
    ("references",),
)


def _tree(
    parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[str, ...]]:
    groups = [item for item in parser._actions if isinstance(item, argparse._SubParsersAction)]
    if not groups:
        yield prefix
    for group in groups:
        for name, child in group.choices.items():
            yield from _tree(child, (*prefix, name))


def test_the_command_tree_is_the_documented_one() -> None:
    assert tuple(_tree(build_parser())) == COMMANDS


@pytest.mark.parametrize("command", COMMANDS, ids=" ".join)
def test_every_command_answers_help(cli: Cli, command: tuple[str, ...]) -> None:
    result = cli(*command, "--help")
    assert result.code == 0
    assert result.out.startswith(f"usage: dispatchatlas {' '.join(command)}")
    assert "--json" in result.out
    assert "--debug" in result.out


def test_version_flag(cli: Cli) -> None:
    result = cli("--version")
    assert (result.code, result.out) == (0, f"dispatchatlas {__version__}\n")


def test_python_m_dispatchatlas_runs_the_command_line() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "dispatchatlas", "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"dispatchatlas {__version__}\n"


def test_version_command_reports_python_and_backends(cli: Cli) -> None:
    report = cli("version", "--json").json()
    assert report["dispatchatlas"] == __version__
    assert report["python"]["version"] == ".".join(map(str, sys.version_info[:3]))
    assert {item["package"] for item in report["dependencies"]} >= {"numpy"}
    optional = {item["package"]: item for item in report["optional"]}
    assert optional["ortools"]["extra"] == "exact"
    assert all(item["extra"] != "all" for item in report["optional"])
    assert set(report["plugins"]) == {"dispatchatlas.solvers", "dispatchatlas.benchmarks"}
    text = cli("version")
    assert text.code == 0
    assert f"dispatchatlas      {__version__}" in text.out
    assert 'pip install "dispatchatlas[exact]"' in text.out


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((), "the following arguments are required: COMMAND"),
        (("solvers",), "the following arguments are required: COMMAND"),
        (("frobnicate",), "invalid choice: 'frobnicate'"),
        (("solvers", "list", "--colour"), "unrecognized arguments: --colour"),
        (("benchmarks", "generate", "job-shop", "--set", "jobs"), "expected KEY=VALUE"),
        (("benchmarks", "generate", "job-shop", "--seed", "-1"), "at least 0"),
        (("solve", "p.json", "--solver", "heft", "--time-limit", "0"), "must be above 0"),
        (("solve", "p.json", "--solver", "heft", "--max-evaluations", "x"), "an integer"),
        (("solve", "p.json", "--solver", "heft", "--target", "nan"), "must be finite"),
        (("campaign", "run", "c.toml", "--output", "o", "--workers", "0"), "or 'auto'"),
        (("solve", "p.json"), "the following arguments are required: --solver"),
        (("references", "--json", "--format", "bibtex"), "exclude each other"),
    ],
)
def test_usage_errors_exit_with_2(cli: Cli, arguments: tuple[str, ...], message: str) -> None:
    result = cli(*arguments)
    assert result.code == 2
    assert "usage: dispatchatlas" in result.err
    assert message in result.err
    assert result.out == ""


def test_a_parameter_set_twice_is_a_usage_error(cli: Cli) -> None:
    result = cli("benchmarks", "generate", "job-shop", "--set", "jobs=3", "--set", "jobs=4")
    assert result.code == 2
    assert "--set gives 'jobs' twice" in result.err


def test_an_error_is_one_line_with_a_hint(cli: Cli) -> None:
    result = cli("solvers", "show", "no-such-solver")
    assert result.code == 1
    lines = result.err.splitlines()
    assert lines[0] == (
        "dispatchatlas: error: unknown solver 'no-such-solver'; "
        "run 'dispatchatlas solvers list' to see ids"
    )
    assert lines[1].startswith("dispatchatlas: hint: ")
    assert len(lines) == 2
    assert "Traceback" not in result.err


@pytest.mark.parametrize("position", ["before", "after"])
def test_debug_shows_the_traceback_before_or_after_the_command(cli: Cli, position: str) -> None:
    command = ("solvers", "show", "no-such-solver")
    arguments = ("--debug", *command) if position == "before" else (*command, "--debug")
    result = cli(*arguments)
    assert result.code == 1
    assert "Traceback (most recent call last)" in result.err
    assert "ConfigError" in result.err
    assert "dispatchatlas: error: unknown solver" in result.err


def test_an_unexpected_exception_is_reported_with_a_debug_hint(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> None:
        raise RuntimeError("the registry exploded")

    monkeypatch.setattr(solvers, "default_registry", broken)
    result = cli("solvers", "list")
    assert result.code == 1
    assert "dispatchatlas: error: the registry exploded" in result.err
    assert "rerun with --debug" in result.err


def test_an_interrupt_exits_with_130(cli: Cli, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(solvers, "default_registry", interrupted)
    result = cli("solvers", "list")
    assert result.code == 130
    assert result.err == "dispatchatlas: interrupted\n"


def test_logs_go_to_stderr_and_quiet_hides_information(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = logging.getLogger("dispatchatlas")
    handlers, level = list(logger.handlers), logger.level

    def noisy(arguments: argparse.Namespace) -> int:
        logging.getLogger("dispatchatlas.campaigns").info("progress line")
        logging.getLogger("dispatchatlas.campaigns").warning("warning line")
        sys.stdout.write("result line\n")
        return 0

    monkeypatch.setattr(version, "_show", noisy)
    shown = cli("version")
    hidden = cli("--quiet", "version")
    assert shown.out == hidden.out == "result line\n"
    assert shown.err == "dispatchatlas: info: progress line\ndispatchatlas: warning: warning line\n"
    assert hidden.err == "dispatchatlas: warning: warning line\n"
    assert (list(logger.handlers), logger.level) == (handlers, level)


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("10", 10),
        ("-2.5", -2.5),
        ("1e3", 1000.0),
        ("true", True),
        ("null", None),
        ("[1, 2]", (1, 2)),
        ('{"a": [1]}', {"a": (1,)}),
        ('"7"', "7"),
        ("random-key", "random-key"),
        ("device,edge", "device,edge"),
        ("NaN", "NaN"),
        ("Infinity", "Infinity"),
        ("", ""),
    ],
)
def test_set_values_are_json_when_they_parse_and_text_otherwise(raw: str, value: object) -> None:
    assert parse_value(raw) == value
    assert type(parse_value(raw)) is type(value)
