"""Fixtures of the command-line tests."""

from __future__ import annotations

import pytest

from tests.cli.runner import Cli


@pytest.fixture
def cli(capsys: pytest.CaptureFixture[str]) -> Cli:
    """Run the command line in-process with captured output."""
    return Cli(capsys)
