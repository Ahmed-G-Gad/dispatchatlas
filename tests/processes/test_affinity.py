from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from dispatchatlas.campaigns.processors import allowed_cpus
from dispatchatlas.processes import affinity
from dispatchatlas.processes.affinity import (
    confined,
    cpus_of_mask,
    current_cpus,
    describe_cpus,
    mask_of_cpus,
    parse_cpus,
    pin_self,
)
from tests.processes.helpers import ROOT, needs_pinning


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0xF0", (4, 5, 6, 7)),
        ("0X3", (0, 1)),
        ("4-7,12", (4, 5, 6, 7, 12)),
        (" 3 ", (3,)),
        ("9,2,5", (2, 5, 9)),
    ],
)
def test_processor_sets_are_read_from_masks_ranges_and_lists(
    text: str, expected: tuple[int, ...]
) -> None:
    assert parse_cpus(text) == expected


@pytest.mark.parametrize("text", ["", "0x0", "0xZZ", "1,1", "7-3", "-1", "a", "1,,2", "2-"])
def test_a_processor_set_that_is_not_one_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match=r"processor|names|range|mask"):
        parse_cpus(text)


def test_masks_and_processor_lists_round_trip() -> None:
    cpus = cpus_of_mask(0xF00F0)
    assert cpus == (4, 5, 6, 7, 16, 17, 18, 19)
    assert mask_of_cpus(cpus) == 0xF00F0
    assert describe_cpus(cpus) == "4-7, 16-19"
    assert cpus_of_mask(1 << 63) == (63,)
    assert cpus_of_mask(0) == ()
    with pytest.raises(ValueError, match="not negative"):
        mask_of_cpus([-1])


@needs_pinning
def test_a_process_pins_itself_and_reads_its_mask_back() -> None:
    cpu = allowed_cpus()[-1]
    script = (
        "import json\n"
        "from dispatchatlas.processes.affinity import pin_self, current_cpus\n"
        f"report = pin_self({cpu})\n"
        "print(json.dumps([report.wanted, report.read, report.inherited, list(current_cpus())]))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=True
    )
    wanted, read, inherited, cpus = json.loads(completed.stdout.strip().splitlines()[-1])
    assert wanted == read == 1 << cpu
    assert cpus == [cpu]
    assert inherited is not None
    assert inherited & (1 << cpu)


@needs_pinning
def test_pinning_outside_the_current_set_is_refused() -> None:
    outside = (os.cpu_count() or 1) + 8  # on no processor of the machine
    with pytest.raises(RuntimeError, match="it may use"):
        pin_self(outside)


@needs_pinning
def test_a_mask_read_back_that_differs_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu = allowed_cpus()[-1]
    reads = iter([mask_of_cpus(allowed_cpus()), 1 << (cpu + 1)])
    monkeypatch.setattr(affinity, "_read_mask", lambda: next(reads))
    monkeypatch.setattr(affinity, "_set_cpus", lambda pid, cpus: None)
    with pytest.raises(RuntimeError, match="read back"):
        pin_self(cpu)


def test_pinning_is_refused_where_the_platform_cannot_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(affinity, "pinning_refusal_text", lambda: "no affinity call")
    with pytest.raises(RuntimeError, match="no affinity call"):
        pin_self(0)


@needs_pinning
def test_confined_restores_the_earlier_set() -> None:
    before = current_cpus()
    assert before is not None
    with confined(before[-1]):
        assert current_cpus() == (before[-1],)
    assert current_cpus() == before
