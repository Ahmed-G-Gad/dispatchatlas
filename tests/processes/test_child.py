from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any

import pytest

from dispatchatlas.campaigns.processors import allowed_cpus, confined
from dispatchatlas.processes import affinity
from dispatchatlas.processes.affinity import mask_of_cpus
from dispatchatlas.processes.child import (
    SCHEMA,
    clean_message,
    describe_error,
    locate_callable,
    run_child,
)
from tests.processes import child_targets
from tests.processes.helpers import ROOT, needs_pinning, slot_cpus


def _document(target: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "key": ["k", 1],
        "target": f"tests.processes.child_targets:{target}",
        "arguments": extra.pop("arguments", {}),
        "cpu": None,
        "group": None,
        "attempt": 1,
        **extra,
    }


def _child(document: dict[str, Any] | str) -> subprocess.CompletedProcess[str]:
    payload = document if isinstance(document, str) else json.dumps(document)
    return subprocess.run(
        [sys.executable, "-m", "dispatchatlas.processes.child"],
        input=payload,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=120,
    )


def test_a_call_returns_its_row_with_facts() -> None:
    reply, code = run_child(_document("echo", arguments={"x": 1}))
    assert code == 0
    assert reply["row"] == {"x": 1}
    facts = reply["facts"]
    assert facts["pid"] > 0
    assert set(facts["threads"]) >= {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
    assert facts["started_utc"] <= facts["ended_utc"]
    assert facts["wall_seconds"] >= 0
    assert facts["peak_rss_bytes"] is None or facts["peak_rss_bytes"] > 0
    assert facts["cpu"] is None


def test_each_failing_step_has_its_error_type() -> None:
    assert run_child(_document("missing"))[0]["error"]["type"] == "TargetNotFound"
    assert run_child(_document("not_a_row"))[0]["error"]["type"] == "RowNotMapping"
    assert run_child({**_document("echo"), "schema": "other/1"})[0]["error"]["type"] == (
        "BadDocument"
    )
    reply, code = run_child(
        _document("fail", arguments={"message": r"bad C:\data\secret\x.csv"}, clean_tokens=["bad"])
    )
    assert code == 1
    assert reply["error"] == {"type": "ValueError", "message": "<token> <path>"}


def test_messages_lose_paths_frames_and_tokens() -> None:
    text = (
        r"failed at C:\x\y.py and \\server\share\f with /home/u/p/f.py; "
        'File "/srv/app/m.py", line 3, in run\n'
        "ProjectName and projectname at 1/2 of version 3.12/4"
    )
    cleaned = clean_message(text, tokens=["projectname"])
    for gone in (r"C:\x", r"\\server", "/home/u", "File ", "ProjectName", "projectname"):
        assert gone not in cleaned
    assert "1/2" in cleaned
    assert "3.12/4" in cleaned
    assert cleaned.count("<token>") == 2
    assert len(clean_message("x" * 10_000)) == 2000


def test_errors_are_named_plainly_for_builtins_and_qualified_otherwise() -> None:
    assert describe_error(ValueError("v"))["type"] == "ValueError"
    assert (
        describe_error(json.JSONDecodeError("m", "d", 0))["type"] == "json.decoder.JSONDecodeError"
    )
    assert describe_error(KeyError())["message"] == "KeyError"
    assert locate_callable("tests.processes.child_targets:echo") is child_targets.echo
    with pytest.raises(TypeError):
        locate_callable("no-colon")


def test_the_reply_is_the_last_line_although_the_function_printed() -> None:
    completed = _child(_document("prints_then_row", arguments={"a": 1}))
    assert completed.returncode == 0, completed.stderr
    reply = json.loads(completed.stdout.strip().splitlines()[-1])
    assert reply["row"] == {"a": 1}
    assert "junk that is not the reply" in completed.stderr


def test_a_failure_replies_with_an_error_and_the_traceback_goes_to_stderr() -> None:
    completed = _child(_document("fail", arguments={"message": "boom"}))
    assert completed.returncode == 1
    reply = json.loads(completed.stdout.strip().splitlines()[-1])
    assert reply["error"] == {"type": "ValueError", "message": "boom"}
    assert "Traceback" in completed.stderr
    assert "Traceback" not in completed.stdout
    bad = _child("not json")
    assert bad.returncode == 1
    assert json.loads(bad.stdout.strip().splitlines()[-1])["error"]["type"] == "BadDocument"


@needs_pinning
def test_a_child_created_on_its_processor_pins_itself_there() -> None:
    cpu = allowed_cpus()[-1]
    with confined(cpu):
        completed = _child(_document("records_env", cpu=cpu))
    assert completed.returncode == 0, completed.stderr
    reply = json.loads(completed.stdout.strip().splitlines()[-1])
    assert reply["facts"]["affinity_mask"] == 1 << cpu
    assert reply["facts"]["affinity_inherited"] == 1 << cpu
    assert reply["row"]["mask"] == 1 << cpu


def test_the_child_entry_loads_neither_campaigns_nor_numpy() -> None:
    script = (
        "import sys, runpy\n"
        "import dispatchatlas.processes.child\n"
        "print(sorted(name for name in sys.modules if name.startswith(('numpy', "
        "'dispatchatlas.campaigns', 'dispatchatlas.core', 'dispatchatlas.solvers'))))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "[]"


_FILE_NOT_FOUND = str(
    FileNotFoundError(
        2, "No such file or directory", "C:\\Users\\My Project\\secret dir\\data.json"
    )
)


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        (r"cannot read C:\Users\Someone\My Project\data.json", "cannot read <path>"),
        (_FILE_NOT_FOUND, "No such file or directory: '<path>'"),
        (r"\\fileserver\share\Team Folder\x.json", "<path>"),
        ("D:/work/run 7/rows.jsonl", "<path>"),
        ("/home/someone/my project/x.csv", "<path>"),
        (r"C:\Program Files\x y\f.json", "<path>"),
        ("open '/home/someone/my project/x.csv' failed", "open '<path>' failed"),
        ('read "C:\\Users\\O\'Brien\\a b.json"', 'read "<path>"'),
    ],
)
def test_paths_with_spaces_are_removed_whole(text: str, kept: str) -> None:
    cleaned = clean_message(text, tokens=["secret"])
    assert kept in cleaned
    for fragment in ("Project", "Folder", "Files", "secret", "data.json", "x.json", "x.csv"):
        assert fragment not in cleaned
    assert "a b" not in cleaned
    assert "rows.jsonl" not in cleaned
    assert re.search(r"[A-Za-z]:[\\/]", cleaned) is None


def test_paths_without_spaces_are_cleaned_as_before() -> None:
    assert clean_message(r"bad C:\data\secret\x.csv", tokens=["bad"]) == "<token> <path>"
    assert clean_message(r"Secret failed reading C:\data\secret\in.csv", tokens=["secret"]) == (
        "<token> failed reading <path>"
    )
    assert clean_message(r"see \\server\share\f now") == "see <path> now"
    assert clean_message("1/2 of version 3.12/4") == "1/2 of version 3.12/4"
    assert clean_message("[Errno 2] No such file: 'C:\\\\nospace\\\\dir\\\\f.json'") == (
        "[Errno 2] No such file: '<path>'"
    )


def test_the_error_type_loses_the_tokens_too() -> None:
    class SecretBoom(Exception):  # noqa: N818 - the test needs a token in the class name
        pass

    SecretBoom.__module__ = "secretjobs.mod"
    described = describe_error(SecretBoom("boom in secret"), tokens=["secret"])
    assert "secret" not in described["type"].lower()
    assert "<token>" in described["type"]
    assert described["message"] == "boom in <token>"


@pytest.mark.parametrize(("kind", "said"), [("inf", "is inf"), ("set", "is a set")])
def test_a_row_that_is_not_json_names_its_field(kind: str, said: str) -> None:
    completed = _child(_document("unwritable", arguments={"kind": kind, "key": 1}))
    assert completed.returncode == 1, completed.stderr
    reply = json.loads(completed.stdout.strip().splitlines()[-1])
    assert reply["error"]["type"] == "RowNotSerializable"
    assert "field 'x.y'" in reply["error"]["message"]
    assert said in reply["error"]["message"]


def test_output_after_the_reply_goes_to_stderr() -> None:
    completed = _child(_document("prints_at_exit", arguments={"a": 1}))
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1])["row"] == {"a": 1}
    assert "printed at exit" not in completed.stdout
    assert "printed at exit" in completed.stderr


@needs_pinning
def test_a_child_on_a_wider_set_pins_itself_to_its_processor() -> None:
    pair = slot_cpus(2)
    if len(pair) < 2:
        pytest.skip("needs two processors")
    cpu = pair[0]
    with affinity.confined(pair):
        completed = _child(_document("records_env", cpu=cpu))
    assert completed.returncode == 0, completed.stderr
    reply = json.loads(completed.stdout.strip().splitlines()[-1])
    assert reply["facts"]["affinity_inherited"] == mask_of_cpus(pair)
    assert reply["facts"]["affinity_mask"] == 1 << cpu
    assert reply["row"]["mask"] == 1 << cpu
