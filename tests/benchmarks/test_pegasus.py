"""Tests of the Pegasus DAX reader."""

from __future__ import annotations

import math
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from dispatchatlas.benchmarks import pegasus, read_pegasus_dax
from dispatchatlas.benchmarks.continuum import NodeSpec, three_tier_platform
from dispatchatlas.benchmarks.pegasus import DAX_NAMESPACE
from dispatchatlas.core import ConfigError, Link, SerializationError, check_schedule
from dispatchatlas.solvers import solve
from dispatchatlas.solvers.scheduling import Decoder

PLATFORM = three_tier_platform(devices=1, edge=1, fog=1, cloud=1)

# One split, two maps, a reduce, and a report. The file chunk_b is listed with one size
# by its writer and another by its reader, and both maps write a file named part.
DOCUMENT = """\
<?xml version="1.0" encoding="UTF-8"?>
<!-- A hand-written workflow. -->
<adag xmlns="http://pegasus.isi.edu/schema/DAX" version="2.1" count="1" index="0"
      name="map reduce" jobCount="5" fileCount="0" childCount="4">
  <job id="ID00000" namespace="demo" name="split" version="1.0" runtime="10">
    <uses file="raw" link="input" register="true" transfer="true" optional="false"
          type="data" size="5000000"/>
    <uses file="chunk_a" link="output" size="2000000"/>
    <uses file="chunk_b" link="output" size="3000000"/>
  </job>
  <job id="ID00001" namespace="demo" name="map" version="1.0" runtime="20.5">
    <uses file="chunk_a" link="input" size="2000000"/>
    <uses file="part" link="output" size="1000000"/>
  </job>
  <job id="ID00002" namespace="demo" name="map" version="1.0" runtime="30">
    <uses file="chunk_b" link="input" size="3100000"/>
    <uses file="part" link="output" size="500000"/>
    <uses file="log" link="output" size="7"/>
  </job>
  <job id="ID00003" namespace="demo" name="reduce" version="1.0" runtime="5">
    <uses file="part" link="input" size="1000000"/>
    <uses file="merged" link="output" size="250000"/>
  </job>
  <job id="ID00004" namespace="demo" name="report" version="1.0" runtime="0">
    <uses file="merged" link="input" size="250000"/>
    <uses file="summary" link="output" size="100"/>
  </job>
  <child ref="ID00001">
    <parent ref="ID00000"/>
  </child>
  <child ref="ID00002">
    <parent ref="ID00000"/>
  </child>
  <child ref="ID00003">
    <parent ref="ID00001"/>
    <parent ref="ID00002"/>
  </child>
  <child ref="ID00004">
    <parent ref="ID00003"/>
  </child>
</adag>
"""
JOBS = [
    ("ID00000", "split", 10.0),
    ("ID00001", "map", 20.5),
    ("ID00002", "map", 30.0),
    ("ID00003", "reduce", 5.0),
    ("ID00004", "report", 0.0),
]
EDGES = [
    ("ID00000", "ID00001", 2.0),
    ("ID00000", "ID00002", 3.0),
    ("ID00001", "ID00003", 1.0),
    ("ID00002", "ID00003", 0.5),
    ("ID00003", "ID00004", 0.25),
]
ONE_JOB = '<adag name="one"><job id="j" runtime="1"/></adag>\n'
ENTITY = (
    '<!DOCTYPE adag [<!ENTITY a "expanded">]>\n<adag name="&a;"><job id="j" runtime="1"/></adag>\n'
)


def write(folder: Path, text: str, name: str = "map-reduce.xml") -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def changed(*replacements: tuple[str, str]) -> str:
    """The document with the first occurrence of each old text replaced by the new one."""
    text = DOCUMENT
    for old, new in replacements:
        assert old in text
        text = text.replace(old, new, 1)
    return text


def edges(path: Path) -> list[tuple[str, str, float]]:
    problem = read_pegasus_dax(path, PLATFORM)
    return [(d.before, d.after, d.data) for d in problem.dependencies]


def forbid_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the test fail if the reader builds an XML parser or gives it a document."""

    def parser_called(*arguments: object, **options: object) -> ET.Element:
        raise AssertionError("the XML parser was given the document")

    monkeypatch.setattr(ET, "fromstring", parser_called)
    monkeypatch.setattr(ET, "XMLParser", parser_called)


def test_reads_tasks_placements_and_data(tmp_path):
    problem = read_pegasus_dax(write(tmp_path, DOCUMENT), PLATFORM)
    assert problem.id == "map-reduce"
    assert [task.id for task in problem.tasks] == [key for key, _, _ in JOBS]
    for task, (_, transformation, runtime) in zip(problem.tasks, JOBS, strict=True):
        assert [mode.location for mode in task.modes] == ["edge-0", "fog-0", "cloud-0"]
        for mode in task.modes:
            assert mode.location is not None
            assert mode.duration == PLATFORM.duration(runtime, mode.location)
        assert task.group == problem.id
        assert task.release == 0.0
        assert dict(task.metadata) == {"namespace": "demo", "transformation": transformation}
    assert [(d.before, d.after, d.data) for d in problem.dependencies] == EDGES
    assert problem.network == PLATFORM.network
    assert problem.objectives == ("makespan",)
    assert dict(problem.metadata) == {
        "name": "map reduce",
        "schema_version": "2.1",
        "source": "pegasus-dax",
    }


def test_a_dependency_carries_the_sizes_its_parent_gives(tmp_path):
    # The sizes the readers list do not matter while the writers give one.
    text = changed(
        ('<uses file="part" link="input" size="1000000"/>', '<uses file="part" link="input"/>'),
        ('link="input" size="3100000"', 'link="input" size="9"'),
        ('<uses file="merged" link="input" size="250000"/>', '<uses file="merged" link="input"/>'),
    )
    assert edges(write(tmp_path, text)) == EDGES


def test_a_size_missing_at_the_parent_is_read_at_the_child_or_refused(tmp_path):
    text = changed(('link="output" size="3000000"', 'link="output"'))
    expected = [edge if edge[1] != "ID00002" else (*edge[:2], 3.1) for edge in EDGES]
    assert edges(write(tmp_path, text)) == expected
    text = changed(
        ('link="output" size="3000000"', 'link="output"'),
        ('link="input" size="3100000"', 'link="input"'),
    )
    message = "file 'chunk_b' passed from job 'ID00000' to job 'ID00002' has no size"
    with pytest.raises(SerializationError, match=message):
        read_pegasus_dax(write(tmp_path, text), PLATFORM)


def test_sizes_of_files_no_dependency_carries_are_optional(tmp_path):
    text = changed(
        ('type="data" size="5000000"', 'type="data"'),
        ('<uses file="log" link="output" size="7"/>', '<uses file="log" link="output"/>'),
        (' size="100"', ""),
    )
    assert edges(write(tmp_path, text)) == EDGES


def test_documents_without_a_namespace_and_with_unread_content_are_read(tmp_path):
    reference = read_pegasus_dax(write(tmp_path, DOCUMENT), PLATFORM)
    text = changed(
        (' xmlns="http://pegasus.isi.edu/schema/DAX" version="2.1"', ' xmlns:x="urn:example:x"'),
        ('jobCount="5"', 'jobCount="25"'),
        ('  <job id="ID00000"', '  <filename file="raw" link="input"/>\n  <job id="ID00000"'),
        (
            'runtime="10">',
            'runtime="10" cores="160" level="1">\n    <argument>-v <filename file="raw"/>'
            '</argument>\n    <profile namespace="env" key="k">v</profile>',
        ),
        ("</adag>", '  <x:job id="ID00000"/>\n  <x:child ref="ghost"/>\n</adag>'),
    )
    problem = read_pegasus_dax(write(tmp_path, text), PLATFORM)
    assert problem.tasks == reference.tasks
    assert problem.dependencies == reference.dependencies
    assert dict(problem.metadata) == {"name": "map reduce", "source": "pegasus-dax"}


def test_problems_are_named_after_files_then_after_the_workflow(tmp_path):
    assert read_pegasus_dax(write(tmp_path, DOCUMENT, "Montage_25.xml"), PLATFORM).id == (
        "Montage_25"
    )
    spaced = write(tmp_path, DOCUMENT, "my flow (1).dax")
    assert read_pegasus_dax(str(spaced), PLATFORM).id == "my-flow-1"
    assert read_pegasus_dax(spaced, PLATFORM, problem_id="chosen").id == "chosen"
    assert read_pegasus_dax(write(tmp_path, DOCUMENT, "---.xml"), PLATFORM).id == "map-reduce"
    nameless = changed(('name="map reduce" ', ""))
    problem = read_pegasus_dax(write(tmp_path, nameless, "---.xml"), PLATFORM)
    assert problem.id == "pegasus-dax"
    assert "name" not in problem.metadata


@pytest.mark.parametrize("problem_id", ["", " ", " chosen", "chosen ", "chosen\n", 7])
def test_an_unusable_problem_id_is_a_configuration_error(tmp_path, problem_id):
    path = write(tmp_path, DOCUMENT)
    with pytest.raises(ConfigError, match="problem_id must be a non-empty string without"):
        read_pegasus_dax(path, PLATFORM, problem_id=problem_id)


def test_tiers_units_and_objectives_are_configurable(tmp_path):
    path = write(tmp_path, DOCUMENT)
    problem = read_pegasus_dax(
        path,
        PLATFORM,
        tiers=["device", "cloud"],
        bytes_per_unit=1.0,
        objectives=("makespan", "energy"),
    )
    assert [mode.location for mode in problem.tasks[0].modes] == ["device-0", "cloud-0"]
    assert problem.dependencies[0].data == 2_000_000.0
    assert problem.objectives == ("makespan", "energy")
    with pytest.raises(ConfigError):
        read_pegasus_dax(path, three_tier_platform(edge=0, fog=0, cloud=0))
    with pytest.raises(ConfigError):
        read_pegasus_dax(path, PLATFORM, bytes_per_unit=0.0)
    with pytest.raises(ConfigError):
        read_pegasus_dax(path, PLATFORM, objectives="speed")
    with pytest.raises(OSError, match="absent"):
        read_pegasus_dax(tmp_path / "absent.xml", PLATFORM)


def test_inout_files_are_read_and_written_and_none_files_are_neither(tmp_path):
    both = changed(
        ('<uses file="part" link="output" size="1000000"/>', '<uses file="part" link="inout"/>'),
        ('<uses file="part" link="input" size="1000000"/>', '<uses file="part" link="inout"/>'),
    )
    with pytest.raises(SerializationError, match="'part' passed from job 'ID00001'"):
        read_pegasus_dax(write(tmp_path, both), PLATFORM)
    both = changed(
        (
            '<uses file="part" link="output" size="1000000"/>',
            '<uses file="part" link="inout" size="1000000"/>',
        ),
        ('<uses file="part" link="input" size="1000000"/>', '<uses file="part" link="inout"/>'),
    )
    assert edges(write(tmp_path, both)) == EDGES
    neither = changed(
        ('<uses file="part" link="output" size="1000000"/>', '<uses file="part" link="none"/>')
    )
    expected = [edge if edge[0] != "ID00001" else (*edge[:2], 0.0) for edge in EDGES]
    assert edges(write(tmp_path, neither)) == expected


def test_only_child_elements_create_dependencies(tmp_path):
    # ID00003 writes merged and ID00004 reads it, yet no child element joins them.
    apart = changed(('  <child ref="ID00004">\n    <parent ref="ID00003"/>\n  </child>\n', ""))
    assert edges(write(tmp_path, apart)) == EDGES[:4]
    # A pair that shares no file carries no data, and a repeated pair is counted once,
    # wherever its child element stands.
    more = changed(
        ('<parent ref="ID00003"/>', '<parent ref="ID00003"/>\n    <parent ref="ID00000"/>'),
        (
            '  <job id="ID00000"',
            '  <child ref="ID00004"><parent ref="ID00003"/></child>\n  <job id="ID00000"',
        ),
        ("</adag>", '  <child ref="ID00002"/>\n</adag>'),
    )
    assert edges(write(tmp_path, more)) == [*EDGES[:2], ("ID00000", "ID00004", 0.0), *EDGES[2:]]


def test_numbers_may_be_spaced_or_scaled_and_runtimes_may_be_zero(tmp_path):
    # The white space of XML may surround a number: a space, a tab, a carriage return, and a
    # line feed, the last three written here as character references.
    text = changed(
        ('runtime="20.5"', 'runtime=" 2.05e1 "'),
        ('runtime="30"', 'runtime="&#9;30.&#13;&#10;"'),
        ('runtime="0"', 'runtime="-0.0"'),
        ('link="output" size="2000000"', 'link="output" size="2E6"'),
        ('link="output" size="250000"', 'link="output" size=".25e+6"'),
    )
    reference = read_pegasus_dax(write(tmp_path, DOCUMENT), PLATFORM)
    problem = read_pegasus_dax(write(tmp_path, text), PLATFORM)
    assert problem == reference
    assert problem.task("ID00004").min_duration == 0.0
    assert str(problem.task("ID00004").min_duration) == "0.0"


def test_long_malformed_numbers_are_refused_in_linear_time_and_quoted_in_part(tmp_path):
    digits = "9" * 100_000
    for value in (f"{digits}x", f"1.{digits}x", f".{digits}x", f"1e{digits}x", f"1e-{digits}x"):
        path = write(tmp_path, changed(('runtime="20.5"', f'runtime="{value}"')))
        started = time.perf_counter()
        with pytest.raises(SerializationError) as caught:
            read_pegasus_dax(path, PLATFORM)
        # A pattern that tries every way of sharing the digits between two of its parts
        # needs minutes for one such value; a linear one needs a fraction of a second.
        assert time.perf_counter() - started < 30.0
        assert str(caught.value) == (
            f"{path}: runtime of job 'ID00001' must be a number, "
            f"found {value[:40]!r} and {len(value) - 40} more characters"
        )
    path = write(tmp_path, changed(('runtime="20.5"', f'runtime="{"9" * 400}"')))
    message = f"must be finite and non-negative, found {'9' * 40!r} and 360 more characters"
    with pytest.raises(SerializationError, match=re.escape(message)):
        read_pegasus_dax(path, PLATFORM)


def test_a_runtime_too_large_for_a_node_is_refused(tmp_path):
    path = write(tmp_path, changed(('runtime="20.5"', 'runtime="1e308"')))
    durations = [mode.duration for mode in read_pegasus_dax(path, PLATFORM).task("ID00001").modes]
    assert durations == [PLATFORM.duration(1e308, node) for node in ("edge-0", "fog-0", "cloud-0")]
    assert all(math.isfinite(duration) for duration in durations)
    # On a node of speed 0.5 the same job would run for twice 1e308 seconds.
    slow = three_tier_platform(
        devices=0, edge=1, fog=0, cloud=0, nodes={"edge": NodeSpec(cores=2, speed=0.5)}
    )
    message = "job 'ID00001' with runtime 1e+308 cannot become a task on the platform"
    with pytest.raises(SerializationError, match=re.escape(message)) as caught:
        read_pegasus_dax(path, slow)
    assert str(caught.value).startswith(f"{path}: ")


def test_sizes_too_large_for_the_data_unit_are_refused(tmp_path):
    path = write(tmp_path, changed(('link="output" size="2000000"', 'link="output" size="1e308"')))
    assert edges(path)[0] == ("ID00000", "ID00001", 1e302)
    message = "the files passed from job 'ID00000' to job 'ID00001' are too large"
    with pytest.raises(SerializationError, match=message) as caught:
        read_pegasus_dax(path, PLATFORM, bytes_per_unit=1e-9)
    assert str(caught.value).startswith(f"{path}: ")


DECLARATIONS = {
    "internal entities": (
        '<?xml version="1.0"?>\n<!DOCTYPE adag [\n  <!ENTITY a "aaaaaaaaaa">\n'
        '  <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">\n]>\n'
        '<adag name="&b;"><job id="j" runtime="1"/></adag>\n',
        "DOCTYPE",
    ),
    "external entity": (
        '<!DOCTYPE adag [<!ENTITY e SYSTEM "file:///etc/hostname">]>\n'
        '<adag name="&e;"><job id="j" runtime="1"/></adag>\n',
        "DOCTYPE",
    ),
    "external subset": (
        '<!DOCTYPE adag SYSTEM "https://example.invalid/dax.dtd">\n' + ONE_JOB,
        "DOCTYPE",
    ),
    "lower case": ("<!doctype adag>\n" + ONE_JOB, "DOCTYPE"),
    "in a comment": ("<!-- <!DOCTYPE adag> -->\n" + ONE_JOB, "DOCTYPE"),
    "entity alone": ('<adag><!ENTITY a "b"><job id="j" runtime="1"/></adag>\n', "ENTITY"),
}


@pytest.mark.parametrize("case", DECLARATIONS)
def test_declarations_are_refused_before_the_parser_runs(tmp_path, monkeypatch, case):
    text, kind = DECLARATIONS[case]
    path = write(tmp_path, text)
    forbid_parsing(monkeypatch)
    with pytest.raises(SerializationError, match=re.escape(f"(<!{kind})")) as caught:
        read_pegasus_dax(path, PLATFORM)
    assert str(path) in str(caught.value)


WIDE = {
    "a document type and an entity": ENTITY,
    "a declared encoding": '<?xml version="1.0" encoding="UTF-16"?>' + ENTITY,
    "a leading space": " " + ENTITY,
    "no declaration": ONE_JOB,
}


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"])
@pytest.mark.parametrize("case", WIDE)
def test_utf_16_and_utf_32_without_a_byte_order_mark_are_refused_before_the_parser_runs(
    tmp_path, monkeypatch, case, encoding
):
    # Decoded as UTF-8, these bytes are text with NUL characters between the letters, in
    # which a search finds no declaration, while the parser would read UTF-16 from them.
    data = WIDE[case].encode(encoding)
    assert "<!DOCTYPE" not in data.decode("utf-8")
    path = tmp_path / "wide.xml"
    path.write_bytes(data)
    forbid_parsing(monkeypatch)
    with pytest.raises(SerializationError, match="not UTF-8 text: it contains a NUL") as caught:
        read_pegasus_dax(path, PLATFORM)
    assert str(caught.value).startswith(f"{path}: ")


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le", "utf-16-be"])
@pytest.mark.parametrize("case", ["internal entities", "external entity", "external subset"])
def test_the_tree_builder_refuses_a_document_type_the_first_checks_miss(
    tmp_path, monkeypatch, case, encoding
):
    # Without the checks on the decoded text the parser is given the declaration. It reads
    # text with a NUL among its first two bytes as UTF-16, so it meets the declaration in
    # all three encodings, and its tree builder refuses the document.
    monkeypatch.setattr(pegasus, "_decode", lambda data: data.decode("utf-8"))
    path = tmp_path / "declared.xml"
    path.write_bytes(DECLARATIONS[case][0].encode(encoding))
    with pytest.raises(SerializationError, match=re.escape("(<!DOCTYPE)")) as caught:
        read_pegasus_dax(path, PLATFORM)
    assert str(caught.value).startswith(f"{path}: ")


def test_a_document_without_declarations_reaches_the_parser(tmp_path):
    problem = read_pegasus_dax(write(tmp_path, ONE_JOB), PLATFORM)
    assert [task.id for task in problem.tasks] == ["j"]
    assert dict(problem.tasks[0].metadata) == {}
    assert problem.dependencies == ()


def test_files_are_utf_8_with_an_optional_byte_order_mark(tmp_path):
    reference = read_pegasus_dax(write(tmp_path, DOCUMENT), PLATFORM)
    path = tmp_path / "map-reduce.xml"
    path.write_bytes(b"\xef\xbb\xbf" + DOCUMENT.encode("utf-8"))
    assert read_pegasus_dax(path, PLATFORM) == reference
    accented = changed(('name="map reduce"', 'name="café"'))
    path.write_bytes(accented.encode("utf-8"))
    assert read_pegasus_dax(path, PLATFORM).metadata["name"] == "café"
    # The parser is given the decoded text, so the encoding a document declares is not used.
    mislabelled = accented.replace('encoding="UTF-8"', 'encoding="ISO-8859-1"')
    path.write_bytes(mislabelled.encode("utf-8"))
    assert read_pegasus_dax(path, PLATFORM).metadata["name"] == "café"
    for encoding in ("latin-1", "utf-16", "utf-16-le", "utf-16-be", "utf-32"):
        path.write_bytes(accented.encode(encoding))
        with pytest.raises(SerializationError, match="not UTF-8 text"):
            read_pegasus_dax(path, PLATFORM)


MALFORMED = [
    ([(DOCUMENT, "runtime,10\n")], "invalid XML"),
    ([("</adag>", "")], "invalid XML"),
    (
        [('name="map reduce"', 'name="map\x00reduce"')],
        "the document is not UTF-8 text: it contains a NUL character",
    ),
    ([("<adag ", "<workflow "), ("</adag>", "</workflow>")], "the root element must be adag"),
    ([(DAX_NAMESPACE, "urn:example:other")], "the root element must be adag"),
    ([(' id="ID00001"', "")], "job element 1 needs a non-empty id attribute"),
    ([(' id="ID00001"', ' id=" "')], "job element 1 needs a non-empty id attribute"),
    ([(' id="ID00001"', ' id="ID00001 "')], "id 'ID00001 ' of job element 1 has surrounding"),
    ([(' id="ID00001"', ' id="ID00000"')], "id 'ID00000' is used by two job elements"),
    ([(' runtime="20.5"', "")], "job 'ID00001' needs a non-empty runtime attribute"),
    ([('runtime="20.5"', 'runtime=""')], "job 'ID00001' needs a non-empty runtime attribute"),
    ([('runtime="20.5"', 'runtime="-20.5"')], "runtime of job 'ID00001' must be finite and non"),
    ([('runtime="20.5"', 'runtime="1e999"')], "runtime of job 'ID00001' must be finite and non"),
    ([('runtime="20.5"', 'runtime="fast"')], "runtime of job 'ID00001' must be a number"),
    ([('runtime="20.5"', 'runtime="nan"')], "runtime of job 'ID00001' must be a number"),
    ([('runtime="20.5"', 'runtime="inf"')], "runtime of job 'ID00001' must be a number"),
    ([('runtime="20.5"', 'runtime="20,5"')], "runtime of job 'ID00001' must be a number"),
    ([('runtime="20.5"', 'runtime="2_0"')], "runtime of job 'ID00001' must be a number"),
    # Digits outside ASCII: Arabic-Indic, full-width, and Devanagari.
    (
        [('runtime="20.5"', 'runtime="\N{ARABIC-INDIC DIGIT TWO}\N{ARABIC-INDIC DIGIT ZERO}"')],
        "runtime of job 'ID00001' must be a number",
    ),
    (
        [('runtime="20.5"', 'runtime="\N{FULLWIDTH DIGIT TWO}\N{FULLWIDTH DIGIT ZERO}"')],
        "runtime of job 'ID00001' must be a number",
    ),
    (
        [('runtime="20.5"', 'runtime="\N{DEVANAGARI DIGIT NINE}"')],
        "runtime of job 'ID00001' must be a number",
    ),
    # A no-break space is not white space of XML.
    (
        [('runtime="20.5"', 'runtime="\N{NO-BREAK SPACE}20.5"')],
        "runtime of job 'ID00001' must be a number",
    ),
    (
        [('link="input" size="2000000"', 'link="sideways" size="2000000"')],
        "file 'chunk_a' of job 'ID00001' has link 'sideways', expected one of input, output",
    ),
    (
        [(' link="input" size="2000000"', ' size="2000000"')],
        "uses element 0 of job 'ID00001' needs a non-empty link attribute",
    ),
    (
        [('file="chunk_a" link="input"', 'link="input"')],
        "uses element 0 of job 'ID00001' needs a non-empty file attribute",
    ),
    (
        [('link="input" size="2000000"', 'link="input" size="-2000000"')],
        "size of file 'chunk_a' of job 'ID00001' must be finite and non-negative",
    ),
    (
        [('link="input" size="2000000"', 'link="input" size="2 MB"')],
        "size of file 'chunk_a' of job 'ID00001' must be a number",
    ),
    (
        [
            (
                'link="input" size="2000000"',
                'link="input" size="\N{FULLWIDTH DIGIT TWO}\N{FULLWIDTH DIGIT ZERO}"',
            )
        ],
        "size of file 'chunk_a' of job 'ID00001' must be a number",
    ),
    (
        [('size="7"/>', 'size="7"/>\n    <uses file="log" link="output" size="8"/>')],
        "job 'ID00002' lists output file 'log' with two different sizes",
    ),
    (
        [
            (
                'link="input" size="3100000"/>',
                'link="input"/>\n    <uses file="raw" link="input" size="1"/>\n'
                '    <uses file="raw" link="inout" size="2"/>',
            )
        ],
        "job 'ID00002' lists input file 'raw' with two different sizes",
    ),
    # Two files of a finite size each, whose total is beyond the largest float.
    (
        [
            (
                '<uses file="chunk_a" link="output" size="2000000"/>',
                '<uses file="chunk_a" link="output" size="1e308"/>\n'
                '    <uses file="spare" link="output" size="1e308"/>',
            ),
            (
                '<uses file="chunk_a" link="input" size="2000000"/>',
                '<uses file="chunk_a" link="input" size="2000000"/>\n'
                '    <uses file="spare" link="input"/>',
            ),
        ],
        "the files passed from job 'ID00000' to job 'ID00001' are too large",
    ),
    ([('<child ref="ID00004">', '<child ref="ghost">')], "child element 3 refers to unknown job"),
    ([('<child ref="ID00004">', "<child>")], "child element 3 needs a non-empty ref attribute"),
    (
        [('<parent ref="ID00003"/>', '<parent ref="ghost"/>')],
        "parent element 0 of child 'ID00004' refers to unknown job 'ghost'",
    ),
    (
        [('<parent ref="ID00003"/>', "<parent/>")],
        "parent element 0 of child 'ID00004' needs a non-empty ref attribute",
    ),
    ([('<parent ref="ID00003"/>', '<parent ref="ID00004"/>')], "job 'ID00004' depends on itself"),
    (
        [
            (
                '<parent ref="ID00003"/>',
                '<parent ref="ID00003"/>\n  </child>\n  <child ref="ID00000">\n'
                '    <parent ref="ID00004"/>',
            )
        ],
        "cycle",
    ),
    ([(DOCUMENT, '<adag name="empty"><child ref="a"/></adag>')], "the adag element has no job"),
]


@pytest.mark.parametrize(("replacements", "message"), MALFORMED)
def test_malformed_documents_are_refused(tmp_path, replacements, message):
    path = write(tmp_path, changed(*replacements))
    with pytest.raises(SerializationError, match=re.escape(message)) as caught:
        read_pegasus_dax(path, PLATFORM)
    assert str(caught.value).startswith(f"{path}: ")


def test_transfers_appear_in_decoded_schedules(tmp_path):
    problem = read_pegasus_dax(write(tmp_path, DOCUMENT), PLATFORM)
    decoder = Decoder(problem)
    # split on the edge, the first map in the cloud: 2 MB cross the metro and wide-area links.
    decoded = decoder.decode(list(range(problem.size)), [0, 2, 0, 0, 0])
    link = PLATFORM.network.link("edge", "cloud")
    assert isinstance(link, Link)
    transfer = link.latency + 2.0 / link.bandwidth
    assert decoded.starts[1] == pytest.approx(decoded.ends[0] + transfer)
    local = decoder.decode(list(range(problem.size)), [0, 0, 0, 0, 0])
    assert local.starts[1] == pytest.approx(local.ends[0])
    assert check_schedule(problem, decoded.schedule(problem)) == ()


def test_a_registered_solver_schedules_a_read_workflow(tmp_path):
    problem = read_pegasus_dax(write(tmp_path, DOCUMENT), PLATFORM)
    result = solve(problem, "heft")
    assert result.schedule is not None
    assert check_schedule(problem, result.schedule) == ()
    assert result.value is not None
    assert result.value > 0.0
