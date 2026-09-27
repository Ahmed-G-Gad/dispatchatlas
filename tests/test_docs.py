"""The documentation: README counts, README quickstarts, API coverage, and project metadata."""

from __future__ import annotations

import contextlib
import io
import pkgutil
import re
import shlex
import shutil
import tomllib
from pathlib import Path

import pytest

import dispatchatlas
from dispatchatlas.analysis.figures import CATALOG
from dispatchatlas.benchmarks import default_registry as default_families
from dispatchatlas.benchmarks import suites
from dispatchatlas.cli import main
from dispatchatlas.core import registered_objectives
from dispatchatlas.solvers import default_registry as default_solvers

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
COUNT = re.compile(
    r"\*\*(?P<count>\d+) (?P<what>benchmark families|published suites|solvers|objectives|"
    r"figure families)\*\*"
)
PUBLICATION_FIELDS = {"authors", "year", "title", "venue", "doi", "url"}


def _section(text: str, heading: str) -> str:
    """Return the text of a ``##`` section, up to the next ``##`` heading."""
    start = text.index(f"## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start : end if end >= 0 else len(text)]


def _blocks(text: str, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", text, flags=re.DOTALL)


def test_the_readme_counts_match_the_registries() -> None:
    expected = {
        "benchmark families": len(default_families(plugins=False)),
        "published suites": len(suites.suites()),
        "solvers": len(default_solvers(plugins=False)),
        "objectives": len(registered_objectives()),
        "figure families": len(CATALOG),
    }
    found = [(match["what"], int(match["count"])) for match in COUNT.finditer(README)]
    assert sorted(what for what, _ in found) == sorted(expected)
    assert dict(found) == expected


def test_the_python_quickstart_prints_what_the_readme_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    section = _section(README, "Quickstart in Python")
    (code,) = _blocks(section, "python")
    (shown,) = _blocks(section, "text")
    monkeypatch.chdir(tmp_path)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "README.md", "exec"), {"__name__": "readme"})
    assert output.getvalue() == shown


@pytest.mark.slow
def test_the_command_line_quickstart_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (script,) = _blocks(_section(README, "Quickstart on the command line"), "bash")
    commands = [shlex.split(line) for line in script.splitlines() if line.strip()]
    assert commands
    assert all(command[0] == "dispatchatlas" for command in commands)
    campaigns = tmp_path / "examples" / "campaigns"
    campaigns.mkdir(parents=True)
    shutil.copy(ROOT / "examples" / "campaigns" / "quickstart.toml", campaigns)
    monkeypatch.chdir(tmp_path)
    for command in commands:
        assert main(command[1:]) == 0, " ".join(command)
    assert "33 match, 0 mismatch" in capsys.readouterr().out


def test_every_public_module_is_in_the_api_reference() -> None:
    public = {"dispatchatlas"} | {
        info.name
        for info in pkgutil.walk_packages(dispatchatlas.__path__, "dispatchatlas.")
        if not any(part.startswith("_") for part in info.name.split("."))
    }
    documented = set()
    for page in (ROOT / "docs" / "api").glob("*.md"):
        documented |= set(re.findall(r"^::: (\S+)$", page.read_text(encoding="utf-8"), re.M))
    assert sorted(public - documented) == []
    assert sorted(documented - public) == []


def test_every_api_page_is_in_the_navigation() -> None:
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    pages = {f"api/{path.name}" for path in (ROOT / "docs" / "api").glob("*.md")}
    assert {page for page in pages if page not in config} == set()


def test_the_publications_file_is_well_formed() -> None:
    document = tomllib.loads((ROOT / "docs" / "publications.toml").read_text(encoding="utf-8"))
    assert set(document) <= {"publication"}
    for entry in document.get("publication", []):
        assert set(entry) <= PUBLICATION_FIELDS
        assert {"authors", "year", "title", "venue"} <= set(entry)
        assert isinstance(entry["year"], int)


def test_the_citation_metadata_names_this_version() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    (version,) = re.findall(r"^version: (\S+)$", citation, flags=re.M)
    assert version == dispatchatlas.__version__
    assert re.search(r"^cff-version: 1\.2\.0$", citation, flags=re.M)


def test_the_changelog_has_an_entry_for_this_version() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(rf"^## \[{re.escape(dispatchatlas.__version__)}\] - ", changelog, re.M)
