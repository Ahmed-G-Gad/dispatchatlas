"""The bibliography: files load, keys are well formed and consistent, and citations resolve."""

from __future__ import annotations

import re
import tomllib
import unicodedata
from collections import defaultdict
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType

import pytest

from dispatchatlas import references
from dispatchatlas.benchmarks import default_registry as family_registry
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.references import Reference, bibliography, cite, reference, to_bibtex
from dispatchatlas.solvers import default_registry as solver_registry

FOLDER = Path(str(files("dispatchatlas.data").joinpath("references")))
FIELDS = {"authors": str, "year": int, "title": str, "venue": str, "doi": str, "url": str}
KEY = re.compile(r"^(?P<name>[a-z]+)(?P<year>[0-9]{4})(?P<word>[a-z0-9]+)$")
DOI = re.compile(r"^10\.[0-9]{4,9}/\S+$")


def _files() -> list[Path]:
    paths = sorted(FOLDER.glob("*.toml"))
    assert paths, f"no bibliography files in {FOLDER}"
    return paths


def _entries() -> list[tuple[str, str, dict[str, object]]]:
    """Every ``(file name, key, fields)`` of every bibliography file."""
    return [
        (path.name, key, fields)
        for path in _files()
        for key, fields in tomllib.loads(path.read_text(encoding="utf-8")).items()
    ]


def _blocks(path: Path) -> dict[str, str]:
    """The source text of each table of a TOML file, without trailing comments or blanks."""
    blocks: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        header = re.match(r"^\[([^\]]+)\]\s*$", line)
        if header:
            current = blocks.setdefault(header.group(1), [])
        if current is not None:
            current.append(line)
    trimmed: dict[str, str] = {}
    for key, lines in blocks.items():
        while lines and (not lines[-1].strip() or lines[-1].lstrip().startswith("#")):
            lines.pop()
        trimmed[key] = "\n".join(lines)
    return trimmed


def _surname_forms(authors: str) -> set[str]:
    """Lower-case ASCII spellings of the first author's surname, word by word."""
    first = re.split(r",| and ", authors)[0]
    words = [word for word in first.split() if not word.endswith(".")]
    forms = set()
    for count in range(1, len(words) + 1):
        text = unicodedata.normalize("NFKD", "".join(words[:count]))
        ascii_text = "".join(c for c in text if not unicodedata.combining(c)).lower()
        forms.add(re.sub(r"[^a-z]", "", ascii_text))
    return forms


def _cited() -> dict[str, set[str]]:
    """Every source cited by a registered solver or benchmark family, with its citers."""
    cited: dict[str, set[str]] = defaultdict(set)
    for info in solver_registry(plugins=False).infos():
        for source in info.references:
            cited[source].add(f"solver {info.id}")
    for family in family_registry(plugins=False).infos():
        for source in family.references:
            cited[source].add(f"family {family.id}")
    return cited


# -------------------------------------------------------------------- files


def test_every_file_parses_into_well_typed_entries() -> None:
    for name, key, fields in _entries():
        assert set(fields) <= set(FIELDS), f"{name}: {key} has unknown fields"
        for field in ("authors", "year", "title", "venue"):
            assert field in fields, f"{name}: {key} lacks {field}"
        for field, value in fields.items():
            assert isinstance(value, FIELDS[field]), f"{name}: {key}.{field} has the wrong type"
            if isinstance(value, str):
                assert value.strip() == value, f"{name}: {key}.{field}"
                assert value, f"{name}: {key}.{field}"
        year = fields["year"]
        assert isinstance(year, int), f"{name}: {key} year {year}"
        assert 1900 <= year <= 2100, f"{name}: {key} year {year}"


def test_keys_follow_the_citation_key_format() -> None:
    for name, key, fields in _entries():
        match = KEY.match(key)
        assert match, f"{name}: key {key!r} is not <surname><year><word>"
        assert int(match.group("year")) == fields["year"], f"{name}: {key} year"
        authors = fields["authors"]
        assert isinstance(authors, str)
        assert match.group("name") in _surname_forms(authors), f"{name}: {key} vs {authors}"


def test_keys_defined_in_several_files_are_byte_identical() -> None:
    sources: dict[str, dict[str, str]] = defaultdict(dict)
    for path in _files():
        for key, text in _blocks(path).items():
            sources[key][path.name] = text
    for key, texts in sources.items():
        assert len(set(texts.values())) == 1, f"{key} differs between {sorted(texts)}"


def test_links_are_well_formed() -> None:
    for name, key, fields in _entries():
        doi, url = fields.get("doi"), fields.get("url")
        if doi is not None:
            assert isinstance(doi, str), f"{name}: {key} DOI {doi!r}"
            assert DOI.match(doi), f"{name}: {key} DOI {doi!r}"
        if url is not None:
            assert isinstance(url, str), f"{name}: {key} URL"
            assert url.startswith("https://"), f"{name}: {key} URL"
            assert " " not in url


# -------------------------------------------------------------- the library


def test_the_bibliography_merges_every_file() -> None:
    merged = bibliography()
    assert isinstance(merged, MappingProxyType)
    assert list(merged) == sorted(merged)
    assert set(merged) == {key for _, key, _ in _entries()}
    for _, key, fields in _entries():
        assert merged[key] == Reference(key=key, **fields)  # type: ignore[arg-type]


def test_every_cited_source_resolves() -> None:
    known = bibliography()
    for source, citers in sorted(_cited().items()):
        if source.startswith("doi:"):
            assert DOI.match(source[4:]), f"{sorted(citers)} cite a malformed DOI {source!r}"
        elif source.startswith("https://"):
            assert " " not in source
        else:
            assert source in known, f"{sorted(citers)} cite unknown key {source!r}"


def test_reference_lookup_and_rendering() -> None:
    entry = reference("kolisch1996serial")
    assert entry.key == "kolisch1996serial"
    assert entry.link == "https://doi.org/10.1016/0377-2217(95)00357-6"
    assert str(entry) == (
        "R. Kolisch (1996). Serial and parallel resource-constrained project scheduling "
        "methods revisited: Theory and computation. European Journal of Operational "
        "Research 90(2):320-333. https://doi.org/10.1016/0377-2217(95)00357-6"
    )
    with pytest.raises(ConfigError, match="unknown reference key 'nobody2000nothing'"):
        reference("nobody2000nothing")


def test_links_prefer_the_doi_and_fall_back_to_the_url() -> None:
    both = Reference("a2000b", "A. A", 2000, "T", "V", doi="10.1/x", url="https://x.org")
    assert both.link == "https://doi.org/10.1/x"
    assert Reference("a2000b", "A. A", 2000, "T", "V", url="https://x.org").link == "https://x.org"
    bare = Reference("a2000b", "A. A", 2000, "T", "V")
    assert bare.link is None
    assert str(bare) == "A. A (2000). T. V."


def test_cite_renders_keys_dois_and_urls() -> None:
    assert cite("kolisch1996serial") == str(reference("kolisch1996serial"))
    assert cite("doi:10.1287/ijoc.6.2.154") == "https://doi.org/10.1287/ijoc.6.2.154"
    assert cite("10.1287/ijoc.6.2.154") == "https://doi.org/10.1287/ijoc.6.2.154"
    assert cite("https://example.org/paper") == "https://example.org/paper"


def test_conflicting_definitions_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "references"
    folder.mkdir()
    entry = '[doe2001thing]\nauthors = "J. Doe"\nyear = 2001\ntitle = "Thing"\nvenue = "V"\n'
    (folder / "a.toml").write_text(entry, encoding="utf-8")
    (folder / "b.toml").write_text(entry, encoding="utf-8")
    (folder / "notes.txt").write_text("ignored", encoding="utf-8")
    monkeypatch.setattr(references, "files", lambda package: tmp_path)
    assert list(references._load()) == ["doe2001thing"]
    (folder / "c.toml").write_text(entry.replace("Thing", "Other"), encoding="utf-8")
    with pytest.raises(ConfigError, match="reference 'doe2001thing' is defined differently"):
        references._load()


# ------------------------------------------------------------------- BibTeX


def test_a_journal_article_becomes_an_article_entry() -> None:
    assert reference("kolisch1996serial").bibtex() == (
        "@article{kolisch1996serial,\n"
        "  author = {Kolisch, R.},\n"
        "  title = {{Serial and parallel resource-constrained project scheduling methods "
        "revisited: Theory and computation}},\n"
        "  journal = {European Journal of Operational Research},\n"
        "  volume = {90},\n"
        "  number = {2},\n"
        "  pages = {320--333},\n"
        "  year = {1996},\n"
        "  doi = {10.1016/0377-2217(95)00357-6}\n"
        "}"
    )


def test_bibtex_names_keep_multi_word_surnames_whole() -> None:
    authors = "C. A. Coello Coello, M.-Y. Wu and M. Reyes Sierra"
    entry = Reference("coello2004study", authors, 2004, "T", "Journal 1:2")
    names = "{Coello Coello}, C. A. and Wu, M.-Y. and {Reyes Sierra}, M."
    assert f"  author = {{{names}}},\n" in entry.bibtex()
    group = Reference("google2020team", "Google Research", 2020, "T", "V")
    assert "  author = {{Google Research}},\n" in group.bibtex()


def test_bibtex_falls_back_to_misc_marks_editors_and_escapes_latex() -> None:
    title = "Costs & 100% of #1_best"
    book = Reference(
        "doe2001costs",
        "A. Doe and B. Roe (eds.)",
        2001,
        title,
        "Publisher, City",
        url="https://x.org/a_b",
    )
    assert book.bibtex() == (
        "@misc{doe2001costs,\n"
        "  editor = {Doe, A. and Roe, B.},\n"
        "  title = {{Costs \\& 100\\% of \\#1\\_best}},\n"
        "  howpublished = {Publisher, City},\n"
        "  year = {2001},\n"
        "  url = {https://x.org/a_b}\n"
        "}"
    )
    proceedings = Reference("x2020things", "X. Y", 2020, "T", "Proceedings of Things 3:1-2")
    assert proceedings.bibtex().startswith("@misc{x2020things,")


def test_to_bibtex_writes_the_chosen_keys_or_every_entry() -> None:
    chosen = to_bibtex(["wilcoxon1945individual", "kolisch1996serial"])
    assert chosen.startswith("@article{wilcoxon1945individual,")
    assert chosen.index("{wilcoxon1945individual,") < chosen.index("{kolisch1996serial,")
    assert chosen.endswith("}\n\n")
    assert to_bibtex([]) == ""
    everything = to_bibtex()
    headers = [line for line in everything.splitlines() if line.startswith("@")]
    assert len(headers) == len(bibliography())
    for entry in bibliography().values():
        text = entry.bibtex()
        assert text in everything
        assert text.count("{") - text.count("\\{") == text.count("}") - text.count("\\}")
    with pytest.raises(ConfigError, match="unknown reference key 'nobody2000nothing'"):
        to_bibtex(["nobody2000nothing"])
