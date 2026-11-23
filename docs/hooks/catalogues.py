"""Generated content of the documentation: catalogues, bibliography, and reference tables.

MkDocs runs this hook while it builds the site (``hooks:`` in ``mkdocs.yml``).
A page asks for generated content with a marker, an HTML comment of the form
``<!-- dispatchatlas:NAME -->`` on a line of its own. The hook replaces it with
Markdown computed from the package the site is built with: the solver and
benchmark registries, the published suites and catalogs, the bibliography, the
objectives, the findings of the campaign check, the statistical methods, the
figure catalog, the help of the command line, the publications list, and the
citation metadata. Generated
pages therefore always describe the code they document. An unknown marker
stops the build.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import tomllib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.exceptions import PluginError
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

from dispatchatlas.analysis.figures import CATALOG
from dispatchatlas.analysis.methods import implemented_methods
from dispatchatlas.benchmarks import catalogs, suites
from dispatchatlas.benchmarks import default_registry as default_families
from dispatchatlas.benchmarks.characterize import DEFINITIONS
from dispatchatlas.benchmarks.formats import EXTENSIONS, FORMATS
from dispatchatlas.campaigns.checking import CODES
from dispatchatlas.cli import build_parser
from dispatchatlas.core import FEATURES, SHAPES, SOLVER_FAMILIES, registered_objectives
from dispatchatlas.references import Reference, bibliography
from dispatchatlas.solvers import default_registry as default_solvers

MARKER = re.compile(r"^<!-- dispatchatlas:(?P<name>[a-z-]+) -->$", re.MULTILINE)
ROOT = Path(__file__).resolve().parents[2]
SOLVERS_PAGE = "catalogue/solvers.md"
BENCHMARKS_PAGE = "catalogue/benchmarks.md"
BIBLIOGRAPHY_PAGE = "catalogue/bibliography.md"
PUBLICATION_FIELDS = {"authors", "year", "title", "venue", "doi", "url"}


def on_page_markdown(markdown: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    """Replace every marker of a page with its generated Markdown."""

    def generate(match: re.Match[str]) -> str:
        name = match["name"]
        if name not in GENERATORS:
            raise PluginError(f"{page.file.src_uri}: unknown marker 'dispatchatlas:{name}'")
        return GENERATORS[name](page.file.src_uri)

    return MARKER.sub(generate, markdown)


# ----------------------------------------------------------------- helpers


def _link(page: str, target: str, anchor: str) -> str:
    """Return the relative Markdown path from ``page`` to ``target#anchor``."""
    if page == target:
        return f"#{anchor}"
    path = PurePosixPath(target).relative_to(PurePosixPath(page).parent, walk_up=True)
    return f"{path}#{anchor}"


def _api(obj: object) -> str:
    """Return an optional cross-reference to the API documentation of a class."""
    identifier = f"{obj.__module__}.{obj.__qualname__}"  # type: ignore[attr-defined]
    name = html.escape(obj.__qualname__)  # type: ignore[attr-defined]
    return f'<autoref identifier="{identifier}" optional><code>{name}</code></autoref>'


def _surnames(reference: Reference) -> str:
    """Return the short author label of an entry: one, two, or the first et al."""
    names = re.split(r",\s*|\s+and\s+", re.sub(r"\s*\(eds?\.\)\s*$", "", reference.authors))
    surnames = []
    for name in names:
        words = name.split()
        while len(words) > 1 and words[0].endswith("."):
            words = words[1:]
        surnames.append(" ".join(words))
    if len(surnames) == 1:
        return surnames[0]
    if len(surnames) == 2:
        return f"{surnames[0]} and {surnames[1]}"
    return f"{surnames[0]} et al."


def _sources(keys: Iterable[str], page: str) -> str:
    """Return the sources of a solver, family, or method as links to the bibliography."""
    entries = bibliography()
    links = []
    for key in keys:
        if key in entries:
            entry = entries[key]
            label = f"{_surnames(entry)} ({entry.year})"
            links.append(f"[{label}]({_link(page, BIBLIOGRAPHY_PAGE, key)})")
        else:
            url = (
                f"https://doi.org/{key.removeprefix('doi:')}"
                if key.startswith(("10.", "doi:"))
                else key
            )
            links.append(f"<{url}>")
    return "; ".join(links) if links else "none"


def _value(value: Any) -> str:
    """Render a parameter default as the Python literal it is, in code."""
    return f"`{value!r}`"


def _cell(text: str) -> str:
    """Escape a text for a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


# ----------------------------------------------------------------- solvers


def solvers(page: str) -> str:
    """The solver catalogue: an overview table, then every solver by family."""
    registry = default_solvers(plugins=False)
    infos = registry.infos()
    order = list(SOLVER_FAMILIES)
    groups: dict[str, list[Any]] = defaultdict(list)
    for info in infos:
        groups[info.family].append(info)
    families = sorted(groups, key=lambda family: (order.index(family), family))
    lines = [
        f"DispatchAtlas registers {len(infos)} solvers in {len(families)} families. "
        "Every solver can be run by its id from Python, from the command line, and "
        "in campaign files.",
        "",
        "| Solver | Family | Runs | Extra |",
        "| --- | --- | --- | --- |",
    ]
    for family in families:
        for info in groups[family]:
            lines.append(
                f"| [`{info.id}`](#{info.id}) | {family} | {_traits(info)} | "
                f"{', '.join(info.requires) or '-'} |"
            )
    for family in families:
        lines += ["", f"## {family.replace('-', ' ').capitalize()} {{ #{family} }}", ""]
        lines += [SOLVER_FAMILIES.get(family, ""), ""]
        for info in groups[family]:
            lines += _solver(info, registry, page)
    return "\n".join(lines)


def _traits(info: Any) -> str:
    traits = ["stochastic" if info.stochastic else "deterministic"]
    traits.append("until a stop criterion" if info.iterative else "to completion")
    if info.multi_objective:
        traits.append("multi-objective")
    return ", ".join(traits)


def _solver(info: Any, registry: Any, page: str) -> list[str]:
    solver = registry.create(info.id)
    parameters = ", ".join(f"`{name}` = {_value(value)}" for name, value in info.parameters.items())
    missing = sorted(set(FEATURES) - set(info.supports))
    items = [
        f"- **Id**: `{info.id}`; implemented by {_api(type(solver))}",
        f"- **Runs**: {_traits(info)}",
        f"- **Parameters**: {parameters or 'none'}",
        f"- **Does not model**: {', '.join(missing)}" if missing else "- **Models**: every feature",
    ]
    if info.shape is not None:
        items.append(f"- **Shape**: `{info.shape}`. {SHAPES[info.shape]}")
    if info.requires:
        items.append(f"- **Needs**: the {', '.join(f'`{extra}`' for extra in info.requires)} extra")
    items.append(f"- **Sources**: {_sources(info.references, page)}")
    return [f"### {info.name} {{ #{info.id} }}", "", info.summary, "", *items, ""]


# -------------------------------------------------------------- benchmarks


def families(page: str) -> str:
    """Every registered benchmark family, with parameters, objectives, and sources."""
    registry = default_families(plugins=False)
    infos = registry.infos()
    lines = [
        f"{len(infos)} families are registered.",
        "",
        "| Family | Problem class | Summary |",
        "| --- | --- | --- |",
    ]
    lines += [f"| [`{i.id}`](#{i.id}) | {i.problem_class} | {_cell(i.summary)} |" for i in infos]
    for info in infos:
        generator = registry.create(info.id)
        lines += [
            "",
            f"### {info.name} {{ #{info.id} }}",
            "",
            info.summary,
            "",
            f"- **Id**: `{info.id}`; implemented by {_api(type(generator))}",
            f"- **Problem class**: {info.problem_class}",
            f"- **Objectives**: {', '.join(f'`{name}`' for name in info.objectives)}",
            f"- **Sources**: {_sources(info.references, page)}",
            "",
            "| Parameter | Default |",
            "| --- | --- |",
        ]
        lines += [f"| `{name}` | {_value(value)} |" for name, value in info.parameters.items()]
        lines += ["", "```bash", f"dispatchatlas benchmarks generate {info.id} --seed 1", "```"]
    return "\n".join(lines)


def suite_table(page: str) -> str:
    """Every published suite, with its instances, source, and files."""
    rows = suites.suites()
    lines = [
        f"{len(rows)} suites are registered.",
        "",
        "| Suite | Problem class | Instances | Loads |",
        "| --- | --- | ---: | --- |",
    ]
    for suite in rows:
        loads = "offline, from the published seeds" if suite.generated else "from downloaded files"
        lines.append(
            f"| [`{suite.id}`](#{suite.id}) | {suite.problem_class} | {len(suite.instances)} | "
            f"{loads} |"
        )
    for suite in rows:
        first, last = suite.instances[0], suite.instances[-1]
        lines += [
            "",
            f"### {suite.name} {{ #{suite.id} }}",
            "",
            suite.notes,
            "",
            f"- **Id**: `{suite.id}`; instances `{first}` to `{last}`",
            f"- **Source**: <{suite.source}>",
        ]
        if suite.files is not None:
            lines.append(f"- **Files to download**: {suite.files}")
        if suite.bounds:
            optimal = sum(bound.optimal for bound in suite.bounds.values())
            lines.append(
                f"- **Bounds**: published for {len(suite.bounds)} instances, "
                f"{optimal} of them proven optimal"
            )
        lines.append(f"- **Sources**: {_sources(suite.references, page)}")
        directory = "" if suite.generated else " --directory downloads"
        lines += ["", "```bash", f"dispatchatlas suites load {suite.id} {first}{directory}", "```"]
    return "\n".join(lines)


def catalog_table(page: str) -> str:
    """The built-in instance catalogs."""
    lines = ["| Catalog | Instances | Families | Description |", "| --- | ---: | --- | --- |"]
    for name, catalog in catalogs().items():
        names = ", ".join(f"[`{family}`](#{family})" for family in catalog.families)
        lines.append(f"| `{name}` | {catalog.size} | {names} | {_cell(catalog.description)} |")
    return "\n".join(lines)


def format_table(page: str) -> str:
    """The instance-file formats and the extensions that imply them."""
    implied: dict[str, list[str]] = defaultdict(list)
    for extension, name in EXTENSIONS.items():
        implied[name].append(f"`{extension}`")
    lines = ["| Format | Reader | Implied by |", "| --- | --- | --- |"]
    for name, reader in FORMATS.items():
        lines.append(
            f"| `{name}` | {_api(reader)} | {', '.join(implied[name]) or 'the format argument'} |"
        )
    return "\n".join(lines)


def instance_features(page: str) -> str:
    """The descriptive features of an instance, with their definitions."""
    lines = ["| Feature | Definition |", "| --- | --- |"]
    lines += [f"| `{name}` | {_cell(text)} |" for name, text in DEFINITIONS.items()]
    return "\n".join(lines)


# --------------------------------------------------------------- reference


def objectives(page: str) -> str:
    """The registered objectives."""
    lines = ["| Objective | Sense | Unit | Measures |", "| --- | --- | --- | --- |"]
    lines += [
        f"| `{o.name}` | {o.sense} | {o.unit} | {_cell(o.description)} |"
        for o in registered_objectives()
    ]
    return "\n".join(lines)


def problem_features(page: str) -> str:
    """The structural features a problem can exhibit."""
    lines = ["| Feature | Meaning |", "| --- | --- |"]
    lines += [f"| `{name}` | {text} |" for name, text in FEATURES.items()]
    return "\n".join(lines)


def check_findings(page: str) -> str:
    """The findings of the check of a campaign directory, with severity and meaning."""
    lines = ["| Code | Severity | Meaning |", "| --- | --- | --- |"]
    lines += [
        f"| `{code}` | {severity} | {_cell(text)} |" for code, (severity, text) in CODES.items()
    ]
    return "\n".join(lines)


def solver_families(page: str) -> str:
    """The standard solver families, with the number of registered solvers in each."""
    counts: dict[str, int] = defaultdict(int)
    for info in default_solvers(plugins=False).infos():
        counts[info.family] += 1
    lines = ["| Family | Solvers | Mechanism |", "| --- | ---: | --- |"]
    lines += [
        f"| [`{name}`]({_link(page, SOLVERS_PAGE, name)}) | {counts[name]} | {text} |"
        for name, text in SOLVER_FAMILIES.items()
    ]
    return "\n".join(lines)


def methods(page: str) -> str:
    """The statistical methods the analysis implements, with their sources."""
    lines = ["| Method | Kind | What it does | Sources |", "| --- | --- | --- | --- |"]
    for method in implemented_methods():
        lines.append(
            f"| {method.name} | {method.family} | {_cell(method.summary)} | "
            f"{_sources(method.references, page)} |"
        )
    return "\n".join(lines)


def figures(page: str) -> str:
    """The figure families, in the order a bundle writes them."""
    lines = ["| Figure | File | Shows | Sources |", "| --- | --- | --- | --- |"]
    for spec in CATALOG:
        stem = f"`{spec.prefix}<problem>`" if spec.prefix else f"`{spec.id}`"
        lines.append(
            f"| {spec.title} | {stem} | {_cell(spec.caption)} | "
            f"{_sources(spec.references, page) if spec.references else '-'} |"
        )
    return "\n".join(lines)


def references_list(page: str) -> str:
    """The bibliography, with the solvers, families, suites, and methods citing each entry."""
    citing: dict[str, list[str]] = defaultdict(list)
    for info in default_solvers(plugins=False).infos():
        for key in info.references:
            citing[key].append(f"solver [`{info.id}`]({_link(page, SOLVERS_PAGE, info.id)})")
    for family in default_families(plugins=False).infos():
        for key in family.references:
            citing[key].append(f"family [`{family.id}`]({_link(page, BENCHMARKS_PAGE, family.id)})")
    for suite in suites.suites():
        for key in suite.references:
            citing[key].append(f"suite [`{suite.id}`]({_link(page, BENCHMARKS_PAGE, suite.id)})")
    for spec in CATALOG:
        for key in spec.references:
            citing[key].append(f"figure {spec.title.lower()}")
    for method in implemented_methods():
        for key in method.references:
            citing[key].append(f"method {method.name}")
    entries = bibliography()
    lines = [f"The bibliography holds {len(entries)} entries, sorted by citation key.", ""]
    for key, entry in entries.items():
        link = f" <{entry.link}>" if entry.link else ""
        text = f"**`{key}`**: {html.escape(str(entry).removesuffix(' ' + (entry.link or '')))}"
        lines.append(f"{text}{link}")
        if citing[key]:
            lines.append(f"<br>Cited by {', '.join(citing[key])}.")
        lines += [f"{{ #{key} }}", ""]
    return "\n".join(lines)


# --------------------------------------------------------- command line


def command_line(page: str) -> str:
    """The help of every command and subcommand, as the installed command line prints it."""
    saved = {name: os.environ.get(name) for name in ("COLUMNS", "NO_COLOR")}
    os.environ.update(COLUMNS="88", NO_COLOR="1")
    try:
        lines: list[str] = []
        for path, parser in _parsers(build_parser(), ["dispatchatlas"]):
            if len(path) > 1:
                level = "#" * len(path)
                command = " ".join(path)
                lines += ["", f"{level} `{command}` {{ #{'-'.join(path[1:])} }}", ""]
            lines += ["```text", parser.format_help().rstrip(), "```"]
        return "\n".join(lines)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _parsers(
    parser: argparse.ArgumentParser, path: list[str]
) -> Iterable[tuple[list[str], argparse.ArgumentParser]]:
    yield path, parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                yield from _parsers(child, [*path, name])


# ----------------------------------------------------------- publications


def publications(page: str) -> str:
    """The publications listed in ``docs/publications.toml``, newest first."""
    path = ROOT / "docs" / "publications.toml"
    entries = tomllib.loads(path.read_text(encoding="utf-8")).get("publication", [])
    for number, entry in enumerate(entries, start=1):
        unknown = set(entry) - PUBLICATION_FIELDS
        missing = {"authors", "year", "title", "venue"} - set(entry)
        if unknown or missing:
            raise PluginError(
                f"{path.name}: publication {number} has unknown fields {sorted(unknown)} "
                f"or lacks {sorted(missing)}"
            )
    if not entries:
        return "The list has no entries."
    lines = []
    for entry in sorted(entries, key=lambda item: (-int(item["year"]), item["authors"])):
        link = f"https://doi.org/{entry['doi']}" if "doi" in entry else entry.get("url")
        text = f"{entry['authors']} ({entry['year']}). {entry['title']}. {entry['venue']}."
        lines.append(f"- {text} <{link}>" if link else f"- {text}")
    return "\n".join(lines)


# --------------------------------------------------------------- citation


def citation(page: str) -> str:
    """A BibTeX entry for the software, read from ``CITATION.cff``."""
    document = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    authors = " and ".join(
        f"{person['family-names']}, {person['given-names']}" for person in document["authors"]
    )
    fields = [
        ("author", authors),
        ("title", f"{{{document['title']}}}"),
        ("version", document["version"]),
        ("url", document["repository-code"]),
        ("license", document["license"]),
    ]
    if "date-released" in document:
        fields.append(("year", str(document["date-released"])[:4]))
    if "doi" in document:
        fields.append(("doi", document["doi"]))
    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
    return f"```bibtex\n@software{{dispatchatlas,\n{body}\n}}\n```"


GENERATORS: Mapping[str, Callable[[str], str]] = {
    "solvers": solvers,
    "families": families,
    "suites": suite_table,
    "catalogs": catalog_table,
    "formats": format_table,
    "instance-features": instance_features,
    "objectives": objectives,
    "problem-features": problem_features,
    "solver-families": solver_families,
    "check-findings": check_findings,
    "methods": methods,
    "figures": figures,
    "bibliography": references_list,
    "cli": command_line,
    "publications": publications,
    "citation": citation,
}
"""The markers a page can use, and the function that generates each."""
