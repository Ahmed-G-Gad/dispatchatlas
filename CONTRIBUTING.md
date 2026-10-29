# Contributing to DispatchAtlas

Contributions are welcome: bug reports, fixes, new solvers and benchmark
families, and documentation. This guide explains how to set up a development
environment, the conventions the code follows, and how to check a change
before you open a pull request. Everyone who takes part agrees to the
[code of conduct](https://github.com/ahmed-g-gad/dispatchatlas/blob/main/CODE_OF_CONDUCT.md).

## Reporting issues

Open an [issue](https://github.com/ahmed-g-gad/dispatchatlas/issues) for a bug,
a question, or a proposal. For a bug, include the output of
`dispatchatlas version`, the smallest problem or campaign file that shows it,
the command or code you ran, and what you expected. Report security issues
privately, as described in the
[security policy](https://github.com/ahmed-g-gad/dispatchatlas/blob/main/SECURITY.md).

A new solver or benchmark family does not have to live in this repository: a
separate package can register it through an entry point, as the
[extending guide](https://dispatchatlas.ahmedgad.com/extending/) explains.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) to manage its environment.
The lock file `uv.lock` pins every dependency, so all contributors and the
continuous integration run the same versions. The continuous integration runs
uv 0.12.19. Change the lock file with the same version, `uvx uv@0.12.19 lock`,
so that it changes only where your change requires.

```bash
git clone https://github.com/ahmed-g-gad/dispatchatlas.git
cd dispatchatlas
uv sync                        # .venv with the package and the development tools
uv sync --extra exact          # also the backends of the exact solvers
uv run pre-commit install      # run the formatting and lint hooks on every commit
```

## Checks

Every pull request must pass the same checks as the continuous integration:

```bash
uv run ruff format --check .   # formatting; `uv run ruff format .` fixes it
uv run ruff check .            # lint rules, including docstrings
uv run mypy                    # strict type checking of src, tests, and examples
uv run mypy --platform win32   # the same, as Windows sees the code
uv run mypy --platform darwin  # the same, as macOS sees the code
uv run pytest                  # the test suite
uv run pytest -m "not slow"    # the suite without the slow tests
uv run pre-commit run --all-files
```

Tests that need the exact solvers' backends are marked `exact` and skip
themselves when the backends are missing; run them after
`uv sync --extra exact`.

The documentation is built with MkDocs from `docs/` and `mkdocs.yml`:

```bash
uv run --group docs mkdocs build --strict   # fails on any warning
uv run --group docs mkdocs serve            # preview at http://127.0.0.1:8000
```

The solver and benchmark catalogues, the bibliography, the command-line
reference, and several tables are generated from the code while the site
builds (`docs/hooks/catalogues.py`), so change the code, not those pages. The
feature counts in `README.md` are checked against the registries by
`tests/test_docs.py`.

## Conventions

### Code

- Python 3.12 or newer, `from __future__ import annotations`, full type
  hints, and `mypy --strict` clean.
- Line length 100, formatted by `ruff format`, and clean under `ruff check`
  with the rules in `pyproject.toml`.
- No `print` in library code: log through
  `dispatchatlas.core.logs.get_logger(__name__)`.
- Raise the classes of `dispatchatlas.core.errors`. An invalid parameter value
  raises `ValueError` in `__post_init__`; the registries report it as a
  `ConfigError`.
- Randomness comes only from the generator a component is given
  (`Search.rng`, `Search.numpy_rng()`, `random.Random(seed)`, or
  `numpy.random.default_rng(seed)`), never from global state. Equal inputs and
  seed must give identical outputs.
- Optional dependencies (`ortools`, `pulp`, `highspy`, `pyscipopt`,
  `gurobipy`) are imported inside the functions that need them, and their
  absence raises `MissingDependencyError`.

### Docstrings

- Every public module, class, and function has a Google-style docstring: a
  one-line summary, then plain prose on what the object does and why.
- Describe algorithms by mechanism: what the operators do to the search. Name
  the published method and cite it. Metaphorical vocabulary appears only in
  the official names of methods.
- Describe the current behaviour only: no history, dates, or comparisons with
  other versions, and no promotional adjectives.

### The solver contract

- One dataclass per solver. Its parameters are fields with defaults, checked
  in `__post_init__`.
- Its metadata is `info: ClassVar[SolverInfo]`: a kebab-case `id`, a `name`, a
  `family` (a key of `SOLVER_FAMILIES` in `dispatchatlas.core.solver`), a
  one-sentence `summary` of the mechanism ending with a period, `references`
  (bibliography keys), `stochastic`, `iterative`, `multi_objective`,
  `supports` (the problem features the solver models correctly), and
  `requires` (the extras it needs).
- Order-based solvers run inside `Search`: `with search:` around the loop,
  `search.next_iteration()` at the top of every main-loop iteration,
  `search.evaluate(order, modes)` for every candidate, `search.check()` inside
  long loops that evaluate nothing, and `return search.result()`. This makes
  every solver honour time, evaluation, iteration, target, and patience
  limits, and report exact counts.
- An iterative solver works with any single bounded criterion: only
  `max_evaluations`, only `max_iterations`, or only `time_limit`.
- The module exposes `register(registry)`, which registers every solver it
  defines, and is listed in `BUILT_IN_MODULES` of
  `dispatchatlas.solvers.registry`.
- `tests/solvers/test_contract.py` runs every registered solver through the
  contract. Add tests of the algorithm itself: hand-computed small cases,
  known optima of tiny instances, properties of the operators, and, where it
  applies, that the search beats random sampling at an equal number of
  evaluations on a seeded instance.

### Bibliography

Solvers, benchmark families, suites, figures, and statistical methods cite
their sources by key. The entries live in TOML files under
`src/dispatchatlas/data/references/`, one per area, with the fields `authors`,
`year`, `title`, `venue`, and optionally `doi` or `url`. A key is the first
author's surname in lowercase, the year, and the first significant word of
the title, for example `deb2002fast`.

Verify every entry, its authors, title, venue, year, and DOI, against a
bibliographic source such as the publisher's page or Crossref. Never invent a
DOI: if you cannot verify one, give a stable URL instead, or leave the entry
out. A key defined in two files must be identical in both.
`tests/test_references.py` checks the format of every entry and that every
cited key resolves.

### Tests

Every change comes with tests. Keep them fast, and mark a test that takes more
than a few seconds with `@pytest.mark.slow`. Tests must not depend on the
speed of the machine beyond generous margins.

## Pull requests

1. Create a branch from `main` and keep the change focused on one topic.
2. Add tests and documentation for what you change, and a line under the
   upcoming version in `CHANGELOG.md`.
3. Run the checks above.
4. Open the pull request and describe what the change does and why. The
   continuous integration runs the checks on Linux, macOS, and Windows.

## Releases

A maintainer sets the version in `pyproject.toml` and `CITATION.cff`, adds
`date-released` to `CITATION.cff`, dates the version's heading in
`CHANGELOG.md`, and publishes a GitHub release with the tag `v<version>`. The
release workflow then builds the source distribution and the wheel, checks
them, and publishes them to PyPI with trusted publishing.
