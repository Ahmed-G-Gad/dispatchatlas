"""Package hygiene: version, type marker, public names, and lazy optional dependencies."""

from __future__ import annotations

import importlib
import importlib.metadata
import pkgutil
import re
import subprocess
import sys
import tomllib
from importlib.resources import files
from pathlib import Path

import dispatchatlas

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ("ortools", "pulp", "highspy", "matplotlib", "torch", "sklearn")


def _public_modules() -> list[str]:
    names = [
        info.name
        for info in pkgutil.walk_packages(dispatchatlas.__path__, "dispatchatlas.")
        if not any(part.startswith("_") for part in info.name.split("."))
    ]
    return ["dispatchatlas", *sorted(names)]


def test_the_version_matches_the_project_metadata() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert dispatchatlas.__version__ == project["version"]
    assert dispatchatlas.__version__ == importlib.metadata.version("dispatchatlas")
    assert re.fullmatch(r"\d+\.\d+\.\d+", dispatchatlas.__version__)
    assert dispatchatlas.__all__ == ["__version__"]


def test_the_package_ships_its_type_marker() -> None:
    assert files("dispatchatlas").joinpath("py.typed").is_file()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packages = project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "src/dispatchatlas" in packages
    assert (ROOT / "src" / "dispatchatlas" / "py.typed").is_file()
    assert "Typing :: Typed" in project["project"]["classifiers"]


def test_every_public_name_exists() -> None:
    checked = 0
    for name in _public_modules():
        module = importlib.import_module(name)
        exported = getattr(module, "__all__", None)
        if exported is None:
            continue
        assert all(isinstance(item, str) for item in exported), name
        assert len(set(exported)) == len(exported), f"{name}.__all__ repeats a name"
        missing = [item for item in exported if not hasattr(module, item)]
        assert not missing, f"{name}.__all__ names missing attributes {missing}"
        checked += 1
    assert checked >= 20


def test_no_package_name_shadows_one_of_its_submodules() -> None:
    for name in _public_modules():
        package = importlib.import_module(name)
        for info in pkgutil.iter_modules(getattr(package, "__path__", [])):
            if info.name.startswith("_"):
                continue
            submodule = importlib.import_module(f"{name}.{info.name}")
            assert getattr(package, info.name) is submodule, f"{name}.{info.name} is shadowed"
    import dispatchatlas.core.objectives as module  # noqa: PLC0415

    assert module is importlib.import_module("dispatchatlas.core.objectives")


def test_importing_the_package_loads_no_optional_dependency() -> None:
    script = "\n".join(
        [
            "import sys",
            f"heavy = set({HEAVY!r})",
            "def loaded():",
            "    print(sorted({name.split('.')[0] for name in sys.modules} & heavy))",
            "import dispatchatlas",
            "loaded()",
            "import dispatchatlas.core, dispatchatlas.solvers, dispatchatlas.benchmarks",
            "import dispatchatlas.campaigns, dispatchatlas.analysis, dispatchatlas.references",
            "from dispatchatlas.solvers import default_registry",
            "from dispatchatlas.benchmarks import default_registry as families",
            "default_registry(plugins=False)",
            "families(plugins=False)",
            "loaded()",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["[]", "[]"]
