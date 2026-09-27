"""Benchmark families, published instance suites, and instance-file readers.

Generate a problem from a registered family::

    from dispatchatlas.benchmarks import generate
    problem = generate("job-shop", seed=1, jobs=10, machines=5)

Rebuild a published instance, or read one from a file::

    from dispatchatlas.benchmarks import load_instance, taillard_flow_shop
    ta001 = taillard_flow_shop("ta001")
    ft06 = load_instance("ft06.txt", format="orlib-job-shop")

The submodules hold the rest: :mod:`~dispatchatlas.benchmarks.suites` (reference
suites and their known bounds), :mod:`~dispatchatlas.benchmarks.catalog` (named
instance sets written to disk with checksums),
:mod:`~dispatchatlas.benchmarks.characterize` (instance features), and
:mod:`~dispatchatlas.benchmarks.continuum` (edge-fog-cloud platforms).
"""

from dispatchatlas.benchmarks.catalog import build_catalog, catalogs, load_directory, materialize
from dispatchatlas.benchmarks.characterize import features
from dispatchatlas.benchmarks.continuum import three_tier_platform
from dispatchatlas.benchmarks.formats import load_instance
from dispatchatlas.benchmarks.registry import (
    PLUGIN_GROUP,
    FamilyInfo,
    FamilyRegistry,
    Generator,
    default_registry,
    generate,
    load_plugins,
)
from dispatchatlas.benchmarks.taillard import (
    taillard_flow_shop,
    taillard_job_shop,
    taillard_open_shop,
)
from dispatchatlas.benchmarks.wfcommons import read_wfcommons

__all__ = [
    "PLUGIN_GROUP",
    "FamilyInfo",
    "FamilyRegistry",
    "Generator",
    "build_catalog",
    "catalogs",
    "default_registry",
    "features",
    "generate",
    "load_directory",
    "load_instance",
    "load_plugins",
    "materialize",
    "read_wfcommons",
    "taillard_flow_shop",
    "taillard_job_shop",
    "taillard_open_shop",
    "three_tier_platform",
]
