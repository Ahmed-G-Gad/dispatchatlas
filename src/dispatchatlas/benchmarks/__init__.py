"""Benchmark families, published instance suites, and instance-file readers.

Generate a problem from a registered family::

    from dispatchatlas.benchmarks import generate
    problem = generate("job-shop", seed=1, jobs=10, machines=5)

Rebuild a published instance, or read one from a file::

    from dispatchatlas.benchmarks import demirkol_job_shop, load_instance, taillard_flow_shop
    ta001 = taillard_flow_shop("ta001")
    dmu01 = demirkol_job_shop("dmu01")
    ft06 = load_instance("ft06.txt", format="orlib-job-shop")

The submodules hold the rest: :mod:`~dispatchatlas.benchmarks.suites` (reference
suites and their known bounds), :mod:`~dispatchatlas.benchmarks.bounds` (best known
bounds with every source's printed values),
:mod:`~dispatchatlas.benchmarks.demirkol` (the job shops of Demirkol, Mehta and Uzsoy),
:mod:`~dispatchatlas.benchmarks.compilations` (dated compilations of best known bounds and
the trivial lower bound), :mod:`~dispatchatlas.benchmarks.catalog`
(named instance sets written to disk with checksums),
:mod:`~dispatchatlas.benchmarks.characterize` (instance features),
:mod:`~dispatchatlas.benchmarks.continuum` (edge-fog-cloud platforms), and the
readers that map published workloads onto such a platform:
:mod:`~dispatchatlas.benchmarks.wfcommons` (WfCommons workflow instances),
:mod:`~dispatchatlas.benchmarks.pegasus` (Pegasus DAX workflow files), and
:mod:`~dispatchatlas.benchmarks.alibaba` (the batch jobs of the Alibaba 2018
cluster trace).
"""

from dispatchatlas.benchmarks.alibaba import iter_alibaba_batch, parse_alibaba_task_name
from dispatchatlas.benchmarks.catalog import build_catalog, catalogs, load_directory, materialize
from dispatchatlas.benchmarks.characterize import features
from dispatchatlas.benchmarks.continuum import three_tier_platform
from dispatchatlas.benchmarks.demirkol import demirkol_job_shop
from dispatchatlas.benchmarks.formats import load_instance
from dispatchatlas.benchmarks.pegasus import read_pegasus_dax
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
    taillard_job_shop_seeded,
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
    "demirkol_job_shop",
    "features",
    "generate",
    "iter_alibaba_batch",
    "load_directory",
    "load_instance",
    "load_plugins",
    "materialize",
    "parse_alibaba_task_name",
    "read_pegasus_dax",
    "read_wfcommons",
    "taillard_flow_shop",
    "taillard_job_shop",
    "taillard_job_shop_seeded",
    "taillard_open_shop",
    "three_tier_platform",
]
