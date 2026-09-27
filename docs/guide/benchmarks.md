# Benchmarks

DispatchAtlas provides problems in four ways: **families** generate them from
parameters and a seed, **published suites** rebuild or read the instances of
the literature, **file readers** load single instances, and **catalogs** write
named sets of generated instances to disk. The
[benchmark catalogue](../catalogue/benchmarks.md) lists everything that is
registered, with every parameter and source.

## Families

A benchmark family is a parameterised generator. Equal parameters and seed
give the same problem, identified by an id that records both:

```python
from dispatchatlas.benchmarks import default_registry, generate

problem = generate("flow-shop", seed=7, jobs=10, machines=5)
print(problem.id)  # flow-shop-10x5-7

registry = default_registry()
print(registry.ids())
print(dict(registry.info("rcpsp").parameters))
```

```bash
dispatchatlas benchmarks list
dispatchatlas benchmarks show rcpsp
dispatchatlas benchmarks generate rcpsp --seed 3 --set activities=30 --set resource_strength=0.3 --output rcpsp.json
```

The built-in families cover:

- **Shops**: `flow-shop`, `job-shop`, `open-shop`, `flexible-job-shop`, and
  `hybrid-flow-shop`. Processing times follow Taillard (1993), integers drawn
  uniformly from `[1, 99]` by default. Due dates and weights, when asked for,
  follow the weighted-tardiness benchmark of Singer and Pinedo (1998).
- **Parallel machines**: `parallel-machines`, unrelated machines with optional
  sequence-dependent setups, release dates, and due dates, after Vallada and
  Ruiz (2011).
- **Projects**: `rcpsp` and `multi-mode-rcpsp`, networks and resources drawn
  as ProGen draws them (Kolisch, Sprecher and Drexl 1995), from the network
  complexity, resource factor, and resource strength.
- **Task graphs**: `workflow`, the random task graphs on heterogeneous
  processors behind the HEFT experiments (Topcuoglu, Hariri and Wu 2002), with
  a set communication-to-computation ratio.
- **Continuum**: `continuum-offloading`, `continuum-workflow`, and
  `continuum-pipeline` place the tasks of applications issued by devices on
  device, edge, fog, and cloud nodes (see below).

Every family names its instances `<family>-<size>-<seed>`, with the
parameters that set the instance apart inserted before the seed, as in
`rcpsp-30x4-nc1.5-rf0.5-rs0.3-7`, and records its parameters and seed in the
problem's metadata. The docstring of each family
module states exactly what it draws and where it departs from its source.

### The edge-fog-cloud continuum

The continuum families model placement as a decision. A platform
(`three_tier_platform`) holds devices that issue work and edge, fog, and cloud
nodes of growing speed and distance; a node is a resource whose capacity is
its number of cores, and a network joins the tiers. Every compute task has one
mode per node it may run on, with duration `work / speed` on one core. Each
application starts and ends on its device, through zero-duration source and
sink tasks, so offloading pays the upload and the download. Nodes and links
carry power, price, latency, and bandwidth, so `makespan`,
`mean_response_time`, `energy`, and `cost` all measure consequences of
placement, and trade off against each other.

The default hardware and link parameters are illustrative orders of
magnitude, chosen so that no tier is best on every objective. They are not
measurements: override them (`NodeSpec`, `Link`) for any study that depends on
their values. The building blocks are public, so other workloads can use the
same model:

```python
from dispatchatlas.benchmarks.continuum import (
    Application,
    NodeSpec,
    build_problem,
    three_tier_platform,
)

fast_cloud = NodeSpec(cores=16, speed=6.0, idle_power=60.0, busy_power=300.0)
platform = three_tier_platform(devices=2, cloud=2, nodes={"cloud": fast_cloud})
request = Application("a0", "device-0", release=0.0, work=(2.0,), upload=(1.5,), download=(0.1,))
problem = build_problem("one-request", platform, [request], due_factor=2.0)
```

`read_wfcommons` maps a recorded execution of a real scientific workflow, in
the WfCommons format (Coleman et al. 2022), onto such a platform.

## Taillard's instances

Taillard (1993) defined his 120 flow-shop, 80 job-shop, and 60 open-shop
instances by a portable random number generator and one or two seeds per
instance. DispatchAtlas implements that generator and carries the published
seed tables, so every instance is rebuilt offline and identical to the
published data:

```python
from dispatchatlas.benchmarks import taillard_flow_shop, taillard_job_shop, taillard_open_shop

ta001 = taillard_flow_shop("ta001")  # 20 jobs, 5 machines, permutation flow shop
ta01 = taillard_job_shop("ta01")  # 15 jobs, 15 machines
tai4x4 = taillard_open_shop("tai4x4-1")  # 4 jobs, 4 machines
```

The open-shop suite also records the bounds published with its seeds. With
the `exact` extra installed, CP-SAT proves the published optimum of
`tai4x4-1`:

```python
from dispatchatlas.benchmarks import suites
from dispatchatlas.solvers import solve

print(solve(ta001, "neh").value)  # 1286.0
result = solve(tai4x4, "cp-sat", time_limit=30)
print(result.status, result.value)  # optimal 193.0
bound = suites.suite("taillard-open-shop").bounds["tai4x4-1"]
print(bound.lower, bound.upper, bound.optimal)  # 186 193 True
```

## Published suites

`dispatchatlas.benchmarks.suites` describes each published instance set: its
instances, where they come from, the works that define them, and any bounds
published with them. Taillard's suites are **generated**. The OR-Library job
and flow shops, Brandimarte's flexible job shops, and the PSPLIB projects are
**file** suites: the library does not bundle their files, and reads them from
a directory where you downloaded them from the suite's source.

```python
from dispatchatlas.benchmarks import suites

ta002 = suites.load("taillard-flow-shop", "ta002")
ft06 = suites.load("orlib-job-shop", "ft06", directory="downloads/orlib")
```

```bash
dispatchatlas suites list
dispatchatlas suites show psplib-j30        # instances, source, and the files to download
dispatchatlas suites load orlib-job-shop ft06 --directory downloads/orlib --output ft06.json
```

The directory is searched recursively for the file each instance needs, so
the downloaded archives can be extracted as they are.

## File formats

`load_instance(path, format=...)` reads one instance file with the reader of
its format. The `.fjs`, `.sm`, and `.mm` extensions imply their format;
OR-Library files, usually `.txt`, need `format`, and files holding many
instances need `instance`:

```python
from dispatchatlas.benchmarks import load_instance

ft06 = load_instance("jobshop1.txt", format="orlib-job-shop", instance="ft06")
mk01 = load_instance("Mk01.fjs")
j301 = load_instance("j301_1.sm")
```

Every reader builds its problem with the same builders as the generated
families, so a read instance and a generated one have the same shape, and a
malformed file raises `SerializationError` naming the file and the offending
value. PSPLIB instances drop their dummy source and sink; a nonrenewable
resource that can constrain a schedule is refused, since the model has
renewable resources only.

## Catalogs

A catalog is a named, declarative list of family settings, each with an
instance count. `build_catalog` generates every instance, deriving each seed
from one root seed with `derive_seed` under the namespace
`<catalog>/<entry>` and the instance index, so an instance never depends on
the entries listed before it. `materialize` writes a set of problems as
`problems/<id>.json` with a `manifest.json` that records each file's SHA-256
digest, family, parameters, and seed; nothing in either depends on the clock
or the machine, so writing the same problems twice gives identical bytes.
`load_directory` reads them back and checks every file against its digest.

```bash
dispatchatlas catalogs list
dispatchatlas catalogs build smoke --output instances/smoke --seed 1
```

```python
from dispatchatlas.benchmarks import build_catalog, load_directory, materialize

problems = build_catalog("continuum", seed=1)
materialize(problems, "instances/continuum")
again = load_directory("instances/continuum")
```

A campaign reads a written catalog as a `directory` source, or builds a
catalog itself as a `catalog` source (see [Campaigns](campaigns.md)).

## Instance features

`features(problem)` summarises a problem as numbers that describe its size,
structure, flexibility, resources, communication, and timing, so instance sets
can be compared, stratified, and related to solver performance. Each feature
is a deterministic function of the problem alone; the
[catalogue](../catalogue/benchmarks.md#instance-features) defines every one.

```python
from dispatchatlas.benchmarks import features, taillard_flow_shop

values = features(taillard_flow_shop("ta001"))
print(values["critical_path"], values["parallelism"], values["area_bound"])
```

## Adding families

A family is a dataclass whose fields are its parameters and whose
`generate(seed)` method returns a problem. Other packages register families
through the `dispatchatlas.benchmarks` entry-point group, as
[Extending](../extending.md#a-benchmark-family) shows.
