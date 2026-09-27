# Benchmark catalogue

Every benchmark family, published suite, instance catalog, file format, and
instance feature of DispatchAtlas, generated from the registries when this
site is built. The [benchmarks guide](../guide/benchmarks.md) explains how to
use them.

## Families

A family generates one problem per seed from its parameters. Parameters not
given keep the defaults listed here.

<!-- dispatchatlas:families -->

## Published suites

<!-- dispatchatlas:suites -->

## Catalogs

A catalog is a named list of family settings with an instance count each.
`dispatchatlas catalogs build NAME --output DIR` writes its instances with a
manifest of their digests.

<!-- dispatchatlas:catalogs -->

## File formats

`load_instance` and `dispatchatlas solve --format` read these formats.
OR-Library files hold many instances; name one with `instance`.

<!-- dispatchatlas:formats -->

## Instance features

`dispatchatlas.benchmarks.features(problem)` returns these numbers for any
problem.

<!-- dispatchatlas:instance-features -->
