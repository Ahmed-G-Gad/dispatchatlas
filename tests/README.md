# Tests

The suite follows the package. Each folder tests the subpackage of the same name: `core/`,
`benchmarks/`, `solvers/`, `processes/`, `campaigns/`, `analysis/`, `tuning/`, and `cli/`. The files at the top test what
spans them: the documentation (`test_docs.py`), the examples (`test_examples.py`), the package
metadata (`test_package.py`), and the bibliography (`test_references.py`). `helpers.py` holds the
problem factories that several folders share.

`CONTRIBUTING.md` gives the commands that run the suite, its markers `slow` and `exact`, and the
conventions every test follows.
