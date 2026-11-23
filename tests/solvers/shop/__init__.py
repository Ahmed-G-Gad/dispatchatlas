"""Tests of the job-shop solvers and of the helpers they share."""

import pytest

# The contract helper reports its failures with assertions that show their values.
pytest.register_assert_rewrite("tests.solvers.shop.contract")
