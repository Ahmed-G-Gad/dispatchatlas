# Example plugin

An installable package that adds one solver to DispatchAtlas: `insertion-descent`, a descent by
random insertion moves that restarts when it stops improving. `pyproject.toml` declares the entry
point that DispatchAtlas reads, and `src/dispatchatlas_example_plugin/__init__.py` holds the solver.

```bash
pip install ./examples/plugin
dispatchatlas solvers show insertion-descent
```

The [extending guide](https://dispatchatlas.ahmedgad.com/extending/) explains each part, and
`tests/test_examples.py` checks the plugin as the suite checks the built-in solvers.
