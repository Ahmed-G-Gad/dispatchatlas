# Command line

The `dispatchatlas` command is a thin layer over the library: every command
calls the public API and prints the result. It is also available as
`python -m dispatchatlas`.

## Conventions

- Results go to standard output, as aligned text or, with `--json`, as JSON.
  Log messages and errors go to standard error; `--quiet` keeps only warnings
  and errors, and `--debug` adds debug messages and the traceback of an
  error.
- A problem argument is a problem document (`.json`), an instance file in a
  published format (see `--format`), or `-` for a problem document on
  standard input, so commands can be chained:

    ```bash
    dispatchatlas benchmarks generate flow-shop --seed 3 | dispatchatlas solve - --solver neh
    ```

- `--set KEY=VALUE` sets a solver or family parameter. The value is read as
  JSON when it parses, else as text: `--set tenure=10`,
  `--set local_search=false`, `--set tiers=edge,cloud`.
- The exit code is 0 on success, 1 when the operation fails (an invalid
  problem, a schedule that breaks a constraint, failed runs, a run that does
  not reproduce), 2 on a usage error, and 130 when the command is
  interrupted.

## Commands

The help below is generated from the installed command line when this site
is built.

<!-- dispatchatlas:cli -->
