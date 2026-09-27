# Solver catalogue

Every solver of DispatchAtlas, generated from the solver registry when this
site is built. For each solver: what it does, the parameters it takes with
their defaults, the problem features it does not model, the extra it needs,
and the sources it implements. The same facts are available from the command
line with `dispatchatlas solvers show ID`, and in Python from
`default_registry().info(id)`.

"Stochastic" solvers draw from the run's seed, "deterministic" ones ignore
it. Solvers that run "until a stop criterion" need a time, evaluation, or
iteration limit; the others finish on their own and still honour the limits
they are given (see [Solvers](../guide/solvers.md)).

<!-- dispatchatlas:solvers -->
