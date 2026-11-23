# Tuning

Automatic configuration of a solver's parameters by iterated racing: parameter spaces, the racing of configurations with the Friedman test and Conover's comparisons, and elitist iterated racing with a budget of runs. The caller supplies the function that runs the experiments; `process_run_batch` is one, which runs them in worker processes and resumes from their rows.

::: dispatchatlas.tuning
    options:
      members: false

::: dispatchatlas.tuning.space

::: dispatchatlas.tuning.racing

::: dispatchatlas.tuning.irace

::: dispatchatlas.tuning.batch
