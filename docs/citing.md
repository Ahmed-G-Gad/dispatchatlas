# Citing

If you use DispatchAtlas in your research, please cite the version you used.
The citation metadata lives in
[`CITATION.cff`](https://github.com/ahmed-g-gad/dispatchatlas/blob/main/CITATION.cff),
which GitHub renders as APA and BibTeX under "Cite this repository". The
BibTeX entry below is generated from that file:

<!-- dispatchatlas:citation -->

## Citing the methods

Every solver, benchmark family, suite, figure, and statistical method records
the works it implements. When you compare methods or use a benchmark, cite
those works too. The command line exports them as BibTeX:

```bash
dispatchatlas references --solver nsga-ii --solver heft --format bibtex > methods.bib
dispatchatlas references --suite taillard-flow-shop --format bibtex > benchmarks.bib
```

The analysis bundle of a campaign lists, in its `index.md`, the sources of
every statistical method and figure it used. The full list is in the
[bibliography](catalogue/bibliography.md).
