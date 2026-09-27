# Publications

Research that uses DispatchAtlas. The list is read from
[`docs/publications.toml`](https://github.com/ahmed-g-gad/dispatchatlas/blob/main/docs/publications.toml)
when the site is built.

<!-- dispatchatlas:publications -->

## Adding a publication

If you used DispatchAtlas in published work, open a pull request that adds one
`[[publication]]` table for it to `docs/publications.toml`, with these fields:

| Field | Required | Value |
| --- | --- | --- |
| `authors` | yes | the authors as printed in the work, joined by commas and "and" |
| `year` | yes | the year of publication, as an integer |
| `title` | yes | the title of the work |
| `venue` | yes | the journal or proceedings, with volume, number, and pages |
| `doi` | when the work has one | the DOI, without `https://doi.org/` |
| `url` | when it has no DOI | a stable URL of the work |

The build refuses an entry with other fields, or without a required one. The
list shows the newest work first.
