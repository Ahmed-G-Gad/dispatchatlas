# Security policy

## Supported versions

Security fixes are made in the latest release.

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |

## Reporting a vulnerability

Please do not report security issues in public issues. Report them privately
through GitHub's private vulnerability reporting (the "Report a
vulnerability" button in the repository's Security tab), or by email to
me@ahmedgad.com. Describe the issue, the affected version, and the steps that
reproduce it. You will receive a reply acknowledging the report, and news as
the issue is investigated and fixed. Once a fix is released, the release notes
describe the issue, with credit to the reporter unless they prefer otherwise.

## What DispatchAtlas trusts

DispatchAtlas reads problem documents, campaign files, run records, and
instance files as data: JSON, TOML, and the text formats of published suites.
It never executes code from them. Names from campaign files that become file
and directory names are restricted to letters, digits, `.`, `_`, and `-`.

Plugins are different: a package that registers solvers or benchmark families
through an entry point runs its code with your permissions whenever
DispatchAtlas builds its registries. Install only plugins you trust, or call
`default_registry(plugins=False)` to load the built-in solvers and families
alone.
