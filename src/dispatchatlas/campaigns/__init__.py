"""Reproducible, resumable experiment campaigns.

A campaign runs every solver of a roster on every problem of its instance
sources, under every stopping protocol, with seeds shared across solvers, and
stores one JSON record per run::

    from dispatchatlas.campaigns import load_campaign, read_runs, run

    campaign = load_campaign("campaign.toml")
    summary = run(campaign, "results")  # resumes when called again
    records = read_runs(summary.directory)

The modules, in the order a campaign meets them:

- :mod:`~dispatchatlas.campaigns.config` and :mod:`~dispatchatlas.campaigns.io`:
  the campaign and its strict TOML or JSON file, and the config hash;
- :mod:`~dispatchatlas.campaigns.planning`: problems, seeds, run ids, and the
  plan, checked by :mod:`~dispatchatlas.campaigns.compatibility` and
  :mod:`~dispatchatlas.campaigns.fairness`;
- :mod:`~dispatchatlas.campaigns.estimation`: run counts and time bounds;
- :mod:`~dispatchatlas.campaigns.runner`, :mod:`~dispatchatlas.campaigns.backends`,
  :mod:`~dispatchatlas.campaigns.execution`,
  :mod:`~dispatchatlas.campaigns.timeout`, and
  :mod:`~dispatchatlas.campaigns.processors`: execution with retries, hard time
  limits, resumption, and the processors the workers run on;
- :mod:`~dispatchatlas.campaigns.records`, :mod:`~dispatchatlas.campaigns.store`,
  and :mod:`~dispatchatlas.campaigns.environment`: what is written, where, and
  in which environment;
- :mod:`~dispatchatlas.campaigns.replaying`: verification that runs reproduce.
"""

from dispatchatlas.campaigns.compatibility import Caveat, Compatibility, Exclusion
from dispatchatlas.campaigns.config import (
    Campaign,
    CatalogSource,
    DirectorySource,
    Execution,
    FamilySource,
    FileSource,
    InstanceSource,
    Protocol,
    SolverSpec,
    SuiteSource,
)
from dispatchatlas.campaigns.environment import capture_environment
from dispatchatlas.campaigns.estimation import Estimate, estimate
from dispatchatlas.campaigns.fairness import Fairness
from dispatchatlas.campaigns.io import (
    campaign_from_dict,
    campaign_to_dict,
    load_campaign,
    loads_campaign,
)
from dispatchatlas.campaigns.planning import (
    Plan,
    PlannedProblem,
    PlannedRun,
    check_compatibility,
    plan,
)
from dispatchatlas.campaigns.records import (
    Category,
    FailureRecord,
    Outcome,
    Provenance,
    RunRecord,
    SessionRecord,
)
from dispatchatlas.campaigns.replaying import ReplayEntry, ReplayReport, Verdict, replay
from dispatchatlas.campaigns.runner import RunSummary, run
from dispatchatlas.campaigns.store import (
    Strays,
    campaign_directory,
    read_campaign,
    read_failures,
    read_plan,
    read_runs,
    read_sessions,
    read_strays,
)

__all__ = [
    "Campaign",
    "CatalogSource",
    "Category",
    "Caveat",
    "Compatibility",
    "DirectorySource",
    "Estimate",
    "Exclusion",
    "Execution",
    "FailureRecord",
    "Fairness",
    "FamilySource",
    "FileSource",
    "InstanceSource",
    "Outcome",
    "Plan",
    "PlannedProblem",
    "PlannedRun",
    "Protocol",
    "Provenance",
    "ReplayEntry",
    "ReplayReport",
    "RunRecord",
    "RunSummary",
    "SessionRecord",
    "SolverSpec",
    "Strays",
    "SuiteSource",
    "Verdict",
    "campaign_directory",
    "campaign_from_dict",
    "campaign_to_dict",
    "capture_environment",
    "check_compatibility",
    "estimate",
    "load_campaign",
    "loads_campaign",
    "plan",
    "read_campaign",
    "read_failures",
    "read_plan",
    "read_runs",
    "read_sessions",
    "read_strays",
    "replay",
    "run",
]
