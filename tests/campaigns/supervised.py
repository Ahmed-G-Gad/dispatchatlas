"""A campaign whose only run never returns, for the test that kills its supervising process.

Run as ``python -m tests.campaigns.supervised <directory> <marker>``. The pid
of this process is written beside the marker before the campaign starts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dispatchatlas.campaigns import FamilySource, SolverSpec, run
from tests.campaigns.solvers import registry, tiny_campaign


def main(directory: str, marker: str) -> None:
    """Run the campaign in ``directory``; its solver writes its pids to ``marker``."""
    campaign = tiny_campaign(
        "orphans",
        solvers=(SolverSpec("test-nested", parameters={"marker": marker, "seconds": 120.0}),),
        instances=(FamilySource("flow-shop", (1,), {"jobs": 3, "machines": 2}),),
        replicates=1,
        backend="process",
    )
    Path(marker + ".supervisor").write_text(str(os.getpid()), encoding="ascii")
    run(campaign, Path(directory), registry=registry())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
