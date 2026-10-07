"""
The tasks for the tool-naming confirmation (docs/RUNBOOK.md, 1h).

Every task that cannot pass without one of the three tools the policy
describes but does not name -- escalation, shipment lookup, goodwill voucher:
a required action that no other tool satisfies. A task that a read through
`get_order` also satisfies (an address change after dispatch, say) is left
out; naming `get_shipment` predicts nothing there, and it would only dilute
the test. Minus the 32 tasks of the first naming run, which produced the
hypothesis (I-tools2) and its first test (J), so they cannot also confirm it.
The rule for reading the run is written in the runbook before the run.

    TASKS=$(python scripts/naming_tasks.py)          # 57 ids, comma-separated
    python scripts/naming_tasks.py --count           # 57
    TASKS=$(python scripts/naming_tasks.py --first-run)   # the 32, for 1i
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The 32 tasks of I-tools2 and of the first naming run (RUNBOOK 1g).
FIRST_RUN = (
    "ACAD-VN", "CCNRD-PH", "CCNRD-TH", "CCRTOM-PH", "CCRTOM-TH", "CCSO-VN.vi", "CHE-MY",
    "CHE-VN", "CWP-ID", "CWP-SG.sg-en", "DRE-PH", "HPRR-PH", "HPRR-SG.sg-en", "HRWR-MY",
    "HRWR-SG.sg-en", "HVPRF-PH", "IVF-MY", "IVF-SG.sg-en", "LCOW-MY", "LCOW-TH",
    "OOWDE-MY", "OOWDE-VN.vi", "OOWOV-TH", "OOWOV-VN", "PPDNC-SG", "PPDNC-TH",
    "PRWR-VN.vi", "T04", "T07", "T09", "T14", "T15",
)


def needs_named(task) -> bool:
    """A required action only one of the three named tools satisfies."""
    from pasarbench.harness.prompts import NAMED_TOOLS
    return any(set(a.tools) <= set(NAMED_TOOLS) for a in task.checks.required_actions)


def tasks() -> list[str]:
    from pasarbench.rescore import all_tasks
    return [tid for tid, t in sorted(all_tasks().items())
            if tid not in FIRST_RUN and needs_named(t)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--count", action="store_true", help="print how many, not which")
    ap.add_argument("--first-run", action="store_true",
                    help="the 32 tasks of the first run instead (the guardrail run, 1i)")
    a = ap.parse_args(argv)
    ids = list(FIRST_RUN) if a.first_run else tasks()
    print(len(ids) if a.count else ",".join(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
