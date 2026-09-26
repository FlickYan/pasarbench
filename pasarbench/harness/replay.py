"""
Regenerate every tool result in a trace by replaying its calls.

Traces store each call's name, arguments, ok/error and the LENGTH of the
payload the agent saw -- not the payload itself. That left any reader of a
trace, human or judge, unable to check a fact the agent read from a lookup:
"the item value is THB 1,500" looks invented when you cannot see the lookup
that said so.

The environment is deterministic -- a fixed clock (db.NOW), counter ids, no
randomness -- so replaying the recorded calls in order against the task's
fresh database reproduces each payload. The recorded length verifies it call
by call: a replayed payload is used only if it matches the length the agent
actually received, and anything that does not reproduce is reported, never
silently substituted.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from ..db import Database
from ..tools import DEFAULT_TOOLS, call
from .exposure import distractor_names, register_distractors

# Calls the loop answered without dispatching. None of them touched the
# database, so they are reproduced from the recorded error, not re-run.
NO_DISPATCH = ("unknown tool", "arguments were not valid JSON",
               "tool call budget exhausted")


def _payload(out: dict[str, Any]) -> str:
    # Exactly as the loop serialises it, so lengths are comparable.
    return json.dumps(out, ensure_ascii=False, default=str)


def replay_payloads(recs: list[dict[str, Any]], task) -> dict[str, Any]:
    """Payload per tool call, in trace order, or None where it did not reproduce.

    Returns {"payloads": [...], "verified": n, "total": m}. search_tools
    results depend on which distractors the ranker could see, which changed
    between versions (WHAT_FAILED #20): both the shared registry and the arm's
    own pool are tried on a throwaway copy, and whichever reproduces the
    recorded length is then run once on the real database, so the action log
    -- which some tools read -- matches the original run exactly.
    """
    head = next((r for r in recs if r.get("type") == "header"), {})
    steps = [r for r in recs if r.get("type") == "step"]
    spec = head.get("exposure") or ""
    universe = None
    if spec.startswith("search-"):
        n = int(spec.split("-")[1]) - len(DEFAULT_TOOLS)
        register_distractors(300, 0)          # the old shared registry held 300
        universe = set(DEFAULT_TOOLS) | set(distractor_names(n, 0))

    db = Database.fresh(task.db_patch)
    payloads: list[str | None] = []
    verified = total = 0
    for st in steps:
        for tc, tr in zip(st.get("tool_calls", []), st.get("tool_results", [])):
            total += 1
            want = tr.get("result_chars")
            err = tr.get("error") or ""
            if any(err.startswith(p) for p in NO_DISPATCH):
                p = _payload({"ok": False, "error": err})
            elif tc["name"] == "search_tools":
                p = None
                for uni in (None, universe):
                    trial = copy.deepcopy(db)
                    trial.search_universe = uni
                    if len(_payload(call(trial, tc["name"], tc.get("arguments") or {}))) == want:
                        db.search_universe = uni
                        p = _payload(call(db, tc["name"], tc.get("arguments") or {}))
                        break
                if p is None:                  # neither reproduced: still keep the
                    db.search_universe = None  # action log in step with the run
                    call(db, tc["name"], tc.get("arguments") or {})
            else:
                p = _payload(call(db, tc["name"], tc.get("arguments") or {}))
            if p is not None and want is not None and len(p) == want:
                verified += 1
                payloads.append(p)
            else:
                payloads.append(None)
    return {"payloads": payloads, "verified": verified, "total": total}
