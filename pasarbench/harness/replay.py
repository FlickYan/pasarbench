"""
Rebuild what an episode saw and did by replaying its recorded tool calls.

Traces store each call's name, arguments, ok/error and the LENGTH of the
payload the agent saw -- from v19 also a digest of it -- not the payload itself.
That left any reader of a trace, human or judge, unable to check a fact the
agent read from a lookup: "the item value is THB 1,500" looks invented when you
cannot see the lookup that said so.

The environment is deterministic -- a fixed clock (db.NOW), counter ids, no
randomness -- so replaying the recorded calls in order against the task's fresh
database reproduces each payload, and leaves the database in the state the
episode ended in. Two things use that: the judges, shown the payloads
(scripts/run_judges.py --payloads), and re-scoring an old run under today's
checks (pasarbench/rescore.py), from the rebuilt state.

VERIFIED BY WHAT. A replayed payload is used only if it matches what the agent
received: by digest where the trace has one (v19 on), else by length. Length is
weak. v18 drew generated tracking numbers from Python's salted str hash, so
every process built the same shipment with a different number of the same
length, and a judge was shown a number the agent never saw -- verified
(WHAT_FAILED #33). So in traces without digests a generated shipment is never
counted as reproduced: its tracking number cannot be recovered.

WHICH WORLD. v19 changed the dates in 87 task worlds (WHAT_FAILED #32). A
trace recorded before then read the old dates, so it is replayed against the
world as it was (`pre_v19_patch`); later traces record a `world_digest`, and one
whose world is not today's is replayed anyway, its digests saying call by call
what still matches. Anything that does not reproduce is reported, never
silently substituted.

WHICH CHECKS. v27 made the tools check each argument against its schema
(tools.CHECKS, WHAT_FAILED #35). A run records the version it ran under, and a
trace without one ran before it: replayed under today's checks, an escalation
it recorded with category "duplicate_refund_claim" or order "unknown" would be
refused, and the state rebuilt would not be the one the episode ended in.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ..db import Database
from ..tasks import world_digest
from ..tools import DEFAULT_TOOLS, call
from .exposure import distractor_names, register_distractors

# Calls the loop answered without dispatching. None of them touched the
# database, so they are reproduced from the recorded error, not re-run.
NO_DISPATCH = ("unknown tool", "arguments were not valid JSON",
               "tool call budget exhausted")


def _payload(out: dict[str, Any]) -> str:
    # Exactly as the loop serialises it, so lengths and digests are comparable.
    return json.dumps(out, ensure_ascii=False, default=str)


def _digest(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------
# The world before v19
# --------------------------------------------------------------------------

_V18_PLACED, _V18_SHIPPED = "2026-11-01 09:00", "2026-11-02 08:00"
_V18_PEAK, _V18_LIVE = "2026-11-10 23:30", "2026-10-20 19:00"


def pre_v19_patch(task) -> dict[str, dict[str, dict[str, Any]]]:
    """The task's database patch as the generator built it before v19.

    v19 changed dates only (WHAT_FAILED #32): every generated order was placed
    on 1 Nov (the peak orders on the 10th) and dispatched on the 2nd, paid
    when placed, and its live was on 20 Oct -- whether it was delivered in
    November or three weeks before it was placed. T01 delivered an order before
    the seed's dispatch date, and T14's earlier refund predated its delivery.
    tests/test_generated.py checks this against digests of the v18 worlds.
    Tracking numbers are left as today's: the old ones came from a salted hash
    and no one can rebuild them."""
    patch = copy.deepcopy(task.db_patch)
    # Generated rows carry a G prefix (GO- orders, GY- payments, GH- shipments,
    # GC- claims); hand-written patches only update seed rows.
    for oid, o in patch.get("orders", {}).items():
        if oid.startswith("GO-") and o.get("created") != _V18_PEAK:
            o["created"] = _V18_PLACED
    for pid, p in patch.get("payments", {}).items():
        if pid.startswith("GY-") and p.get("paid") is not None:
            p["paid"] = _V18_PLACED
    for sid, s in patch.get("shipments", {}).items():
        if sid.startswith("GH-"):
            s["shipped"] = _V18_SHIPPED
    for cid, c in patch.get("livestream_claims", {}).items():
        if cid.startswith("GC-"):
            c["timestamp"] = _V18_LIVE
    if task.task_id == "T01":
        patch["shipments"]["SH2"].pop("shipped", None)
    if task.task_id == "T14":
        patch["refunds"]["REF9999"]["created"] = "2026-11-05 09:00"
    return patch


def _world(head: dict[str, Any], task) -> tuple[dict, str]:
    """(patch the episode ran against, "same" | "changed" | "pre-v19")."""
    recorded = head.get("world_digest")
    if not recorded:
        return pre_v19_patch(task), "pre-v19"
    return task.db_patch, ("same" if recorded == world_digest(task) else "changed")


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------

@dataclass
class Replayed:
    db: Database                       # the state after every recorded call
    payloads: list[str | None]         # per call, trace order; None: not reproduced
    total: int = 0
    verified: int = 0                  # payloads that match the recorded digest/length
    by: str = "length"                 # "digest" when the trace records digests
    world: str = "pre-v19"             # "same" | "changed" | "pre-v19"
    # calls whose success differs from the recording: the rebuilt state may
    # not be the one the episode ended in
    diverged: list[str] = field(default_factory=list)
    # calls whose payload differs from the recording: the world is not the one
    # the episode ran in (an older generator, a changed task)
    mismatched: list[str] = field(default_factory=list)
    # payloads no replay can check, and why: v18's salted tracking numbers
    unrecoverable: list[str] = field(default_factory=list)

    @property
    def unverified(self) -> list[str]:
        return self.mismatched + self.unrecoverable


def replay(recs: list[dict[str, Any]], task) -> Replayed:
    """Replay every recorded call of one episode, in order, on a fresh database.

    search_tools results depend on which distractors the ranker could see,
    which changed between versions (WHAT_FAILED #20): the shared registry and
    the arm's own pool are both tried on a throwaway copy, and whichever
    reproduces the recorded payload is then run once on the real database, so
    the action log -- which some tools read -- matches the original run."""
    head = next((r for r in recs if r.get("type") == "header"), {})
    steps = [r for r in recs if r.get("type") == "step"]
    spec = head.get("exposure") or "default"
    universe = None
    if spec != "default":
        register_distractors(300, 0)          # the old shared registry held 300
    if spec.startswith("search-"):
        n = int(spec.split("-")[1]) - len(DEFAULT_TOOLS)
        universe = set(DEFAULT_TOOLS) | set(distractor_names(n, 0))

    patch, world = _world(head, task)
    checks = int(head.get("tool_checks") or 1)
    db = Database.fresh(patch)
    digests = any("result_sha1" in tr for st in steps for tr in st.get("tool_results", []))
    # A generated shipment's tracking number is unrecoverable in a trace
    # without digests (#33): no payload that carries one is ever "verified".
    salted = set() if digests else {s["order_id"] for sid, s in
                                    task.db_patch.get("shipments", {}).items()
                                    if sid.startswith("GH-")}
    out = Replayed(db=db, payloads=[], by="digest" if digests else "length", world=world)

    def matches(p: str, tr: dict[str, Any]) -> bool:
        if tr.get("result_sha1"):
            return _digest(p) == tr["result_sha1"]
        return tr.get("result_chars") is not None and len(p) == tr["result_chars"]

    for st in steps:
        for tc, tr in zip(st.get("tool_calls", []), st.get("tool_results", [])):
            out.total += 1
            # As recorded. Version 1 read any empty value as no arguments; a
            # version-2 run refuses a non-object, so a recorded [] or "" must
            # reach the tools as itself to be refused again.
            name, args = tc["name"], tc.get("arguments")
            if args is None or (checks < 2 and not args):
                args = {}
            err = tr.get("error") or ""
            where = f"step {st.get('step')} {name}"
            if any(err.startswith(p) for p in NO_DISPATCH):
                res = {"ok": False, "error": err}
            elif name == "search_tools":
                res = None
                for uni in (None, universe):
                    trial = copy.deepcopy(db)
                    trial.search_universe = uni
                    if matches(_payload(call(trial, name, args, checks=checks)), tr):
                        db.search_universe = uni
                        res = call(db, name, args, checks=checks)
                        break
                if res is None:                # neither reproduced: still keep the
                    db.search_universe = None  # action log in step with the run
                    res = call(db, name, args, checks=checks)
            else:
                res = call(db, name, args, checks=checks)
            p = _payload(res)
            # Success is what decides the state: a call that failed changed
            # nothing but its line in the action log, and no check reads an
            # error's wording (which can differ between Python versions).
            if bool(res.get("ok")) != bool(tr.get("ok")):
                out.diverged.append(f"{where}: recorded ok={tr.get('ok')} {err[:60]!r}, "
                                    f"replayed ok={res.get('ok')} "
                                    f"{(res.get('error') or '')[:60]!r}")
            if name == "get_shipment" and args.get("order_id") in salted:
                out.payloads.append(None)
                out.unrecoverable.append(f"{where}: tracking number from a salted hash (#33)")
                if not matches(p, tr):         # the length at least must agree
                    out.mismatched.append(f"{where}: payload differs from the recorded length")
            elif matches(p, tr):
                out.verified += 1
                out.payloads.append(p)
            else:
                out.payloads.append(None)
                out.mismatched.append(f"{where}: payload differs from the recorded "
                                      f"{'digest' if tr.get('result_sha1') else 'length'}")
    return out


def replay_payloads(recs: list[dict[str, Any]], task) -> dict[str, Any]:
    """Payload per tool call, in trace order, or None where it did not reproduce.

    Returns {"payloads": [...], "verified": n, "total": m, "by": "digest" |
    "length", "world": ...}."""
    r = replay(recs, task)
    return {"payloads": r.payloads, "verified": r.verified, "total": r.total,
            "by": r.by, "world": r.world}
