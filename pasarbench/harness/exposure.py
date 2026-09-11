"""
Tool exposure.

THE EXPERIMENT, AND THE CONFOUND THAT RUINS IT
----------------------------------------------
"Agents get worse with more tools" is easy to show and tells you nothing,
because two different things degrade at once:

    TOKEN COST       300 schemas is ~25k tokens injected on every single call
    SELECTION        300 near-neighbours make picking the right one harder

A two-arm study (20 tools vs 300 tools) cannot separate them, so its finding is
uninterpretable. Four arms can:

    oracle       only the ~6 tools the task actually needs      ceiling
    all-20       the real registry, no distractors              baseline
    all-N        registry + distractors, N in {50, 100, 300}    both effects
    random-N     N tools sampled at random but ALWAYS containing
                 the needed ones                                selection only,
                                                                token cost held
                                                                at the same N
    search-N     only meta-tools visible; the agent retrieves   the proposed fix

`random-N` is the arm that does the work. It has the same token cost as
`all-N` and the same guarantee as `oracle` that the answer is reachable. If
`random-N` tracks `all-N`, the damage is token cost and dilution. If `random-N`
tracks `oracle`, the damage is selection.

PROGRESSIVE DISCLOSURE
----------------------
`search-N` exposes a small core plus `search_tools`. Tools returned by a search
become callable on subsequent steps. Which tools are visible is DERIVED from
the action log by replaying past `search_tools` calls -- no extra state, so it
survives interrupt/resume for free.
"""

from __future__ import annotations

import random
import re
from typing import Any, Protocol

from ..db import Database
from ..tools import DEFAULT_TOOLS, TOOLS, _ok, tool
from .types import EpisodeState

# --------------------------------------------------------------------------
# Distractors
# --------------------------------------------------------------------------
# Plausible internal tools of a real marketplace. They must be believable
# enough to be genuine near-neighbours -- a distractor nobody would ever
# consider does not test selection -- and completely inert: none mutates the
# world, and no verifier check references any of them.

_DOMAINS = [
    ("seller", ["metrics", "rating", "payout", "warehouse", "sla_breach",
                "onboarding_status", "commission_tier", "penalty_points"]),
    ("campaign", ["voucher_pool", "flash_slot", "banner", "budget",
                  "eligibility", "performance", "co_funding"]),
    ("catalog", ["attribute_quality", "duplicate_listing", "image_audit",
                 "category_mapping", "restricted_keyword", "price_history"]),
    ("logistics", ["lane_capacity", "courier_sla", "pickup_slot", "zone_coverage",
                   "oversize_flag", "cod_remittance"]),
    ("risk", ["fraud_score", "chargeback_history", "velocity_check",
              "device_fingerprint", "blocklist_status"]),
    ("finance", ["settlement_batch", "tax_invoice", "fx_rate_lock",
                 "wallet_ledger", "escrow_release"]),
    ("content", ["livestream_schedule", "short_video_stats", "creator_tier",
                 "affiliate_link", "comment_moderation"]),
    ("account", ["loyalty_points", "tier_history", "notification_pref",
                 "address_book", "linked_device", "consent_record"]),
    ("support", ["ticket_queue", "macro_library", "csat_survey", "agent_roster",
                 "sla_timer", "handoff_history"]),
    ("marketplace", ["fee_schedule", "policy_version", "region_config",
                     "currency_rounding", "holiday_calendar", "peak_window"]),
]
_VERBS = ["get", "list", "check", "summarise", "lookup"]


def distractor_names(n: int, seed: int = 0) -> list[str]:
    combos = [f"{v}_{d}_{f}" for d, fields in _DOMAINS for f in fields for v in _VERBS]
    rng = random.Random(seed)
    rng.shuffle(combos)
    return combos[:n]


_REGISTERED: set[str] = set()


def register_distractors(n: int, seed: int = 0) -> list[str]:
    """Idempotent. Registers inert read-only tools into the global registry."""
    names = distractor_names(n, seed)
    for name in names:
        if name in TOOLS:
            continue
        _make_distractor(name)
        _REGISTERED.add(name)
    return names


def _make_distractor(name: str) -> None:
    domain = name.split("_")[1]
    desc = (f"Internal {domain} operations lookup. Returns {domain} records for "
            f"reporting. Not used for customer refunds, returns or cancellations.")

    @tool(name, desc, {"entity_id": {"type": "string"},
                       "window": {"type": "string",
                                  "enum": ["7d", "30d", "90d"]}}, ["entity_id"])
    def _fn(db: Database, entity_id: str, window: str = "30d", _n=name) -> dict:
        # Inert: returns an empty, plausible payload. Deliberately NOT an error
        # -- an error would teach the agent that unfamiliar tools fail, which
        # is a different lesson from the one this experiment is measuring.
        return _ok(db, _n, {"entity_id": entity_id, "window": window},
                   records=[], window=window,
                   note="no records for this entity in the selected window")


# --------------------------------------------------------------------------
# search_tools meta-tool
# --------------------------------------------------------------------------

CORE_VISIBLE = ["search_tools", "get_order", "list_user_orders", "search_policy",
                "verify_identity"]


@tool("search_tools",
      "Find tools by what you want to do. Returns matching tool names and "
      "descriptions; those tools then become available to call.",
      {"query": {"type": "string",
                 "description": "what you are trying to do, e.g. 'refund a cash order'"},
       "limit": {"type": "integer", "description": "max results, default 6"}},
      ["query"])
def search_tools(db: Database, query: str, limit: int = 6) -> dict:
    hits = _rank_tools(query, limit)
    return _ok(db, "search_tools", {"query": query, "limit": limit},
               tools=[{"name": n, "description": d} for n, d in hits],
               note=None if hits else "no match; try different words")


def _rank_tools(query: str, limit: int = 6) -> list[tuple[str, str]]:
    """Keyword ranking over names and descriptions.

    Deliberately keyword-based, not embedding-based, and that IS a finding
    waiting to happen: a Thai or Vietnamese query scores zero against English
    tool descriptions. `diagnose.py` measures exactly that, and it is one of
    the mechanisms behind the multilingual gap rather than a bug to hide.
    """
    terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
    scored = []
    for name, spec in TOOLS.items():
        if name == "search_tools":
            continue
        fn = spec["schema"]["function"]
        hay = (name + " " + fn["description"]).lower()
        score = sum(3 if t in name.lower() else hay.count(t) for t in terms)
        if score:
            scored.append((score, name, fn["description"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(n, d) for _, n, d in scored[:limit]]


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------

class ExposureStrategy(Protocol):
    name: str
    def tools_for(self, state: EpisodeState, db: Database, task) -> list[str]: ...


class AllTools:
    """Every real tool, plus `n_distractors` plausible irrelevant ones."""

    def __init__(self, n_distractors: int = 0, seed: int = 0):
        self.extra = register_distractors(n_distractors, seed) if n_distractors else []
        self.n = len(DEFAULT_TOOLS) + len(self.extra)
        self.name = f"all-{self.n}"

    def tools_for(self, state, db, task) -> list[str]:
        return list(DEFAULT_TOOLS) + self.extra


class OracleTools:
    """Only the tools the reference solution uses. The ceiling arm.

    Any gap between this and `all-20` is the cost of the real registry before a
    single distractor is added -- worth knowing separately, and usually
    non-zero.
    """
    name = "oracle"

    def __init__(self, solutions: dict[str, list[tuple[str, dict]]]):
        self.solutions = solutions

    def tools_for(self, state, db, task) -> list[str]:
        needed = {n for n, _ in self.solutions.get(task.task_id, [])}
        needed |= {"get_order", "search_policy", "verify_identity"}
        return sorted(needed & set(TOOLS))


class RandomSubset:
    """N tools at random, but ALWAYS containing everything the task needs.

    The control arm. Same token cost as `all-N`, same reachability guarantee as
    `oracle`. Whichever of those two it tracks tells you which effect is doing
    the damage.
    """

    def __init__(self, n: int, solutions: dict[str, list[tuple[str, dict]]],
                 n_distractors: int = 300, seed: int = 0):
        self.n, self.solutions, self.seed = n, solutions, seed
        self.pool = list(DEFAULT_TOOLS) + register_distractors(n_distractors, seed)
        self.name = f"random-{n}"

    def tools_for(self, state, db, task) -> list[str]:
        needed = {n for n, _ in self.solutions.get(task.task_id, [])}
        needed |= {"get_order", "search_policy", "verify_identity"}
        needed &= set(TOOLS)
        rng = random.Random(f"{self.seed}:{task.task_id}")
        rest = [t for t in self.pool if t not in needed]
        rng.shuffle(rest)
        return sorted(needed | set(rest[:max(0, self.n - len(needed))]))


class ToolSearch:
    """Small core + `search_tools`. Retrieved tools become callable.

    Visibility is DERIVED from the action log by replaying past searches, so
    there is no extra state to serialise and resume works unchanged.
    """

    def __init__(self, n_distractors: int = 300, limit: int = 6, seed: int = 0):
        register_distractors(n_distractors, seed)
        self.limit = limit
        self.n = len(DEFAULT_TOOLS) + n_distractors
        self.name = f"search-{self.n}"

    def tools_for(self, state, db, task) -> list[str]:
        visible = [t for t in CORE_VISIBLE if t in TOOLS]
        for action in db.action_log:
            if action.tool != "search_tools" or not action.ok:
                continue
            q = action.args.get("query", "")
            for name, _ in _rank_tools(q, action.args.get("limit") or self.limit):
                if name not in visible:
                    visible.append(name)
        return visible


def schema_tokens(names: list[str]) -> int:
    """Rough token cost of injecting these schemas, for the cost axis.

    Approximate (4 chars/token) and labelled as such. Real numbers come from
    the backend's reported usage; this is for planning an arm, not for a table.
    """
    import json
    total = sum(len(json.dumps(TOOLS[n]["schema"])) for n in names if n in TOOLS)
    return total // 4


def build_exposure(spec: str, solutions: dict[str, list[tuple[str, dict]]],
                   seed: int = 0) -> ExposureStrategy:
    """`oracle` | `all-20` | `all-100` | `random-100` | `search-300`"""
    if spec == "oracle":
        return OracleTools(solutions)
    kind, _, num = spec.partition("-")
    n = int(num) if num else 0
    if kind == "all":
        return AllTools(max(0, n - len(DEFAULT_TOOLS)), seed)
    if kind == "random":
        return RandomSubset(n, solutions, 300, seed)
    if kind == "search":
        return ToolSearch(max(0, n - len(DEFAULT_TOOLS)), seed=seed)
    raise ValueError(f"unknown exposure spec {spec!r}")
