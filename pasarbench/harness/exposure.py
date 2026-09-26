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

    oracle       every lookup + only the write actions the     ceiling
                 task takes (removes choices, never facts)
    all-20       the real registry, no distractors              baseline
    all-N        registry + distractors, N in {50, 100, 300}    both effects
    random-N     N tools sampled at random but ALWAYS containing
                 the oracle's set; the rest mostly distractors  composition
                                                                control at the
                                                                same N
    search-N     only meta-tools visible; the agent retrieves   the proposed fix

`random-N` is the arm that does the work. It has the same tool count and
schema cost as `all-N`, and everything the oracle exposes; what it drops is
most of the unneeded REAL write tools, swapped for distractors. If `random-N`
tracks `all-N`, the damage comes with the number of tools -- count, context
length and dilution move together here and are not separable. If `random-N`
tracks `oracle`, the damage comes from the plausible wrong actions only `all-N`
still holds. Neither verdict is read until the loss itself clears a paired
test; at 32 tasks, a few episodes' difference usually does not.

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
from typing import Any, Iterable, Protocol

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
    # Rank within the arm's own tools, the same pool visibility is derived from
    # -- otherwise the agent could be shown a tool it then cannot call.
    hits = _rank_tools(query, limit, getattr(db, "search_universe", None))
    return _ok(db, "search_tools", {"query": query, "limit": limit},
               tools=[{"name": n, "description": d} for n, d in hits],
               note=None if hits else "no match; try different words")


def _rank_tools(query: str, limit: int = 6,
                universe: Iterable[str] | None = None) -> list[tuple[str, str]]:
    """Keyword ranking over names and descriptions.

    Deliberately keyword-based, not embedding-based, and that IS a finding
    waiting to happen: a Thai or Vietnamese query scores zero against English
    tool descriptions. `diagnose.py` measures exactly that, and it is one of
    the mechanisms behind the multilingual gap rather than a bug to hide.
    """
    terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
    # `universe` confines the search to one arm's tools. Without it the ranker
    # searched the GLOBAL registry, which holds every distractor any earlier arm
    # registered: after random-100 (300 distractors), "search-300" was really
    # searching 320 tools, and its difficulty depended on the order arms ran in.
    pool = set(universe) if universe is not None else None
    scored = []
    for name, spec in TOOLS.items():
        if name == "search_tools" or (pool is not None and name not in pool):
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


# Always visible in the reduced arms: how any agent learns an order's facts.
BASE_LOOKUPS = {"get_order", "search_policy", "verify_identity"}

# Arguments a reference solution hard-codes but a real agent has to LOOK UP,
# and the read-only tool that reveals each. A reference solution is an oracle:
# it passes order_item_id="OI3" because it already knows, so it never calls
# get_order_items -- and an arm built from "the tools the solution calls" then
# hides the only way a real agent can learn "OI3". That made the oracle arm
# unsolvable on every trap that needs an item: exactly 0.00 on four traps,
# below the arm it was meant to be the ceiling of. See WHAT_FAILED #18.
ARG_SOURCES = {
    "order_item_id": "get_order_items",
    "product_id": "get_order_items",
}


# Every read-only lookup in the real registry: tools that answer questions and
# change no table. (verify_identity marks the session verified, which is
# session state, not data.) tests/test_exposure.py calls each one against a
# live database and fails if any table changes.
LOOKUPS = {"get_user_profile", "get_order", "list_user_orders", "get_order_items",
           "get_shipment", "get_product", "get_payment", "search_policy",
           "get_livestream_claims", "check_return_eligibility",
           "calculate_refund_amount", "verify_identity"}


def needed_tools(solution: list[tuple[str, dict]]) -> set[str]:
    """What an agent that has to FIND OUT needs: every lookup, plus the
    ACTIONS the reference solution takes.

    An oracle may remove choices, never information. The first definition
    exposed only what the reference solution calls, which hid the lookups a
    real agent needs to learn a value (get_order_items -> the item id). The
    second added lookups for values, and still hid the ones a careful agent
    needs to RULE OUT an exception: the policy lets a livestream claim override
    the return window, so an agent denying an out-of-window return checks
    get_livestream_claims first -- the reference solution skips it because it
    already knows there is no claim. Your audit caught that one: random-100
    agents called it six times on tasks whose solution never does.

    So the rule is structural rather than a list of cases: expose every lookup,
    and restrict only the write actions. The oracle arm then isolates what it
    is for -- choosing the right ACTION with no distractors -- and cannot
    starve the agent of a fact. See WHAT_FAILED #18.
    """
    need = {n for n, _ in solution} | BASE_LOOKUPS | LOOKUPS
    for _, args in solution:
        need |= {ARG_SOURCES[k] for k in args if k in ARG_SOURCES}
    return need & set(TOOLS)


class OracleTools:
    """Only the tools the task needs. The ceiling arm.

    "Needs" means needs for an agent that has to discover the facts -- see
    `needed_tools`. If this arm ever scores below `all-20`, it is not a ceiling
    and nothing measured against it can be read; tool_scaling_report checks.
    """
    name = "oracle"

    def __init__(self, solutions: dict[str, list[tuple[str, dict]]]):
        self.solutions = solutions

    def tools_for(self, state, db, task) -> list[str]:
        return sorted(needed_tools(self.solutions.get(task.task_id, [])))


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
        # Same definition as the oracle, deliberately: the control arm's
        # reachability guarantee is only as good as this set.
        needed = needed_tools(self.solutions.get(task.task_id, []))
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
        self.universe = set(DEFAULT_TOOLS) | set(register_distractors(n_distractors, seed))
        self.limit = limit
        self.n = len(DEFAULT_TOOLS) + n_distractors
        self.name = f"search-{self.n}"

    def tools_for(self, state, db, task) -> list[str]:
        # Called at the top of every step, so this is set before the agent can
        # make its first search_tools call.
        db.search_universe = self.universe
        visible = [t for t in CORE_VISIBLE if t in TOOLS]
        for action in db.action_log:
            if action.tool != "search_tools" or not action.ok:
                continue
            q = action.args.get("query", "")
            for name, _ in _rank_tools(q, action.args.get("limit") or self.limit,
                                       universe=self.universe):
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
