"""
Reward.

PasarBench's verifier is a programmatic pass/fail with no human and no learned
reward model in the path. That makes it a *verifiable* reward -- the R in RLVR
-- and it is the reason this environment can be trained against at all.

THE HARD PART IS NOT COMPUTING A REWARD. IT IS NOT GETTING HACKED.
------------------------------------------------------------------
The naive shaped reward gives partial credit per satisfied `required_action`.
A policy discovers within a few hundred steps that calling every tool in the
registry harvests most of that credit without solving anything. Three defences,
all enforced here and all tested:

  1. Partial credit is CLIPPED below the pass floor (`clip_partial` <
     `w_pass`). A passing episode always beats a non-passing one, whatever
     partial credit the latter accumulated. This is asserted as a property
     over the whole task set in tests/test_reward.py.

  2. Any forbidden action ZEROES partial credit. Not a subtraction -- a gate.
     Otherwise a policy learns to trade one policy violation against several
     satisfied requirements, which is precisely the behaviour a CS org cannot
     ship.

  3. Tool calls beyond the reference count are penalised. Without this the
     cheapest way to satisfy `required_actions` is to enumerate the tool space.

Group-relative methods (GRPO) get their baseline for free here: roll out k
times on the same task, and the verifier separates them with no critic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..db import Database
from ..tasks import Task
from ..verifier import Result, verify


@dataclass
class RewardConfig:
    w_pass: float = 1.0          # terminal bonus for a fully correct episode
    w_db: float = 0.30           # partial: fraction of db asserts satisfied
    w_required: float = 0.20     # partial: fraction of required actions present
    p_forbidden: float = 1.00    # per violation, subtractive AND gating
    p_order: float = 0.30        # per ordering violation
    p_extra_step: float = 0.005  # per step beyond the reference path
    clip_partial: float = 0.55   # ceiling on partial credit; MUST be < w_pass
    floor: float = -1.0          # reward is clamped to [floor, w_pass]

    def __post_init__(self) -> None:
        if self.clip_partial >= self.w_pass:
            raise ValueError(
                "clip_partial must be strictly below w_pass, or a policy can "
                "score higher by failing usefully than by passing"
            )


@dataclass
class RewardBreakdown:
    task_id: str
    reward: float
    passed: bool
    db_frac: float
    required_frac: float
    n_forbidden: int
    n_order_violations: int
    extra_steps: int
    gated: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _components(task: Task, db: Database) -> tuple[float, float, int, int]:
    checks = task.checks
    log = db.action_log

    db_hits = sum(1 for a in checks.db_asserts if a.check(db)[0])
    db_frac = db_hits / len(checks.db_asserts) if checks.db_asserts else 1.0

    req_hits = sum(1 for spec in checks.required_actions
                   if any(spec.matches(x) for x in log))
    req_frac = req_hits / len(checks.required_actions) if checks.required_actions else 1.0

    n_forbidden = sum(1 for spec in checks.forbidden_actions
                      for x in log if spec.matches(x))

    n_order = 0
    for first, second in checks.ordered_actions:
        i = next((n for n, x in enumerate(log) if first.matches(x)), None)
        j = next((n for n, x in enumerate(log) if second.matches(x)), None)
        if j is not None and (i is None or i > j):
            n_order += 1

    return db_frac, req_frac, n_forbidden, n_order


def compute_reward(task: Task, db: Database, steps: int,
                   reference_steps: int | None = None,
                   cfg: RewardConfig | None = None,
                   result: Result | None = None) -> RewardBreakdown:
    cfg = cfg or RewardConfig()
    result = result or verify(task, db, n_turns=steps)
    db_frac, req_frac, n_forbidden, n_order = _components(task, db)

    ref = reference_steps if reference_steps is not None else 6
    extra = max(0, steps - ref)
    notes: list[str] = []

    if result.passed:
        r = cfg.w_pass - cfg.p_extra_step * extra
        if extra:
            notes.append(f"passed but took {extra} steps over reference")
        return RewardBreakdown(task.task_id, round(max(r, cfg.clip_partial), 4), True,
                               db_frac, req_frac, n_forbidden, n_order, extra,
                               notes=notes)

    gated = n_forbidden > 0
    if gated:
        # Defence 2: a policy violation forfeits partial credit entirely.
        partial = 0.0
        notes.append(f"partial credit gated by {n_forbidden} forbidden action(s)")
    else:
        partial = cfg.w_db * db_frac + cfg.w_required * req_frac
        partial = min(partial, cfg.clip_partial)   # Defence 1

    r = (partial
         - cfg.p_forbidden * n_forbidden
         - cfg.p_order * n_order
         - cfg.p_extra_step * extra)                # Defence 3

    return RewardBreakdown(task.task_id, round(max(r, cfg.floor), 4), False,
                           db_frac, req_frac, n_forbidden, n_order, extra,
                           gated=gated, notes=notes)


def group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """GRPO-style group-relative advantage: standardise within the group of k
    rollouts on the same task. No critic, no value head.

    If every rollout in a group scores identically the group carries no
    learning signal at all -- return zeros rather than dividing by ~0 and
    amplifying noise into a gradient. Track how often this happens: an
    all-pass or all-fail rate above ~70% means your task mix is badly
    calibrated for the current policy and you are burning rollout compute.
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    if std < eps:
        return [0.0] * n
    return [round((r - mean) / std, 6) for r in rewards]


def degenerate_group_rate(groups: list[list[float]]) -> float:
    """Fraction of rollout groups with zero signal. Report this every run."""
    if not groups:
        return 0.0
    dead = sum(1 for g in groups if len(set(g)) <= 1)
    return round(dead / len(groups), 4)
