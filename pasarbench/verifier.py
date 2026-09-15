"""
PasarBench verifier.

Scoring is on final database state plus the action log. Never on text.

A task passes only if ALL of these hold:
  - every db_assert is satisfied
  - every required_action appears in the log with matching args (subset match)
  - no forbidden_action appears
  - every ordered pair (A before B) holds in the log

Report pass^1 (any single run) and pass^k (all k runs of the same task pass).
pass^k is the number that matters: a CS org cares about reliability, not
average success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .db import Database


@dataclass
class DBAssert:
    """One predicate over final DB state."""
    table: str
    match: dict[str, Any] = field(default_factory=dict)
    count: int | None = None          # exact number of matching rows
    min_count: int | None = None
    row_id: str | None = None         # check a specific row's fields
    fields: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    def check(self, db: Database) -> tuple[bool, str]:
        if self.row_id is not None:
            r = db.row(self.table, self.row_id)
            if r is None:
                return False, f"{self.table}[{self.row_id}] missing"
            for k, v in self.fields.items():
                if r.get(k) != v:
                    return False, f"{self.table}[{self.row_id}].{k}={r.get(k)!r}, expected {v!r}"
            return True, ""
        rows = [r for r in db.t(self.table).values()
                if all(r.get(k) == v for k, v in self.match.items())]
        if self.count is not None and len(rows) != self.count:
            return False, (f"{self.table} matching {self.match}: "
                           f"found {len(rows)}, expected {self.count}")
        if self.min_count is not None and len(rows) < self.min_count:
            return False, (f"{self.table} matching {self.match}: "
                           f"found {len(rows)}, expected >= {self.min_count}")
        if self.fields and rows:
            for k, v in self.fields.items():
                if not any(r.get(k) == v for r in rows):
                    return False, f"no row in {self.table} has {k}={v!r}"
        return True, ""


@dataclass
class ActionSpec:
    """Match an action in the log. args is a SUBSET match.

    `alt` lists other tools that establish the same fact. Requiring one
    SPECIFIC read tool when several would do tests the model's tool
    preference, not its policy compliance -- an agent that calls
    get_shipment instead of get_order has not broken any rule.
    """
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    must_succeed: bool = True
    alt: tuple[str, ...] = ()

    def matches(self, action) -> bool:
        is_alt = action.tool != self.tool and action.tool in self.alt
        if action.tool != self.tool and not is_alt:
            return False
        if self.must_succeed and not action.ok:
            return False
        # Alternatives match on NAME ONLY. They were listed precisely because
        # they establish the same fact by a different route, so they do not
        # carry the primary tool's arguments: list_user_orders takes user_id,
        # not order_id. Holding them to the primary's args made the alternative
        # unreachable and reintroduced exactly the tool-preference test the
        # `alt` mechanism exists to remove.
        if is_alt:
            return True
        return all(action.args.get(k) == v for k, v in self.args.items())


@dataclass
class TaskCheck:
    db_asserts: list[DBAssert] = field(default_factory=list)
    required_actions: list[ActionSpec] = field(default_factory=list)
    forbidden_actions: list[ActionSpec] = field(default_factory=list)
    ordered_actions: list[tuple[ActionSpec, ActionSpec]] = field(default_factory=list)


@dataclass
class Result:
    task_id: str
    passed: bool
    failures: list[str]
    n_actions: int
    n_turns: int = 0
    tokens: int = 0

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        head = f"[{mark}] {self.task_id}  ({self.n_actions} actions, {self.n_turns} turns)"
        if self.passed:
            return head
        return head + "\n      " + "\n      ".join(self.failures)


def verify(task, db: Database, n_turns: int = 0, tokens: int = 0) -> Result:
    failures: list[str] = []
    log = db.action_log

    for a in task.checks.db_asserts:
        ok, msg = a.check(db)
        if not ok:
            failures.append(f"db: {a.label or msg} -- {msg}")

    for spec in task.checks.required_actions:
        if not any(spec.matches(x) for x in log):
            failures.append(f"missing required action: {spec.tool} {spec.args or ''}".strip())

    for spec in task.checks.forbidden_actions:
        hits = [x for x in log if spec.matches(x)]
        if hits:
            failures.append(f"forbidden action taken: {spec.tool} {hits[0].args}")

    for first, second in task.checks.ordered_actions:
        i = next((n for n, x in enumerate(log) if first.matches(x)), None)
        j = next((n for n, x in enumerate(log) if second.matches(x)), None)
        if j is not None and (i is None or i > j):
            failures.append(f"ordering violated: {first.tool} must precede {second.tool}")

    return Result(task_id=task.task_id, passed=not failures, failures=failures,
                  n_actions=len(log), n_turns=n_turns, tokens=tokens)


def pass_hat_k(results_by_task: dict[str, list[Result]]) -> dict[str, float]:
    """pass^1 = mean over all runs. pass^k = fraction of tasks where ALL runs passed."""
    all_runs = [r for rs in results_by_task.values() for r in rs]
    if not all_runs:
        return {"pass^1": 0.0, "pass^k": 0.0, "n_tasks": 0, "k": 0}
    k = max(len(rs) for rs in results_by_task.values())
    p1 = sum(r.passed for r in all_runs) / len(all_runs)
    pk = sum(all(r.passed for r in rs) for rs in results_by_task.values()) / len(results_by_task)
    return {"pass^1": round(p1, 4), "pass^k": round(pk, 4),
            "n_tasks": len(results_by_task), "k": k}
