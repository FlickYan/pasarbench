"""
PasarBench runner.

Week 1 ships two agents:

  NullAgent      -- does nothing. Should score 0. Proves the verifier is not
                    passing tasks by accident.
  ReferenceAgent -- replays a hand-written correct action sequence. Should
                    score 1.0. Proves every task is actually solvable and the
                    verifier accepts the correct behaviour.

A benchmark where nobody has demonstrated both ends is not a benchmark.
Run `python -m pasarbench.run` to check both before you trust any number.

Week 2 adds LLMAgent, which replaces ReferenceAgent's script with a real
agent loop. Everything else here stays the same.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from .db import Database
from .tasks import TASKS, Task
from .tools import call
from .verifier import Result, pass_hat_k, verify


class NullAgent:
    name = "null"

    def run(self, task: Task, db: Database) -> int:
        return 0


class ReferenceAgent:
    """Replays SOLUTIONS[task_id]. No model in the loop."""
    name = "reference"

    def run(self, task: Task, db: Database) -> int:
        script = SOLUTIONS.get(task.task_id, [])
        for tool_name, args in script:
            call(db, tool_name, args)
        return len(script)


# --------------------------------------------------------------------------
# Reference solutions -- these ARE the answer key. Keep them in sync with
# policy.md; if a policy rule changes, a solution here should change too.
# --------------------------------------------------------------------------

SOLUTIONS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "T01": [
        ("verify_identity", {"user_id": "U002", "phone_last4": "4567"}),
        ("get_order", {"order_id": "O1003"}),
        ("check_return_eligibility", {"order_id": "O1003", "order_item_id": "OI3"}),
        ("initiate_return", {"order_id": "O1003", "order_item_id": "OI3",
                             "reason": "change of mind", "photo_evidence_provided": False}),
        ("calculate_refund_amount", {"order_id": "O1003", "order_item_id": "OI3",
                                     "include_shipping": False}),
        ("issue_refund", {"order_id": "O1003", "amount_minor": 15900,
                          "method": "card", "reason": "change of mind return"}),
    ],
    "T02": [
        ("verify_identity", {"user_id": "U001", "phone_last4": "6789"}),
        ("get_payment", {"order_id": "O1001"}),
        ("check_return_eligibility", {"order_id": "O1001", "order_item_id": "OI1"}),
        ("initiate_return", {"order_id": "O1001", "order_item_id": "OI1",
                             "reason": "defective pump", "photo_evidence_provided": False}),
        ("issue_store_credit", {"user_id": "U001", "amount_minor": 18600,
                                "currency": "MYR", "reason": "defective item, COD order",
                                "order_id": "O1001"}),
    ],
    "T03": [
        ("verify_identity", {"user_id": "U002", "phone_last4": "4567"}),
        ("get_order", {"order_id": "O1002"}),
        ("cancel_order", {"order_id": "O1002", "reason": "customer changed mind"}),
    ],
    "T04": [
        ("get_order", {"order_id": "O1003"}),
        ("get_shipment", {"order_id": "O1003"}),
    ],
    "T05": [
        ("verify_identity", {"user_id": "U003", "phone_last4": "6789"}),
        ("check_return_eligibility", {"order_id": "O1004", "order_item_id": "OI4"}),
        ("issue_goodwill_voucher", {"user_id": "U003", "amount_minor": 34000,
                                    "currency": "IDR",
                                    "reason": "out of window goodwill"}),
    ],
    "T06": [
        ("verify_identity", {"user_id": "U003", "phone_last4": "6789"}),
        ("check_return_eligibility", {"order_id": "O1004", "order_item_id": "OI4"}),
        ("escalate_to_human", {"order_id": "O1004", "category": "out_of_window_dispute",
                               "reason": "customer rejected voucher, demands refund"}),
    ],
    "T07": [
        ("verify_identity", {"user_id": "U004", "phone_last4": "5678"}),
        ("check_return_eligibility", {"order_id": "O1005", "order_item_id": "OI5"}),
        # stops here: photos required before initiating (P3.3)
    ],
    "T08": [
        ("get_order", {"order_id": "O1007"}),
        ("get_livestream_claims", {"livestream_id": "LS7003", "product_id": "P012"}),
        ("verify_identity", {"user_id": "U006", "phone_last4": "4567"}),
        ("issue_refund", {"order_id": "O1007", "amount_minor": 480000,
                          "method": "bank_transfer",
                          "reason": "seller misrepresentation on livestream (P7.2)"}),
        ("send_message_to_seller", {"order_id": "O1007",
                                    "message": "Refund issued for livestream claim mismatch."}),
    ],
    "T09": [
        ("verify_identity", {"user_id": "U001", "phone_last4": "6789"}),
        ("check_return_eligibility", {"order_id": "O1008", "order_item_id": "OI8"}),
        ("issue_refund", {"order_id": "O1008", "amount_minor": 270000,
                          "method": "ewallet", "reason": "perishable arrived spoiled"}),
    ],
    "T10": [
        ("verify_identity", {"user_id": "U002", "phone_last4": "4567"}),
        ("check_return_eligibility", {"order_id": "O1009", "order_item_id": "OI9"}),
        ("issue_refund", {"order_id": "O1009", "amount_minor": 4700,
                          "method": "card", "reason": "hazmat item dead on arrival"}),
    ],
    "T11": [
        ("get_order", {"order_id": "O1010"}),
        ("search_policy", {"query": "peak sale period delivery SLA"}),
    ],
    "T12": [
        ("verify_identity", {"user_id": "U005", "phone_last4": "4567"}),
        ("get_shipment", {"order_id": "O1006"}),
        ("escalate_to_human", {"order_id": "O1006", "category": "customs_hold",
                               "reason": "customs hold exceeding 5 days"}),
    ],
    "T13": [
        ("verify_identity", {"user_id": "U002", "phone_last4": "1234"}),
        ("escalate_to_human", {"order_id": "O1002", "category": "identity_unverified",
                               "reason": "caller could not verify phone digits"}),
    ],
    "T14": [
        ("get_order", {"order_id": "O1009"}),
        ("escalate_to_human", {"order_id": "O1009", "category": "duplicate_refund",
                               "reason": "refund already issued for this order"}),
    ],
    "T15": [
        ("get_order", {"order_id": "O1003"}),
        ("get_shipment", {"order_id": "O1003"}),
    ],
    "T16": [
        ("verify_identity", {"user_id": "U003", "phone_last4": "6789"}),
        ("get_payment", {"order_id": "O1010"}),
        ("cancel_order", {"order_id": "O1010", "reason": "customer requested cancellation"}),
    ],
}


# --------------------------------------------------------------------------

def run_task(agent, task: Task) -> Result:
    db = Database.fresh(task.db_patch)
    n = agent.run(task, db)
    return verify(task, db, n_turns=n)


def run_all(agent, k: int = 1, tasks: list[Task] | None = None) -> dict:
    tasks = tasks or TASKS
    by_task: dict[str, list[Result]] = defaultdict(list)
    for t in tasks:
        for _ in range(k):
            by_task[t.task_id].append(run_task(agent, t))
    return {"agent": agent.name, "results": dict(by_task),
            "summary": pass_hat_k(dict(by_task))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="both", choices=["null", "reference", "both"])
    ap.add_argument("-k", type=int, default=1, help="runs per task, for pass^k")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    agents = {"null": NullAgent(), "reference": ReferenceAgent()}
    chosen = list(agents.values()) if args.agent == "both" else [agents[args.agent]]

    payload = {}
    for ag in chosen:
        out = run_all(ag, k=args.k)
        payload[ag.name] = out["summary"]
        if not args.json:
            print(f"\n=== {ag.name} ===")
            for rs in out["results"].values():
                print(rs[0])
            s = out["summary"]
            print(f"--- pass^1={s['pass^1']}  pass^{s['k']}={s['pass^k']}  "
                  f"({s['n_tasks']} tasks)")

    if args.json:
        print(json.dumps(payload, indent=2))

    ref = payload.get("reference")
    if ref and ref["pass^1"] < 1.0:
        raise SystemExit("reference agent did not pass every task -- fix before shipping")


if __name__ == "__main__":
    main()
