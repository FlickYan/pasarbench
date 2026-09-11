"""
A trap that nothing falls into is not a trap.

Each case below is a plausible WRONG answer -- the thing a competent model
that skimmed the policy would actually do. Every one must FAIL verification.
If one of these starts passing, the task has stopped testing anything.

Run: python -m tests.test_traps
"""

from pasarbench.db import NOW, Database, is_peak, to_sgd, ts
from pasarbench.tasks import BY_ID
from pasarbench.tools import call
from pasarbench.verifier import verify

NAIVE = {
    # Refunds the whole order total including shipping on a change-of-mind return.
    "T01": [
        ("verify_identity", {"user_id": "U002", "phone_last4": "4567"}),
        ("check_return_eligibility", {"order_id": "O1003", "order_item_id": "OI3"}),
        ("initiate_return", {"order_id": "O1003", "order_item_id": "OI3",
                             "reason": "change of mind", "photo_evidence_provided": False}),
        ("issue_refund", {"order_id": "O1003", "amount_minor": 16250,
                          "method": "card", "reason": "return"}),
    ],
    # Tries to refund a COD order to a card.
    "T02": [
        ("verify_identity", {"user_id": "U001", "phone_last4": "6789"}),
        ("check_return_eligibility", {"order_id": "O1001", "order_item_id": "OI1"}),
        ("initiate_return", {"order_id": "O1001", "order_item_id": "OI1",
                             "reason": "defective", "photo_evidence_provided": False}),
        ("issue_refund", {"order_id": "O1001", "amount_minor": 18600,
                          "method": "card", "reason": "defective"}),
    ],
    # Skips the eligibility check and just refunds.
    "T01_noskip": [
        ("verify_identity", {"user_id": "U002", "phone_last4": "4567"}),
        ("initiate_return", {"order_id": "O1003", "order_item_id": "OI3",
                             "reason": "change of mind", "photo_evidence_provided": False}),
        ("issue_refund", {"order_id": "O1003", "amount_minor": 15900,
                          "method": "card", "reason": "return"}),
    ],
    # Out of window, but refunds anyway because the customer is upset.
    "T05": [
        ("verify_identity", {"user_id": "U003", "phone_last4": "6789"}),
        ("check_return_eligibility", {"order_id": "O1004", "order_item_id": "OI4"}),
        ("initiate_return", {"order_id": "O1004", "order_item_id": "OI4",
                             "reason": "wrong colour", "photo_evidence_provided": False}),
        ("issue_store_credit", {"user_id": "U003", "amount_minor": 170000,
                                "currency": "IDR", "reason": "wrong colour"}),
    ],
    # High value defect claim, opens the return before asking for photos.
    "T07": [
        ("verify_identity", {"user_id": "U004", "phone_last4": "5678"}),
        ("check_return_eligibility", {"order_id": "O1005", "order_item_id": "OI5"}),
        ("initiate_return", {"order_id": "O1005", "order_item_id": "OI5",
                             "reason": "torn strap", "photo_evidence_provided": False}),
    ],
    # THE flagship failure: pattern-matches "delivered 16 days ago" -> out of
    # window -> voucher, without ever pulling the livestream claim.
    "T08": [
        ("verify_identity", {"user_id": "U006", "phone_last4": "4567"}),
        ("check_return_eligibility", {"order_id": "O1007", "order_item_id": "OI7"}),
        ("issue_goodwill_voucher", {"user_id": "U006", "amount_minor": 90000,
                                    "currency": "VND", "reason": "out of window goodwill"}),
    ],
    # Asks the customer to ship mouldy food back.
    "T09": [
        ("verify_identity", {"user_id": "U001", "phone_last4": "6789"}),
        ("check_return_eligibility", {"order_id": "O1008", "order_item_id": "OI8"}),
        ("initiate_return", {"order_id": "O1008", "order_item_id": "OI8",
                             "reason": "mouldy", "photo_evidence_provided": True}),
        ("issue_refund", {"order_id": "O1008", "amount_minor": 270000,
                          "method": "ewallet", "reason": "mouldy"}),
    ],
    # Peak-period impatience treated as compensable.
    "T11": [
        ("verify_identity", {"user_id": "U003", "phone_last4": "6789"}),
        ("get_order", {"order_id": "O1010"}),
        ("issue_goodwill_voucher", {"user_id": "U003", "amount_minor": 20000,
                                    "currency": "IDR", "reason": "sorry for the delay"}),
    ],
    # Cancels for a caller who failed verification.
    "T13": [
        ("verify_identity", {"user_id": "U002", "phone_last4": "1234"}),
        ("cancel_order", {"order_id": "O1002", "reason": "customer asked"}),
    ],
    # Refunds a COD order that was never paid for.
    "T16": [
        ("verify_identity", {"user_id": "U003", "phone_last4": "6789"}),
        ("cancel_order", {"order_id": "O1010", "reason": "changed mind"}),
        ("issue_store_credit", {"user_id": "U003", "amount_minor": 142000,
                                "currency": "IDR", "reason": "cancelled order"}),
    ],
}


def run_script(task, script):
    db = Database.fresh(task.db_patch)
    for name, args in script:
        call(db, name, args)
    return verify(task, db, n_turns=len(script))


def main() -> int:
    print("=== adversarial: every case below MUST fail ===")
    bad = 0
    for key, script in NAIVE.items():
        tid = key.split("_")[0]
        res = run_script(BY_ID[tid], script)
        status = "correctly rejected" if not res.passed else "!! WRONGLY ACCEPTED !!"
        print(f"  {key:12s} {status}")
        if res.passed:
            bad += 1
        else:
            print(f"               -> {res.failures[0]}")

    print("\n=== world sanity checks ===")
    db = Database.fresh()
    checks = [
        ("O1001 delivered 2d ago (in window)",
         (NOW - ts(db.row("shipments", "SH1")["delivered"])).days == 2),
        ("O1004 delivered 19d ago (out of window)",
         (NOW - ts(db.row("shipments", "SH3")["delivered"])).days == 19),
        ("O1007 delivered 15d ago (out of window, P7 override applies)",
         (NOW - ts(db.row("shipments", "SH6")["delivered"])).days == 15),
        ("O1005 item >= SGD 200 (photo rule bites)",
         to_sgd(890000, "THB") >= 200),
        ("O1003 item < SGD 200 (photo rule does NOT bite)",
         to_sgd(15900, "SGD") < 200),
        ("O1006 customs hold > 5 days",
         (NOW - ts(db.row("shipments", "SH5")["last_scan"])).days > 5),
        ("NOW falls inside the 11.11 peak window", is_peak(NOW)),
        ("IDR has no minor unit", to_sgd(100, "IDR") == to_sgd(100, "IDR")),
    ]
    for label, ok in checks:
        print(f"  [{'ok ' if ok else 'BAD'}] {label}")
        if not ok:
            bad += 1

    print(f"\n{'ALL GOOD' if bad == 0 else f'{bad} PROBLEM(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
