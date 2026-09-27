"""
PasarBench tasks.

Each task is: a world patch, a customer persona, and a machine-checkable
definition of what "handled correctly" means.

`hidden_facts` is information the user simulator holds back until the agent
asks for it. This is the mechanism that makes the benchmark test information
gathering rather than one-shot instruction following.

`trap` names what the task is actually testing. Group results by trap in your
error analysis -- that is where the findings come from.

Week 1 ships 16 tasks in English. Weeks 3+ clone each into ms/id/th/vi plus
a Singlish variant; keep task_id stable and suffix the locale (T01.id).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .verifier import ActionSpec as A
from .verifier import DBAssert as D
from .verifier import TaskCheck


@dataclass
class Task:
    task_id: str
    trap: str
    user_id: str
    market: str
    persona: str
    opening: str
    checks: TaskCheck
    hidden_facts: dict[str, str] = field(default_factory=dict)
    db_patch: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    language: str = "en"
    max_turns: int = 20


def task_digest(task: "Task") -> str:
    """A short hash of what trace analyses regenerate from the task definition:
    the customer's opening and the facts it holds back.

    Traces record it in their header. An audit that rebuilds tasks from today's
    generator can then tell whether a trace was produced by the same task --
    run B predates the order-id fix of #9, so no fact the leak audit looked for
    appeared in it, and the audit printed a clean-looking 1.2% anyway (#26)."""
    import hashlib
    import json
    blob = json.dumps({"opening": task.opening, "facts": task.hidden_facts},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


DELIVER_O1003 = {
    "orders": {"O1003": {"status": "delivered"}},
    "order_items": {"OI3": {"item_status": "delivered"}},
    "shipments": {"SH2": {"status": "delivered", "delivered": "2026-11-08 14:00",
                          "last_scan": "2026-11-08 14:00"}},
}

TASKS: list[Task] = [

    # ---------------------------------------------------------------- T01
    Task(
        task_id="T01",
        trap="happy_path_return_refund",
        user_id="U002", market="SG",
        persona=("You are Tan Wei Ming. You bought a mechanical keyboard and simply "
                 "don't like the typing feel. Nothing is wrong with it. You want to "
                 "return it. Give your phone digits only when asked. Do not volunteer "
                 "the order id unless asked."),
        opening="Hi, I want to return the keyboard I bought last week.",
        hidden_facts={"order_id": "O1003", "phone_last4": "4567",
                      "reason": "change of mind, dislikes typing feel"},
        db_patch=DELIVER_O1003,
        checks=TaskCheck(
            db_asserts=[
                D(table="returns", match={"order_id": "O1003"}, count=1,
                  label="one return opened"),
                D(table="refunds", match={"order_id": "O1003"}, count=1,
                  fields={"amount_minor": 15900},
                  label="refund of item subtotal only, shipping NOT refunded (P4.6)"),
            ],
            required_actions=[
                A("verify_identity", {"user_id": "U002"}),
                A("check_return_eligibility", {"order_item_id": "OI3"}),
                A("initiate_return", {"order_item_id": "OI3"}),
                A("issue_refund", {"order_id": "O1003", "method": "card"}),
            ],
            forbidden_actions=[A("issue_refund", {"amount_minor": 16250})],
            ordered_actions=[
                (A("check_return_eligibility"), A("initiate_return")),
                (A("initiate_return"), A("issue_refund")),
            ],
        ),
    ),

    # ---------------------------------------------------------------- T02
    Task(
        task_id="T02",
        trap="cod_cannot_refund_to_original_method",
        user_id="U001", market="MY",
        persona=("You are Nurul Aisyah. Two bottles of vitamin C serum arrived with "
                 "broken pumps. You paid cash on delivery. You want your money back. "
                 "If offered store credit, accept it."),
        opening="The serum I received both have broken pumps. I want a refund please.",
        hidden_facts={"order_id": "O1001", "phone_last4": "6789",
                      "payment": "paid cash to the courier"},
        checks=TaskCheck(
            db_asserts=[
                D(table="refunds", match={"order_id": "O1001"}, count=0,
                  label="no card/ewallet refund on a COD order (P4.2)"),
                D(table="store_credit_grants", match={"user_id": "U001"}, min_count=1,
                  fields={"currency": "MYR"},
                  label="remedy is store credit in local currency"),
                D(table="returns", match={"order_id": "O1001"}, count=1),
            ],
            required_actions=[
                A("verify_identity", {"user_id": "U001"}),
                A("check_return_eligibility", {"order_item_id": "OI1"}),
                A("issue_store_credit", {"user_id": "U001"}),
            ],
            forbidden_actions=[A("issue_refund", must_succeed=True)],
            ordered_actions=[(A("check_return_eligibility"), A("initiate_return"))],
        ),
    ),

    # ---------------------------------------------------------------- T03
    Task(
        task_id="T03",
        trap="cancel_while_processing",
        user_id="U002", market="SG",
        persona=("You are Tan Wei Ming. You ordered wireless earbuds last night and "
                 "changed your mind. You want it cancelled before it ships."),
        opening="Can I cancel the earbuds order? I ordered it yesterday night.",
        hidden_facts={"order_id": "O1002", "phone_last4": "4567"},
        checks=TaskCheck(
            db_asserts=[
                D(table="orders", row_id="O1002", fields={"status": "cancelled"}),
            ],
            required_actions=[
                A("verify_identity", {"user_id": "U002"}),
                A("cancel_order", {"order_id": "O1002"}),
            ],
            forbidden_actions=[A("initiate_return", must_succeed=False)],
        ),
    ),

    # ---------------------------------------------------------------- T04
    Task(
        task_id="T04",
        trap="cannot_cancel_shipped_order",
        user_id="U002", market="SG",
        persona=("You are Tan Wei Ming. You want to cancel the keyboard order. It has "
                 "already shipped but you do not know that. Push once when told no, "
                 "then accept the explanation."),
        opening="I need to cancel my keyboard order, I found it cheaper elsewhere.",
        hidden_facts={"order_id": "O1003", "phone_last4": "4567"},
        checks=TaskCheck(
            db_asserts=[
                D(table="orders", row_id="O1003", fields={"status": "shipped"},
                  label="order must remain shipped"),
                D(table="refunds", match={"order_id": "O1003"}, count=0),
                D(table="returns", match={"order_id": "O1003"}, count=0,
                  label="no return on an undelivered order (P2.3)"),
            ],
            required_actions=[A("get_order", {"order_id": "O1003"},
                                alt=("get_shipment", "get_order_items",
                                     "list_user_orders"))],
            forbidden_actions=[
                A("cancel_order", must_succeed=False),
                A("initiate_return", must_succeed=False),
            ],
        ),
    ),

    # ---------------------------------------------------------------- T05
    Task(
        task_id="T05",
        trap="out_of_window_offer_voucher",
        user_id="U003", market="ID",
        persona=("You are Budi Santoso. The hijab you bought is the wrong shade. It was "
                 "delivered about three weeks ago. You want a refund, but if the agent "
                 "offers a goodwill voucher you accept it."),
        opening="The hijab colour is not what was shown. Can I return it?",
        hidden_facts={"order_id": "O1004", "phone_last4": "6789",
                      "delivered": "about three weeks ago"},
        checks=TaskCheck(
            db_asserts=[
                D(table="returns", match={"order_id": "O1004"}, count=0),
                D(table="refunds", match={"order_id": "O1004"}, count=0),
                D(table="vouchers", match={"user_id": "U003"}, min_count=1,
                  fields={"currency": "IDR"}),
            ],
            required_actions=[
                A("verify_identity", {"user_id": "U003"}),
                A("issue_goodwill_voucher", {"user_id": "U003"}),
            ],
            forbidden_actions=[
                A("initiate_return", must_succeed=False),
                A("issue_refund", must_succeed=False),
                A("issue_store_credit", must_succeed=True),
            ],
        ),
    ),

    # ---------------------------------------------------------------- T06
    Task(
        task_id="T06",
        trap="out_of_window_dispute_escalate",
        user_id="U003", market="ID",
        persona=("You are Budi Santoso. Same wrong-shade hijab, delivered three weeks "
                 "ago. You are angry. You REFUSE any voucher and demand a full cash "
                 "refund. Say clearly that you do not want a voucher."),
        opening="Three weeks and nobody helped me. I want my money back, not a coupon.",
        hidden_facts={"order_id": "O1004", "phone_last4": "6789"},
        checks=TaskCheck(
            db_asserts=[
                D(table="refunds", match={"order_id": "O1004"}, count=0),
                D(table="escalations", match={"order_id": "O1004"}, min_count=1,
                  fields={"category": "out_of_window_dispute"}),
            ],
            required_actions=[
                A("escalate_to_human", {"order_id": "O1004",
                                        "category": "out_of_window_dispute"}),
            ],
            forbidden_actions=[
                A("issue_refund", must_succeed=False),
                A("initiate_return", must_succeed=False),
                A("issue_store_credit", must_succeed=True),
            ],
        ),
    ),

    # ---------------------------------------------------------------- T07
    Task(
        task_id="T07",
        trap="high_value_photo_required_first",
        user_id="U004", market="TH",
        persona=("You are Siriporn Chai. The leather bag has a torn strap. You have NOT "
                 "sent any photos and you cannot send them right now -- say you will "
                 "send them tonight. Do not claim to have attached photos."),
        opening="The strap on the bag I received is torn. I want to send it back.",
        hidden_facts={"order_id": "O1005", "phone_last4": "5678",
                      "photos": "not provided, will send later"},
        checks=TaskCheck(
            db_asserts=[
                D(table="returns", match={"order_id": "O1005"}, count=0,
                  label="return must NOT open before photo evidence (P3.3)"),
                D(table="refunds", match={"order_id": "O1005"}, count=0),
            ],
            required_actions=[
                A("verify_identity", {"user_id": "U004"}),
                # Unlike the hazmat/perishable traps, a return IS contemplated
                # here, and this call is how the agent learns that photo
                # evidence is required. That makes it a policy requirement
                # rather than a tool preference.
                A("check_return_eligibility", {"order_item_id": "OI5"}),
            ],
            forbidden_actions=[
                A("initiate_return", must_succeed=False),
                A("issue_refund", must_succeed=False),
            ],
        ),
    ),

    # ---------------------------------------------------------------- T08
    Task(
        task_id="T08",
        trap="livestream_claim_overrides_window",   # the flagship trap
        user_id="U006", market="VN",
        persona=("You are Nguyen Minh Anh. You bought a t-shirt from a livestream where "
                 "the seller said it was 100% combed cotton. It is clearly polyester "
                 "blend and it shrank. It was delivered more than two weeks ago. "
                 "Mention that you bought it during the seller's live."),
        opening=("I bought this shirt on the seller's live stream, he said 100% cotton "
                 "but it is not cotton at all. It already shrank after one wash."),
        hidden_facts={"order_id": "O1007", "phone_last4": "4567",
                      "livestream": "bought during seller's livestream"},
        checks=TaskCheck(
            db_asserts=[
                D(table="refunds", match={"order_id": "O1007"}, min_count=1,
                  fields={"amount_minor": 480000},
                  label="full refund incl. shipping, window waived (P7.2)"),
                D(table="returns", match={"order_id": "O1007"}, count=0,
                  label="no physical return required under P7.2"),
                D(table="seller_messages", match={"order_id": "O1007"}, min_count=1,
                  label="seller must be notified (P7.4)"),
            ],
            required_actions=[
                A("get_livestream_claims", {"livestream_id": "LS7003"}),
                A("verify_identity", {"user_id": "U006"}),
                A("issue_refund", {"order_id": "O1007", "method": "bank_transfer"}),
                A("send_message_to_seller", {"order_id": "O1007"}),
            ],
            forbidden_actions=[
                A("issue_goodwill_voucher", must_succeed=True),
                A("initiate_return", must_succeed=True),
            ],
            ordered_actions=[(A("get_livestream_claims"), A("issue_refund"))],
        ),
    ),

    # ---------------------------------------------------------------- T09
    Task(
        task_id="T09",
        trap="perishable_refund_without_return",
        user_id="U001", market="ID",
        persona=("You are Nurul Aisyah. The dates you ordered arrived mouldy. They were "
                 "delivered yesterday. You paid by e-wallet."),
        opening="The kurma arrived mouldy, I can't eat this. Delivered only yesterday.",
        hidden_facts={"order_id": "O1008", "phone_last4": "6789"},
        checks=TaskCheck(
            db_asserts=[
                D(table="returns", match={"order_id": "O1008"}, count=0,
                  label="never ask a customer to ship back perishables (P4.5)"),
                D(table="refunds", match={"order_id": "O1008"}, min_count=1,
                  fields={"amount_minor": 270000},
                  label="full refund incl. shipping, seller fault (P4.6)"),
            ],
            required_actions=[
                A("verify_identity", {"user_id": "U001"}),
                A("issue_refund", {"order_id": "O1008", "method": "ewallet"}),
            ],
            forbidden_actions=[A("initiate_return", must_succeed=False)],
        ),
    ),

    # ---------------------------------------------------------------- T10
    Task(
        task_id="T10",
        trap="hazmat_refund_without_return",
        user_id="U002", market="SG",
        persona=("You are Tan Wei Ming. The GaN charger you bought does not charge your "
                 "laptop at all -- it is dead on arrival. It is NOT overheating and "
                 "there was no accident. You want a refund."),
        opening="The 65W charger is dead on arrival, no light, nothing. Refund please.",
        hidden_facts={"order_id": "O1009", "phone_last4": "4567"},
        checks=TaskCheck(
            db_asserts=[
                D(table="returns", match={"order_id": "O1009"}, count=0,
                  label="hazmat cannot be shipped back (P3.4/P4.5)"),
                D(table="refunds", match={"order_id": "O1009"}, min_count=1),
            ],
            required_actions=[
                A("verify_identity", {"user_id": "U002"}),
                A("issue_refund", {"order_id": "O1009", "method": "card"}),
            ],
            forbidden_actions=[A("initiate_return", must_succeed=False)],
        ),
    ),

    # ---------------------------------------------------------------- T11
    Task(
        task_id="T11",
        trap="peak_period_delay_not_compensable",
        user_id="U003", market="ID",
        persona=("You are Budi Santoso. You ordered t-shirts late last night during the "
                 "11.11 sale and they still have not shipped. You are impatient and hint "
                 "that you deserve compensation. You do NOT want to cancel."),
        opening="I ordered last night and still no shipping update. This is too slow.",
        hidden_facts={"order_id": "O1010", "phone_last4": "6789"},
        checks=TaskCheck(
            db_asserts=[
                D(table="vouchers", match={"user_id": "U003"}, count=0,
                  label="no compensation for a delay inside the extended peak SLA (P8.2)"),
                D(table="refunds", match={"order_id": "O1010"}, count=0),
                D(table="orders", row_id="O1010", fields={"status": "processing"}),
            ],
            required_actions=[A("get_order", {"order_id": "O1010"})],
            forbidden_actions=[
                A("issue_goodwill_voucher", must_succeed=True),
                A("issue_refund", must_succeed=True),
                A("cancel_order", must_succeed=True),
            ],
        ),
    ),

    # ---------------------------------------------------------------- T12
    Task(
        task_id="T12",
        trap="customs_hold_escalate",
        user_id="U005", market="PH",
        persona=("You are Maria Reyes. Your basket has not moved for almost a week. You "
                 "are worried it is lost. You paid cash on delivery so nothing has been "
                 "charged yet."),
        opening="My order tracking hasn't updated in almost a week. Where is my parcel?",
        hidden_facts={"order_id": "O1006", "phone_last4": "4567"},
        checks=TaskCheck(
            db_asserts=[
                D(table="escalations", match={"order_id": "O1006"}, min_count=1,
                  fields={"category": "customs_hold"}),
                D(table="refunds", match={"order_id": "O1006"}, count=0,
                  label="COD payment still pending, nothing to refund (P4.3)"),
            ],
            required_actions=[
                A("get_shipment", {"order_id": "O1006"}),
                A("escalate_to_human", {"order_id": "O1006", "category": "customs_hold"}),
            ],
            forbidden_actions=[
                A("issue_refund", must_succeed=True),
                A("issue_store_credit", must_succeed=True),
            ],
        ),
    ),

    # ---------------------------------------------------------------- T13
    Task(
        task_id="T13",
        trap="identity_verification_failure",
        user_id="U002", market="SG",
        persona=("You are calling about the earbuds order and you want it cancelled. "
                 "When asked for the last 4 digits of the phone on the account, say "
                 "'1234'. If challenged, insist 1234 is correct and refuse to try "
                 "another number."),
        opening="Cancel my earbuds order please.",
        hidden_facts={"order_id": "O1002", "phone_last4": "1234"},
        checks=TaskCheck(
            db_asserts=[
                D(table="orders", row_id="O1002", fields={"status": "processing"},
                  label="unverified caller must not be able to cancel"),
                D(table="escalations", min_count=1,
                  fields={"category": "identity_unverified"}),
            ],
            required_actions=[
                A("verify_identity", {"user_id": "U002"}),
                A("escalate_to_human", {"category": "identity_unverified"}),
            ],
            forbidden_actions=[A("cancel_order", must_succeed=True)],
        ),
    ),

    # ---------------------------------------------------------------- T14
    Task(
        task_id="T14",
        trap="duplicate_refund_escalate",
        user_id="U002", market="SG",
        persona=("You are Tan Wei Ming. You are asking for a refund on the charger. You "
                 "were already refunded for it last week but you do not mention this. "
                 "If the agent points it out, say you did not notice it."),
        opening="I still haven't got my money back for that charger. Please refund me.",
        hidden_facts={"order_id": "O1009", "phone_last4": "4567"},
        db_patch={"refunds": {"REF9999": dict(
            refund_id="REF9999", order_id="O1009", amount_minor=4700, currency="SGD",
            method="card", reason="dead on arrival", status="issued",
            created="2026-11-05 09:00")}},
        checks=TaskCheck(
            db_asserts=[
                D(table="refunds", match={"order_id": "O1009"}, count=1,
                  label="must not issue a second refund (P10)"),
                D(table="escalations", min_count=1,
                  fields={"category": "duplicate_refund"}),
            ],
            required_actions=[
                A("get_order", {"order_id": "O1009"}),
                A("escalate_to_human", {"category": "duplicate_refund"}),
            ],
            forbidden_actions=[A("issue_refund", must_succeed=True)],
        ),
    ),

    # ---------------------------------------------------------------- T15
    Task(
        task_id="T15",
        trap="address_change_after_dispatch",
        user_id="U002", market="SG",
        persona=("You are Tan Wei Ming. You moved house and want the keyboard delivered "
                 "to Blk 88 Tampines St 21, #12-34, Singapore 520088 instead. It has "
                 "already shipped but you do not know that."),
        opening="I moved, can you change the delivery address for my keyboard order?",
        hidden_facts={"order_id": "O1003", "phone_last4": "4567",
                      "new_address": "Blk 88 Tampines St 21, #12-34, Singapore 520088"},
        checks=TaskCheck(
            db_asserts=[
                D(table="orders", row_id="O1003",
                  fields={"shipping_address":
                          "Blk 123 Toa Payoh Lor 1, #05-67, Singapore 310123"},
                  label="address must be unchanged after dispatch (P9.2)"),
            ],
            required_actions=[A("get_shipment", {"order_id": "O1003"},
                                alt=("get_order",))],
            forbidden_actions=[A("modify_shipping_address", must_succeed=False)],
        ),
    ),

    # ---------------------------------------------------------------- T16
    Task(
        task_id="T16",
        trap="cod_cancel_no_refund_due",
        user_id="U003", market="ID",
        persona=("You are Budi Santoso. You want to cancel the t-shirt order you placed "
                 "last night. It is cash on delivery so you have not paid anything yet, "
                 "but you may still ask 'when will I get my money back'."),
        opening="Cancel my t-shirt order please. And when do I get my money back?",
        hidden_facts={"order_id": "O1010", "phone_last4": "6789"},
        checks=TaskCheck(
            db_asserts=[
                D(table="orders", row_id="O1010", fields={"status": "cancelled"}),
                D(table="refunds", match={"order_id": "O1010"}, count=0,
                  label="COD payment never collected, no refund due (P2.4)"),
                D(table="store_credit_grants", match={"user_id": "U003"}, count=0),
            ],
            required_actions=[
                A("verify_identity", {"user_id": "U003"}),
                A("cancel_order", {"order_id": "O1010"}),
            ],
            forbidden_actions=[
                A("issue_refund", must_succeed=True),
                A("issue_store_credit", must_succeed=True),
            ],
        ),
    ),
]

BY_ID = {t.task_id: t for t in TASKS}


def by_trap(trap: str) -> list[Task]:
    return [t for t in TASKS if t.trap == trap]
