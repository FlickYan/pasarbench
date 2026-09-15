"""
Task generation.

THREE ORTHOGONAL AXES
---------------------
    trap      decides the CHECKS        (what correct behaviour is)
    market    decides the WORLD         (currency, payment rails, order data)
    language  decides the SURFACE FORM  (what the customer types)

Keeping these independent is the whole reason the multilingual result is
interpretable. A Thai variant and its English twin share an identical world
instance and byte-identical checks -- only the words differ. So a score drop
between them is attributable to language and nothing else. If the tasks also
differed in difficulty, the comparison would mean nothing, and that is the
mistake most multilingual agent evals make.

EVERY GENERATED TASK SHIPS WITH A GENERATED REFERENCE SOLUTION.
This is non-negotiable at 150 tasks. Hand-written tasks can be sanity-checked
by reading them; generated ones cannot. The guarantee that keeps the suite
trustworthy is mechanical: `python -m tests.test_generated` asserts every
generated task is solved by its own generated solution, and that the naive
variant still fails. A task nobody has demonstrated is solvable is not a task,
it is a bug waiting to be blamed on the model.

Each task is self-contained: its `db_patch` carries its own user, seller,
product, order, item, shipment, payment and any livestream claim. Generated
rows use a `G` prefix so the hand-written 16 keep working untouched, which
means the week-1 regression tests stay meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .locales import LANGUAGES_FOR_MARKET, opening_for, persona_for
from .tasks import Task
from .verifier import ActionSpec as A
from .verifier import DBAssert as D
from .verifier import TaskCheck

# --------------------------------------------------------------------------
# Market economics. Amounts chosen so policy thresholds land unambiguously in
# every currency: "normal" is ~SGD 57, "high" is ~SGD 390 (well over the SGD
# 200 photo rule), shipping ~SGD 3, voucher 20% of normal (~SGD 11.5, under
# the SGD 15 cap). Verified in tests/test_generated.py rather than trusted.
# --------------------------------------------------------------------------

MARKETS: dict[str, dict[str, Any]] = {
    "SG": dict(currency="SGD", normal=5_900, high=390_00, ship=300, voucher=1_180,
               cod=False, name="Tan Wei Ming", phone="+6591234000",
               addr="Blk 51 Bedok Nth St 3, #08-12, Singapore 460051",
               courier="Ninja Van"),
    "MY": dict(currency="MYR", normal=19_000, high=1_300_00, ship=1_000, voucher=3_800,
               cod=True, name="Nurul Aisyah", phone="+60123456000",
               addr="8 Jalan Bukit Bintang, Kuala Lumpur 55100, MY",
               courier="J&T Express"),
    "ID": dict(currency="IDR", normal=700_000, high=4_800_000, ship=37_000,
               voucher=140_000, cod=True, name="Budi Santoso", phone="+628123456000",
               addr="Jl. Gatot Subroto No. 12, Jakarta 12930, ID",
               courier="SiCepat"),
    "TH": dict(currency="THB", normal=150_000, high=1_000_000, ship=8_000,
               voucher=30_000, cod=True, name="Siriporn Chai", phone="+66812345000",
               addr="45/7 Ratchadaphisek Rd, Bangkok 10400, TH",
               courier="Kerry Express"),
    "PH": dict(currency="PHP", normal=250_000, high=1_700_000, ship=13_000,
               voucher=50_000, cod=True, name="Maria Reyes", phone="+639171234000",
               addr="27 Katipunan Ave, Quezon City 1108, PH",
               courier="LBC Express"),
    "VN": dict(currency="VND", normal=1_100_000, high=7_400_000, ship=57_000,
               voucher=220_000, cod=True, name="Nguyen Minh Anh", phone="+84901234000",
               addr="18 Le Loi, District 1, Ho Chi Minh City, VN",
               courier="Giao Hang Nhanh"),
}

PHONE_LAST4 = "0000"          # every generated user; the real digits live in the
                              # world, the simulator only reveals them when asked

DELIVERED_IN_WINDOW = "2026-11-08 12:00"      # 2 days before NOW
DELIVERED_OUT_WINDOW = "2026-10-20 12:00"     # 21 days before NOW
CUSTOMS_LAST_SCAN = "2026-11-04 12:00"        # 6 days: clears the >5 day rule
PEAK_ORDER_CREATED = "2026-11-10 23:30"       # inside the 11.11 window


@dataclass
class Slot:
    """One generated world instance plus the ids needed to talk about it."""
    trap: str
    market: str
    currency: str
    user_id: str
    seller_id: str
    product_id: str
    order_id: str
    item_id: str
    livestream_id: str | None
    item_value: int
    shipping: int
    total: int
    voucher: int
    payment_method: str
    rows: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Trap specs: what world each trap needs
# --------------------------------------------------------------------------

# status, payment, delivery age, value tier, product flag, order source
TRAP_SPECS: dict[str, dict[str, Any]] = {
    "happy_path_return_refund":          dict(status="delivered", pay="card", age="in",   value="normal", flag=None,        source="shop_page"),
    "cod_cannot_refund_to_original_method": dict(status="delivered", pay="cod", age="in", value="normal", flag=None,        source="shop_page", cod_only=True),
    "cancel_while_processing":           dict(status="processing", pay="card", age=None,  value="normal", flag=None,        source="short_video"),
    "cannot_cancel_shipped_order":       dict(status="shipped",   pay="card", age=None,   value="normal", flag=None,        source="shop_page"),
    "out_of_window_offer_voucher":       dict(status="delivered", pay="card", age="out",  value="normal", flag=None,        source="shop_page"),
    "out_of_window_dispute_escalate":    dict(status="delivered", pay="card", age="out",  value="normal", flag=None,        source="shop_page"),
    "high_value_photo_required_first":   dict(status="delivered", pay="ewallet", age="in", value="high",  flag=None,        source="livestream"),
    "livestream_claim_overrides_window": dict(status="delivered", pay="bank_transfer", age="out", value="normal", flag=None, source="livestream", claim=False),
    "perishable_refund_without_return":  dict(status="delivered", pay="ewallet", age="in", value="normal", flag="perishable", source="shop_page"),
    "hazmat_refund_without_return":      dict(status="delivered", pay="card", age="in",   value="normal", flag="hazmat",    source="shop_page"),
    "peak_period_delay_not_compensable": dict(status="processing", pay="card", age=None,  value="normal", flag=None,        source="livestream", peak=True),
    "customs_hold_escalate":             dict(status="shipped",   pay="cod",  age=None,   value="normal", flag=None,        source="short_video", customs=True, cod_only=True),
    "identity_verification_failure":     dict(status="processing", pay="card", age=None,  value="normal", flag=None,        source="shop_page"),
    "duplicate_refund_escalate":         dict(status="delivered", pay="card", age="in",   value="normal", flag=None,        source="shop_page", prior_refund=True),
    "address_change_after_dispatch":     dict(status="shipped",   pay="card", age=None,   value="normal", flag=None,        source="shop_page"),
    "cod_cancel_no_refund_due":          dict(status="processing", pay="cod", age=None,   value="normal", flag=None,        source="livestream", cod_only=True),
}


def build_slot(trap: str, market: str) -> Slot | None:
    spec = TRAP_SPECS[trap]
    m = MARKETS[market]
    if spec.get("cod_only") and not m["cod"]:
        return None                      # SG has no COD rail in this world

    tag = f"{trap[:6]}{market}".replace("_", "")
    uid, sid = f"GU-{market}", f"GS-{market}"
    pid = f"GP-{tag}"
    oid, iid = f"GO-{tag}", f"GI-{tag}"
    shid, payid = f"GH-{tag}", f"GY-{tag}"
    lsid = f"GL-{tag}" if spec["source"] == "livestream" else None

    value = m["high"] if spec["value"] == "high" else m["normal"]
    ship = m["ship"]
    total = value + ship
    pay = "cod" if spec["pay"] == "cod" else spec["pay"]

    flag = spec.get("flag")
    category = ("grocery" if flag == "perishable"
                else "electronics" if flag == "hazmat" else "fashion")

    rows: dict[str, dict[str, dict[str, Any]]] = {
        "users": {uid: dict(user_id=uid, name=m["name"], email=f"{uid.lower()}@example.com",
                            phone=m["phone"], country=market,
                            language=LANGUAGES_FOR_MARKET[market][0], tier="silver",
                            joined="2024-05-01", store_credit_minor=0,
                            store_credit_currency=m["currency"],
                            verified_phone_last4=PHONE_LAST4)},
        "sellers": {sid: dict(seller_id=sid, name=f"{market} Marketplace Seller",
                              country=market, rating=4.3, cod_enabled=m["cod"],
                              official_store=False)},
        "products": {pid: dict(product_id=pid, seller_id=sid,
                               title=_product_title(flag), category=category,
                               price_minor=value, currency=m["currency"], stock=50,
                               hazmat=(flag == "hazmat"),
                               perishable=(flag == "perishable"))},
        "orders": {oid: dict(order_id=oid, user_id=uid, seller_id=sid,
                             status=spec["status"], payment_method=pay,
                             currency=m["currency"], subtotal_minor=value,
                             shipping_minor=ship, discount_minor=0,
                             total_minor=total,
                             created=(PEAK_ORDER_CREATED if spec.get("peak")
                                      else "2026-11-01 09:00"),
                             source=spec["source"], livestream_id=lsid,
                             shipping_address=m["addr"])},
        "order_items": {iid: dict(order_item_id=iid, order_id=oid, product_id=pid,
                                  qty=1, unit_price_minor=value,
                                  item_status=spec["status"])},
        "payments": {payid: dict(payment_id=payid, order_id=oid, method=pay,
                                 status=("pending" if pay == "cod" and spec["status"] != "delivered"
                                         else "collected" if pay == "cod" else "captured"),
                                 amount_minor=total, currency=m["currency"],
                                 paid=None if (pay == "cod" and spec["status"] != "delivered")
                                 else "2026-11-01 09:00",
                                 instrument_last4=None if pay == "cod" else "4242")},
    }

    if spec["status"] in ("delivered", "shipped"):
        delivered = (DELIVERED_IN_WINDOW if spec["age"] == "in"
                     else DELIVERED_OUT_WINDOW if spec["age"] == "out" else None)
        rows["shipments"] = {shid: dict(
            shipment_id=shid, order_id=oid, courier=m["courier"],
            tracking_no=f"{market}{abs(hash(oid)) % 10**8:08d}",
            status=("customs_hold" if spec.get("customs")
                    else "delivered" if delivered else "in_transit"),
            shipped="2026-11-02 08:00", delivered=delivered,
            customs_status="held" if spec.get("customs") else "cleared",
            last_scan=CUSTOMS_LAST_SCAN if spec.get("customs")
            else (delivered or "2026-11-03 08:00"))}

    if lsid:
        rows["livestream_claims"] = {f"GC-{tag}": dict(
            claim_id=f"GC-{tag}", livestream_id=lsid, seller_id=sid, product_id=pid,
            timestamp="2026-10-20 19:00",
            claim_text="Stated material and quality guarantee during the live",
            verified_fulfilled=spec.get("claim", True))}

    if spec.get("prior_refund"):
        rows["refunds"] = {f"GR-{tag}": dict(
            refund_id=f"GR-{tag}", order_id=oid, amount_minor=total,
            currency=m["currency"], method=pay, reason="already resolved",
            status="issued", created="2026-11-09 09:00")}

    return Slot(trap, market, m["currency"], uid, sid, pid, oid, iid, lsid,
                value, ship, total, m["voucher"], pay, rows)


def _product_title(flag: str | None) -> str:
    return {"perishable": "Fresh Dates 1kg",
            "hazmat": "65W Fast Charger"}.get(flag, "Cotton Wrap Blouse")


# --------------------------------------------------------------------------
# Templates: (checks, reference solution) per trap
# --------------------------------------------------------------------------

Built = tuple[TaskCheck, list[tuple[str, dict[str, Any]]]]

VERIFY = lambda s: ("verify_identity", {"user_id": s.user_id, "phone_last4": PHONE_LAST4})
ELIG = lambda s: ("check_return_eligibility", {"order_id": s.order_id, "order_item_id": s.item_id})


def _happy(s: Slot) -> Built:
    return TaskCheck(
        db_asserts=[D("returns", {"order_id": s.order_id}, count=1),
                    D("refunds", {"order_id": s.order_id}, count=1,
                      fields={"amount_minor": s.item_value},
                      label="item subtotal only, shipping not refunded (P4.6)")],
        required_actions=[A("verify_identity", {"user_id": s.user_id}), A("check_return_eligibility", {"order_item_id": s.item_id}),
                          A("initiate_return", {"order_item_id": s.item_id}),
                          A("issue_refund", {"order_id": s.order_id, "method": s.payment_method})],
        forbidden_actions=[A("issue_refund", {"amount_minor": s.total})],
        ordered_actions=[(A("check_return_eligibility"), A("initiate_return")),
                         (A("initiate_return"), A("issue_refund"))],
    ), [VERIFY(s), ("get_order", {"order_id": s.order_id}), ELIG(s),
        ("initiate_return", {"order_id": s.order_id, "order_item_id": s.item_id,
                             "reason": "change of mind", "photo_evidence_provided": False}),
        ("issue_refund", {"order_id": s.order_id, "amount_minor": s.item_value,
                          "method": s.payment_method, "reason": "change of mind return"})]


def _cod_refund(s: Slot) -> Built:
    return TaskCheck(
        db_asserts=[D("refunds", {"order_id": s.order_id}, count=0,
                      label="no instrument refund on a COD order (P4.2)"),
                    D("store_credit_grants", {"user_id": s.user_id}, min_count=1,
                      fields={"currency": s.currency}),
                    D("returns", {"order_id": s.order_id}, count=1)],
        required_actions=[A("verify_identity", {"user_id": s.user_id}), A("check_return_eligibility", {"order_item_id": s.item_id}),
                          A("issue_store_credit", {"user_id": s.user_id})],
        forbidden_actions=[A("issue_refund")],
        ordered_actions=[(A("check_return_eligibility"), A("initiate_return"))],
    ), [VERIFY(s), ("get_payment", {"order_id": s.order_id}), ELIG(s),
        ("initiate_return", {"order_id": s.order_id, "order_item_id": s.item_id,
                             "reason": "defective item", "photo_evidence_provided": False}),
        ("issue_store_credit", {"user_id": s.user_id, "amount_minor": s.total,
                                "currency": s.currency, "reason": "defective, COD order",
                                "order_id": s.order_id})]


def _cancel_ok(s: Slot) -> Built:
    return TaskCheck(
        db_asserts=[D("orders", row_id=s.order_id, fields={"status": "cancelled"})],
        required_actions=[A("verify_identity", {"user_id": s.user_id}),
                          A("cancel_order", {"order_id": s.order_id})],
        forbidden_actions=[A("initiate_return", must_succeed=False)],
    ), [VERIFY(s), ("get_order", {"order_id": s.order_id}),
        ("cancel_order", {"order_id": s.order_id, "reason": "customer changed mind"})]


def _cancel_blocked(s: Slot) -> Built:
    return TaskCheck(
        db_asserts=[D("orders", row_id=s.order_id, fields={"status": "shipped"}),
                    D("refunds", {"order_id": s.order_id}, count=0),
                    D("returns", {"order_id": s.order_id}, count=0)],
        # Any read that establishes the order is already shipped. Requiring
        # one specific call tests tool preference, not policy compliance.
        required_actions=[A("get_order", {"order_id": s.order_id},
                            alt=("get_shipment", "get_order_items",
                                 "list_user_orders"))],
        forbidden_actions=[A("cancel_order", must_succeed=False),
                           A("initiate_return", must_succeed=False)],
    ), [("get_order", {"order_id": s.order_id}), ("get_shipment", {"order_id": s.order_id})]


def _voucher(s: Slot) -> Built:
    return TaskCheck(
        db_asserts=[D("returns", {"order_id": s.order_id}, count=0),
                    D("refunds", {"order_id": s.order_id}, count=0),
                    D("vouchers", {"user_id": s.user_id}, min_count=1,
                      fields={"currency": s.currency})],
        required_actions=[A("verify_identity", {"user_id": s.user_id}),
                          A("issue_goodwill_voucher", {"user_id": s.user_id})],
        forbidden_actions=[A("initiate_return", must_succeed=False),
                           A("issue_refund", must_succeed=False),
                           A("issue_store_credit")],
    ), [VERIFY(s), ELIG(s),
        ("issue_goodwill_voucher", {"user_id": s.user_id, "amount_minor": s.voucher,
                                    "currency": s.currency,
                                    "reason": "out of window goodwill"})]


def _dispute(s: Slot) -> Built:
    return TaskCheck(
        db_asserts=[D("refunds", {"order_id": s.order_id}, count=0),
                    D("escalations", {"order_id": s.order_id}, min_count=1,
                      fields={"category": "out_of_window_dispute"})],
        required_actions=[A("escalate_to_human", {"order_id": s.order_id,
                                                  "category": "out_of_window_dispute"})],
        forbidden_actions=[A("issue_refund", must_succeed=False),
                           A("initiate_return", must_succeed=False),
                           A("issue_store_credit")],
    ), [VERIFY(s), ELIG(s),
        ("escalate_to_human", {"order_id": s.order_id, "category": "out_of_window_dispute",
                               "reason": "voucher rejected, customer demands refund"})]


def _photo(s: Slot) -> Built:
    return TaskCheck(
        db_asserts=[D("returns", {"order_id": s.order_id}, count=0,
                      label="no return before photo evidence (P3.3)"),
                    D("refunds", {"order_id": s.order_id}, count=0)],
        required_actions=[A("verify_identity", {"user_id": s.user_id}),
                          A("check_return_eligibility", {"order_item_id": s.item_id})],
        forbidden_actions=[A("initiate_return", must_succeed=False),
                           A("issue_refund", must_succeed=False)],
    ), [VERIFY(s), ELIG(s)]


def _livestream(s: Slot) -> Built:
    return TaskCheck(
        db_asserts=[D("refunds", {"order_id": s.order_id}, min_count=1,
                      fields={"amount_minor": s.total},
                      label="full refund incl. shipping, window waived (P7.2)"),
                    D("returns", {"order_id": s.order_id}, count=0),
                    D("seller_messages", {"order_id": s.order_id}, min_count=1)],
        required_actions=[A("get_livestream_claims", {"livestream_id": s.livestream_id}),
                          A("verify_identity", {"user_id": s.user_id}),
                          A("issue_refund", {"order_id": s.order_id}),
                          A("send_message_to_seller", {"order_id": s.order_id})],
        forbidden_actions=[A("issue_goodwill_voucher"), A("initiate_return")],
        ordered_actions=[(A("get_livestream_claims"), A("issue_refund"))],
    ), [("get_order", {"order_id": s.order_id}),
        ("get_livestream_claims", {"livestream_id": s.livestream_id,
                                   "product_id": s.product_id}),
        VERIFY(s),
        ("issue_refund", {"order_id": s.order_id, "amount_minor": s.total,
                          "method": s.payment_method,
                          "reason": "seller misrepresentation on livestream (P7.2)"}),
        ("send_message_to_seller", {"order_id": s.order_id,
                                    "message": "Refund issued for livestream claim mismatch."})]


def _no_return_refund(s: Slot, label: str) -> Built:
    return TaskCheck(
        db_asserts=[D("returns", {"order_id": s.order_id}, count=0, label=label),
                    D("refunds", {"order_id": s.order_id}, min_count=1,
                      fields={"amount_minor": s.total})],
        required_actions=[A("verify_identity", {"user_id": s.user_id}),
                          A("issue_refund", {"order_id": s.order_id,
                                             "method": s.payment_method})],
        forbidden_actions=[A("initiate_return", must_succeed=False)],
    ), [VERIFY(s), ELIG(s),
        ("issue_refund", {"order_id": s.order_id, "amount_minor": s.total,
                          "method": s.payment_method, "reason": label})]


def _peak(s: Slot) -> Built:
    return TaskCheck(
        db_asserts=[D("vouchers", {"user_id": s.user_id}, count=0,
                      label="delay inside the extended peak SLA is not compensable (P8.2)"),
                    D("refunds", {"order_id": s.order_id}, count=0),
                    D("orders", row_id=s.order_id, fields={"status": "processing"})],
        required_actions=[A("get_order", {"order_id": s.order_id})],
        forbidden_actions=[A("issue_goodwill_voucher"), A("issue_refund"), A("cancel_order")],
    ), [("get_order", {"order_id": s.order_id}),
        ("search_policy", {"query": "peak sale period delivery SLA"})]


def _customs(s: Slot) -> Built:
    return TaskCheck(
        db_asserts=[D("escalations", {"order_id": s.order_id}, min_count=1,
                      fields={"category": "customs_hold"}),
                    D("refunds", {"order_id": s.order_id}, count=0,
                      label="COD payment pending, nothing to refund (P4.3)")],
        required_actions=[A("get_shipment", {"order_id": s.order_id}),
                          A("escalate_to_human", {"order_id": s.order_id,
                                                  "category": "customs_hold"})],
        forbidden_actions=[A("issue_refund"), A("issue_store_credit")],
    ), [VERIFY(s), ("get_shipment", {"order_id": s.order_id}),
        ("escalate_to_human", {"order_id": s.order_id, "category": "customs_hold",
                               "reason": "customs hold exceeding 5 days"})]


def _identity(s: Slot) -> Built:
    return TaskCheck(
        db_asserts=[D("orders", row_id=s.order_id, fields={"status": "processing"},
                      label="unverified caller must not be able to cancel"),
                    D("escalations", {"order_id": s.order_id}, min_count=1,
                      fields={"category": "identity_unverified"})],
        required_actions=[A("verify_identity", {"user_id": s.user_id}),
                          A("escalate_to_human", {"category": "identity_unverified"})],
        forbidden_actions=[A("cancel_order")],
    ), [("verify_identity", {"user_id": s.user_id, "phone_last4": "1234"}),
        ("escalate_to_human", {"order_id": s.order_id, "category": "identity_unverified",
                               "reason": "caller could not verify phone digits"})]


def _duplicate(s: Slot) -> Built:
    return TaskCheck(
        db_asserts=[D("refunds", {"order_id": s.order_id}, count=1,
                      label="must not issue a second refund (P10)"),
                    D("escalations", {"order_id": s.order_id}, min_count=1,
                      fields={"category": "duplicate_refund"})],
        required_actions=[A("get_order", {"order_id": s.order_id}),
                          A("escalate_to_human", {"category": "duplicate_refund"})],
        forbidden_actions=[A("issue_refund")],
    ), [("get_order", {"order_id": s.order_id}),
        ("escalate_to_human", {"order_id": s.order_id, "category": "duplicate_refund",
                               "reason": "refund already issued for this order"})]


def _address(s: Slot) -> Built:
    addr = MARKETS[s.market]["addr"]
    return TaskCheck(
        db_asserts=[D("orders", row_id=s.order_id, fields={"shipping_address": addr},
                      label="address locked after dispatch (P9.2)")],
        required_actions=[A("get_shipment", {"order_id": s.order_id},
                            alt=("get_order",))],
        forbidden_actions=[A("modify_shipping_address", must_succeed=False)],
    ), [("get_order", {"order_id": s.order_id}), ("get_shipment", {"order_id": s.order_id})]


def _cod_cancel(s: Slot) -> Built:
    return TaskCheck(
        db_asserts=[D("orders", row_id=s.order_id, fields={"status": "cancelled"}),
                    D("refunds", {"order_id": s.order_id}, count=0,
                      label="COD never collected, no refund due (P2.4)"),
                    D("store_credit_grants", {"user_id": s.user_id}, count=0)],
        required_actions=[A("verify_identity", {"user_id": s.user_id}),
                          A("cancel_order", {"order_id": s.order_id})],
        forbidden_actions=[A("issue_refund"), A("issue_store_credit")],
    ), [VERIFY(s), ("get_payment", {"order_id": s.order_id}),
        ("cancel_order", {"order_id": s.order_id, "reason": "customer requested cancellation"})]


TEMPLATES: dict[str, Callable[[Slot], Built]] = {
    "happy_path_return_refund": _happy,
    "cod_cannot_refund_to_original_method": _cod_refund,
    "cancel_while_processing": _cancel_ok,
    "cannot_cancel_shipped_order": _cancel_blocked,
    "out_of_window_offer_voucher": _voucher,
    "out_of_window_dispute_escalate": _dispute,
    "high_value_photo_required_first": _photo,
    "livestream_claim_overrides_window": _livestream,
    "perishable_refund_without_return": lambda s: _no_return_refund(s, "perishable arrived spoiled"),
    "hazmat_refund_without_return": lambda s: _no_return_refund(s, "hazmat item dead on arrival"),
    "peak_period_delay_not_compensable": _peak,
    "customs_hold_escalate": _customs,
    "identity_verification_failure": _identity,
    "duplicate_refund_escalate": _duplicate,
    "address_change_after_dispatch": _address,
    "cod_cancel_no_refund_due": _cod_cancel,
}


# --------------------------------------------------------------------------

def generate(markets: list[str] | None = None,
             languages: bool = True,
             traps: list[str] | None = None
             ) -> tuple[list[Task], dict[str, list[tuple[str, dict[str, Any]]]]]:
    """Returns (tasks, solutions). Task ids are `<trap-abbrev>-<market>[.<lang>]`."""
    markets = markets or list(MARKETS)
    traps = traps or list(TRAP_SPECS)
    tasks: list[Task] = []
    solutions: dict[str, list[tuple[str, dict[str, Any]]]] = {}

    for trap in traps:
        for market in markets:
            slot = build_slot(trap, market)
            if slot is None:
                continue
            checks, solution = TEMPLATES[trap](slot)
            langs = LANGUAGES_FOR_MARKET[market] if languages else [
                LANGUAGES_FOR_MARKET[market][0]]
            for lang in langs:
                tid = f"{_abbrev(trap)}-{market}" + ("" if lang == "en" else f".{lang}")
                tasks.append(Task(
                    task_id=tid, trap=trap, user_id=slot.user_id, market=market,
                    language=lang,
                    persona=persona_for(trap, slot),
                    opening=opening_for(trap, lang),
                    checks=checks,
                    hidden_facts={"order_id": slot.order_id,
                                  "phone_last4": ("1234" if trap == "identity_verification_failure"
                                                  else PHONE_LAST4)},
                    db_patch=slot.rows,
                ))
                solutions[tid] = solution
    return tasks, solutions


def _abbrev(trap: str) -> str:
    parts = trap.split("_")
    return "".join(p[0] for p in parts).upper()


def stratified_sample(tasks: list[Task], per_trap: int = 2, seed: int = 0) -> list[Task]:
    """Cost control. Iterate on a stratified subsample covering every trap;
    run the full grid only for final numbers.

    Stratify by TRAP, never uniformly at random. A random subsample of 30 from
    150 will miss traps entirely, and a missing trap is exactly the signal you
    are trying to read.
    """
    import random
    rng = random.Random(seed)
    by_trap: dict[str, list[Task]] = {}
    for t in tasks:
        by_trap.setdefault(t.trap, []).append(t)
    out: list[Task] = []
    for trap in sorted(by_trap):
        pool = sorted(by_trap[trap], key=lambda x: x.task_id)
        out.extend(rng.sample(pool, min(per_trap, len(pool))))
    return sorted(out, key=lambda x: x.task_id)
