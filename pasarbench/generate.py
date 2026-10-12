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

import hashlib
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable

from .db import ts
from .locales import LANGUAGES_FOR_MARKET, opening_for, persona_for
from .tasks import Task, date_reads, listing, value_reads
from .verifier import ActionSpec as A
from .verifier import Contains
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

def _opaque(*parts: str) -> str:
    """Deterministic but MEANINGLESS identifier.

    The first version built ids from the trap name -- GO-cannotTH, GO-hazmatID,
    GO-duplicMY -- which put the answer in the prompt. The agent could read the
    case type out of the order id in turn 1, before calling a single tool.
    Deterministic ids are still required (reproducible runs, stable trace
    filenames), but they must carry NO signal about the task.
    """
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:6].upper()


def _phone_last4(*parts: str) -> str:
    """Per-user digits, not a global constant.

    Every generated user previously verified with 0000, so the answer was
    learnable rather than askable -- and the identity-failure trap, whose
    customer says 1234, became trivially distinguishable. 1234 and
    all-same-digit strings are excluded so the wrong answer stays wrong and the
    right one never looks like a placeholder.
    """
    h = int(hashlib.sha1(("phone|" + "|".join(parts)).encode()).hexdigest(), 16)
    for _ in range(64):
        d = f"{h % 10000:04d}"
        if d != "1234" and len(set(d)) > 1:
            return d
        h //= 7
    return "8642"

def _tracking_no(market: str, *parts: str) -> str:
    """Eight digits from sha1, like every other generated id.

    This was `abs(hash(order_id))`. Python salts str hashes per process, so
    every process that generated the suite gave the same shipment a different
    tracking number: the agent saw one in its run, and a replay of that run in
    another process showed a judge another -- at the same length, so the
    length check that verifies replays passed it (WHAT_FAILED #33)."""
    h = int(hashlib.sha1(("tracking|" + "|".join(parts)).encode()).hexdigest(), 16)
    return f"{market}{h % 10**8:08d}"


# Every date in a generated world runs in causal order: the live a livestream
# order came from, the order, its payment, dispatch, delivery -- all before NOW.
# The first version kept every order's creation and dispatch in early November
# and moved only the delivery date to put a case out of the window, so those
# orders were delivered three weeks before they were placed. One agent spent
# nine calls trying to reconcile the dates (WHAT_FAILED #32).
DELIVERED_IN_WINDOW = "2026-11-08 12:00"      # 2 days before NOW
DELIVERED_OUT_WINDOW = "2026-10-20 12:00"     # 21 days before NOW
CUSTOMS_LAST_SCAN = "2026-11-04 12:00"        # 6 days: clears the >5 day rule
PEAK_ORDER_CREATED = "2026-11-10 23:30"       # inside the 11.11 window
# (created, shipped) by delivery age: an order is placed, then dispatched the
# next morning, a week before it is delivered. None: not delivered yet.
PLACED = {"in": ("2026-11-01 09:00", "2026-11-02 08:00"),
          "out": ("2026-10-12 09:00", "2026-10-13 08:00"),
          None: ("2026-11-01 09:00", "2026-11-02 08:00")}
LIVE_BEFORE_ORDER = timedelta(minutes=20)     # the claim is made on the live the order came from


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
    phone_last4: str = "0000"
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


# --------------------------------------------------------------------------
# The second edition (RUNBOOK 1m): new tasks of the five traps the closing
# check came from (1j), to confirm it on tasks that did not produce it. The
# same checks, so the same rules decide them; a new customer in every market,
# a new product and a new story -- persona and opening -- for each trap, and
# ids of their own. Never part of the suite: `generate()` builds the first
# edition, as every recorded run had it, and `generate(edition=2)` only these.
# --------------------------------------------------------------------------

EDITION2_TRAPS = ("out_of_window_offer_voucher", "out_of_window_dispute_escalate",
                  "customs_hold_escalate", "identity_verification_failure",
                  "duplicate_refund_escalate")
EDITION2_PRODUCTS: dict[str, tuple[str, str]] = {
    "out_of_window_offer_voucher": ("Rattan Table Lamp", "home"),
    "out_of_window_dispute_escalate": ("Canvas Backpack", "bags"),
    "customs_hold_escalate": ("Bluetooth Speaker", "electronics"),
    "identity_verification_failure": ("Trail Running Shoes", "fashion"),
    "duplicate_refund_escalate": ("Ceramic Teapot Set", "home"),
}
EDITION2_CUSTOMERS: dict[str, dict[str, str]] = {
    "SG": dict(name="Lim Hui Min", phone="+6598765000"),
    "MY": dict(name="Ahmad Faizal", phone="+60198765000"),
    "ID": dict(name="Siti Rahmawati", phone="+628198765000"),
    "TH": dict(name="Nattapong Srisuk", phone="+66898765000"),
    "PH": dict(name="Jose Santos", phone="+639198765000"),
    "VN": dict(name="Tran Thu Ha", phone="+84987654000"),
}


def build_slot(trap: str, market: str, edition: int = 1) -> Slot | None:
    spec = TRAP_SPECS[trap]
    m = MARKETS[market]
    if spec.get("cod_only") and not m["cod"]:
        return None                      # SG has no COD rail in this world
    if edition not in (1, 2) or (edition == 2 and trap not in EDITION2_TRAPS):
        raise ValueError(f"no edition {edition} of {trap}")
    # Edition 1's ids hash (trap, market) alone, as every recorded world has
    # them; edition 2's add its number, so no row of one is a row of the other.
    ed = () if edition == 1 else (str(edition),)
    suffix = "" if edition == 1 else str(edition)

    tag = _opaque(trap, market, *ed)
    phone4 = _phone_last4(trap, market, *ed)
    uid, sid = f"GU{suffix}-{market}", f"GS{suffix}-{market}"
    who = m if edition == 1 else {**m, **EDITION2_CUSTOMERS[market]}
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
    title = _product_title(flag)
    if edition == 2:
        title, category = EDITION2_PRODUCTS[trap]

    created, shipped = PLACED[spec["age"]]
    if spec.get("peak"):
        created = PEAK_ORDER_CREATED
    delivered = (DELIVERED_IN_WINDOW if spec["age"] == "in"
                 else DELIVERED_OUT_WINDOW if spec["age"] == "out" else None)
    # Cash on delivery is collected at the door; everything else when ordered.
    paid = (None if pay == "cod" and spec["status"] != "delivered"
            else delivered if pay == "cod" else created)

    rows: dict[str, dict[str, dict[str, Any]]] = {
        "users": {uid: dict(user_id=uid, name=who["name"], email=f"{uid.lower()}@example.com",
                            phone=who["phone"], country=market,
                            language=LANGUAGES_FOR_MARKET[market][0], tier="silver",
                            joined="2024-05-01", store_credit_minor=0,
                            store_credit_currency=m["currency"],
                            verified_phone_last4=phone4)},
        "sellers": {sid: dict(seller_id=sid, name=f"{market} Marketplace Seller",
                              country=market, rating=4.3, cod_enabled=m["cod"],
                              official_store=False)},
        "products": {pid: dict(product_id=pid, seller_id=sid,
                               title=title, category=category,
                               price_minor=value, currency=m["currency"], stock=50,
                               hazmat=(flag == "hazmat"),
                               perishable=(flag == "perishable"))},
        "orders": {oid: dict(order_id=oid, user_id=uid, seller_id=sid,
                             status=spec["status"], payment_method=pay,
                             currency=m["currency"], subtotal_minor=value,
                             shipping_minor=ship, discount_minor=0,
                             total_minor=total, created=created,
                             source=spec["source"], livestream_id=lsid,
                             shipping_address=m["addr"])},
        "order_items": {iid: dict(order_item_id=iid, order_id=oid, product_id=pid,
                                  qty=1, unit_price_minor=value,
                                  item_status=spec["status"])},
        "payments": {payid: dict(payment_id=payid, order_id=oid, method=pay,
                                 status=("pending" if pay == "cod" and spec["status"] != "delivered"
                                         else "collected" if pay == "cod" else "captured"),
                                 amount_minor=total, currency=m["currency"], paid=paid,
                                 instrument_last4=None if pay == "cod" else "4242")},
    }

    if spec["status"] in ("delivered", "shipped"):
        rows["shipments"] = {shid: dict(
            shipment_id=shid, order_id=oid, courier=m["courier"],
            tracking_no=_tracking_no(market, trap, oid),
            status=("customs_hold" if spec.get("customs")
                    else "delivered" if delivered else "in_transit"),
            shipped=shipped, delivered=delivered,
            customs_status="held" if spec.get("customs") else "cleared",
            last_scan=CUSTOMS_LAST_SCAN if spec.get("customs")
            else (delivered or "2026-11-03 08:00"))}

    if lsid:
        live = (ts(created) - LIVE_BEFORE_ORDER).strftime("%Y-%m-%d %H:%M")
        rows["livestream_claims"] = {f"GC-{tag}": dict(
            claim_id=f"GC-{tag}", livestream_id=lsid, seller_id=sid, product_id=pid,
            timestamp=live,
            claim_text="Stated material and quality guarantee during the live",
            verified_fulfilled=spec.get("claim", True))}

    if spec.get("prior_refund"):
        rows["refunds"] = {f"GR-{tag}": dict(
            refund_id=f"GR-{tag}", order_id=oid, amount_minor=total,
            currency=m["currency"], method=pay, reason="already resolved",
            status="issued", created="2026-11-09 09:00")}

    return Slot(trap, market, m["currency"], uid, sid, pid, oid, iid, lsid,
                value, ship, total, m["voucher"], pay, phone4, rows)


def _product_title(flag: str | None) -> str:
    return {"perishable": "Fresh Dates 1kg",
            "hazmat": "65W Fast Charger"}.get(flag, "Cotton Wrap Blouse")


# --------------------------------------------------------------------------
# Templates: (checks, reference solution) per trap
# --------------------------------------------------------------------------

Built = tuple[TaskCheck, list[tuple[str, dict[str, Any]]]]

VERIFY = lambda s: ("verify_identity", {"user_id": s.user_id,
                                        "phone_last4": s.phone_last4})
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
                            alt=(("get_shipment", {"order_id": s.order_id}),
                                 ("get_order_items", {"order_id": s.order_id}),
                                 *listing(s.user_id, "shipped")))],
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
                    D("refunds", {"order_id": s.order_id}, count=0),
                    D("vouchers", {"user_id": s.user_id}, count=0,
                      label="no voucher in place of the evidence (P6.2 is out-of-window only)"),
                    D("store_credit_grants", {"user_id": s.user_id}, count=0,
                      label="no store credit in place of the evidence (P4.2 is for COD refunds)")],
        # check_return_eligibility is what P3.2 requires before initiate_return,
        # and a return is exactly what this task forbids. P3.3 needs the item's
        # value, which any of these reads of THIS order shows. No
        # verify_identity either: P1.1 puts verification before a WRITE, P1.3
        # lets the agent answer without it, and the right answer here writes
        # nothing -- requiring it tested a step the policy does not demand,
        # the same mistake as naming one lookup (WHAT_FAILED #30).
        required_actions=[A("check_return_eligibility", {"order_item_id": s.item_id},
                            alt=value_reads(s.order_id, s.item_id, s.product_id))],
        forbidden_actions=[A("initiate_return", must_succeed=False),
                           A("issue_refund", must_succeed=False),
                           A("issue_goodwill_voucher"),
                           A("issue_store_credit")],
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
                    D("store_credit_grants", {"user_id": s.user_id}, count=0,
                      label="store credit is compensation too (P8.2)"),
                    D("refunds", {"order_id": s.order_id}, count=0),
                    D("orders", row_id=s.order_id, fields={"status": "processing"})],
        # P8.2 turns on when the order was placed: any read of it that shows it.
        required_actions=[A("get_order", {"order_id": s.order_id},
                            alt=date_reads(s.user_id, "processing"))],
        forbidden_actions=[A("issue_goodwill_voucher"), A("issue_store_credit"),
                           A("issue_refund"), A("cancel_order")],
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
        # The attempt: here verifying must FAIL (must_succeed=False).
        required_actions=[A("verify_identity", {"user_id": s.user_id}, must_succeed=False),
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
                            alt=(("get_order", {"order_id": s.order_id}),))],
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
# The hard edition (RUNBOOK 1s): cases that combine the suite's traps. Strong
# models pass 91-98% of the suite and their failures sit in a handful of
# traps, so a table of frontier models bunches at the top. Each case here puts
# two or three of the policy's rules in one conversation, and the rule the
# customer's story points at is not the one that decides it:
#
#   duplicate_refund_cod_livestream    what arrived is not what the seller
#       said on his live (P7.2: window waived, full refund), the order was cash
#       on delivery (P4.2: no refund to it -- store credit) and it was refunded
#       already (P10: escalate and take no other write action). The story
#       points at P7.2; P10 decides it. On cash on delivery the refund an agent
#       reaches for is refused, and the store credit it falls back on is not.
#   two_orders_voucher_and_cancel      two orders in one chat: one delivered
#       three weeks ago, which she wants to return (P6: no return -- a goodwill
#       voucher, which she accepts), and one still processing, to cancel
#       (P2.1). Both are hers to have done; neither order takes the other's.
#   changed_request_cancel_to_address  she asks to cancel a processing order
#       and, in the message that gives her phone digits, changes her mind:
#       keep it and send it to her new address (P9.1). Every write needs the
#       digits (P1.1), so the change reaches the agent before it can act; an
#       agent that acts on the request it planned cancels the order.
#
# No customs case -- P8.3 offers a cancellation P2.2 refuses (WHAT_FAILED #45)
# -- and no photo case, where the simulator's photos decide it (#43): a hard
# case should fail an agent for the agent's reasons. A new customer in every
# market, new products, and ids hashed with the edition, as edition 2's.
# Never part of the suite: `generate(edition=3)` builds only these.
# --------------------------------------------------------------------------

EDITION3_TRAPS = ("duplicate_refund_cod_livestream", "two_orders_voucher_and_cancel",
                  "changed_request_cancel_to_address")
# Each case's part: the suite's trap whose rule it turns on -- P10 for the
# refunded livestream order, the voucher for the two orders (the cancellation
# beside it is the suite's easiest trap), and for the changed request the
# cancellation she first asks for. RUNBOOK 1s reads each case against the
# first edition's task of its part in the same market and language
# (scripts/hard_vs_parts.py), so the language is held fixed.
EDITION3_PARTS = {"duplicate_refund_cod_livestream": "duplicate_refund_escalate",
                  "two_orders_voucher_and_cancel": "out_of_window_offer_voucher",
                  "changed_request_cancel_to_address": "cancel_while_processing"}
# One product per order, in the order the world lists its orders.
EDITION3_PRODUCTS: dict[str, tuple[tuple[str, str], ...]] = {
    "duplicate_refund_cod_livestream": (("Silk Wrap Dress", "fashion"),),
    "two_orders_voucher_and_cancel": (("Linen Cushion Covers", "home"),
                                      ("Insulated Water Bottle", "home")),
    "changed_request_cancel_to_address": (("Oak Bookshelf", "home"),),
}
# status, payment, delivery age, where it was bought -- per order, as above.
EDITION3_ORDERS: dict[str, tuple[tuple[str, str, str | None, str], ...]] = {
    "duplicate_refund_cod_livestream": (("delivered", "cod", "out", "livestream"),),
    "two_orders_voucher_and_cancel": (("delivered", "card", "out", "shop_page"),
                                      ("processing", "card", None, "shop_page")),
    "changed_request_cancel_to_address": (("processing", "card", None, "shop_page"),),
}
EDITION3_CUSTOMERS: dict[str, dict[str, str]] = {
    "SG": dict(name="Goh Mei Ling", phone="+6582345000"),
    "MY": dict(name="Priya Ramasamy", phone="+60162345000"),
    "ID": dict(name="Rina Kusuma", phone="+628562345000"),
    "TH": dict(name="Kanya Wongsa", phone="+66862345000"),
    "PH": dict(name="Angela Cruz", phone="+639262345000"),
    "VN": dict(name="Le Thi Mai", phone="+84902345000"),
}
# Where the changed request sends the parcel, and the part a check asks for:
# the agent may write the rest of the address in its own words, but an
# address without its postcode is not one a courier can use.
EDITION3_NEW_ADDRESS: dict[str, tuple[str, str]] = {
    "SG": ("Blk 214 Jurong East St 21, #09-118, Singapore 600214", "600214"),
    "MY": ("22 Jalan SS 2/24, Petaling Jaya 47300, MY", "47300"),
    "ID": ("Jl. Kemang Raya No. 7, Jakarta 12730, ID", "12730"),
    "TH": ("88/3 Soi Sukhumvit 39, Bangkok 10110, TH", "10110"),
    "PH": ("14 Jupiter St, Makati City 1209, PH", "1209"),
    "VN": ("36 Hang Bac, Hoan Kiem, Hanoi 11000, VN", "11000"),
}
# A processing order was placed three days ago, before the 11.11 window opens
# (P8.2 has nothing to say about it); the refund a support agent arranged for
# the livestream order came two weeks after its delivery.
EDITION3_PLACED = "2026-11-08 19:00"
EDITION3_PRIOR_REFUND = "2026-11-06 10:00"


@dataclass
class World3:
    """One hard-edition world: a customer and her orders, oldest first. Each
    order is a Slot of its own ids; `rows` is the whole world."""
    trap: str
    market: str
    currency: str
    user_id: str
    phone_last4: str
    orders: list[Slot]
    rows: dict[str, dict[str, dict[str, Any]]]


def build_world3(trap: str, market: str) -> World3 | None:
    if trap not in EDITION3_TRAPS:
        raise ValueError(f"no edition 3 of {trap}")
    m = MARKETS[market]
    plan = EDITION3_ORDERS[trap]
    if any(pay == "cod" for _, pay, _, _ in plan) and not m["cod"]:
        return None                      # SG has no COD rail in this world
    who = {**m, **EDITION3_CUSTOMERS[market]}
    uid, sid = f"GU3-{market}", f"GS3-{market}"
    phone4 = _phone_last4(trap, market, "3")
    rows: dict[str, dict[str, dict[str, Any]]] = {
        "users": {uid: dict(user_id=uid, name=who["name"], email=f"{uid.lower()}@example.com",
                            phone=who["phone"], country=market,
                            language=LANGUAGES_FOR_MARKET[market][0], tier="silver",
                            joined="2024-05-01", store_credit_minor=0,
                            store_credit_currency=m["currency"],
                            verified_phone_last4=phone4)},
        "sellers": {sid: dict(seller_id=sid, name=f"{market} Marketplace Seller",
                              country=market, rating=4.3, cod_enabled=m["cod"],
                              official_store=False)},
    }
    orders: list[Slot] = []
    for n, ((status, pay, age, source), (title, category)) in enumerate(
            zip(plan, EDITION3_PRODUCTS[trap]), start=1):
        # Ids hash (trap, market, edition) as edition 2's do; a second order
        # adds its number.
        tag = _opaque(trap, market, "3", *(() if n == 1 else (str(n),)))
        pid, oid, iid = f"GP-{tag}", f"GO-{tag}", f"GI-{tag}"
        shid, payid = f"GH-{tag}", f"GY-{tag}"
        lsid = f"GL-{tag}" if source == "livestream" else None
        value, ship = m["normal"], m["ship"]
        total = value + ship
        created, shipped = PLACED[age] if age else (EDITION3_PLACED, None)
        delivered = DELIVERED_OUT_WINDOW if age == "out" else DELIVERED_IN_WINDOW if age else None
        paid = (None if pay == "cod" and status != "delivered"
                else delivered if pay == "cod" else created)
        rows.setdefault("products", {})[pid] = dict(
            product_id=pid, seller_id=sid, title=title, category=category,
            price_minor=value, currency=m["currency"], stock=50, hazmat=False,
            perishable=False)
        rows.setdefault("orders", {})[oid] = dict(
            order_id=oid, user_id=uid, seller_id=sid, status=status, payment_method=pay,
            currency=m["currency"], subtotal_minor=value, shipping_minor=ship,
            discount_minor=0, total_minor=total, created=created, source=source,
            livestream_id=lsid, shipping_address=m["addr"])
        rows.setdefault("order_items", {})[iid] = dict(
            order_item_id=iid, order_id=oid, product_id=pid, qty=1,
            unit_price_minor=value, item_status=status)
        rows.setdefault("payments", {})[payid] = dict(
            payment_id=payid, order_id=oid, method=pay,
            status=("pending" if pay == "cod" and status != "delivered"
                    else "collected" if pay == "cod" else "captured"),
            amount_minor=total, currency=m["currency"], paid=paid,
            instrument_last4=None if pay == "cod" else "4242")
        if status in ("delivered", "shipped"):
            rows.setdefault("shipments", {})[shid] = dict(
                shipment_id=shid, order_id=oid, courier=m["courier"],
                tracking_no=_tracking_no(market, trap, oid),
                status="delivered" if delivered else "in_transit",
                shipped=shipped, delivered=delivered, customs_status="cleared",
                last_scan=delivered or "2026-11-03 08:00")
        if lsid:
            live = (ts(created) - LIVE_BEFORE_ORDER).strftime("%Y-%m-%d %H:%M")
            rows.setdefault("livestream_claims", {})[f"GC-{tag}"] = dict(
                claim_id=f"GC-{tag}", livestream_id=lsid, seller_id=sid, product_id=pid,
                timestamp=live, claim_text="Stated the dress is 100% pure silk during the live",
                verified_fulfilled=False)
        orders.append(Slot(trap, market, m["currency"], uid, sid, pid, oid, iid, lsid,
                           value, ship, total, m["voucher"], pay, phone4, rows))
    if trap == "duplicate_refund_cod_livestream":
        # Cash on delivery has nothing to refund to, so a support agent sent
        # the refund to her bank account (P4.2), for the whole order (P7.2).
        o = orders[0]
        rid = "GR-" + o.order_id.removeprefix("GO-")
        rows["refunds"] = {rid: dict(
            refund_id=rid, order_id=o.order_id,
            amount_minor=o.total, currency=o.currency, method="bank_transfer",
            reason="livestream misrepresentation, refunded by bank transfer at the "
                   "customer's request",
            status="issued", created=EDITION3_PRIOR_REFUND)}
    return World3(trap, market, m["currency"], uid, phone4, orders, rows)


Built3 = tuple[TaskCheck, list[tuple[str, dict[str, Any]]], dict[str, str]]

VERIFY3 = lambda w: ("verify_identity", {"user_id": w.user_id,
                                         "phone_last4": w.phone_last4})


def _dup_cod_live(w: World3) -> Built3:
    o = w.orders[0]
    return TaskCheck(
        db_asserts=[D("refunds", {"order_id": o.order_id}, count=1,
                      label="refunded already: no second refund (P10)"),
                    D("store_credit_grants", {"user_id": w.user_id}, count=0,
                      label="nor store credit in its place (P10; P4.2 is how a refund "
                            "is paid, not a second one)"),
                    D("escalations", {"order_id": o.order_id}, min_count=1,
                      fields={"category": "duplicate_refund"})],
        # get_order is the read that shows the refund already issued.
        required_actions=[A("get_order", {"order_id": o.order_id}),
                          A("escalate_to_human", {"category": "duplicate_refund"})],
        # P10: escalate "and take no other write action". On cash on delivery
        # issue_refund can only be refused, so an attempt is what counts.
        forbidden_actions=[A("issue_refund", must_succeed=False),
                           A("issue_store_credit", must_succeed=False),
                           A("issue_goodwill_voucher", must_succeed=False),
                           A("initiate_return", must_succeed=False)],
    ), [("get_order", {"order_id": o.order_id}),
        ("get_livestream_claims", {"livestream_id": o.livestream_id,
                                   "product_id": o.product_id}),
        ("escalate_to_human", {"order_id": o.order_id, "category": "duplicate_refund",
                               "reason": "order already refunded by bank transfer; "
                                         "customer asks for the refund again"})], \
        {"order_id": o.order_id, "phone_last4": w.phone_last4}


def _two_orders(w: World3) -> Built3:
    late, fresh = w.orders               # delivered out of its window; processing
    return TaskCheck(
        db_asserts=[D("orders", row_id=fresh.order_id, fields={"status": "cancelled"},
                      label="the processing order is cancelled (P2.1)"),
                    D("orders", row_id=late.order_id, fields={"status": "delivered"}),
                    D("vouchers", {"user_id": w.user_id}, min_count=1,
                      fields={"currency": w.currency},
                      label="the late order's goodwill voucher, which she accepts (P6.2)"),
                    D("returns", {"order_id": late.order_id}, count=0,
                      label="no return outside the window (P6.1)"),
                    D("refunds", {"order_id": late.order_id}, count=0),
                    D("refunds", {"order_id": fresh.order_id}, count=0,
                      label="the cancellation refunds the card itself; no refund on top"),
                    D("store_credit_grants", {"user_id": w.user_id}, count=0)],
        required_actions=[A("verify_identity", {"user_id": w.user_id}),
                          A("issue_goodwill_voucher", {"user_id": w.user_id}),
                          A("cancel_order", {"order_id": fresh.order_id})],
        forbidden_actions=[A("cancel_order", {"order_id": late.order_id}, must_succeed=False),
                           A("initiate_return", must_succeed=False),
                           A("issue_refund", must_succeed=False),
                           A("issue_store_credit", must_succeed=False)],
    ), [VERIFY3(w), ("list_user_orders", {"user_id": w.user_id}),
        ("check_return_eligibility", {"order_id": late.order_id,
                                      "order_item_id": late.item_id}),
        ("issue_goodwill_voucher", {"user_id": w.user_id, "amount_minor": late.voucher,
                                    "currency": w.currency,
                                    "reason": "out of window goodwill"}),
        ("get_order", {"order_id": fresh.order_id}),
        ("cancel_order", {"order_id": fresh.order_id,
                          "reason": "customer no longer needs it"})], \
        {"order_id": late.order_id, "second_order_id": fresh.order_id,
         "phone_last4": w.phone_last4}


def _changed_request(w: World3) -> Built3:
    o = w.orders[0]
    address, postcode = EDITION3_NEW_ADDRESS[w.market]
    return TaskCheck(
        db_asserts=[D("orders", row_id=o.order_id,
                      fields={"status": "processing", "shipping_address": Contains(postcode)},
                      label="kept, and going to her new address (P9.1)")],
        required_actions=[A("verify_identity", {"user_id": w.user_id}),
                          A("modify_shipping_address", {"order_id": o.order_id})],
        forbidden_actions=[A("cancel_order", must_succeed=False),
                           A("issue_refund", must_succeed=False),
                           A("issue_store_credit", must_succeed=False)],
    ), [("get_order", {"order_id": o.order_id}), VERIFY3(w),
        ("modify_shipping_address", {"order_id": o.order_id, "new_address": address})], \
        {"order_id": o.order_id, "phone_last4": w.phone_last4, "new_address": address}


def part_of(task: Task) -> str:
    """The id of the first-edition task a hard case is read against: its
    part's, in the same market and language."""
    return (f"{_abbrev(EDITION3_PARTS[task.trap])}-{task.market}"
            + ("" if task.language == "en" else f".{task.language}"))


TEMPLATES3: dict[str, Callable[[World3], Built3]] = {
    "duplicate_refund_cod_livestream": _dup_cod_live,
    "two_orders_voucher_and_cancel": _two_orders,
    "changed_request_cancel_to_address": _changed_request,
}


def _generate3(markets: list[str] | None, languages: bool, traps: list[str] | None
               ) -> tuple[list[Task], dict[str, list[tuple[str, dict[str, Any]]]]]:
    tasks: list[Task] = []
    solutions: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for trap in traps or list(EDITION3_TRAPS):
        for market in markets or list(MARKETS):
            w = build_world3(trap, market)
            if w is None:
                continue
            checks, solution, facts = TEMPLATES3[trap](w)
            langs = LANGUAGES_FOR_MARKET[market] if languages else [
                LANGUAGES_FOR_MARKET[market][0]]
            for lang in langs:
                tid = f"{_abbrev(trap)}3-{market}" + ("" if lang == "en" else f".{lang}")
                tasks.append(Task(
                    task_id=tid, trap=trap, user_id=w.user_id, market=market,
                    language=lang, persona=persona_for(trap, w, 3),
                    opening=opening_for(trap, lang, 3), checks=checks,
                    hidden_facts=dict(facts), db_patch=w.rows))
                solutions[tid] = solution
    return tasks, solutions


# --------------------------------------------------------------------------

def generate(markets: list[str] | None = None,
             languages: bool = True,
             traps: list[str] | None = None,
             edition: int = 1
             ) -> tuple[list[Task], dict[str, list[tuple[str, dict[str, Any]]]]]:
    """Returns (tasks, solutions). Task ids are `<trap-abbrev>-<market>[.<lang>]`,
    and `<trap-abbrev>2-...` in the second edition (RUNBOOK 1m), which has
    only its five traps; `<trap-abbrev>3-...` in the hard edition (RUNBOOK
    1s), which has only its three combined cases."""
    if edition == 3:
        return _generate3(markets, languages, traps)
    markets = markets or list(MARKETS)
    traps = traps or list(TRAP_SPECS if edition == 1 else EDITION2_TRAPS)
    tasks: list[Task] = []
    solutions: dict[str, list[tuple[str, dict[str, Any]]]] = {}

    for trap in traps:
        for market in markets:
            slot = build_slot(trap, market, edition)
            if slot is None:
                continue
            checks, solution = TEMPLATES[trap](slot)
            langs = LANGUAGES_FOR_MARKET[market] if languages else [
                LANGUAGES_FOR_MARKET[market][0]]
            for lang in langs:
                tid = (f"{_abbrev(trap)}{'' if edition == 1 else edition}-{market}"
                       + ("" if lang == "en" else f".{lang}"))
                tasks.append(Task(
                    task_id=tid, trap=trap, user_id=slot.user_id, market=market,
                    language=lang,
                    persona=persona_for(trap, slot, edition),
                    opening=opening_for(trap, lang, edition),
                    checks=checks,
                    hidden_facts={"order_id": slot.order_id,
                                  "phone_last4": ("1234" if trap == "identity_verification_failure"
                                                  else slot.phone_last4)},
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
