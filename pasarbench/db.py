"""
PasarBench database layer.

Design notes
------------
1. Everything is plain dicts so a task's success can be checked by comparing
   final state, not by string-matching the agent's prose. This is the single
   most important design decision in the benchmark (see tau-bench).

2. Money is stored as INTEGER MINOR UNITS with a per-currency exponent.
   IDR and VND have no minor unit (exponent 0), SGD/MYR/THB/PHP have 2.
   This is real SEA texture and a genuine trap: an agent that assumes
   "divide by 100" will be wrong on half the markets.

3. NOW is frozen. Every relative-time policy rule (the 14-day return window,
   peak-sale SLA) is deterministic. Never call datetime.now() anywhere.

4. Database.action_log records every write attempt, successful or not.
   The verifier reads it to check required/forbidden/ordered actions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

SGT = timezone(timedelta(hours=8))

# Frozen clock. 11.11 sale period -- makes peak-season tasks natural.
NOW = datetime(2026, 11, 11, 10, 0, 0, tzinfo=SGT)

CURRENCY_EXPONENT = {
    "SGD": 2,
    "MYR": 2,
    "THB": 2,
    "PHP": 2,
    "IDR": 0,
    "VND": 0,
}

# Approximate SGD conversion, used ONLY for policy thresholds that are
# defined in SGD (e.g. the photo-evidence threshold). Fixed, not live.
SGD_RATE = {
    "SGD": 1.0,
    "MYR": 0.30,
    "THB": 0.038,
    "PHP": 0.023,
    "IDR": 0.000082,
    "VND": 0.000053,
}

PEAK_PERIODS = [
    ("2026-09-09", "2026-09-11"),
    ("2026-11-10", "2026-11-13"),
    ("2026-12-12", "2026-12-14"),
]


def ts(s: str) -> datetime:
    """Parse 'YYYY-MM-DD HH:MM' as SGT."""
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=SGT)


def to_sgd(amount_minor: int, currency: str) -> float:
    exp = CURRENCY_EXPONENT[currency]
    return (amount_minor / (10**exp)) * SGD_RATE[currency]


def fmt_money(amount_minor: int, currency: str) -> str:
    exp = CURRENCY_EXPONENT[currency]
    if exp == 0:
        return f"{currency} {amount_minor:,}"
    return f"{currency} {amount_minor / (10 ** exp):,.2f}"


def is_peak(when: datetime) -> bool:
    d = when.strftime("%Y-%m-%d")
    return any(start <= d <= end for start, end in PEAK_PERIODS)


# --------------------------------------------------------------------------
# Seed data
# --------------------------------------------------------------------------

def seed() -> dict[str, dict[str, dict[str, Any]]]:
    """Build the world. Every task starts from a deep copy of this."""

    users = {
        "U001": dict(user_id="U001", name="Nurul Aisyah", email="nurul.a@example.com",
                     phone="+60123456789", country="MY", language="ms",
                     tier="gold", joined="2023-04-11", store_credit_minor=0,
                     store_credit_currency="MYR", verified_phone_last4="6789"),
        "U002": dict(user_id="U002", name="Tan Wei Ming", email="wm.tan@example.com",
                     phone="+6591234567", country="SG", language="en",
                     tier="silver", joined="2024-01-20", store_credit_minor=500,
                     store_credit_currency="SGD", verified_phone_last4="4567"),
        "U003": dict(user_id="U003", name="Budi Santoso", email="budi.s@example.com",
                     phone="+628123456789", country="ID", language="id",
                     tier="bronze", joined="2025-06-02", store_credit_minor=0,
                     store_credit_currency="IDR", verified_phone_last4="6789"),
        "U004": dict(user_id="U004", name="Siriporn Chai", email="siriporn.c@example.com",
                     phone="+66812345678", country="TH", language="th",
                     tier="gold", joined="2022-11-30", store_credit_minor=0,
                     store_credit_currency="THB", verified_phone_last4="5678"),
        "U005": dict(user_id="U005", name="Maria Reyes", email="m.reyes@example.com",
                     phone="+639171234567", country="PH", language="en",
                     tier="bronze", joined="2026-02-14", store_credit_minor=0,
                     store_credit_currency="PHP", verified_phone_last4="4567"),
        "U006": dict(user_id="U006", name="Nguyen Minh Anh", email="minhanh.n@example.com",
                     phone="+84901234567", country="VN", language="vi",
                     tier="silver", joined="2024-08-08", store_credit_minor=0,
                     store_credit_currency="VND", verified_phone_last4="4567"),
    }

    sellers = {
        "S001": dict(seller_id="S001", name="GlowUp Beauty Official", country="MY",
                     rating=4.8, cod_enabled=True, official_store=True),
        "S002": dict(seller_id="S002", name="TechDeal SG", country="SG",
                     rating=4.2, cod_enabled=False, official_store=False),
        "S003": dict(seller_id="S003", name="Toko Berkah Jaya", country="ID",
                     rating=3.9, cod_enabled=True, official_store=False),
        "S004": dict(seller_id="S004", name="Bangkok Fashion House", country="TH",
                     rating=4.5, cod_enabled=True, official_store=False),
        "S005": dict(seller_id="S005", name="Manila Home Living", country="PH",
                     rating=4.0, cod_enabled=True, official_store=False),
    }

    products = {
        "P001": dict(product_id="P001", seller_id="S001", title="Vitamin C Serum 30ml",
                     category="beauty", price_minor=8900, currency="MYR", stock=120,
                     hazmat=False, perishable=False),
        "P002": dict(product_id="P002", seller_id="S001", title="Retinol Night Cream 50g",
                     category="beauty", price_minor=12900, currency="MYR", stock=45,
                     hazmat=False, perishable=False),
        "P003": dict(product_id="P003", seller_id="S002", title="Wireless Earbuds Pro",
                     category="electronics", price_minor=13900, currency="SGD", stock=8,
                     hazmat=False, perishable=False),
        "P004": dict(product_id="P004", seller_id="S002", title="65W GaN Charger",
                     category="electronics", price_minor=4500, currency="SGD", stock=200,
                     hazmat=True, perishable=False),
        "P005": dict(product_id="P005", seller_id="S003", title="Hijab Voal Premium",
                     category="fashion", price_minor=85000, currency="IDR", stock=300,
                     hazmat=False, perishable=False),
        "P006": dict(product_id="P006", seller_id="S003", title="Kurma Ajwa 1kg",
                     category="grocery", price_minor=250000, currency="IDR", stock=60,
                     hazmat=False, perishable=True),
        "P007": dict(product_id="P007", seller_id="S004", title="Silk Wrap Dress",
                     category="fashion", price_minor=189000, currency="THB", stock=22,
                     hazmat=False, perishable=False),
        "P008": dict(product_id="P008", seller_id="S004", title="Leather Crossbody Bag",
                     category="fashion", price_minor=890000, currency="THB", stock=15,
                     hazmat=False, perishable=False),
        "P009": dict(product_id="P009", seller_id="S005", title="Rattan Storage Basket",
                     category="home", price_minor=89900, currency="PHP", stock=80,
                     hazmat=False, perishable=False),
        "P010": dict(product_id="P010", seller_id="S001", title="Sheet Mask Bundle x20",
                     category="beauty", price_minor=5900, currency="MYR", stock=500,
                     hazmat=False, perishable=False),
        "P011": dict(product_id="P011", seller_id="S002", title="Mechanical Keyboard 75%",
                     category="electronics", price_minor=15900, currency="SGD", stock=12,
                     hazmat=False, perishable=False),
        "P012": dict(product_id="P012", seller_id="S003", title="Kaos Polos Cotton Combed",
                     category="fashion", price_minor=65000, currency="IDR", stock=400,
                     hazmat=False, perishable=False),
    }

    # source: shop_page | short_video | livestream
    orders = {
        # --- MY, COD, delivered 3 days ago, in return window -------------
        "O1001": dict(order_id="O1001", user_id="U001", seller_id="S001",
                      status="delivered", payment_method="cod", currency="MYR",
                      subtotal_minor=17800, shipping_minor=800, discount_minor=0,
                      total_minor=18600, created="2026-11-02 14:30",
                      source="livestream", livestream_id="LS7001",
                      shipping_address="12 Jalan Ampang, Kuala Lumpur 50450, MY"),
        # --- SG, card, still processing, cancellable --------------------
        "O1002": dict(order_id="O1002", user_id="U002", seller_id="S002",
                      status="processing", payment_method="card", currency="SGD",
                      subtotal_minor=13900, shipping_minor=0, discount_minor=1000,
                      total_minor=12900, created="2026-11-10 21:05",
                      source="short_video", livestream_id=None,
                      shipping_address="Blk 123 Toa Payoh Lor 1, #05-67, Singapore 310123"),
        # --- SG, card, already shipped, NOT cancellable -----------------
        "O1003": dict(order_id="O1003", user_id="U002", seller_id="S002",
                      status="shipped", payment_method="card", currency="SGD",
                      subtotal_minor=15900, shipping_minor=350, discount_minor=0,
                      total_minor=16250, created="2026-11-08 09:14",
                      source="shop_page", livestream_id=None,
                      shipping_address="Blk 123 Toa Payoh Lor 1, #05-67, Singapore 310123"),
        # --- ID, COD, delivered 20 days ago, OUTSIDE window -------------
        "O1004": dict(order_id="O1004", user_id="U003", seller_id="S003",
                      status="delivered", payment_method="cod", currency="IDR",
                      subtotal_minor=170000, shipping_minor=15000, discount_minor=0,
                      total_minor=185000, created="2026-10-14 11:20",
                      source="shop_page", livestream_id=None,
                      shipping_address="Jl. Sudirman No. 45, Jakarta Selatan 12190, ID"),
        # --- TH, ewallet, delivered 5 days ago, HIGH VALUE (photo req) --
        "O1005": dict(order_id="O1005", user_id="U004", seller_id="S004",
                      status="delivered", payment_method="ewallet", currency="THB",
                      subtotal_minor=890000, shipping_minor=5000, discount_minor=0,
                      total_minor=895000, created="2026-11-01 16:45",
                      source="livestream", livestream_id="LS7002",
                      shipping_address="88/12 Sukhumvit Soi 24, Bangkok 10110, TH"),
        # --- PH, COD, in transit, customs/courier delay -----------------
        "O1006": dict(order_id="O1006", user_id="U005", seller_id="S005",
                      status="shipped", payment_method="cod", currency="PHP",
                      subtotal_minor=89900, shipping_minor=12000, discount_minor=0,
                      total_minor=101900, created="2026-11-02 08:00",
                      source="short_video", livestream_id=None,
                      shipping_address="14 Kalayaan Ave, Quezon City 1101, PH"),
        # --- VN, bank_transfer, delivered, livestream claim mismatch ----
        "O1007": dict(order_id="O1007", user_id="U006", seller_id="S003",
                      status="delivered", payment_method="bank_transfer", currency="VND",
                      subtotal_minor=450000, shipping_minor=30000, discount_minor=0,
                      total_minor=480000, created="2026-10-20 19:30",
                      source="livestream", livestream_id="LS7003",
                      shipping_address="25 Nguyen Hue, District 1, Ho Chi Minh City, VN"),
        # --- MY, ewallet, delivered 1 day ago, perishable grocery -------
        "O1008": dict(order_id="O1008", user_id="U001", seller_id="S003",
                      status="delivered", payment_method="ewallet", currency="IDR",
                      subtotal_minor=250000, shipping_minor=20000, discount_minor=0,
                      total_minor=270000, created="2026-11-05 10:10",
                      source="shop_page", livestream_id=None,
                      shipping_address="12 Jalan Ampang, Kuala Lumpur 50450, MY"),
        # --- SG, card, delivered, hazmat item ---------------------------
        "O1009": dict(order_id="O1009", user_id="U002", seller_id="S002",
                      status="delivered", payment_method="card", currency="SGD",
                      subtotal_minor=4500, shipping_minor=200, discount_minor=0,
                      total_minor=4700, created="2026-11-03 13:00",
                      source="shop_page", livestream_id=None,
                      shipping_address="Blk 123 Toa Payoh Lor 1, #05-67, Singapore 310123"),
        # --- ID, COD, processing, peak-period order ---------------------
        "O1010": dict(order_id="O1010", user_id="U003", seller_id="S003",
                      status="processing", payment_method="cod", currency="IDR",
                      subtotal_minor=130000, shipping_minor=12000, discount_minor=0,
                      total_minor=142000, created="2026-11-10 23:50",
                      source="livestream", livestream_id="LS7004",
                      shipping_address="Jl. Sudirman No. 45, Jakarta Selatan 12190, ID"),
    }

    order_items = {
        "OI1": dict(order_item_id="OI1", order_id="O1001", product_id="P001",
                    qty=2, unit_price_minor=8900, item_status="delivered"),
        "OI2": dict(order_item_id="OI2", order_id="O1002", product_id="P003",
                    qty=1, unit_price_minor=13900, item_status="processing"),
        "OI3": dict(order_item_id="OI3", order_id="O1003", product_id="P011",
                    qty=1, unit_price_minor=15900, item_status="shipped"),
        "OI4": dict(order_item_id="OI4", order_id="O1004", product_id="P005",
                    qty=2, unit_price_minor=85000, item_status="delivered"),
        "OI5": dict(order_item_id="OI5", order_id="O1005", product_id="P008",
                    qty=1, unit_price_minor=890000, item_status="delivered"),
        "OI6": dict(order_item_id="OI6", order_id="O1006", product_id="P009",
                    qty=1, unit_price_minor=89900, item_status="shipped"),
        "OI7": dict(order_item_id="OI7", order_id="O1007", product_id="P012",
                    qty=1, unit_price_minor=450000, item_status="delivered"),
        "OI8": dict(order_item_id="OI8", order_id="O1008", product_id="P006",
                    qty=1, unit_price_minor=250000, item_status="delivered"),
        "OI9": dict(order_item_id="OI9", order_id="O1009", product_id="P004",
                    qty=1, unit_price_minor=4500, item_status="delivered"),
        "OI10": dict(order_item_id="OI10", order_id="O1010", product_id="P012",
                     qty=2, unit_price_minor=65000, item_status="processing"),
    }

    shipments = {
        "SH1": dict(shipment_id="SH1", order_id="O1001", courier="J&T Express",
                    tracking_no="JT880012345MY", status="delivered",
                    shipped="2026-11-04 09:00", delivered="2026-11-08 15:22",
                    customs_status="cleared", last_scan="2026-11-08 15:22"),
        "SH2": dict(shipment_id="SH2", order_id="O1003", courier="Ninja Van",
                    tracking_no="NVSG99887766", status="in_transit",
                    shipped="2026-11-09 11:00", delivered=None,
                    customs_status="n/a", last_scan="2026-11-10 08:30"),
        "SH3": dict(shipment_id="SH3", order_id="O1004", courier="SiCepat",
                    tracking_no="SC00112233ID", status="delivered",
                    shipped="2026-10-15 10:00", delivered="2026-10-22 14:05",
                    customs_status="n/a", last_scan="2026-10-22 14:05"),
        "SH4": dict(shipment_id="SH4", order_id="O1005", courier="Kerry Express",
                    tracking_no="KEX55443322TH", status="delivered",
                    shipped="2026-11-02 08:30", delivered="2026-11-06 12:40",
                    customs_status="cleared", last_scan="2026-11-06 12:40"),
        # Stuck in customs, last scan 6 days ago -- during peak period
        "SH5": dict(shipment_id="SH5", order_id="O1006", courier="LBC Express",
                    tracking_no="LBC77665544PH", status="customs_hold",
                    shipped="2026-11-03 07:15", delivered=None,
                    customs_status="held", last_scan="2026-11-04 19:00"),
        "SH6": dict(shipment_id="SH6", order_id="O1007", courier="Giao Hang Nhanh",
                    tracking_no="GHN12309876VN", status="delivered",
                    shipped="2026-10-21 08:00", delivered="2026-10-26 17:10",
                    customs_status="cleared", last_scan="2026-10-26 17:10"),
        "SH7": dict(shipment_id="SH7", order_id="O1008", courier="J&T Express",
                    tracking_no="JT880098765ID", status="delivered",
                    shipped="2026-11-06 09:30", delivered="2026-11-10 11:00",
                    customs_status="cleared", last_scan="2026-11-10 11:00"),
        "SH8": dict(shipment_id="SH8", order_id="O1009", courier="Ninja Van",
                    tracking_no="NVSG11223344", status="delivered",
                    shipped="2026-11-04 10:00", delivered="2026-11-07 16:00",
                    customs_status="n/a", last_scan="2026-11-07 16:00"),
    }

    payments = {
        "PAY1": dict(payment_id="PAY1", order_id="O1001", method="cod",
                     status="collected", amount_minor=18600, currency="MYR",
                     paid="2026-11-08 15:22", instrument_last4=None),
        "PAY2": dict(payment_id="PAY2", order_id="O1002", method="card",
                     status="authorized", amount_minor=12900, currency="SGD",
                     paid="2026-11-10 21:05", instrument_last4="4242"),
        "PAY3": dict(payment_id="PAY3", order_id="O1003", method="card",
                     status="captured", amount_minor=16250, currency="SGD",
                     paid="2026-11-08 09:14", instrument_last4="4242"),
        "PAY4": dict(payment_id="PAY4", order_id="O1004", method="cod",
                     status="collected", amount_minor=185000, currency="IDR",
                     paid="2026-10-22 14:05", instrument_last4=None),
        "PAY5": dict(payment_id="PAY5", order_id="O1005", method="ewallet",
                     status="captured", amount_minor=895000, currency="THB",
                     paid="2026-11-01 16:45", instrument_last4="8821"),
        "PAY6": dict(payment_id="PAY6", order_id="O1006", method="cod",
                     status="pending", amount_minor=101900, currency="PHP",
                     paid=None, instrument_last4=None),
        "PAY7": dict(payment_id="PAY7", order_id="O1007", method="bank_transfer",
                     status="captured", amount_minor=480000, currency="VND",
                     paid="2026-10-20 19:30", instrument_last4="3310"),
        "PAY8": dict(payment_id="PAY8", order_id="O1008", method="ewallet",
                     status="captured", amount_minor=270000, currency="IDR",
                     paid="2026-11-05 10:10", instrument_last4="7702"),
        "PAY9": dict(payment_id="PAY9", order_id="O1009", method="card",
                     status="captured", amount_minor=4700, currency="SGD",
                     paid="2026-11-03 13:00", instrument_last4="4242"),
        "PAY10": dict(payment_id="PAY10", order_id="O1010", method="cod",
                      status="pending", amount_minor=142000, currency="IDR",
                      paid=None, instrument_last4=None),
    }

    # The TikTok-specific table: what the seller actually said on the live.
    livestream_claims = {
        "LC1": dict(claim_id="LC1", livestream_id="LS7001", seller_id="S001",
                    product_id="P001", timestamp="2026-11-02 14:12",
                    claim_text="Free gift sheet mask bundle with every 2 bottles",
                    verified_fulfilled=False),
        "LC2": dict(claim_id="LC2", livestream_id="LS7002", seller_id="S004",
                    product_id="P008", timestamp="2026-11-01 16:30",
                    claim_text="Genuine cowhide leather, 1 year warranty",
                    verified_fulfilled=True),
        "LC3": dict(claim_id="LC3", livestream_id="LS7003", seller_id="S003",
                    product_id="P012", timestamp="2026-10-20 19:15",
                    claim_text="100% cotton combed 30s, pre-shrunk",
                    verified_fulfilled=False),
        "LC4": dict(claim_id="LC4", livestream_id="LS7004", seller_id="S003",
                    product_id="P012", timestamp="2026-11-10 23:40",
                    claim_text="Ships within 24 hours guaranteed",
                    verified_fulfilled=False),
    }

    return {
        "users": users,
        "sellers": sellers,
        "products": products,
        "orders": orders,
        "order_items": order_items,
        "shipments": shipments,
        "payments": payments,
        "livestream_claims": livestream_claims,
        # mutable output tables, start empty
        "returns": {},
        "refunds": {},
        "store_credit_grants": {},
        "vouchers": {},
        "escalations": {},
        "seller_messages": {},
        # scratchpad for the note-taking context strategy. NOT part of the
        # world: no verifier check may reference it, or the strategy would
        # be able to score points by writing notes.
        "notes": {},
    }


# --------------------------------------------------------------------------
# Database wrapper
# --------------------------------------------------------------------------

@dataclass
class Action:
    tool: str
    args: dict[str, Any]
    ok: bool
    error: str | None = None
    # The call ran and the answer was no: verify_identity with the wrong
    # digits returns ok with verified=False. A check that needs the call to
    # have WORKED reads this (verifier.ActionSpec.must_succeed).
    denied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args, "ok": self.ok, "error": self.error,
                "denied": self.denied}


@dataclass
class Database:
    tables: dict[str, dict[str, dict[str, Any]]] = field(default_factory=seed)
    action_log: list[Action] = field(default_factory=list)
    verified: set = field(default_factory=set)
    _counter: int = 0

    @classmethod
    def fresh(cls, patch: dict[str, dict[str, dict[str, Any]]] | None = None) -> "Database":
        """New DB from seed, optionally with per-task overrides applied."""
        db = cls(tables=copy.deepcopy(seed()))
        if patch:
            for table, rows in patch.items():
                db.tables.setdefault(table, {})
                for row_id, fields in rows.items():
                    # deepcopy is load-bearing: without it a new row is ALIASED
                    # into the database, the first episode's mutations leak back
                    # into the task definition, and every later run starts from
                    # a corrupted world. Invisible at k=1, silently wrong for
                    # every pass^k.
                    if row_id in db.tables[table]:
                        db.tables[table][row_id].update(copy.deepcopy(fields))
                    else:
                        db.tables[table][row_id] = copy.deepcopy(fields)
        return db

    def next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter:04d}"

    def log(self, tool: str, args: dict[str, Any], ok: bool, error: str | None = None,
            denied: bool = False) -> None:
        self.action_log.append(Action(tool=tool, args=args, ok=ok, error=error,
                                      denied=denied))

    # convenience accessors ------------------------------------------------
    def t(self, name: str) -> dict[str, dict[str, Any]]:
        return self.tables[name]

    def row(self, table: str, row_id: str) -> dict[str, Any] | None:
        return self.tables[table].get(row_id)

    def where(self, table: str, **kw: Any) -> list[dict[str, Any]]:
        out = []
        for r in self.tables[table].values():
            if all(r.get(k) == v for k, v in kw.items()):
                out.append(r)
        return out

    def write_actions(self) -> list[Action]:
        from .tools import WRITE_TOOLS
        return [a for a in self.action_log if a.tool in WRITE_TOOLS]

    # -- serialisation, so an episode can be interrupted and resumed ------
    def to_dict(self) -> dict[str, Any]:
        return {
            "tables": self.tables,
            "action_log": [a.to_dict() for a in self.action_log],
            "verified": sorted(self.verified),
            "counter": self._counter,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Database":
        db = cls(tables=copy.deepcopy(d["tables"]))
        db.action_log = [Action(**a) for a in d["action_log"]]
        db.verified = set(d["verified"])
        db._counter = d["counter"]
        return db
