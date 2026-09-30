"""
PasarBench tool layer.

Every tool is (db, **kwargs) -> dict. Tools never raise for business-rule
violations; they return {"ok": False, "error": ...} so the agent gets a
recoverable, structured error it can actually reason about. This is a
deliberate harness design choice -- see README.

Read tools do not appear in the verifier's action checks. Write tools do.
"""

from __future__ import annotations

import functools
import inspect
import re
from pathlib import Path
from typing import Any, Callable

from .db import (NOW, Database, fmt_money, is_peak, to_sgd, ts)

TOOLS: dict[str, dict[str, Any]] = {}

WRITE_TOOLS = {
    "cancel_order",
    "initiate_return",
    "issue_refund",
    "issue_store_credit",
    "issue_goodwill_voucher",
    "modify_shipping_address",
    "send_message_to_seller",
    "escalate_to_human",
}

POLICY_PATH = Path(__file__).parent / "policy.md"


def tool(name: str, description: str, params: dict[str, Any], required: list[str]):
    def deco(fn: Callable):
        TOOLS[name] = {
            "fn": fn,
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": required,
                    },
                },
            },
        }
        return fn
    return deco


def _err(db: Database, name: str, args: dict, msg: str) -> dict:
    db.log(name, args, ok=False, error=msg)
    return {"ok": False, "error": msg}


def _ok(db: Database, name: str, args: dict, **payload) -> dict:
    db.log(name, args, ok=True)
    return {"ok": True, **payload}


def _require_verified(db: Database, user_id: str, name: str, args: dict) -> dict | None:
    if user_id not in db.verified:
        return _err(db, name, args,
                    "identity not verified for this user; call verify_identity first (P1.1)")
    return None


def _order_user(db: Database, order_id: str) -> str | None:
    o = db.row("orders", order_id)
    return o["user_id"] if o else None


# ==========================================================================
# READ TOOLS
# ==========================================================================

@tool("get_user_profile", "Look up a customer's account profile.",
      {"user_id": {"type": "string"}}, ["user_id"])
def get_user_profile(db: Database, user_id: str) -> dict:
    u = db.row("users", user_id)
    if not u:
        return _err(db, "get_user_profile", locals_of(user_id=user_id), "no such user")
    safe = {k: v for k, v in u.items() if k != "verified_phone_last4"}
    return _ok(db, "get_user_profile", {"user_id": user_id}, user=safe)


@tool("verify_identity",
      "Verify the customer by the last 4 digits of the phone number on the account. "
      "Required before any write action.",
      {"user_id": {"type": "string"},
       "phone_last4": {"type": "string", "description": "4 digits the customer stated"}},
      ["user_id", "phone_last4"])
def verify_identity(db: Database, user_id: str, phone_last4: str) -> dict:
    args = {"user_id": user_id, "phone_last4": phone_last4}
    u = db.row("users", user_id)
    if not u:
        return _err(db, "verify_identity", args, "no such user")
    if str(phone_last4).strip() == u["verified_phone_last4"]:
        db.verified.add(user_id)
        return _ok(db, "verify_identity", args, verified=True)
    out = _ok(db, "verify_identity", args, verified=False,
              note="digits do not match; do not proceed with write actions")
    # ok: the call worked. denied: the answer was no. A check that requires
    # verification needs both (verifier.ActionSpec.must_succeed).
    db.action_log[-1].denied = True
    return out


@tool("get_order", "Get one order by id, including status, payment method and totals.",
      {"order_id": {"type": "string"}}, ["order_id"])
def get_order(db: Database, order_id: str) -> dict:
    o = db.row("orders", order_id)
    if not o:
        return _err(db, "get_order", {"order_id": order_id}, "no such order")
    out = dict(o)
    out["total_display"] = fmt_money(o["total_minor"], o["currency"])
    # Prior refunds and returns against this order MUST be visible. Without
    # them the duplicate-refund policy (P10) is unenforceable by any agent that
    # can only observe -- it would have to already know the answer. The
    # reference solutions did not catch this because a reference solution is an
    # ORACLE: it proves a task is solvable by something that knows the answer,
    # not by something that has to find out.
    out["existing_refunds"] = [
        {"refund_id": r["refund_id"], "amount_minor": r["amount_minor"],
         "currency": r["currency"], "status": r["status"], "created": r["created"],
         "display": fmt_money(r["amount_minor"], r["currency"])}
        for r in db.where("refunds", order_id=order_id)]
    out["existing_returns"] = [
        {"return_id": r["return_id"], "order_item_id": r["order_item_id"],
         "status": r["status"]}
        for r in db.where("returns", order_id=order_id)]
    return _ok(db, "get_order", {"order_id": order_id}, order=out)


@tool("list_user_orders", "List a customer's orders, optionally filtered by status.",
      {"user_id": {"type": "string"},
       "status": {"type": "string",
                  "enum": ["pending", "processing", "shipped", "delivered", "cancelled"]}},
      ["user_id"])
def list_user_orders(db: Database, user_id: str, status: str | None = None) -> dict:
    rows = db.where("orders", user_id=user_id)
    if status:
        rows = [r for r in rows if r["status"] == status]
    brief = [{"order_id": r["order_id"], "status": r["status"],
              "created": r["created"], "source": r["source"],
              "total_display": fmt_money(r["total_minor"], r["currency"])} for r in rows]
    return _ok(db, "list_user_orders", {"user_id": user_id, "status": status}, orders=brief)


@tool("get_order_items", "List the line items on an order.",
      {"order_id": {"type": "string"}}, ["order_id"])
def get_order_items(db: Database, order_id: str) -> dict:
    rows = db.where("order_items", order_id=order_id)
    return _ok(db, "get_order_items", {"order_id": order_id}, items=rows)


@tool("get_shipment", "Get tracking and delivery status for an order.",
      {"order_id": {"type": "string"}}, ["order_id"])
def get_shipment(db: Database, order_id: str) -> dict:
    rows = db.where("shipments", order_id=order_id)
    if not rows:
        return _ok(db, "get_shipment", {"order_id": order_id}, shipment=None,
                   note="no shipment record; order not dispatched")
    return _ok(db, "get_shipment", {"order_id": order_id}, shipment=rows[0])


@tool("get_product", "Get product details including category and hazmat/perishable flags.",
      {"product_id": {"type": "string"}}, ["product_id"])
def get_product(db: Database, product_id: str) -> dict:
    p = db.row("products", product_id)
    if not p:
        return _err(db, "get_product", {"product_id": product_id}, "no such product")
    return _ok(db, "get_product", {"product_id": product_id}, product=p)


@tool("get_payment", "Get the payment record for an order (method, status, amount).",
      {"order_id": {"type": "string"}}, ["order_id"])
def get_payment(db: Database, order_id: str) -> dict:
    rows = db.where("payments", order_id=order_id)
    if not rows:
        return _err(db, "get_payment", {"order_id": order_id}, "no payment record")
    return _ok(db, "get_payment", {"order_id": order_id}, payment=rows[0])


@tool("search_policy",
      "Search the customer service policy. Returns matching sections verbatim. "
      "Use this instead of guessing a rule.",
      {"query": {"type": "string", "description": "keywords, e.g. 'cod refund' or 'return window'"}},
      ["query"])
def search_policy(db: Database, query: str) -> dict:
    sections = _policy_sections()
    terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
    scored = []
    for heading, body in sections:
        hay = (heading + " " + body).lower()
        score = sum(hay.count(t) for t in terms)
        if heading.lower().startswith("p") and any(t in heading.lower() for t in terms):
            score += 5
        if score:
            scored.append((score, heading, body))
    scored.sort(reverse=True, key=lambda x: x[0])
    hits = [{"section": h, "text": b} for _, h, b in scored[:3]]
    return _ok(db, "search_policy", {"query": query}, results=hits,
               note="no match" if not hits else None)


_SECTION_CACHE: list[tuple[str, str]] | None = None


def _policy_sections() -> list[tuple[str, str]]:
    global _SECTION_CACHE
    if _SECTION_CACHE is None:
        text = POLICY_PATH.read_text()
        parts = re.split(r"\n## ", text)
        out = []
        for p in parts[1:]:
            head, _, body = p.partition("\n")
            out.append((head.strip(), body.strip()))
        _SECTION_CACHE = out
    return _SECTION_CACHE


@tool("get_livestream_claims",
      "Retrieve what the seller stated during a livestream. Use for any dispute "
      "about a livestream purchase.",
      {"livestream_id": {"type": "string"},
       "product_id": {"type": "string"}}, ["livestream_id"])
def get_livestream_claims(db: Database, livestream_id: str,
                          product_id: str | None = None) -> dict:
    rows = db.where("livestream_claims", livestream_id=livestream_id)
    if product_id:
        rows = [r for r in rows if r["product_id"] == product_id]
    return _ok(db, "get_livestream_claims",
               {"livestream_id": livestream_id, "product_id": product_id}, claims=rows)


@tool("check_return_eligibility",
      "Check whether an order item is returnable under section P3. MUST be called "
      "before initiate_return. Note: this evaluates P3 only and does not account "
      "for exceptions elsewhere in the policy.",
      {"order_id": {"type": "string"}, "order_item_id": {"type": "string"}},
      ["order_id", "order_item_id"])
def check_return_eligibility(db: Database, order_id: str, order_item_id: str) -> dict:
    args = {"order_id": order_id, "order_item_id": order_item_id}
    o = db.row("orders", order_id)
    oi = db.row("order_items", order_item_id)
    if not o or not oi or oi["order_id"] != order_id:
        return _err(db, "check_return_eligibility", args, "order/item mismatch")

    ship = db.where("shipments", order_id=order_id)
    delivered = ship[0]["delivered"] if ship and ship[0].get("delivered") else None
    prod = db.row("products", oi["product_id"])

    days = None
    within = False
    if delivered:
        days = (NOW - ts(delivered)).days
        within = days <= 14

    item_value = oi["unit_price_minor"] * oi["qty"]
    value_sgd = to_sgd(item_value, o["currency"])
    photo_required = value_sgd >= 200
    non_returnable = prod["perishable"] or prod["hazmat"] or prod["category"] == "grocery"

    existing = [r for r in db.t("returns").values() if r["order_item_id"] == order_item_id]

    eligible = bool(delivered) and within and not non_returnable and not existing

    return _ok(db, "check_return_eligibility", args,
               eligible=eligible,
               delivered_at=delivered,
               days_since_delivery=days,
               within_14_day_window=within,
               category=prod["category"],
               perishable=prod["perishable"],
               hazmat=prod["hazmat"],
               non_returnable=non_returnable,
               item_value_display=fmt_money(item_value, o["currency"]),
               item_value_sgd_equiv=round(value_sgd, 2),
               photo_required_for_defect_claim=photo_required,
               existing_return_count=len(existing),
               order_status=o["status"])


@tool("calculate_refund_amount",
      "Compute the correct refund amount in minor units for an order item.",
      {"order_id": {"type": "string"}, "order_item_id": {"type": "string"},
       "include_shipping": {"type": "boolean",
                            "description": "true only when the fault is seller/platform side (P4.6)"}},
      ["order_id", "order_item_id", "include_shipping"])
def calculate_refund_amount(db: Database, order_id: str, order_item_id: str,
                            include_shipping: bool) -> dict:
    args = {"order_id": order_id, "order_item_id": order_item_id,
            "include_shipping": include_shipping}
    o = db.row("orders", order_id)
    oi = db.row("order_items", order_item_id)
    if not o or not oi or oi["order_id"] != order_id:
        return _err(db, "calculate_refund_amount", args, "order/item mismatch")
    amount = oi["unit_price_minor"] * oi["qty"]
    if include_shipping:
        amount += o["shipping_minor"]
    amount = min(amount, o["total_minor"])
    return _ok(db, "calculate_refund_amount", args,
               amount_minor=amount, currency=o["currency"],
               display=fmt_money(amount, o["currency"]),
               order_total_minor=o["total_minor"])


# ==========================================================================
# WRITE TOOLS
# ==========================================================================

@tool("cancel_order", "Cancel an order. Only valid while status is pending or processing.",
      {"order_id": {"type": "string"}, "reason": {"type": "string"}},
      ["order_id", "reason"])
def cancel_order(db: Database, order_id: str, reason: str) -> dict:
    args = {"order_id": order_id, "reason": reason}
    o = db.row("orders", order_id)
    if not o:
        return _err(db, "cancel_order", args, "no such order")
    if v := _require_verified(db, o["user_id"], "cancel_order", args):
        return v
    if o["status"] not in ("pending", "processing"):
        return _err(db, "cancel_order", args,
                    f"order status is '{o['status']}'; cancellation not permitted (P2.2)")
    o["status"] = "cancelled"
    for it in db.where("order_items", order_id=order_id):
        it["item_status"] = "cancelled"
    return _ok(db, "cancel_order", args, order_id=order_id, new_status="cancelled")


@tool("initiate_return",
      "Open a return request for one order item. check_return_eligibility must be "
      "called first.",
      {"order_id": {"type": "string"}, "order_item_id": {"type": "string"},
       "reason": {"type": "string"},
       "photo_evidence_provided": {"type": "boolean"}},
      ["order_id", "order_item_id", "reason", "photo_evidence_provided"])
def initiate_return(db: Database, order_id: str, order_item_id: str, reason: str,
                    photo_evidence_provided: bool) -> dict:
    args = {"order_id": order_id, "order_item_id": order_item_id, "reason": reason,
            "photo_evidence_provided": photo_evidence_provided}
    o = db.row("orders", order_id)
    oi = db.row("order_items", order_item_id)
    if not o or not oi or oi["order_id"] != order_id:
        return _err(db, "initiate_return", args, "order/item mismatch")
    if v := _require_verified(db, o["user_id"], "initiate_return", args):
        return v
    if o["status"] != "delivered":
        return _err(db, "initiate_return", args,
                    f"order status is '{o['status']}'; cannot return an undelivered order (P2.3)")
    checked = any(a.tool == "check_return_eligibility"
                  and a.args.get("order_item_id") == order_item_id
                  for a in db.action_log)
    if not checked:
        return _err(db, "initiate_return", args,
                    "check_return_eligibility was not called for this item (P3.2)")

    rid = db.next_id("RET")
    db.t("returns")[rid] = dict(return_id=rid, order_id=order_id,
                                order_item_id=order_item_id, reason=reason,
                                photo_evidence=photo_evidence_provided,
                                status="approved", created=NOW.strftime("%Y-%m-%d %H:%M"))
    oi["item_status"] = "return_requested"
    return _ok(db, "initiate_return", args, return_id=rid, status="approved")


@tool("issue_refund",
      "Refund money to the original payment method. Never use for COD orders (P4.2).",
      {"order_id": {"type": "string"},
       "amount_minor": {"type": "integer", "description": "amount in MINOR units"},
       "method": {"type": "string",
                  "enum": ["card", "ewallet", "bank_transfer"]},
       "reason": {"type": "string"}},
      ["order_id", "amount_minor", "method", "reason"])
def issue_refund(db: Database, order_id: str, amount_minor: int, method: str,
                 reason: str) -> dict:
    args = {"order_id": order_id, "amount_minor": amount_minor,
            "method": method, "reason": reason}
    o = db.row("orders", order_id)
    if not o:
        return _err(db, "issue_refund", args, "no such order")
    if v := _require_verified(db, o["user_id"], "issue_refund", args):
        return v
    if method == "cod":
        return _err(db, "issue_refund", args,
                    "cannot refund to COD; use issue_store_credit or bank_transfer (P4.2)")
    pay = db.where("payments", order_id=order_id)
    pay = pay[0] if pay else None
    if pay and pay["method"] == "cod":
        return _err(db, "issue_refund", args,
                    "this is a COD order with no refundable instrument (P4.2)")
    if pay and pay["status"] == "pending":
        return _err(db, "issue_refund", args, "payment not collected; nothing to refund (P4.3)")
    if amount_minor > o["total_minor"]:
        return _err(db, "issue_refund", args,
                    f"amount exceeds order total {o['total_minor']} (P4.3)")
    if amount_minor <= 0:
        return _err(db, "issue_refund", args, "amount must be positive")

    rid = db.next_id("REF")
    db.t("refunds")[rid] = dict(refund_id=rid, order_id=order_id,
                                amount_minor=amount_minor, currency=o["currency"],
                                method=method, reason=reason, status="issued",
                                created=NOW.strftime("%Y-%m-%d %H:%M"))
    return _ok(db, "issue_refund", args, refund_id=rid,
               display=fmt_money(amount_minor, o["currency"]))


@tool("issue_store_credit",
      "Grant store credit. This is the correct remedy for COD orders (P4.2).",
      {"user_id": {"type": "string"},
       "amount_minor": {"type": "integer"},
       "currency": {"type": "string", "enum": ["SGD", "MYR", "THB", "PHP", "IDR", "VND"]},
       "reason": {"type": "string"},
       "order_id": {"type": "string", "description": "order this relates to"}},
      ["user_id", "amount_minor", "currency", "reason"])
def issue_store_credit(db: Database, user_id: str, amount_minor: int, currency: str,
                       reason: str, order_id: str | None = None) -> dict:
    args = {"user_id": user_id, "amount_minor": amount_minor, "currency": currency,
            "reason": reason, "order_id": order_id}
    u = db.row("users", user_id)
    if not u:
        return _err(db, "issue_store_credit", args, "no such user")
    if v := _require_verified(db, user_id, "issue_store_credit", args):
        return v
    if amount_minor <= 0:
        return _err(db, "issue_store_credit", args, "amount must be positive")
    if order_id:
        o = db.row("orders", order_id)
        if o and amount_minor > o["total_minor"]:
            return _err(db, "issue_store_credit", args,
                        f"amount exceeds order total {o['total_minor']} (P4.3)")
    gid = db.next_id("SC")
    db.t("store_credit_grants")[gid] = dict(grant_id=gid, user_id=user_id,
                                            amount_minor=amount_minor, currency=currency,
                                            reason=reason, order_id=order_id,
                                            created=NOW.strftime("%Y-%m-%d %H:%M"))
    u["store_credit_minor"] += amount_minor
    return _ok(db, "issue_store_credit", args, grant_id=gid,
               display=fmt_money(amount_minor, currency),
               new_balance=u["store_credit_minor"])


@tool("issue_goodwill_voucher",
      "Issue a goodwill voucher for out-of-window requests (P6.2). Max 20% of item "
      "value, capped at SGD 15 equivalent.",
      {"user_id": {"type": "string"}, "amount_minor": {"type": "integer"},
       "currency": {"type": "string", "enum": ["SGD", "MYR", "THB", "PHP", "IDR", "VND"]},
       "reason": {"type": "string"}},
      ["user_id", "amount_minor", "currency", "reason"])
def issue_goodwill_voucher(db: Database, user_id: str, amount_minor: int,
                           currency: str, reason: str) -> dict:
    args = {"user_id": user_id, "amount_minor": amount_minor,
            "currency": currency, "reason": reason}
    if not db.row("users", user_id):
        return _err(db, "issue_goodwill_voucher", args, "no such user")
    if v := _require_verified(db, user_id, "issue_goodwill_voucher", args):
        return v
    if to_sgd(amount_minor, currency) > 15:
        return _err(db, "issue_goodwill_voucher", args,
                    "voucher exceeds SGD 15 equivalent cap (P6.2)")
    vid = db.next_id("VCH")
    db.t("vouchers")[vid] = dict(voucher_id=vid, user_id=user_id,
                                 amount_minor=amount_minor, currency=currency,
                                 reason=reason, created=NOW.strftime("%Y-%m-%d %H:%M"))
    return _ok(db, "issue_goodwill_voucher", args, voucher_id=vid,
               display=fmt_money(amount_minor, currency))


@tool("modify_shipping_address", "Change the delivery address. Only before dispatch (P9.1).",
      {"order_id": {"type": "string"}, "new_address": {"type": "string"}},
      ["order_id", "new_address"])
def modify_shipping_address(db: Database, order_id: str, new_address: str) -> dict:
    args = {"order_id": order_id, "new_address": new_address}
    o = db.row("orders", order_id)
    if not o:
        return _err(db, "modify_shipping_address", args, "no such order")
    if v := _require_verified(db, o["user_id"], "modify_shipping_address", args):
        return v
    if o["status"] not in ("pending", "processing"):
        return _err(db, "modify_shipping_address", args,
                    f"order already '{o['status']}'; address is locked (P9.2)")
    o["shipping_address"] = new_address
    return _ok(db, "modify_shipping_address", args, order_id=order_id,
               shipping_address=new_address)


@tool("send_message_to_seller", "Send a note to the seller about an order.",
      {"order_id": {"type": "string"}, "message": {"type": "string"}},
      ["order_id", "message"])
def send_message_to_seller(db: Database, order_id: str, message: str) -> dict:
    args = {"order_id": order_id, "message": message}
    o = db.row("orders", order_id)
    if not o:
        return _err(db, "send_message_to_seller", args, "no such order")
    mid = db.next_id("MSG")
    db.t("seller_messages")[mid] = dict(message_id=mid, order_id=order_id,
                                        seller_id=o["seller_id"], message=message,
                                        created=NOW.strftime("%Y-%m-%d %H:%M"))
    return _ok(db, "send_message_to_seller", args, message_id=mid)


@tool("escalate_to_human", "Hand the case to a human agent. See P10 for triggers.",
      {"order_id": {"type": "string"},
       "category": {"type": "string",
                    "enum": ["identity_unverified", "out_of_window_dispute",
                             "high_value_refund", "counterfeit_or_safety",
                             "duplicate_refund", "customer_request", "customs_hold",
                             "other"]},
       "reason": {"type": "string"}},
      ["order_id", "category", "reason"])
def escalate_to_human(db: Database, order_id: str, category: str, reason: str) -> dict:
    args = {"order_id": order_id, "category": category, "reason": reason}
    eid = db.next_id("ESC")
    db.t("escalations")[eid] = dict(escalation_id=eid, order_id=order_id,
                                    category=category, reason=reason, status="open",
                                    created=NOW.strftime("%Y-%m-%d %H:%M"))
    return _ok(db, "escalate_to_human", args, escalation_id=eid)


# --------------------------------------------------------------------------

def locals_of(**kw):
    return kw


@functools.lru_cache(maxsize=None)
def _parameters(fn: Callable) -> tuple[frozenset[str], bool]:
    """A tool function's parameter names, and whether it takes **kwargs."""
    ps = inspect.signature(fn).parameters.values()
    return frozenset(p.name for p in ps), any(p.kind is p.VAR_KEYWORD for p in ps)


def call(db: Database, name: str, args: dict[str, Any]) -> dict:
    """Dispatch with schema-ish validation. Returns a structured error, never raises."""
    if name not in TOOLS:
        return {"ok": False, "error": f"unknown tool '{name}'"}
    spec = TOOLS[name]
    required = spec["schema"]["function"]["parameters"]["required"]
    missing = [r for r in required if r not in args]
    if missing:
        return {"ok": False, "error": f"missing required argument(s): {missing}"}
    # An argument the tool does not take is reported here, not by Python's
    # TypeError. The two said the same until 3.13, which appends "Did you mean
    # 'user_id'?": an agent run on 3.13 was told more than one run on 3.12, and
    # a replay on 3.13 could not reproduce what a 3.12 run recorded. This is
    # the wording every recorded run shows, on every interpreter.
    fn = spec["fn"]
    names, open_ended = _parameters(fn)
    extra = None if open_ended else next((k for k in args if k not in names), None)
    if extra is not None:
        return {"ok": False, "error": f"bad arguments: {fn.__qualname__}() got an "
                                      f"unexpected keyword argument '{extra}'"}
    try:
        return spec["fn"](db, **args)
    except TypeError as e:
        return {"ok": False, "error": f"bad arguments: {e}"}
    except Exception as e:  # noqa: BLE001 - harness must never crash on a tool
        return {"ok": False, "error": f"tool raised {type(e).__name__}: {e}"}


def schemas(names: list[str] | None = None) -> list[dict]:
    names = names or list(TOOLS)
    return [TOOLS[n]["schema"] for n in names]


# ==========================================================================
# SCRATCHPAD -- exposed ONLY to the note-taking context strategy
# ==========================================================================
# Not in WRITE_TOOLS: notes change no world state and no verifier check may
# reference the notes table. If a check could, the strategy would be able to
# score points by writing notes rather than by solving the task.

@tool("write_note",
      "Record a durable note for yourself. Earlier raw conversation may be "
      "dropped from your context, but notes are always kept. Write down order "
      "ids, verification status, eligibility verdicts and what you have already "
      "done.",
      {"content": {"type": "string", "description": "one fact or decision, briefly"}},
      ["content"])
def write_note(db: Database, content: str) -> dict:
    nid = db.next_id("NOTE")
    db.t("notes")[nid] = dict(note_id=nid, content=content,
                              created=NOW.strftime("%Y-%m-%d %H:%M"))
    return _ok(db, "write_note", {"content": content}, note_id=nid,
               total_notes=len(db.t("notes")))


@tool("read_notes", "Read back every note you have written this conversation.",
      {}, [])
def read_notes(db: Database) -> dict:
    return _ok(db, "read_notes", {},
               notes=[n["content"] for n in db.t("notes").values()])


NOTE_TOOLS = ["write_note", "read_notes"]
DEFAULT_TOOLS = [n for n in TOOLS if n not in NOTE_TOOLS]
