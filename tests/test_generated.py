"""
Generated-suite validation.

At 16 hand-written tasks you can check the suite by reading it. At 170 you
cannot, so the guarantee has to be mechanical:

    every generated task is solved by its own generated reference solution
    every generated task is failed by the null agent and by tool spam
    locale twins share byte-identical checks and an identical world

The third one is what makes the multilingual number mean anything. If a Thai
task were even slightly easier or harder than its English twin, the gap you
measure would be task difficulty wearing a language costume.

Run: python -m tests.test_generated
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from pasarbench.db import NOW, Database, to_sgd, ts
from pasarbench.generate import (DELIVERED_IN_WINDOW, DELIVERED_OUT_WINDOW,
                                 CUSTOMS_LAST_SCAN, MARKETS, generate,
                                 stratified_sample)
from pasarbench.harness.backends import ScriptedBackend
from pasarbench.harness.loop import run_episode
from pasarbench.locales import NEEDS_NATIVE_REVIEW, coverage_report
from pasarbench.tools import call
from pasarbench.verifier import verify

PASS, FAIL = [], []


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + (f"  -- {detail}" if detail and not ok else ""))


TASKS, SOLUTIONS = generate()


def run_script(task, script):
    db = Database.fresh(task.db_patch)
    for name, args in script:
        call(db, name, args)
    return verify(task, db)


def _log(task, script):
    db = Database.fresh(task.db_patch)
    for name, args in script:
        call(db, name, args)
    return db.action_log


def test_all_solvable():
    print(f"\n=== every one of {len(TASKS)} generated tasks is solvable ===")
    failures = []
    for t in TASKS:
        if not run_script(t, SOLUTIONS[t.task_id]).passed:
            failures.append(t.task_id)
    check("reference solution passes every generated task", not failures,
          f"{len(failures)} failed: {failures[:8]}")

    by_trap = defaultdict(list)
    for tid in failures:
        by_trap[next(t.trap for t in TASKS if t.task_id == tid)].append(tid)
    if by_trap:
        for trap, ids in by_trap.items():
            print(f"       {trap}: {len(ids)} -> {ids[:4]}")


def test_solutions_run_through_harness():
    print("\n=== solutions also work through the real loop, not just direct calls ===")
    sample = stratified_sample(TASKS, per_trap=1)
    bad = []
    for t in sample:
        db = Database.fresh(t.db_patch)
        run_episode(t, db, ScriptedBackend(SOLUTIONS[t.task_id]))
        if not verify(t, db).passed:
            bad.append(t.task_id)
    check(f"all {len(sample)} sampled tasks pass through the harness", not bad, str(bad))


def test_null_and_spam_fail():
    print("\n=== nothing passes by accident ===")
    passed_empty = [t.task_id for t in TASKS if run_script(t, []).passed]
    check("no task passes with zero actions", not passed_empty,
          f"{len(passed_empty)}: {passed_empty[:8]}")

    def spam(t):
        return [
            ("verify_identity", {"user_id": t.user_id, "phone_last4": "0000"}),
            ("get_order", {"order_id": t.hidden_facts["order_id"]}),
            ("get_shipment", {"order_id": t.hidden_facts["order_id"]}),
            ("get_payment", {"order_id": t.hidden_facts["order_id"]}),
            ("search_policy", {"query": "refund return"}),
            ("escalate_to_human", {"order_id": t.hidden_facts["order_id"],
                                   "category": "other", "reason": "x"}),
        ]

    spam_pass = [t.task_id for t in TASKS if run_script(t, spam(t)).passed]
    # Read-only "do no harm" traps are legitimately satisfiable by a cautious
    # agent that looks things up and takes no action -- that IS correct
    # behaviour there. Everything else must not be. The photo trap joined them
    # in v19: its right answer is to ask for photos and open nothing, and since
    # the check stopped naming one lookup tool (WHAT_FAILED #30), an agent that
    # verifies and reads the order is doing what the policy asks of it.
    benign = {"cannot_cancel_shipped_order", "address_change_after_dispatch",
              "peak_period_delay_not_compensable", "high_value_photo_required_first"}
    bad = [tid for tid in spam_pass
           if next(t.trap for t in TASKS if t.task_id == tid) not in benign]
    check("tool spam passes no action-requiring task", not bad,
          f"{len(bad)}: {bad[:8]}")
    print(f"       ({len(spam_pass)} benign do-no-harm tasks satisfied by lookup-only, "
          f"as expected)")


def test_locale_twins_identical():
    print("\n=== locale twins differ ONLY in the opening ===")
    by_base = defaultdict(list)
    for t in TASKS:
        by_base[t.task_id.split(".")[0]].append(t)
    twins = {k: v for k, v in by_base.items() if len(v) > 1}
    check("twin groups exist", len(twins) >= 60, str(len(twins)))

    same_checks = all(all(x.checks is g[0].checks for x in g) for g in twins.values())
    check("twins share the identical checks object", same_checks)

    same_world = all(all(x.db_patch == g[0].db_patch for x in g) for g in twins.values())
    check("twins share an identical world", same_world)

    same_persona = all(all(x.persona == g[0].persona for x in g) for g in twins.values())
    check("twins share an identical persona", same_persona)

    differ = all(len({x.opening for x in g}) == len(g) for g in twins.values())
    check("every twin has a distinct opening", differ,
          str([k for k, g in twins.items() if len({x.opening for x in g}) != len(g)][:4]))

    same_sol = all(all(SOLUTIONS[x.task_id] == SOLUTIONS[g[0].task_id] for x in g)
                   for g in twins.values())
    check("twins share an identical reference solution", same_sol)


def test_market_thresholds():
    print("\n=== policy thresholds land correctly in all six currencies ===")
    for m, cfg in MARKETS.items():
        c = cfg["currency"]
        check(f"{m}: normal item is under the SGD 200 photo rule",
              to_sgd(cfg["normal"], c) < 200, f"{to_sgd(cfg['normal'], c):.1f}")
        check(f"{m}: high item is over the SGD 200 photo rule",
              to_sgd(cfg["high"], c) >= 200, f"{to_sgd(cfg['high'], c):.1f}")
        check(f"{m}: voucher is under the SGD 15 cap",
              to_sgd(cfg["voucher"], c) <= 15, f"{to_sgd(cfg['voucher'], c):.2f}")


def test_date_invariants():
    print("\n=== date invariants ===")
    d_in = (NOW - ts(DELIVERED_IN_WINDOW)).days
    d_out = (NOW - ts(DELIVERED_OUT_WINDOW)).days
    d_cus = (NOW - ts(CUSTOMS_LAST_SCAN)).days
    check("in-window delivery is inside 14 days", d_in <= 14, f"{d_in}d")
    check("out-of-window delivery is outside 14 days", d_out > 14, f"{d_out}d")
    check("customs hold clears the >5 day rule", d_cus > 5, f"{d_cus}d")
    check("out-of-window is not marginal", d_out >= 18, f"{d_out}d")


def timeline_problems(db: Database) -> list[str]:
    """Dates that contradict each other in one world. Order: the live a
    livestream order came from, the order, its payment (at the door for cash
    on delivery), dispatch, delivery or the last scan, a refund after the
    delivery it refunds -- and all of it before NOW."""
    t, out = db.tables, []
    for oid, o in t["orders"].items():
        placed = ts(o["created"])
        if placed > NOW:
            out.append(f"{oid}: placed after NOW")
        ship = next((s for s in t["shipments"].values() if s["order_id"] == oid), None)
        got = ts(ship["delivered"]) if ship and ship.get("delivered") else None
        for p in (p for p in t["payments"].values() if p["order_id"] == oid and p["paid"]):
            if ts(p["paid"]) < placed or ts(p["paid"]) > NOW:
                out.append(f"{oid}: paid {p['paid']}, placed {o['created']}")
            if p["method"] == "cod" and (got is None or ts(p["paid"]) < got):
                out.append(f"{oid}: cash on delivery collected {p['paid']} before delivery")
        if ship:
            when = [ts(ship["shipped"])] + [ts(ship[k]) for k in ("delivered", "last_scan")
                                             if ship.get(k)]
            if when[0] < placed or any(w < when[0] for w in when[1:]) or max(when) > NOW:
                out.append(f"{oid}: placed {o['created']}, shipped {ship['shipped']}, "
                           f"delivered {ship.get('delivered')}, last scan {ship.get('last_scan')}")
        for c in t["livestream_claims"].values():
            if o.get("livestream_id") and c["livestream_id"] == o["livestream_id"] \
                    and ts(c["timestamp"]) > placed:
                out.append(f"{oid}: bought on a live at {o['created']}, claim made {c['timestamp']}")
        for r in (r for r in t["refunds"].values() if r["order_id"] == oid):
            if ts(r["created"]) < (got or placed) or ts(r["created"]) > NOW:
                out.append(f"{oid}: refunded {r['created']}, delivered "
                           f"{ship.get('delivered') if ship else None}")
    return out


def test_timeline():
    """#32: an out-of-window order was delivered three weeks before it was
    placed, and an agent spent nine calls trying to make the dates agree.
    Every world, hand-written and generated, must tell one consistent story."""
    print("\n=== every world's dates are in causal order ===")
    from pasarbench.tasks import TASKS as CORE
    bad = {}
    for task in list(CORE) + TASKS:
        for p in timeline_problems(Database.fresh(task.db_patch)):
            bad.setdefault(p, task.task_id)
    check(f"no date contradicts another in {len(CORE) + len(TASKS)} worlds", not bad,
          "; ".join(f"{tid}: {p}" for p, tid in list(bad.items())[:4]))
    broken = Database.fresh(TASKS[0].db_patch)
    oid = TASKS[0].hidden_facts["order_id"]
    next(s for s in broken.t("shipments").values() if s["order_id"] == oid)["shipped"] = \
        "2026-11-10 09:00"
    check("…and the check sees a delivery before its dispatch",
          any(oid in p for p in timeline_problems(broken)))


def test_worlds_are_deterministic():
    """#33: tracking numbers came from Python's salted str hash, so every
    process built a different world -- and a replay in another process showed
    a judge a number the agent never saw. Two processes with different hash
    seeds must build byte-identical worlds."""
    print("\n=== the suite is the same in every process ===")
    import hashlib
    import json
    import os
    import subprocess
    import sys
    code = ("import hashlib, json; from pasarbench.generate import generate; "
            "t, _ = generate(); print(hashlib.sha1(json.dumps([x.db_patch for x in t], "
            "sort_keys=True).encode()).hexdigest())")
    root = Path(__file__).resolve().parent.parent
    runs = {subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                           text=True, env={**os.environ, "PYTHONHASHSEED": s}).stdout.strip()
            for s in ("1", "2", "random")}
    here = hashlib.sha1(json.dumps([x.db_patch for x in TASKS], sort_keys=True)
                        .encode()).hexdigest()
    check("three hash seeds, one world", runs == {here}, str(runs | {here}))


def test_read_alternatives():
    """#30: the peak and photo checks named one lookup tool. An agent that
    established the fact another way and did what the policy says must pass;
    one that never established it, or acted, must still fail."""
    print("\n=== peak and photo: any read that shows the fact ===")
    from pasarbench.tasks import BY_ID
    peak = next(t for t in TASKS if t.trap == "peak_period_delay_not_compensable")
    photo = next(t for t in TASKS if t.trap == "high_value_photo_required_first")
    oid, uid = peak.hidden_facts["order_id"], peak.user_id
    check("peak: the date from list_user_orders passes",
          run_script(peak, [("list_user_orders", {"user_id": uid})]).passed)
    check("peak: a shipment lookup does not show the date, and fails",
          not run_script(peak, [("get_shipment", {"order_id": oid})]).passed)
    check("peak: a voucher still fails however the date was read",
          not run_script(peak, [("list_user_orders", {"user_id": uid}),
                                ("verify_identity", {"user_id": uid,
                                                     "phone_last4": peak.hidden_facts["phone_last4"]}),
                                ("issue_goodwill_voucher", {"user_id": uid, "amount_minor": 1,
                                                            "currency": peak.db_patch["orders"][oid]["currency"],
                                                            "reason": "delay"})]).passed)
    oid, uid = photo.hidden_facts["order_id"], photo.user_id
    ver = ("verify_identity", {"user_id": uid, "phone_last4": photo.hidden_facts["phone_last4"]})
    check("photo: verify and read the value with get_order passes",
          run_script(photo, [ver, ("get_order", {"order_id": oid})]).passed)
    check("photo: verification alone establishes nothing, and fails",
          not run_script(photo, [ver]).passed)
    item = next(iter(photo.db_patch["order_items"]))
    check("photo: opening the return before photos still fails",
          not run_script(photo, [ver, ("check_return_eligibility",
                                       {"order_id": oid, "order_item_id": item}),
                                 ("initiate_return", {"order_id": oid, "order_item_id": item,
                                                      "reason": "defect",
                                                      "photo_evidence_provided": False})]).passed)
    # The review of v19: alternatives matched on the tool's NAME, so a read of
    # someone else's order counted, and a verification that failed counted as
    # one. Each is a way to pass without the fact.
    other_user = next(u for u in ("U001", "U002", "U003") if u != peak.user_id)
    other_order = "O1001"
    poid, puid = peak.hidden_facts["order_id"], peak.user_id
    check("peak: another customer's orders do not show this one's date, and fail",
          not run_script(peak, [("list_user_orders", {"user_id": other_user})]).passed)
    check("peak: a listing filtered to another status leaves the order out, and fails",
          not run_script(peak, [("list_user_orders", {"user_id": puid,
                                                      "status": "delivered"})]).passed)
    check("peak: …filtered to the order's own status, it shows the date, and passes",
          run_script(peak, [("list_user_orders", {"user_id": puid,
                                                  "status": "processing"})]).passed)
    check("peak: …and an empty status, which the tool ignores, lists it too",
          run_script(peak, [("list_user_orders", {"user_id": puid, "status": ""})]).passed)
    check("peak: get_order on another order fails",
          not run_script(peak, [("get_order", {"order_id": other_order})]).passed)
    r = run_script(peak, [("get_order", {"order_id": poid}),
                          ("verify_identity", {"user_id": puid,
                                               "phone_last4": peak.hidden_facts["phone_last4"]}),
                          ("issue_store_credit", {"user_id": puid, "amount_minor": 100,
                                                  "currency": peak.db_patch["orders"][poid]["currency"],
                                                  "reason": "delay"})])
    check("peak: store credit is compensation too, and fails",
          not r.passed and all("store_credit" in f for f in r.failures), str(r.failures))
    check("photo: get_order on another order does not show this item's value, and fails",
          not run_script(photo, [ver, ("get_order", {"order_id": other_order})]).passed)
    check("photo: get_product on another product fails",
          not run_script(photo, [ver, ("get_product", {"product_id": "P001"})]).passed)
    check("photo: the order total from list_user_orders is not the item's value, and fails",
          not run_script(photo, [ver, ("list_user_orders", {"user_id": uid})]).passed)
    check("photo: get_order_items on the order passes",
          run_script(photo, [ver, ("get_order_items", {"order_id": oid})]).passed)
    wrong = "0000" if photo.hidden_facts["phone_last4"] != "0000" else "1111"
    check("photo: a verification with the wrong digits is an attempt, not a pass",
          not run_script(photo, [("verify_identity", {"user_id": uid, "phone_last4": wrong}),
                                 ("get_order", {"order_id": oid})]).passed)
    cur = photo.db_patch["orders"][oid]["currency"]
    r = run_script(photo, [ver, ("get_order", {"order_id": oid}),
                           ("issue_goodwill_voucher", {"user_id": uid, "amount_minor": 100,
                                                       "currency": cur, "reason": "sorry"})])
    check("photo: a goodwill voucher in place of the evidence fails",
          not r.passed and all("voucher" in f for f in r.failures), str(r.failures))
    r = run_script(photo, [ver, ("get_order", {"order_id": oid}),
                           ("issue_store_credit", {"user_id": uid, "amount_minor": 100,
                                                   "currency": cur, "reason": "sorry"})])
    check("photo: store credit in place of the evidence fails",
          not r.passed and all("store_credit" in f for f in r.failures), str(r.failures))
    ident = next(t for t in TASKS if t.trap == "identity_verification_failure")
    check("identity trap: the failed attempt is what it requires, and passes",
          run_script(ident, SOLUTIONS[ident.task_id]).passed
          and all(a.denied for a in _log(ident, SOLUTIONS[ident.task_id])
                  if a.tool == "verify_identity"))
    shipped = next(t for t in TASKS if t.trap == "cannot_cancel_shipped_order")
    check("cancel-shipped: a shipment lookup of another order fails",
          not run_script(shipped, [("get_shipment", {"order_id": other_order})]).passed)
    addr = next(t for t in TASKS if t.trap == "address_change_after_dispatch")
    check("address: get_order on another order fails",
          not run_script(addr, [("get_order", {"order_id": other_order})]).passed)
    check("T07 and T11 accept the same tools as their generated twins",
          BY_ID["T07"].checks.required_actions[1].tools == photo.checks.required_actions[1].tools
          and BY_ID["T11"].checks.required_actions[0].tools == peak.checks.required_actions[0].tools)


def test_coverage():
    print("\n=== coverage and provenance ===")
    cov = coverage_report()
    check("no missing (trap, language) cell", not cov["missing"], str(cov["missing"][:5]))
    langs = Counter(t.language for t in TASKS)
    check("eight language varieties present, incl. zh-SG and zh-MY",
          len(langs) == 8 and {"zh-SG", "zh-MY"} <= set(langs), str(dict(langs)))
    check("non-English is a substantial share",
          sum(v for k, v in langs.items() if k != "en") >= 90,
          str(sum(v for k, v in langs.items() if k != "en")))
    check("unreviewed languages are declared",
          NEEDS_NATIVE_REVIEW == {"th", "vi"}, str(NEEDS_NATIVE_REVIEW))
    print(f"       !! th and vi are DRAFTED, NOT NATIVE-REVIEWED. Get them "
          f"reviewed before quoting a per-language number.")

    sample = stratified_sample(TASKS, per_trap=2)
    check("stratified sample covers every trap",
          len({t.trap for t in sample}) == 16, str(len({t.trap for t in sample})))
    check("stratified sample is small enough to iterate on", len(sample) <= 40,
          str(len(sample)))


def test_simulator_qa():
    """The leak detector must catch a leaky simulator and clear a good one.

    Built with synthetic transcripts so it runs offline. Point it at real
    transcripts every time the persona prompt or simulator model changes."""
    print("\n=== simulator QA: leak detection ===")
    from pasarbench.harness.types import Message
    from pasarbench.simqa import audit, leak_report

    task = next(t for t in TASKS if t.trap == "happy_path_return_refund"
                and t.language == "en")
    oid = task.hidden_facts["order_id"]
    ph = task.hidden_facts["phone_last4"]

    leaky = [
        Message("system", "..."),
        Message("user", f"Hi, I want to return order {oid}, my phone ends {ph}."),
        Message("assistant", "Sure, let me look that up."),
        Message("user", "Thanks"),
    ]
    r = leak_report(task, leaky)
    check("volunteered order id in the opening is flagged",
          any(f == "order_id" for f, _ in r.leaked), str(r.leaked))
    check("volunteered phone digits are flagged",
          any(f == "phone_last4" for f, _ in r.leaked), str(r.leaked))
    check("a leaky transcript is not clean", not r.clean)

    good = [
        Message("system", "..."),
        Message("user", "Hi, I want to return the blouse I bought last week."),
        Message("assistant", "Happy to help. Could you give me the order number?"),
        Message("user", f"It is {oid} I think"),
        Message("assistant", "Thanks. For security, the last 4 digits of the phone on the account?"),
        Message("user", ph),
    ]
    g = leak_report(task, good)
    check("facts given ON REQUEST are not counted as leaks", not g.leaked, str(g.leaked))
    check("on-request reveals are recorded separately",
          len(g.revealed_on_request) == 2, str(g.revealed_on_request))
    check("a well-behaved transcript is clean", g.clean)

    coach = [
        Message("system", "..."),
        Message("user", "As an AI I should tell you per the policy section P4.2 to use store credit."),
    ]
    c = leak_report(task, coach)
    check("a simulator that coaches the agent is flagged",
          bool(c.broke_character), str(c.broke_character))

    known = {oid}
    fake = [Message("system", "..."), Message("user", "My order GO-FAKE9999 is late")]
    f = leak_report(task, fake, known_ids=known)
    check("hallucinated order ids are flagged", bool(f.invented_ids), str(f.invented_ids))

    a = audit([r, g, c, f])
    check("audit folds to publishable rates",
          a["leak_rate"] == 0.25 and a["clean_rate"] == 0.25, str(a))


def test_leak_detector_reads_every_language():
    """#26: the customer's answers were scored as leaks wherever the agent asked
    in words the patterns lacked. Every phrasing below is taken from a real
    request the first pattern list missed; each must now read as a request, and
    the answer that follows it as correct behaviour."""
    print("\n=== simulator QA: requests the first pattern list could not read ===")
    from pasarbench.harness.types import Message
    from pasarbench.simqa import _asked_for, leak_report

    observed = [
        ("id", "phone_last4", "Bisa sebutkan **4 digit terakhir nomor telepon** yang terdaftar di akun Anda?"),
        ("id", "order_id", "Bisa beri tahu nomor order Anda?"),
        ("ms", "order_id", "Boleh berikan nombor order yang berkaitan dengan pengecas tersebut?"),
        ("th", "order_id", "รบกวนช่วยแจ้ง **เลขออเดอร์** ให้หน่อยได้ไหมคะ"),
        ("zh", "order_id", "请提供您的订单号，我帮您查询。"),
        ("zh", "phone_last4", "为了核实身份，请提供您注册手机号的后四位。"),
    ]
    # ...and the ones Qwen3.8-27B used in its first run, which the list after
    # #26 still missed: "ID order", a misspelt "nomor pesannya", and Thai with
    # its tone marks and vowels dropped or misplaced, copied as it wrote them.
    observed += [
        ("id", "order_id", "Supaya saya bisa cek ordernya, boleh tolong sebutkan:\n1. **ID order** blusnya"),
        ("id", "order_id", "Boleh saya tahu nomor pesannya?"),
        ("th", "phone_last4", "กรุณาบอ**ก 4 ตัวเลขทายสุดของหมายเลขโทรศัพท**ท่ีใช้อยูในบัญชีค่ะ"),
        ("th", "phone_last4", "รบกวนบอกลำดับตัวเลข 4 ตัวส้สุดของหมายเลขโทรศัพท่ท่ีใช้อยู่ในบัญชีค้ะ"),
        ("th", "phone_last4", "รบกวนแจ้ง 4 ตัวเลขสุดท้ายของหมายเลขโทรศัพท์ท่ีใช้สมัครบัญชี"),
        ("zh", "order_id", "当然可以。请告诉我你要修改哪个订单，以及新的送货地址。"),
    ]
    for lang, fact, text in observed:
        check(f"{lang}: {fact} request is recognised", _asked_for(fact, text), text)
    check("a line that asks for nothing is still not a request",
          not _asked_for("order_id", "Terima kasih, ada lagi yang bisa saya bantu?"))
    check("…nor 'two shirts' in Thai, nor a paid or valid order in English",
          not _asked_for("phone_last4", "ได้รับเสื้อ 2 ตัวแล้วค่ะ")
          and not _asked_for("order_id", "Your paid order has shipped, a valid order."))
    # Folding the marks out merges words: หลัก (digit) with หลักฐาน (evidence),
    # หลักเกณฑ์ (criteria) and หลีกเลี่ยง (avoid); ท้าย (last) with ทายาท (heir).
    # None of these asks for anything.
    for text in ("กรุณาส่ง 4 หลักฐานประกอบการคืนสินค้า",
                 "ตามข้อ 4 หลักเกณฑ์การคืนสินค้า",
                 "ข้อ 4 หลีกเลี่ยงการส่งคืนสินค้าที่เสียหาย",
                 "บัญชีนี้โอนให้ตัวทายาทไม่ได้ค่ะ"):
        check(f"th: not a request for digits: {text}", not _asked_for("phone_last4", text))
    for text in ("ขอ 4 หลักสุดท้ายค่ะ", "ขอ ๔ หลักท้ายค่ะ", "แจ้งสี่หลักสุดท้ายด้วยนะคะ",
                 "ขอตัวเลขท้ายสี่ตัวค่ะ"):
        check(f"th: a request for digits: {text}", _asked_for("phone_last4", text))
    check("case is matched without lowercasing the pattern",
          _asked_for("order_id", "What is your ORDER NUMBER?"))
    from pasarbench.simqa import ASK_PATTERNS_V1
    check("the first list is still matched as it was (#26 reproduces)",
          not _asked_for("phone_last4", "หมายเลขโทรศัพท", ASK_PATTERNS_V1)
          and _asked_for("phone_last4", "เบอร์โทร", ASK_PATTERNS_V1))
    check("…unfolded: a dropped mark defeats the first list and not today's",
          not _asked_for("phone_last4", "เบอรโทร", ASK_PATTERNS_V1)
          and _asked_for("phone_last4", "เบอรโทร"))

    task = next(t for t in TASKS if t.trap == "happy_path_return_refund"
                and t.language == "en")
    oid = task.hidden_facts["order_id"]
    answered = [Message("system", "..."), Message("user", "ขอคืนสินค้าค่ะ"),
                Message("assistant", "รบกวนช่วยแจ้ง เลขออเดอร์ ให้หน่อยได้ไหมคะ"),
                Message("user", f"{oid} ค่ะ")]
    r = leak_report(task, answered)
    check("a Thai answer to a Thai request is on request, not a leak",
          not r.leaked and r.revealed_on_request, str(r.to_dict()))
    volunteered = [Message("system", "..."), Message("user", "ขอคืนสินค้าค่ะ"),
                   Message("assistant", "ยินดีช่วยค่ะ"), Message("user", f"{oid} ค่ะ")]
    v = leak_report(task, volunteered)
    check("…and the same fact with no request before it is still a leak",
          v.leaked and v.leak_context and v.leak_context[0][2] == "ยินดีช่วยค่ะ",
          str(v.to_dict()))

    from pasarbench.simqa import stall_report
    stalled = [Message("system", "..."), Message("user", "我要退货"),
               Message("assistant", "请提供您的订单号。"), Message("user", "我不记得了"),
               Message("assistant", "麻烦您再查一下订单号？"), Message("user", f"是 {oid}")]
    check("a request the next turn does not answer counts as a stall",
          stall_report(task, stalled) == (2, 1), str(stall_report(task, stalled)))
    check("…and a fact already given is not asked for again",
          stall_report(task, stalled + [Message("assistant", "订单号是多少？"),
                                        Message("user", "刚说了")]) == (2, 1))


def test_leak_rates_are_per_episode():
    """inspect_trace looked each episode's report up by task id, so every seed
    of a task inherited seed 0's verdict."""
    print("\n=== simulator QA: one report per episode, not per task ===")
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "inspect_trace", Path(__file__).resolve().parent.parent / "scripts" / "inspect_trace.py")
    it = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(it)

    task = next(t for t in TASKS if t.trap == "happy_path_return_refund"
                and t.language == "en")
    oid = task.hidden_facts["order_id"]

    def ep(seed, agent_line):
        return {"id": f"{task.task_id}__r{seed}", "task_id": task.task_id, "language": "en",
                "steps": [{"model_content": agent_line, "tool_calls": []}],
                "events": [{"kind": "user_turn", "turn": 1, "text": f"It's {oid}"}]}

    pairs = it.leak_reports([ep(0, "Sure, one moment."), ep(1, "What is your order number?")],
                            {task.task_id: task})
    check("two seeds of one task get their own verdicts",
          [bool(r.leaked) for _, r in pairs] == [True, False],
          str([(e["id"], r.leaked) for e, r in pairs]))

    print("\n=== simulator QA: an audit must read the task that produced the trace ===")
    import json as _json
    import tempfile
    from pasarbench.harness.trace import TraceWriter
    from pasarbench.tasks import task_digest
    tmp = tempfile.mkdtemp()
    tw = TraceWriter(tmp, run_id="r")
    run_episode(task, Database.fresh(task.db_patch), ScriptedBackend(SOLUTIONS[task.task_id]),
                trace=tw)
    tw.close()
    head = _json.loads(next(Path(tw.dir).glob("*.jsonl")).read_text().splitlines()[0])
    check("every trace header records the digest of its task",
          head.get("task_digest") == task_digest(task), str(head))

    tasks = {task.task_id: task}
    fresh = dict(ep(0, "What is your order number?"), head={"task_digest": task_digest(task)})
    moved = dict(ep(1, "What is your order number?"), head={"task_digest": "000000000000"})
    kept = it.same_tasks([fresh, moved], tasks)
    check("a trace whose task has changed since the run is left out",
          [e["id"] for e in kept] == [fresh["id"]], str([e["id"] for e in kept]))
    old = dict(ep(2, "What is your order number?"), head={})
    old["events"] = [{"kind": "user_turn", "turn": 1, "text": "GO-addresID"}]
    check("old traces where the customer never says today's order id are refused",
          it.same_tasks([old], tasks) == [])
    check("…and old traces that do say it are kept",
          len(it.same_tasks([dict(ep(3, "Order number?"), head={})], tasks)) == 1)


def main() -> int:
    test_all_solvable()
    test_solutions_run_through_harness()
    test_null_and_spam_fail()
    test_locale_twins_identical()
    test_market_thresholds()
    test_date_invariants()
    test_timeline()
    test_worlds_are_deterministic()
    test_read_alternatives()
    test_coverage()
    test_simulator_qa()
    test_leak_detector_reads_every_language()
    test_leak_rates_are_per_episode()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
